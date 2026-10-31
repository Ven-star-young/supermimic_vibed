"""CPU evaluation contracts: scalar hinge records and unreset terminal frames."""

from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from supermimic.scripts import evaluate as evaluation


class DoorEvaluationEnv:
    """A deterministic two-clip stream, without allocating a physics backend."""

    def __init__(self, fail_second_after=None):
        self.device, self.num_envs = "cpu", 2
        self.cfg = SimpleNamespace(auto_reset=False)
        self.fail_second_after = fail_second_after
        motion = SimpleNamespace(
            names=["short", "long"], clip_start=torch.tensor([0, 3]),
            clip_end=torch.tensor([2, 7]), clip_len=torch.tensor([3, 5]),
            root_p=torch.zeros(8, 3), root_R=torch.eye(3).expand(8, 3, 3),
            q=torch.zeros(8, 29), door_angle=torch.tensor([0., .1, .2, 0., .2, .4, .6, .8]),
            door_omega=torch.ones(8),
        )
        motion.door_ref_state = lambda frame: (motion.door_angle[frame], motion.door_omega[frame])
        self.command = SimpleNamespace(
            task=SimpleNamespace(max_step_reward=.1), motion=motion,
            scene=SimpleNamespace(entities={"robot": object(), "door": object()}),
            time_steps=motion.clip_start.clone(), state=self.state,
            set_reset_frames=lambda ids, frames: self.command.time_steps.copy_(frames),
        )
        self.command_manager = SimpleNamespace(get_term=lambda name: self.command)
        self.unwrapped = self

    def reset(self):
        return {}, {}

    def state(self):
        motion, frame = self.command.motion, self.command.time_steps
        phase = (frame - motion.clip_start).float()
        errors = torch.tensor([.1, .2])
        return (motion.root_p[frame], motion.root_R[frame], torch.zeros(2, 3),
                torch.zeros(2, 3), motion.q[frame], torch.zeros(2, 29),
                (motion.door_angle[frame] + phase * errors)[:, None],
                (motion.door_omega[frame] + 2 * errors)[:, None])

    def step(self, action):
        motion = self.command.motion
        self.command.time_steps.copy_(torch.minimum(self.command.time_steps + 1, motion.clip_end))
        terminated = torch.zeros(2, dtype=torch.bool)
        if self.fail_second_after is not None:
            terminated[1] = self.command.time_steps[1] - motion.clip_start[1] >= self.fail_second_after
        truncated = self.command.time_steps >= motion.clip_end
        return {}, torch.full((2,), .1), terminated, truncated, {}


@pytest.mark.parametrize("failure_step", [None, 3, 4])
def test_door_record_and_rmse_include_terminal_reference_frame(tmp_path, failure_step):
    env = DoorEvaluationEnv(failure_step)
    runner = SimpleNamespace(get_inference_policy=lambda device: lambda obs: torch.zeros(2, 29))
    path = tmp_path / "trajectory.npz"
    result = evaluation.evaluate(env, runner, record=path)
    count = 4 if failure_step is None else failure_step
    metrics = result["door_tracking"]
    assert metrics["n_reference_completed"] == (2 if failure_step is None else 1)
    assert result["n_failed"] == (0 if failure_step is None else 1)
    assert metrics["per_clip"]["short"]["observed_transitions"] == 2
    assert metrics["per_clip"]["long"]["observed_transitions"] == count
    assert metrics["per_clip"]["short"]["angle_rmse_rad"] == pytest.approx(np.sqrt((.1**2 + .2**2) / 2))
    expected_long = .2 * np.arange(1, count + 1)
    assert metrics["per_clip"]["long"]["angle_rmse_rad"] == pytest.approx(np.sqrt(np.mean(expected_long**2)))
    assert metrics["per_clip"]["long"]["reference_completed"] is (failure_step is None)
    # A last-frame failure reaches the reference end but does not count as a
    # completed reference. A prior failure keeps the planned final target.
    assert metrics["per_clip"]["long"]["ref_final_theta_rad"] == pytest.approx(.8)
    with np.load(path) as record:
        assert record["theta"].shape == record["omega"].shape == (count + 1, 2)
        assert record["ref_theta"].shape == record["ref_omega"].shape == (8,)
        assert record["root_quat"].shape == (count + 1, 2, 4)
        assert "object_p" not in record and "object_quat" not in record
        np.testing.assert_array_equal(record["f"][0], [0, 3])
        assert record["f"][-1, 1] == 3 + count
        assert record["f"][2, 0] == 2 and record["alive"][2, 0]
        assert not record["alive"][3:, 0].any()
        assert record["theta"][2, 0] == pytest.approx(.4)


def test_evaluation_rejects_auto_reset_before_losing_terminal_state():
    env = DoorEvaluationEnv()
    env.cfg.auto_reset = True
    with pytest.raises(ValueError, match="auto_reset=False"):
        evaluation.evaluate(env, SimpleNamespace())


def test_optional_contact_metrics_follow_first_episode_alive_mask():
    env, masks = DoorEvaluationEnv(), []
    runner = SimpleNamespace(get_inference_policy=lambda device: lambda obs: torch.zeros(2, 29))
    metrics = SimpleNamespace(begin_step=lambda alive: masks.append(alive.clone()),
                              report=lambda: {"per_env": [{"observed_substeps": 8}, {"observed_substeps": 16}]})
    result = evaluation.evaluate(env, runner, metrics=metrics)
    assert torch.stack(masks).tolist() == [[True, True], [True, True], [False, True], [False, True]]
    assert result["door_contact"]["per_clip"]["short"]["observed_substeps"] == 8


@pytest.mark.parametrize("single_file", [False, True])
def test_score_checkpoint_selects_door_motion_and_paired_xml_before_physics(tmp_path, monkeypatch, single_file):
    from supermimic.tasks.open_door.assets import door_xml_path

    for name in ("a", "b"):
        (tmp_path / f"{name}.npz").touch()
        (tmp_path / f"{name}.xml").write_text("<mujoco/>")
    selected = tmp_path / "b.npz" if single_file else tmp_path
    cfg = SimpleNamespace(auto_reset=True, scene=SimpleNamespace(num_envs=32), commands={
        "motion": SimpleNamespace(motion_dir="old", motion_files=(), door_xml=None),
    })
    monkeypatch.setattr(evaluation.torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(evaluation, "load_env_cfg", lambda *args, **kwargs: cfg)

    class PhysicsReached(Exception):
        pass

    def inspect_cfg(actual, *, device):
        assert not actual.auto_reset
        assert actual.scene.num_envs == (1 if single_file else 2)
        assert actual.commands["motion"].motion_dir == str(selected)
        assert door_xml_path(actual.commands["motion"]) == tmp_path / ("b.xml" if single_file else "a.xml")
        raise PhysicsReached

    monkeypatch.setattr(evaluation, "ManagerBasedRlEnv", inspect_cfg)
    args = evaluation.build_parser().parse_args([
        "--task", "Mjlab-SuperMimic-OpenDoor-Unitree-G1", "--motion-dir", str(selected),
    ])
    with pytest.raises(PhysicsReached):
        evaluation.score_checkpoint(tmp_path / "unused.pt", args)


@pytest.fixture
def tuned_door_task(tmp_path, cpu_kin):
    from motion_data import save_motion
    from supermimic.tasks.open_door.env_cfgs import make_env_cfg
    from supermimic.tasks.open_door.task import build_task
    from tests.test_open_door_motion import DOOR_XML, linear_door_clip

    path = tmp_path / "reference.npz"
    save_motion(path, linear_door_clip())
    path.with_suffix(".xml").write_text(DOOR_XML)
    cfg = make_env_cfg()
    command = cfg.commands["motion"]
    command.motion_dir = str(path)
    command.door_mass, command.door_damping = 9.5, 3.25
    command.door_friction, command.robot_door_friction = .35, .8
    command.door_stiffness, command.door_spring_reference = 14., -.07
    command.door_randomization_cfg = replace(command.door_randomization_cfg,
                                             mass=(7., 12.), stiffness=(8., 24.))
    command.door_reward_cfg = replace(command.door_reward_cfg, angle_weight=1.5, penetration_limit=.015)
    command.reward_cfg = replace(command.reward_cfg, w_root_pos=.7)
    command.observation_cfg = replace(command.observation_cfg, gravity_noise=.01)
    command.termination_cfg = replace(command.termination_cfg, height=.3)
    return build_task(device="cpu", cfg=cfg, kin=cpu_kin)


def test_score_checkpoint_restores_task_parameters_before_building_door(tuned_door_task, monkeypatch, tmp_path):
    from supermimic.tasks.open_door.env_cfgs import make_env_cfg
    from supermimic.tasks.open_door.task import build_task

    training = tuned_door_task
    metadata = training.checkpoint_metadata()
    checkpoint = {"task": metadata}
    cfg = make_env_cfg(play=True)
    command = cfg.commands["motion"]
    selected = training.env_cfg.commands["motion"].motion_dir
    # Both the selected motion and an explicit XML stay caller-owned.
    command.door_xml = str(tmp_path / "reference.xml")
    monkeypatch.setattr(evaluation.torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(evaluation, "load_env_cfg", lambda *args, **kwargs: cfg)

    class PhysicsReached(Exception):
        pass

    def inspect_cfg(actual, *, device):
        assert actual.commands["motion"] is command
        assert command.motion_dir == selected and command.door_xml == str(tmp_path / "reference.xml")
        assert command.motion_files == () and command.play
        assert asdict(command.door_reward_cfg) == metadata["door_reward"]
        assert asdict(command.reward_cfg) == metadata["robot_reward"]
        assert asdict(command.observation_cfg) == metadata["observation"]
        assert asdict(command.termination_cfg) == metadata["termination"]
        assert metadata["schema_version"] == 2
        assert asdict(command.door_randomization_cfg) == metadata["door_randomization"]
        # The existing lazy factory must see the restored command instance.
        model = actual.scene.entities["door"].spec_fn().compile()
        assert model.body("door").mass[0] == pytest.approx(9.5)
        assert model.dof_damping[0] == pytest.approx(3.25)
        restored_task = build_task(device="cpu", cfg=actual, kin=training.mdp.kin,
                                   motion=training.motion)
        restored_task.validate_checkpoint(checkpoint)
        raise PhysicsReached

    monkeypatch.setattr(evaluation, "ManagerBasedRlEnv", inspect_cfg)
    args = evaluation.build_parser().parse_args([
        "--task", "Mjlab-SuperMimic-OpenDoor-Unitree-G1", "--motion-dir", selected,
    ])
    with pytest.raises(PhysicsReached):
        evaluation.score_checkpoint(tmp_path / "unused.pt", args)
    assert checkpoint["task"] == metadata


def test_restore_does_not_relax_schema_or_change_other_tasks(tuned_door_task):
    from supermimic.tasks.g1_tracking.env_cfgs import make_env_cfg as tracking_cfg
    from supermimic.tasks.open_door.env_cfgs import make_env_cfg
    from supermimic.tasks.open_door.task import build_task

    training = tuned_door_task
    checkpoint = {"task": training.checkpoint_metadata()}
    baseline = tracking_cfg(play=True)
    old_reward = asdict(baseline.commands["motion"].reward_cfg)
    evaluation._restore_open_door_config(baseline, checkpoint)
    assert asdict(baseline.commands["motion"].reward_cfg) == old_reward

    cfg = make_env_cfg(play=True)
    corrupted = deepcopy(checkpoint)
    corrupted["task"]["dimensions"]["state"] += 1
    evaluation._restore_open_door_config(cfg, corrupted)
    restored = build_task(device="cpu", cfg=cfg, kin=training.mdp.kin, motion=training.motion)
    with pytest.raises(ValueError, match="schema mismatch"):
        restored.validate_checkpoint(corrupted)

    corrupted = deepcopy(checkpoint)
    corrupted["task"]["door_physics"]["door_xml"] = "/unexpected/path.xml"
    with pytest.raises(ValueError, match="door_physics fields"):
        evaluation._restore_open_door_config(cfg, corrupted)
    assert cfg.commands["motion"].door_xml is None


@pytest.mark.parametrize("field,value", [("mass", (6., 12.)), ("enabled", False)])
def test_checkpoint_rejects_changed_door_distribution_on_resume(tuned_door_task, field, value):
    checkpoint = {"task": tuned_door_task.checkpoint_metadata()}
    checkpoint["task"]["door_randomization"][field] = value
    with pytest.raises(ValueError, match="schema mismatch"):
        tuned_door_task.validate_checkpoint(checkpoint)


def test_checkpoint_accepts_json_ranges_but_not_missing_fields(tuned_door_task):
    import json

    checkpoint = {"task": json.loads(json.dumps(tuned_door_task.checkpoint_metadata()))}
    assert isinstance(checkpoint["task"]["door_randomization"]["mass"], list)
    tuned_door_task.validate_checkpoint(checkpoint)
    del checkpoint["task"]["door_randomization"]["damping"]
    with pytest.raises(ValueError, match="Invalid OpenDoor checkpoint door_randomization"):
        tuned_door_task.validate_checkpoint(checkpoint)


def test_restore_rejects_legacy_and_invalid_distribution_before_mutation(tuned_door_task):
    from supermimic.tasks.open_door.env_cfgs import make_env_cfg

    cfg = make_env_cfg(play=True)
    command = cfg.commands["motion"]
    initial_physics = command.door_mass
    initial_distribution = asdict(command.door_randomization_cfg)
    checkpoint = {"task": tuned_door_task.checkpoint_metadata()}
    checkpoint["task"]["door_randomization"]["mass"] = (12., 7.)
    with pytest.raises(ValueError):
        evaluation._restore_open_door_config(cfg, checkpoint)
    assert command.door_mass == initial_physics
    assert asdict(command.door_randomization_cfg) == initial_distribution

    checkpoint = {"task": tuned_door_task.checkpoint_metadata()}
    checkpoint["task"]["schema_version"] = 1
    del checkpoint["task"]["door_randomization"]
    with pytest.raises(ValueError, match="schema mismatch"):
        evaluation._restore_open_door_config(cfg, checkpoint)
    with pytest.raises(ValueError, match="schema mismatch"):
        tuned_door_task.validate_checkpoint(checkpoint)
    assert command.door_mass == initial_physics
    assert asdict(command.door_randomization_cfg) == initial_distribution
