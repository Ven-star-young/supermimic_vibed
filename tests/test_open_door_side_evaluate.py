"""Side-task checkpoint restoration, reference selection and measured contacts."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, replace
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motion_data import save_motion
from supermimic.scripts.evaluate import _restore_open_door_config
from supermimic.tasks.open_door_side.assets import select_motion_files
from supermimic.tasks.open_door_side.contacts import ALLOWED_BODIES, SIDE_CONTACT_SCHEMA
from supermimic.tasks.open_door_side.env_cfgs import make_env_cfg
from supermimic.tasks.open_door_side.evaluation import SideDoorContactMetrics
from supermimic.tasks.open_door_side.mdp import SideDoorRewardCfg
from supermimic.tasks.open_door_side.task import build_task
from tests.test_open_door_motion import DOOR_XML, linear_door_clip


def _reference(path, region="right_side"):
    clip = linear_door_clip()
    if region is not None:
        clip["door_contact_region"] = region
    save_motion(path, clip)
    path.with_suffix(".xml").write_text(DOOR_XML)
    return path


@pytest.fixture
def tuned_side_task(tmp_path, cpu_kin):
    cfg = make_env_cfg()
    command = cfg.commands["motion"]
    command.motion_dir = str(_reference(tmp_path / "right_side.npz"))
    command.door_mass, command.door_damping = 9.5, 3.25
    command.door_randomization_cfg = replace(command.door_randomization_cfg,
                                            mass=(7., 12.), stiffness=(8., 24.))
    command.door_reward_cfg = replace(command.door_reward_cfg, contact_weight=.4,
                                      reference_contact_std=.07,
                                      contact_penetration_tolerance=.004)
    command.reward_cfg = replace(command.reward_cfg, w_root_pos=.7)
    command.observation_cfg = replace(command.observation_cfg, gravity_noise=.01)
    command.termination_cfg = replace(command.termination_cfg, height=.3)
    return build_task(device="cpu", cfg=cfg, kin=cpu_kin)


def test_side_checkpoint_restores_reward_and_physics_without_replacing_factory_command(tuned_side_task):
    training = tuned_side_task
    checkpoint = {"task": training.checkpoint_metadata()}
    original_checkpoint = deepcopy(checkpoint)
    cfg = make_env_cfg(play=True)
    command = cfg.commands["motion"]
    command.motion_dir = training.env_cfg.commands["motion"].motion_dir
    command.door_xml = command.motion_dir.replace(".npz", ".xml")
    selected = (command.motion_dir, command.door_xml)
    _restore_open_door_config(cfg, checkpoint)

    assert cfg.commands["motion"] is command
    assert cfg.scene.entities["door"].spec_fn.args[0] is command
    assert (command.motion_dir, command.door_xml) == selected and command.play
    assert isinstance(command.door_reward_cfg, SideDoorRewardCfg)
    for saved, field in (("door_reward", "door_reward_cfg"),
                         ("door_randomization", "door_randomization_cfg"),
                         ("robot_reward", "reward_cfg"),
                         ("observation", "observation_cfg"),
                         ("termination", "termination_cfg")):
        assert asdict(getattr(command, field)) == checkpoint["task"][saved]
    model = cfg.scene.entities["door"].spec_fn().compile()
    assert model.body("door").mass[0] == pytest.approx(9.5)
    assert model.dof_damping[0] == pytest.approx(3.25)
    restored = build_task(device="cpu", cfg=cfg, kin=training.mdp.kin, motion=training.motion)
    restored.validate_checkpoint(checkpoint)
    assert checkpoint == original_checkpoint


def test_back_and_side_checkpoints_cannot_restore_into_each_other(tuned_side_task):
    from supermimic.tasks.open_door.env_cfgs import make_env_cfg as make_back_cfg
    from supermimic.tasks.open_door.task import build_task as build_back_task

    side = tuned_side_task
    back_cfg = make_back_cfg(play=True)
    back = build_back_task(device="cpu", cfg=back_cfg, kin=side.mdp.kin, motion=side.motion)
    with pytest.raises(ValueError, match="task mismatch"):
        _restore_open_door_config(back_cfg, {"task": side.checkpoint_metadata()})
    with pytest.raises(ValueError, match="task mismatch"):
        _restore_open_door_config(make_env_cfg(play=True), {"task": back.checkpoint_metadata()})
    with pytest.raises(ValueError, match="schema mismatch"):
        side.validate_checkpoint({"task": back.checkpoint_metadata()})
    with pytest.raises(ValueError, match="schema mismatch"):
        back.validate_checkpoint({"task": side.checkpoint_metadata()})


def test_side_selection_accepts_labelled_single_file_and_explicit_subset(tmp_path):
    right = _reference(tmp_path / "right.npz")
    _reference(tmp_path / "back.npz", "back")
    cfg = SimpleNamespace(motion_dir=str(right), motion_files=())
    assert select_motion_files(cfg) == [right]
    cfg.motion_dir, cfg.motion_files = str(tmp_path), (str(right),)
    assert select_motion_files(cfg) == [right]


@pytest.mark.parametrize("region", ["back", "left_side", None, ["right_side"]])
def test_side_selection_rejects_wrong_or_nonscalar_contact_labels(tmp_path, region):
    path = _reference(tmp_path / "wrong.npz", region)
    cfg = SimpleNamespace(motion_dir=str(path), motion_files=())
    with pytest.raises(ValueError, match="door_contact_region='right_side'"):
        select_motion_files(cfg)


def test_side_selection_rejects_a_directory_mixing_back_and_side(tmp_path):
    _reference(tmp_path / "right.npz")
    _reference(tmp_path / "back.npz", "back")
    cfg = SimpleNamespace(motion_dir=str(tmp_path), motion_files=())
    with pytest.raises(ValueError, match="back or unlabelled clips cannot be mixed"):
        select_motion_files(cfg)


@pytest.fixture
def side_contact_native(monkeypatch):
    body_names = ("torso_link", "logo_link", *ALLOWED_BODIES, "right_hand", "right_knee_link")
    body_ids = {f"robot/{name}": i + 1 for i, name in enumerate(body_names)}
    model = SimpleNamespace(ngeom=10, geom_bodyid=np.arange(1, 11),
                            body=lambda name: SimpleNamespace(id=body_ids[name]),
                            geom=lambda name: SimpleNamespace(id=8))
    xpos = torch.zeros(2, 11, 3)
    xmat = torch.eye(3).expand(2, 11, 3, 3).clone()
    xpos[1, 1] = torch.tensor([2., 3., 1.])
    xmat[1, 1] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    local = torch.tensor([[0., -.08, .2]] * 7)
    local[4] = torch.tensor([-.05, 0., .2])  # old back patch must be forbidden
    worlds = torch.tensor([0, 0, 0, 1, 1, 0, 0])
    points = (xmat[worlds, 1] @ local[..., None]).squeeze(-1) + xpos[worlds, 1]
    contacts = SimpleNamespace(
        geom=torch.tensor([[8, 2], [4, 8], [6, 8], [0, 8], [8, 0], [8, 2], [9, 0]]),
        worldid=worlds, pos=points, efc_address=torch.tensor([[0]] * 5 + [[-1], [0]]),
    )
    data = SimpleNamespace(overflow=torch.zeros(2, dtype=torch.int32), nacon=torch.tensor([7]),
                           contact=contacts, xpos=xpos, xmat=xmat)
    forces = torch.tensor([.6, .6, 1.1, 1.2, 2., 100., 100.])
    events = []

    def contact_force(model, data, ids, to_world, output):
        assert not to_world
        events.append("contact_force")
        output[:, 0] = forces[ids.long()]

    monkeypatch.setitem(sys.modules, "mujoco_warp", SimpleNamespace(contact_force=contact_force))
    monkeypatch.setitem(sys.modules, "warp", SimpleNamespace(
        ScopedDevice=lambda device: nullcontext(), from_torch=lambda value, **kwargs: value,
        spatial_vector=object(),
    ))
    sim = SimpleNamespace(mj_model=model, data=data, step=lambda: events.append("physics"),
                          wp_device="cpu", wp_model=None, wp_data=None)
    native = SimpleNamespace(sim=sim, device="cpu", num_envs=2,
                              scene={"robot": SimpleNamespace(indexing=SimpleNamespace(
                                  geom_ids=torch.arange(8)))})
    return native, events, forces


def test_side_contact_classification_matches_patch_and_limb_definition(side_contact_native):
    native, _, _ = side_contact_native
    with SideDoorContactMetrics(native) as metrics:
        # All approved shoulder/forearm shells remain legal regardless of torso
        # location; hand and leg shells cannot become legal by entering the patch.
        local = torch.tensor([[0., -.08, .2]] * 10)
        local[8] = torch.tensor([-.05, 0., .2])
        local[9] = torch.tensor([0., -.04, .2])  # strict torso boundary
        geom = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 0])
        assert metrics.classify_contacts(local, geom).tolist() == [True] * 6 + [False] * 4


def test_side_force_collector_reports_side_regions_in_rotated_worlds(side_contact_native):
    native, events, forces = side_contact_native
    original_step = native.sim.step
    with SideDoorContactMetrics(native) as metrics:
        metrics.begin_step(torch.tensor([True, False]))
        native.sim.step()
        assert events[-2:] == ["physics", "contact_force"]
        assert metrics.contact_steps.tolist() == [[1, 1], [0, 0]]
        forces[:3] = 0.
        metrics.begin_step(torch.tensor([True, True]))
        native.sim.step()
        report = metrics.report()
        assert report["allowed_side_substep_rate"] == pytest.approx(2 / 3)
        assert report["forbidden_door_substep_rate"] == pytest.approx(2 / 3)
        assert report["n_episodes_with_forbidden_contact"] == 2
        assert report["per_env"][0]["allowed_side_substep_rate"] == .5
        assert report["per_env"][1]["allowed_side_substep_rate"] == 1.
        assert "allowed_back_substep_rate" not in report
        assert "back_region_torso_m" not in report
        assert report["side_contact_region"] == SIDE_CONTACT_SCHEMA
        report["side_contact_region"]["allowed_bodies"].clear()
        assert metrics.report()["side_contact_region"]["allowed_bodies"] == list(ALLOWED_BODIES)
    assert native.sim.step is original_step


def test_back_builder_rejects_mixed_side_directory_but_accepts_legacy_back_subset(tmp_path, cpu_kin):
    from supermimic.tasks.open_door.env_cfgs import make_env_cfg as make_back_cfg
    from supermimic.tasks.open_door.task import build_task as build_back_task

    legacy = _reference(tmp_path / "legacy.npz", None)
    back = _reference(tmp_path / "back.npz", "back")
    cfg = make_back_cfg(play=True)
    cfg.commands["motion"].motion_dir = str(tmp_path)
    assert len(build_back_task(device="cpu", cfg=cfg, kin=cpu_kin).motion.clip_len) == 2

    right = _reference(tmp_path / "right.npz", "right_side")
    with pytest.raises(ValueError, match="back-opening task cannot load a different door_contact_region"):
        build_back_task(device="cpu", cfg=cfg, kin=cpu_kin)

    cfg.commands["motion"].motion_files = (str(legacy), str(back))
    assert len(build_back_task(device="cpu", cfg=cfg, kin=cpu_kin).motion.clip_len) == 2

    side_cfg = make_env_cfg(play=True)
    side_cfg.commands["motion"].motion_dir = str(tmp_path)
    side_cfg.commands["motion"].motion_files = (str(right),)
    side_task = build_task(device="cpu", cfg=side_cfg, kin=cpu_kin)
    assert side_task.name == "g1_open_door_side" and len(side_task.motion.clip_len) == 1
