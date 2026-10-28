"""Reference selection and real native clean/perturbed policy initialization."""
from dataclasses import asdict
import importlib.metadata
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motion_data import load_motion, save_motion
from motion_data.chair_split import create_split


@pytest.fixture
def deployment_data(tmp_path):
    import trimesh
    from supermimic.assets.g1.g1_constants import G1_JOINT_NAMES, default_joint_pos_list

    root = tmp_path / "motions"
    time = np.arange(60) / 25.
    for index in range(3):
        mesh = trimesh.creation.box(extents=[.4 + .1 * index, .4, .5])
        angle = .1 * time
        save_motion(root / f"sitting__chair_{index}__000.npz", dict(
            fps=25., joint_names=G1_JOINT_NAMES,
            root_pos=np.stack((.1 * time, time * 0, .85 + time * 0), -1),
            root_quat=np.stack((np.cos(angle / 2), time * 0, time * 0, np.sin(angle / 2)), -1),
            joint_pos=np.tile(default_joint_pos_list(), (len(time), 1))
                      + .01 * np.sin(time[:, None]),
            object_pos=np.stack((1.5 + .02 * time, time * 0, .25 + time * 0), -1),
            object_quat=np.stack((np.cos(angle / 2), time * 0, time * 0, np.sin(angle / 2)), -1),
            object_scale=np.ones(3), object_vertices=np.asarray(mesh.vertices),
            object_faces=np.asarray(mesh.faces),
        ))
    manifest = create_split(root, tmp_path / "split.json", seed=0)
    return root, manifest


def _args(*values):
    from supermimic.scripts.play_policy import build_parser
    return build_parser().parse_args(["--checkpoint", "model.pt", *values])


def test_reference_selection_defaults_to_train_and_explicit_reference_can_be_heldout(deployment_data):
    from supermimic.scripts.play_policy import resolve_references
    root, manifest = deployment_data
    expected = sorted(root / c["path"] for group in manifest["train"] for c in group["clips"])
    assert resolve_references(_args(), manifest, root) == expected
    heldout = root / manifest["test"][0]["clips"][0]["path"]
    assert resolve_references(_args("--partition", "test"), manifest, root) == [heldout]
    assert resolve_references(_args("--partition", "all"), manifest, root) == sorted([*expected, heldout])
    assert resolve_references(_args("--reference", str(heldout)), manifest, root) == [heldout]


def test_selection_detects_modified_reference_without_opening_unselected_partition(deployment_data):
    from supermimic.scripts.play_policy import resolve_references
    root, manifest = deployment_data
    heldout = root / manifest["test"][0]["clips"][0]["path"]
    motion = load_motion(heldout)
    save_motion(heldout, {**motion, "root_pos": motion["root_pos"] + .1})
    assert len(resolve_references(_args(), manifest, root)) == len(manifest["train"])
    with pytest.raises(ValueError, match="(?i)changed|hash|digest"):
        resolve_references(_args("--partition", "test"), manifest, root)
    with pytest.raises(ValueError, match="(?i)changed|hash|digest"):
        resolve_references(_args("--reference", str(heldout)), manifest, root)


def test_user_reference_manifest_loads_only_the_selected_clip(deployment_data):
    from supermimic.scripts.play_policy import resolve_references
    root, manifest = deployment_data
    path = root / "sitting__chair_0__000.npz"
    selection = root / "manifest.json"
    selection.write_text(json.dumps({"version": 1, "references": {"chair_0": path.name}}))
    # Other references can be unavailable when deploying an explicit selection.
    for unselected in root.glob("*.npz"):
        if unselected != path:
            unselected.unlink()
    assert resolve_references(_args("--references", str(selection)), manifest, root) == [path]


def _checkpoint():
    from supermimic.tasks.sitting.mdp import ObjectRewardCfg
    return {
        "task": {"object_reward": asdict(ObjectRewardCfg())},
        "policy_config": {"history_length": 4, "goal_frames": 3},
    }


def test_deployment_uses_checkpoint_contract_and_fixed_evaluation_physics(deployment_data, tmp_path):
    from supermimic.scripts.play_policy import DeploymentCommandCfg, make_deployment_cfg
    root, manifest = deployment_data
    reference = root / manifest["train"][0]["clips"][0]["path"]
    args = _args("--init", "perturbed", "--start-frame", "3", "--cache-dir", str(tmp_path))
    cfg = make_deployment_cfg(_checkpoint(), reference, args)
    command = cfg.commands["motion"]
    assert isinstance(command, DeploymentCommandCfg)
    assert cfg.scene.num_envs == 1 and not cfg.auto_reset
    assert command.play
    assert command.history_length == 4 and command.goal_frames == 3
    assert command.object_physical_cfg.evaluation_mass == 10.
    assert command.object_physical_cfg.evaluation_robot_friction == 1.
    assert command.collision_cache_dir == str(tmp_path)


def _prepare_box_cache(reference, cache, monkeypatch, profile="default"):
    from motion_data import collision
    original = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda name: collision.COACD_VERSION if name == "coacd" else original(name))
    # Only the offline decomposition is replaced: the fixture geometry already
    # is convex. Native MuJoCo/Warp scene construction and resets remain real.
    monkeypatch.setitem(sys.modules, "coacd", SimpleNamespace(set_log_level=lambda _: None,
        Mesh=lambda vertices, faces: (vertices, faces), run_coacd=lambda mesh, **_: [mesh]))
    collision.prepare_collision(load_motion(reference), cache, profile=profile)


def test_compact_deployment_accepts_only_compact_cache(deployment_data, tmp_path, monkeypatch, capsys):
    from motion_data.collision import collision_key
    from supermimic.scripts import play_policy as deploy_policy

    root, manifest = deployment_data
    reference = root / manifest["train"][0]["clips"][0]["path"]
    cache = tmp_path / "compact_cache"
    _prepare_box_cache(reference, cache, monkeypatch, profile="compact")
    assert not (cache / f"{collision_key(load_motion(reference))}.npz").exists()
    checkpoint = _checkpoint()
    checkpoint["infos"] = {"sitting_experiment": {"collision_profile": "compact"}}

    class PhysicsReached(Exception):
        pass

    def inspect_physics(cfg, *, device):
        assert cfg.commands["motion"].collision_profile == "compact"
        raise PhysicsReached

    monkeypatch.setattr(deploy_policy, "ManagerBasedRlEnv", inspect_physics)
    with pytest.raises(PhysicsReached):
        deploy_policy.run_reference(checkpoint, manifest, reference, _args("--cache-dir", str(cache)))
    identity = json.loads(capsys.readouterr().out.split("[deploy] ")[-1])
    assert identity["collision_profile"] == "compact"


def test_compact_deployment_rejects_default_only_cache_before_physics(deployment_data, tmp_path, monkeypatch):
    from supermimic.scripts import play_policy as deploy_policy

    root, manifest = deployment_data
    reference = root / manifest["train"][0]["clips"][0]["path"]
    cache = tmp_path / "default_cache"
    _prepare_box_cache(reference, cache, monkeypatch)
    checkpoint = _checkpoint()
    checkpoint["infos"] = {"sitting_experiment": {"collision_profile": "compact"}}

    def unexpected_physics(*args, **kwargs):
        pytest.fail("Missing compact cache must fail before physics allocation")

    monkeypatch.setattr(deploy_policy, "ManagerBasedRlEnv", unexpected_physics)
    with pytest.raises(FileNotFoundError, match="--profile compact"):
        deploy_policy.run_reference(checkpoint, manifest, reference, _args("--cache-dir", str(cache)))


def test_native_clean_perturbed_reset_and_history_are_reproducible(
    deployment_data, tmp_path, cuda_device, monkeypatch,
):
    from mjlab.envs import ManagerBasedRlEnv
    from supermimic.scripts.play_policy import DeploymentEnv, make_deployment_cfg
    root, manifest = deployment_data
    reference = root / manifest["train"][0]["clips"][0]["path"]
    cache = tmp_path / "cache"
    _prepare_box_cache(reference, cache, monkeypatch)
    args = _args("--start-frame", "3", "--cache-dir", str(cache))
    cfg = make_deployment_cfg(_checkpoint(), reference, args)
    env = ManagerBasedRlEnv(cfg, device=cuda_device)
    try:
        command = env.command_manager.get_term("motion")

        def reset(seed):
            observations, _ = env.reset(seed=seed)
            actual = tuple(value.clone() for value in command.state())
            expected = command.motion.full_ref_state(command.time_steps)
            # Histories must be backfilled with actual post-reset measurements,
            # including perturbations, without training observation noise.
            frame = command.mdp.obs(actual, env.action_manager.action,
                                    command.xi, command.time_steps)
            history = observations["policy_history"].reshape(1, 4, -1)
            torch.testing.assert_close(history, frame[:, None].expand_as(history))
            torch.testing.assert_close(command.xi[:, 33], command.xi.new_tensor([10.]))
            torch.testing.assert_close(command.xi[:, 43], command.xi.new_tensor([1.]))
            assert command.xi[:, 1:33].count_nonzero() == 0
            return actual, expected

        clean, reference_state = reset(17)
        assert command.time_steps.item() == 3
        for actual, expected in zip(clean, reference_state):
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        # The fixture includes nonzero robot/object velocities, so matching the
        # reference checks velocity initialization as well as the visible pose.
        assert clean[2].abs().sum() > 0 and clean[8].abs().sum() > 0

        command.cfg.init_mode = "perturbed"
        perturbed, _ = reset(17)
        repeated, _ = reset(17)
        for first, second in zip(perturbed, repeated):
            torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)
        assert not torch.allclose(perturbed[0], clean[0])
        assert not torch.allclose(perturbed[4], clean[4])
        changed_seed, _ = reset(18)
        assert not torch.allclose(perturbed[0], changed_seed[0])

        command.cfg.noise_scale = 0.
        zero_noise, _ = reset(17)
        for actual, expected in zip(zero_noise, clean):
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)

        command.cfg.random_start = True
        random_state, random_reference = reset(27)
        random_frame = command.time_steps.item()
        assert 0 <= random_frame < int(command.motion.clip_len[0]) - 1
        for actual, expected in zip(random_state, random_reference):
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        reset(27)
        assert command.time_steps.item() == random_frame
        observed_frames = set()
        for seed in (28, 29, 30):
            reset(seed)
            observed_frames.add(command.time_steps.item())
        assert len(observed_frames) > 1

        command.cfg.random_start = False
        command.cfg.start_frame = 3
        wrapped = DeploymentEnv(env, episodes=2, max_steps=1)
        actions = torch.zeros((1, 29), device=cuda_device)
        first = wrapped.step(actions)
        assert len(wrapped.records) == 1 and not wrapped.finished
        assert first[0]["frame"].item() == command.time_steps.item() == 3
        wrapped.step(actions)
        assert wrapped.finished and len(wrapped.records) == 2
        assert all(record["steps"] == 1 for record in wrapped.records)
        # A closed finite rollout must not step terminal physics again.
        wrapped.step(actions)
        assert len(wrapped.records) == 2

        command.cfg.start_frame = int(command.motion.clip_len[0]) - 1
        with pytest.raises(ValueError, match="start-frame"):
            env.reset(seed=17)
    finally:
        env.close()


def test_frozen_actor_runs_heldout_geometry_and_rejects_old_schema(
    deployment_data, tmp_path, cuda_device, monkeypatch,
):
    from dataclasses import asdict
    from mjlab.envs import ManagerBasedRlEnv
    from supermimic.rl.config import SuperMimicRunnerCfg
    from supermimic.rl.runner import SuperMimicRunner
    from supermimic.scripts.play_policy import (
        DeploymentEnv, load_actor, make_deployment_cfg, policy_digest,
    )

    root, manifest = deployment_data
    reference = root / manifest["test"][0]["clips"][0]["path"]
    cache = tmp_path / "cache"
    _prepare_box_cache(reference, cache, monkeypatch)
    cfg = make_deployment_cfg(_checkpoint(), reference, _args("--cache-dir", str(cache)))
    native = ManagerBasedRlEnv(cfg, device=cuda_device)
    try:
        env = DeploymentEnv(native, episodes=1, max_steps=2)
        runner = SuperMimicRunner(env, asdict(SuperMimicRunnerCfg()) | dict(amp="none", no_compile=True),
                                 device=cuda_device)
        state = runner.pi.state_dict()
        checkpoint = dict(task=runner.task.checkpoint_metadata(), policy_config=runner.policy_config,
            pi=state, wm={key: value for key, value in state.items() if key.startswith("geometry_encoder.")},
            # Training-bank metadata is deliberately unrelated to this new scene.
            geometry_bank={"keys": ["training-chair"]},
            infos={"sitting_experiment": {"split_fingerprint": manifest["fingerprint"],
                   "visited_train_shapes": [group["shape_key"] for group in manifest["train"]]}})
        scene_bank = runner.pi.geometry_bank
        load_actor(runner, checkpoint, manifest)
        assert runner.pi.geometry_bank is scene_bank
        before = policy_digest(runner.pi)
        with torch.inference_mode():
            while not env.finished:
                env.step(runner.get_inference_policy()(env.get_observations()))
        assert runner.wm is runner.critic is runner.buffer is None
        assert before == policy_digest(runner.pi)
        assert len(env.records) == 1 and env.records[0]["steps"] == 2
        old = {**checkpoint, "task": {**checkpoint["task"], "schema_version": 3}}
        with pytest.raises(ValueError, match="schema mismatch"):
            load_actor(runner, old, manifest)
    finally:
        native.close()


def test_success_requires_reference_end_and_failure_has_precedence():
    from supermimic.scripts.play_policy import classify_outcome, validate_arguments
    assert classify_outcome(failed=False, reference_end=False) is None
    assert classify_outcome(failed=False, reference_end=True, step_limit=True) == "reference_complete"
    assert classify_outcome(failed=True, reference_end=True) == "tracking_failure"
    assert classify_outcome(failed=False, reference_end=True, timed_out=True) == "timeout"
    assert classify_outcome(failed=False, reference_end=False, step_limit=True) == "step_limit"
    with pytest.raises(ValueError, match="frame 0"):
        validate_arguments(_args("--evaluate", "--random-start"))
    with pytest.raises(ValueError, match="distinct"):
        validate_arguments(_args("--evaluate", "--seeds", "7", "7"))
    args = _args("--evaluate", "--seeds", "19", "83")
    validate_arguments(args)
    assert args.viewer == "none" and args.all and args.episodes == 2


def test_episode_seeds_do_not_depend_on_previous_episode_length(
    deployment_data, tmp_path, cuda_device, monkeypatch,
):
    from mjlab.envs import ManagerBasedRlEnv
    from supermimic.scripts.play_policy import DeploymentEnv, make_deployment_cfg
    root, manifest = deployment_data
    reference = root / manifest["train"][0]["clips"][0]["path"]
    cache = tmp_path / "cache"
    _prepare_box_cache(reference, cache, monkeypatch)
    args = _args("--init", "perturbed", "--cache-dir", str(cache))
    native = ManagerBasedRlEnv(make_deployment_cfg(_checkpoint(), reference, args), device=cuda_device)
    try:
        starts = []
        for steps in (1, 3):
            wrapped = DeploymentEnv(native, episodes=2, max_steps=steps, seeds=[7, 83])
            try:
                while not wrapped.finished:
                    wrapped.step(torch.zeros(1, 29, device=cuda_device))
                assert [record["seed"] for record in wrapped.records] == [7, 83]
                assert all(not record["full_reference_success"] for record in wrapped.records)
                assert all(record["outcome"] == "step_limit" for record in wrapped.records)
                assert all(record["stable_contact_eligible"] for record in wrapped.records)
                starts.append(wrapped.records[1]["initial"]["state"])
            finally:
                wrapped.close_metrics()
        torch.testing.assert_close(torch.tensor(starts[0]), torch.tensor(starts[1]), atol=1e-6, rtol=1e-6)

        wrapped = DeploymentEnv(native, episodes=1, seeds=[29])
        try:
            wrapped.interrupt()
            assert wrapped.records[0]["steps"] == 0
            assert wrapped.records[0]["outcome"] == "interrupted"
            assert not wrapped.records[0]["full_reference_success"]
        finally:
            wrapped.close_metrics()
    finally:
        native.close()


def test_interrupted_batch_report_keeps_all_planned_references_and_seeds(deployment_data, tmp_path, monkeypatch):
    from supermimic.scripts import play_policy as module
    root, manifest = deployment_data
    checkpoint = tmp_path / "model.pt"
    torch.save(dict(iteration=10, infos={"sitting_experiment": {
        "split_fingerprint": manifest["fingerprint"]}}), checkpoint)
    output = tmp_path / "results.json"
    def identity(checkpoint, manifest, reference, args):
        group = next(g for p in ("train", "test") for g in manifest[p]
                     if any(c["path"] == reference.name for c in g["clips"]))
        return dict(reference=str(reference), sha256="fixture", shape_key=group["shape_key"],
                    geometry_partition="train", reference_partition="train", visited_in_checkpoint=True)

    def interrupted(checkpoint, manifest, reference, args, *, on_record, metrics_cfg):
        assert args.trial_seeds == [3, 9] and args.init_mode == "clean"
        report = json.loads(output.read_text())
        assert len(report["references"]) == len(manifest["train"])
        assert all(set(r["conditions"]) == {"clean", "perturbed"} for r in report["references"])
        on_record(dict(episode=1, seed=3, condition="clean", initial={"frame": 0},
            valid=True, outcome="reference_complete", reference_completed=True,
            full_reference_eligible=True, full_reference_success=True,
            stable_contact_eligible=True, stable_contact_success=False))
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "reference_identity", identity)
    monkeypatch.setattr(module, "run_reference", interrupted)
    with pytest.raises(KeyboardInterrupt):
        module.main(["--checkpoint", str(checkpoint), "--split", str(tmp_path / "split.json"),
                     "--evaluate", "--seeds", "3", "9", "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["status"] == "interrupted"
    clean = report["summary"]["task"]["conditions"]["clean"]
    assert clean["planned_trials"] == 2 * len(manifest["train"])
    assert clean["executed_trials"] == clean["full_reference_successes"] == 1
    assert clean["unexecuted_trials"] == clean["planned_trials"] - 1
