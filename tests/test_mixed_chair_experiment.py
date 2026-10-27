"""Simultaneous native chair variants keep references, geometry and Replay aligned."""

from dataclasses import asdict
from copy import deepcopy
import gc

import numpy as np
import pytest
import torch

from test_chair_experiment import chair_dataset, _prepare_boxes, _small_models
from motion_data import load_motion
from motion_data.collision import collision_key, prepare_collision
from motion_data.chair_split import create_split, scenes
from supermimic.tasks.entities.mixed import balanced_assignment
from supermimic.tasks.interaction.assets import geometry_key


def test_mixed_assignment_is_fixed_balanced_and_seeded():
    first = balanced_assignment(512, num_scenes=83, seed=0)
    assert first == balanced_assignment(512, num_scenes=83, seed=0)
    assert first != balanced_assignment(512, num_scenes=83, seed=1)
    counts = np.bincount(first, minlength=83)
    assert counts.min() == 6 and counts.max() == 7 and counts.sum() == 512
    assert sorted(balanced_assignment(4, num_scenes=4, seed=0)) == list(range(4))
    with pytest.raises(ValueError, match="at least"):
        balanced_assignment(3, num_scenes=4, seed=0)


def _assert_membership(command, frames, geometry_ids):
    expected_scene = command.env_scene.expand_as(frames)
    torch.testing.assert_close(command.clip_scene[command.motion.clip_id[frames]], expected_scene)
    expected_ids = torch.tensor(command.scene_geometry_ids, device=frames.device)[expected_scene]
    torch.testing.assert_close(geometry_ids, expected_ids)


@pytest.mark.parametrize("collision_profile", ["default", "compact"])
def test_mixed_native_resets_replay_training_resume_and_heldout(
    chair_dataset, tmp_path, cuda_device, monkeypatch, collision_profile,
):
    from supermimic.rl.mixed_chair_experiment import MixedChairExperimentRunner
    from supermimic.rl.config import SuperMimicRunnerCfg
    from supermimic.scripts.chair_experiment import evaluate_split
    from tools._smoke import reset_frames

    split = create_split(chair_dataset, tmp_path / "split.json")
    cache = tmp_path / "cache"
    _prepare_boxes(chair_dataset, cache, monkeypatch)
    if collision_profile == "compact":
        for path in chair_dataset.glob("*.npz"):
            clip = load_motion(path)
            prepare_collision(clip, cache, profile="compact")
            # No default fallback is available in this test: both training
            # and held-out evaluation must honor checkpoint collision provenance.
            (cache / (collision_key(clip) + ".npz")).unlink(missing_ok=True)
    _small_models(monkeypatch)
    cfg = asdict(SuperMimicRunnerCfg(buffer_steps=16, utd=1., horizon=2, wm_window=2,
        wm_batches=1, wm_batch=4, pi_batch=4, warmup_steps=0, wm_only_iters=0,
        critic_epochs=1, critic_minibatches=1, no_compile=True, amp="none"))
    run = tmp_path / "mixed"
    torch.manual_seed(0)
    runner = MixedChairExperimentRunner(split, chair_dataset, cache, cfg,
        num_envs=8, max_objects=2, log_dir=run, device=cuda_device,
        collision_profile=collision_profile)
    try:
        native = runner.env.unwrapped
        command = native.command_manager.get_term("motion")
        replay = runner.make_replay_env()
        command.cfg.reset_noise = False
        assert command.env_scene.bincount().tolist() == [2, 2, 2, 2]
        assert command.motion is runner.motion is replay.motion
        assert len(command.motion.names) == 8
        ids = torch.arange(8, device=cuda_device)
        assignments = command.env_scene.clone()
        # Random RSI repeatedly samples only each environment's assigned chair.
        for _ in range(5):
            native.reset()
            context = command.dynamics.object_context(command.xi)
            _assert_membership(command, command.time_steps, context[:, 0, 12].long())
            assert context[:, 1].count_nonzero() == 0
            torch.testing.assert_close(command.env_scene, assignments)
        # Exercise both references per chair, not only its anchor clip.
        first_clip = command.env_scene * 2
        frames = command.motion.clip_start[first_clip + ids % 2] + 3
        reset_frames(native, frames)
        for value, expected in zip(command.state(), command.motion.full_ref_state(frames)):
            torch.testing.assert_close(value, expected, atol=3e-5, rtol=3e-5)
        sampled_geometry = context[:, 0, 12].clone()
        observations = runner.env.get_observations()
        torch.testing.assert_close(observations["goal"], command.motion.goal(frames, command.cfg.goal_frames))
        assert torch.isfinite(observations["policy_history"]).all()
        # Partial resets must not replace the geometry or other worlds' state.
        subset = torch.tensor([1, 4, 7], device=cuda_device)
        other = torch.tensor([0, 2, 3, 5, 6], device=cuda_device)
        before_state = tuple(value.clone() for value in command.state())
        before_context = command.xi.clone()
        before_frames = command.time_steps.clone()
        partial = command.motion.clip_start[first_clip[subset] + 1] + 7
        reset_frames(native, partial, subset)
        torch.testing.assert_close(command.time_steps[other], before_frames[other])
        torch.testing.assert_close(command.xi[other], before_context[other])
        for value, previous in zip(command.state(), before_state):
            torch.testing.assert_close(value[other], previous[other], atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(command.dynamics.object_context(command.xi)[:, 0, 12], sampled_geometry)
        wrong_scene = (command.env_scene[subset] + 1) % 4
        wrong_frames = command.motion.clip_start[wrong_scene * 2]
        with pytest.raises(ValueError, match="assigned chair"):
            command.set_reset_frames(subset, wrong_frames)
        with pytest.raises(ValueError, match="assigned chair"):
            command.curriculum.reset(subset, wrong_frames)
        # Context's six inertia components must track the geometry actually
        # assigned to each world, scaled by that episode's randomized mass.
        body = native.scene["chair"].indexing.root_body_id
        mass = native.sim.model.body_mass[:, body]
        default_mass = native.sim.get_default_field("body_mass")[:, body]
        principal = native.sim.get_default_field("body_inertia")[:, body] * (mass / default_mass)[:, None]
        torch.testing.assert_close(native.sim.model.body_inertia[:, body], principal)
        from supermimic.rotations import quat_to_mat
        rotation = quat_to_mat(native.sim.get_default_field("body_iquat")[:, body])
        inertia = rotation @ torch.diag_embed(principal) @ rotation.transpose(-1, -2)
        context = command.dynamics.object_context(command.xi)[:, 0]
        torch.testing.assert_close(context[:, 4:10], inertia[:, [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]])
        # Replay records the same reference clock and scene IDs as the native
        # environment. A terminal reset cannot overwrite the outgoing sample.
        starting_frames, starting_xi = command.time_steps.clone(), command.xi.clone()
        out = replay.step(torch.zeros(8, 29, device=cuda_device))
        torch.testing.assert_close(out["f"], starting_frames)
        torch.testing.assert_close(out["xi"], starting_xi)
        _assert_membership(command, out["f"], command.dynamics.object_context(out["xi"])[:, 0, 12].long())
        runner._init_training_models()
        models = tuple(id(model) for model in (runner.pi, runner.wm, runner.critic))
        env_identity = id(runner.env)
        runner.learn(4)
        assert len(runner.visited_shapes) == 4
        assert models == tuple(id(model) for model in (runner.pi, runner.wm, runner.critic))
        assert runner.pi.geometry_encoder is runner.wm.geometry_encoder is runner.critic.geometry_encoder
        expected_keys = lambda partition: {
            geometry_key(load_motion(chair_dataset / scene["clips"][0]["path"]), collision_profile)
            for scene in scenes(split, partition)}
        assert set(runner.global_bank.keys[1:]) == expected_keys("train")
        assert not set(runner.global_bank.keys[1:]) & expected_keys("test")
        assert torch.isfinite(runner.global_bank.points).all()
        assert torch.isfinite(runner.global_bank.extents).all()
        for field in ("body_mass", "body_inertia", "body_ipos", "geom_size", "geom_aabb"):
            assert torch.isfinite(getattr(native.sim.model, field)).all()
        rows = runner.buffer._row(torch.arange(runner.buffer.size, device=cuda_device))
        replay_geometry = command.dynamics.object_context(runner.buffer.data["xi"][rows])[..., 0, 12].long()
        _assert_membership(command, runner.buffer.data["f"][rows], replay_geometry)
        assert replay_geometry.unique().numel() == 4
        # Crossing the old 50-iteration rotation boundary must do nothing to
        # physics, frame clocks, Replay done flags, or environment identity.
        saved_frames = command.time_steps.clone()
        saved_done = runner.buffer.data["done"].clone()
        runner.current_learning_iteration = 60
        runner.prepare_collection()
        assert id(runner.env) == env_identity
        torch.testing.assert_close(command.time_steps, saved_frames)
        torch.testing.assert_close(runner.buffer.data["done"], saved_done)
        runner.current_learning_iteration = 4
        checkpoint = run / "model_4.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        protocol = saved["task"]["chair_experiment"]
        assert protocol["scene_distribution"] == "fixed_balanced_per_environment"
        assert protocol["num_envs"] == 8 and protocol["version"] == 2
        assert protocol["collision_profile"] == collision_profile
        assert len(saved["infos"]["chair_experiment"]["visited_train_shapes"]) == 4
        assert runner.opt_wm.state and runner.opt_pi.state and runner.opt_v.state
        # Full resume must reject both a different physical approximation and
        # the previous rotating experiment, even with identical tensor shapes.
        for mismatch in ("collision_profile", "scene_distribution"):
            changed = saved | {"task": deepcopy(saved["task"])}
            changed_protocol = changed["task"]["chair_experiment"]
            changed_protocol[mismatch] = (
                "compact" if collision_profile == "default" else "default"
            ) if mismatch == "collision_profile" else "strided_shuffled_visits"
            rejected = tmp_path / (mismatch + ".pt")
            torch.save(changed, rejected)
            with pytest.raises(ValueError, match="task/schema mismatch"):
                runner.load(rejected)
            assert runner.current_learning_iteration == 4
    finally:
        runner.close()
    del runner, native, command, replay
    gc.collect()
    torch.cuda.empty_cache()
    resumed = MixedChairExperimentRunner(split, chair_dataset, cache, cfg,
        num_envs=8, max_objects=2, log_dir=run, device=cuda_device,
        collision_profile=collision_profile)
    try:
        resumed.load(checkpoint)
        assert resumed.current_learning_iteration == 4
        assert len(resumed.visited_shapes) == 4
        torch.testing.assert_close(resumed.env_scene, assignments)
        resumed.learn(1)
        assert resumed.current_learning_iteration == 5
    finally:
        resumed.close()
    del resumed
    gc.collect()
    torch.cuda.empty_cache()
    report = evaluate_split(split, chair_dataset, checkpoint, cache_dir=cache, device=cuda_device,
                            output=tmp_path / "heldout.json")
    assert report["protocol"] == "held_out_geometry_frozen_policy"
    assert report["evaluated_chairs"] == 1 and report["visited_training_chairs"] == 4
    assert len(report["per_scene"][0]["per_clip"]) == 2
    assert not report["incomplete_training"]
