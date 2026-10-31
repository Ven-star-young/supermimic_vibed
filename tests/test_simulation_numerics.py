"""Numerically failed worlds must never become world-model training data."""

from collections import defaultdict

import pytest
import torch

from supermimic.rl.storage import ReplayBuffer
from supermimic.tasks.open_door.models import WorldModel
from tests.test_open_door_env import door_runner  # noqa: F401


def _reset(replay):
    replay.reset(frames=torch.zeros(replay.N, device=replay.device, dtype=torch.long))


def _inject_after_physics(monkeypatch, replay, failure, bad_world=1):
    """Corrupt one native result after a genuine step, before Replay reads it."""
    native_step = replay.wrapped_env.step
    native_reset = replay.reset
    captured = {"resets": []}

    def step(action):
        result = native_step(action)
        captured["physical_state"] = replay.dynamics.pack(replay.state()).clone()
        captured["physical_done"] = (
            replay.env.reset_terminated | replay.env.reset_time_outs
        ).clone()
        if failure == "nan_position":
            replay.env.sim.data.qpos[bad_world, 0] = float("nan")
            replay.env.sim.forward()
        elif failure == "huge_acceleration":
            replay.env.sim.data.qacc[bad_world, 0] = 1e12
        elif failure is not None:
            raise ValueError(failure)
        return result

    def reset(ids=None, frames=None):
        captured["resets"].append(None if ids is None else ids.clone())
        return native_reset(ids, frames)

    monkeypatch.setattr(replay.wrapped_env, "step", step)
    monkeypatch.setattr(replay, "reset", reset)
    return captured


@pytest.mark.parametrize("failure", ["nan_position", "huge_acceleration"])
def test_invalid_physics_resets_only_failed_world_and_returns_safe_transition(
    door_runner, monkeypatch, failure
):
    replay = door_runner.replay_env
    _reset(replay)
    before = replay.dynamics.pack(replay.state()).clone()
    captured = _inject_after_physics(monkeypatch, replay, failure)
    bad_world = 1
    good = torch.tensor([0, 2, 3], device=replay.device)
    try:
        transition = replay.step(torch.zeros(replay.N, 29, device=replay.device))

        assert not captured["physical_done"].any(), "Fixture should isolate numerical failure"
        assert torch.isfinite(transition["s1"]).all()
        assert transition["valid"].tolist() == [True, False, True, True]
        assert transition["done"].tolist() == [False, True, False, False]
        assert transition["term"].tolist() == [False, True, False, False]
        assert len(captured["resets"]) == 1
        assert captured["resets"][0].tolist() == [bad_world]
        assert replay.ep_step.tolist() == [1, 0, 1, 1]
        assert transition["r"][bad_world] == 0
        torch.testing.assert_close(transition["s1"][bad_world], before[bad_world])
        torch.testing.assert_close(transition["s"], before)
        torch.testing.assert_close(transition["s1"][good], captured["physical_state"][good])
        torch.testing.assert_close(
            replay.dynamics.pack(replay.state())[good], captured["physical_state"][good]
        )
        assert torch.isfinite(transition["r"]).all()
        assert torch.isfinite(replay.dynamics.pack(replay.state())).all()
    finally:
        _reset(replay)


def test_valid_terminal_keeps_actual_next_state_before_reset(door_runner, monkeypatch):
    replay = door_runner.replay_env
    _reset(replay)
    replay.reset(
        torch.tensor([0], device=replay.device), replay.motion.clip_end[:1] - 1
    )
    captured = _inject_after_physics(monkeypatch, replay, None)
    transition = replay.step(torch.zeros(replay.N, 29, device=replay.device))

    assert transition["valid"].all()
    assert transition["done"][0] and replay.ep_step[0] == 0
    torch.testing.assert_close(transition["s1"], captured["physical_state"])
    assert not torch.allclose(transition["s1"][0], transition["s"][0])
    assert not torch.allclose(
        transition["s1"][0], replay.dynamics.pack(replay.state())[0]
    )
    expected = replay.mdp.reward(
        replay.dynamics.unpack(transition["s1"]), transition["f"] + 1,
        transition["a"], transition["a_prev"],
    )
    torch.testing.assert_close(transition["r"], expected)


def test_collection_excludes_failed_transition_from_wm_normalizers(door_runner, monkeypatch):
    runner, replay = door_runner, door_runner.replay_env
    _reset(replay)
    world_model = WorldModel(dynamics=replay.dynamics, hidden=(16, 16)).to(replay.device)
    buffer = ReplayBuffer(
        4, replay.N, replay.device, state_dim=73, action_dim=29, context_dim=46
    )
    monkeypatch.setattr(runner, "wm", world_model)
    monkeypatch.setattr(runner, "buffer", buffer)
    _inject_after_physics(monkeypatch, replay, "nan_position")
    try:
        runner.collect(1, defaultdict(list))

        assert buffer.data["valid"][0].tolist() == [True, False, True, True]
        assert world_model.in_norm.count == 3
        assert world_model.out_norm.count == 3
        valid = buffer.data["valid"][0]
        initial = replay.dynamics.unpack(buffer.data["s"][0, valid])
        final = replay.dynamics.unpack(buffer.data["s1"][0, valid])
        inputs = world_model.inputs(
            initial, buffer.data["a"][0, valid],
            replay.dynamics.normalize_context(buffer.data["xi"][0, valid]),
        )
        torch.testing.assert_close(world_model.in_norm.mean, inputs.mean(0))
        torch.testing.assert_close(
            world_model.out_norm.mean, world_model.targets(initial, final).mean(0)
        )
        for normalizer in (world_model.in_norm, world_model.out_norm):
            assert torch.isfinite(normalizer.mean).all()
            assert torch.isfinite(normalizer.var).all()
    finally:
        _reset(replay)
