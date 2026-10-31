"""A transferred WM seeds a fresh learner without importing its source policy."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from supermimic.rl.config import SuperMimicRunnerCfg
from supermimic.rl.models import RunningNorm
from supermimic.rl.runner import SuperMimicRunner
from supermimic.tasks.open_door.env_cfgs import make_env_cfg as make_back_env_cfg
from supermimic.tasks.open_door.models import WorldModel
from supermimic.tasks.open_door.task import OpenDoorTask, build_task as build_back_task
from tests.test_open_door_side_mdp import task as side_task  # noqa: F401


@pytest.fixture
def runners(side_task, monkeypatch):
    # Keep the real door dynamics/normalizers/checkpoint schemas; reduce only the
    # network widths and replace simulator collection in lifecycle tests below.
    monkeypatch.setattr(OpenDoorTask, "make_policy", lambda *args: nn.Sequential(
        RunningNorm(4), nn.Linear(4, 29)))
    monkeypatch.setattr(OpenDoorTask, "make_critic", lambda *args: nn.Sequential(
        RunningNorm(4), nn.Linear(4, 1)))
    monkeypatch.setattr(OpenDoorTask, "make_world_model", lambda self, *args: WorldModel(
        dynamics=self.dynamics, hidden=(16,)))
    back = build_back_task(device="cpu", kin=side_task.mdp.kin, motion=side_task.motion,
                           cfg=make_back_env_cfg(play=True))

    def make(task=side_task, **overrides):
        command = SimpleNamespace(task=task, cfg=task.env_cfg.commands["motion"])
        env = SimpleNamespace(num_envs=2, unwrapped=SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda name: command)))
        cfg = asdict(SuperMimicRunnerCfg(
            buffer_steps=8, wm_window=2, horizon=2, wm_batches=1,
            wm_batch=2, pi_batch=2, warmup_steps=4, wm_only_iters=0,
            critic_epochs=1, critic_minibatches=1, utd=1.,
            no_compile=True, amp="none")) | overrides
        runner = SuperMimicRunner(env, cfg, device="cpu")
        runner._init_training_models()
        return runner

    return make(back), make


@pytest.fixture
def source_checkpoint(runners, tmp_path):
    source, _ = runners
    with torch.no_grad():
        for module in (source.pi, source.wm, source.critic):
            for parameter in module.parameters():
                parameter.fill_(.123)
        for norm in (source.wm.in_norm, source.wm.out_norm):
            norm.mean.fill_(.25)
            norm.var.fill_(.75)
            norm.count.fill_(12000)
    # Include nonempty source optimizer state so accidental full resume is visible.
    for module, optimizer in ((source.pi, source.opt_pi), (source.wm, source.opt_wm),
                              (source.critic, source.opt_v)):
        sum(parameter.sum() for parameter in module.parameters()).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    source.current_learning_iteration = 630
    source.sim_steps, source.wall_time = 1000000, 345.6
    path = tmp_path / "back_model_630.pt"
    source.save(path)
    return path


def snapshot(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def assert_state_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


def stub_updates(runner, monkeypatch):
    events = []
    monkeypatch.setattr(runner, "make_replay_env", lambda: SimpleNamespace(N=2))
    monkeypatch.setattr(runner, "collect", lambda steps, stats: events.append(("collect", steps)))
    monkeypatch.setattr(runner, "wm_update", lambda: (
        events.append(("wm", runner.current_learning_iteration)) or (torch.tensor(1.), torch.tensor(2.))))
    monkeypatch.setattr(runner, "policy_update", lambda: (
        events.append(("ac", runner.current_learning_iteration)) or tuple(torch.tensor(0.) for _ in range(5))))
    return events


def test_back_wm_imports_weights_and_statistics_but_keeps_fresh_side_learner(runners, source_checkpoint):
    source, make = runners
    target = make()
    actor, critic = snapshot(target.pi), snapshot(target.critic)
    provenance = target.load_world_model(source_checkpoint)

    assert_state_equal(target.wm.state_dict(), source.wm.state_dict())
    assert_state_equal(target.pi.state_dict(), actor)
    assert_state_equal(target.critic.state_dict(), critic)
    assert all(not optimizer.state for optimizer in (target.opt_wm, target.opt_pi, target.opt_v))
    assert target.current_learning_iteration == target.sim_steps == target.wall_time == 0
    assert provenance == target.world_model_initialization
    assert provenance == {"path": str(source_checkpoint.resolve()),
                          "sha256": hashlib.sha256(source_checkpoint.read_bytes()).hexdigest(),
                          "iteration": 630, "task": "g1_open_door"}
    assert all(parameter.requires_grad for parameter in target.wm.parameters())


@pytest.mark.parametrize("corruption", ["missing", "extra", "shape", "dtype", "nan", "inf",
                                       "negative_variance", "negative_count"])
def test_corrupt_wm_rejected_without_partially_loading(runners, source_checkpoint, tmp_path, corruption):
    _, make = runners
    target = make()
    before = snapshot(target.wm)
    checkpoint = torch.load(source_checkpoint, weights_only=False)
    weights = checkpoint["wm"]
    key = next(key for key in weights if key.endswith("weight"))
    if corruption == "missing":
        weights.pop(key)
    elif corruption == "extra":
        weights["unexpected.weight"] = torch.zeros(1)
    elif corruption == "shape":
        weights[key] = weights[key][:1]
    elif corruption == "dtype":
        weights[key] = weights[key].double()
    elif corruption in ("nan", "inf"):
        weights[key].flatten()[0] = float(corruption)
    elif corruption == "negative_variance":
        weights["out_norm.var"][0] = -1
    else:
        weights["in_norm.count"].fill_(-1)
    path = tmp_path / "corrupt.pt"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError):
        target.load_world_model(path)
    assert_state_equal(target.wm.state_dict(), before)
    assert target.world_model_initialization is None
    assert not target.opt_wm.state


def test_configured_import_happens_once_before_prefill_and_ac_starts_immediately(
        runners, source_checkpoint, monkeypatch):
    source, make = runners
    target = make(wm_checkpoint=str(source_checkpoint), wm_only_iters=0)
    events = stub_updates(target, monkeypatch)
    real_import = target.load_world_model

    def load(path):
        assert events == []
        result = real_import(path)
        assert_state_equal(target.wm.state_dict(), source.wm.state_dict())
        events.append(("load", str(path)))
        return result

    monkeypatch.setattr(target, "load_world_model", load)
    target.learn(1)
    assert events == [("load", str(source_checkpoint)), ("collect", 4), ("collect", 1),
                      ("wm", 0), ("ac", 0)]
    assert target.current_learning_iteration == 1
    # A later learn call refills replay but neither reimports nor delays AC.
    target.learn(1)
    assert events[-4:] == [("collect", 4), ("collect", 1), ("wm", 1), ("ac", 1)]


def test_full_resume_preserves_import_provenance_without_reopening_original_source(
        runners, source_checkpoint, tmp_path, monkeypatch):
    _, make = runners
    target = make(wm_checkpoint=str(source_checkpoint))
    target.load_world_model(source_checkpoint)
    with torch.no_grad():
        next(target.wm.parameters()).add_(.5)
    target.current_learning_iteration = 7
    target.sim_steps, target.wall_time = 234, 12.5
    path = tmp_path / "side_model_7.pt"
    target.save(path)
    provenance = target.world_model_initialization.copy()
    assert torch.load(path, weights_only=False)["world_model_initialization"] == provenance
    source_checkpoint.unlink()

    resumed = make(wm_checkpoint=str(source_checkpoint))
    resumed.load(path)
    assert resumed.world_model_initialization == provenance
    assert resumed.current_learning_iteration == 7
    assert_state_equal(resumed.wm.state_dict(), target.wm.state_dict())

    def forbidden_import(*args):
        raise AssertionError("Resume reopened the original WM initialization checkpoint")

    monkeypatch.setattr(resumed, "load_world_model", forbidden_import)
    events = stub_updates(resumed, monkeypatch)
    resumed.learn(1)
    assert events[-2:] == [("wm", 7), ("ac", 7)]
    assert resumed.current_learning_iteration == 8


def test_import_cannot_overwrite_an_existing_initialization(runners, source_checkpoint):
    _, make = runners
    target = make()
    target.load_world_model(source_checkpoint)
    with pytest.raises(ValueError):
        target.load_world_model(source_checkpoint)


def test_default_side_training_skips_wm_only_phase_but_retains_replay_prefill():
    path = Path(__file__).resolve().parents[1] / "configs" / "open_door_right_side.json"
    config = json.loads(path.read_text())["training"]
    assert config["wm_only_iters"] == 0
    assert config["warmup_steps"] == 64
