"""Reward fine-tuning transfers Actor/WM while starting a fresh value learner."""

import hashlib

import pytest
import torch

from tests.test_wm_initialization import (
    assert_state_equal, runners, side_task, snapshot, stub_updates,
)  # noqa: F401


@pytest.fixture
def side_checkpoint(runners, tmp_path):
    _, make = runners
    source = make()
    with torch.no_grad():
        for module in (source.pi, source.wm, source.critic):
            for parameter in module.parameters():
                parameter.fill_(.123)
            for name, value in module.state_dict().items():
                if name.endswith(".mean"):
                    value.fill_(.25)
                elif name.endswith(".var"):
                    value.fill_(.75)
                elif name.endswith(".count"):
                    value.fill_(12000)
    # Optimizers and critic deliberately contain trained state that must not be
    # imported into the new reward objective.
    for module, optimizer in ((source.pi, source.opt_pi), (source.wm, source.opt_wm),
                              (source.critic, source.opt_v)):
        sum(parameter.sum() for parameter in module.parameters()).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    source.current_learning_iteration = 10000
    source.sim_steps, source.wall_time = 8000000, 1234.5
    source.world_model_initialization = {
        "path": "/missing/original_back_model.pt", "sha256": "a" * 64,
        "iteration": 10000, "task": "g1_open_door",
    }
    path = tmp_path / "side_model_10000.pt"
    source.save(path)
    return source, path


def test_finetune_imports_actor_wm_and_statistics_with_fresh_critic_and_optimizers(
        runners, side_checkpoint):
    _, make = runners
    source, path = side_checkpoint
    target = make(lr=5e-5)
    critic = snapshot(target.critic)

    provenance = target.load_finetune(path)

    assert_state_equal(target.pi.state_dict(), source.pi.state_dict())
    assert_state_equal(target.wm.state_dict(), source.wm.state_dict())
    assert_state_equal(target.critic.state_dict(), critic)
    assert all(not optimizer.state for optimizer in (target.opt_pi, target.opt_wm, target.opt_v))
    assert all(group["lr"] == 5e-5 for optimizer in (target.opt_pi, target.opt_wm, target.opt_v)
               for group in optimizer.param_groups)
    assert target.current_learning_iteration == target.sim_steps == target.wall_time == 0
    assert provenance == target.finetune_initialization == {
        "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "iteration": 10000, "task": "g1_open_door_side",
    }
    assert target.world_model_initialization == source.world_model_initialization
    assert all(parameter.requires_grad for module in (target.pi, target.wm, target.critic)
               for parameter in module.parameters())


@pytest.mark.parametrize("module", ["pi", "wm"])
@pytest.mark.parametrize("corruption", ["missing", "extra", "shape", "dtype", "nan", "inf",
                                       "negative_variance", "negative_count"])
def test_corrupt_transfer_is_atomic_for_both_modules(
        runners, side_checkpoint, tmp_path, module, corruption):
    _, make = runners
    _, path = side_checkpoint
    target = make()
    before = {name: snapshot(getattr(target, name)) for name in ("pi", "wm", "critic")}
    checkpoint = torch.load(path, weights_only=False)
    weights = checkpoint[module]
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
    else:
        suffix = ".var" if corruption == "negative_variance" else ".count"
        key = next(key for key in weights if key.endswith(suffix))
        weights[key].fill_(-1)
    corrupt_path = tmp_path / "corrupt.pt"
    torch.save(checkpoint, corrupt_path)

    with pytest.raises(ValueError):
        target.load_finetune(corrupt_path)

    for name, expected in before.items():
        assert_state_equal(getattr(target, name).state_dict(), expected)
    assert target.finetune_initialization is target.world_model_initialization is None
    assert all(not optimizer.state for optimizer in (target.opt_pi, target.opt_wm, target.opt_v))


@pytest.mark.parametrize("field", ["history_length", "goal_frames"])
def test_policy_temporal_contract_checked_before_loading(
        runners, side_checkpoint, tmp_path, field):
    _, make = runners
    _, path = side_checkpoint
    target = make()
    actor, model = snapshot(target.pi), snapshot(target.wm)
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["policy_config"][field] += 1
    corrupt_path = tmp_path / "different_temporal_contract.pt"
    torch.save(checkpoint, corrupt_path)

    with pytest.raises(ValueError, match="history/goal"):
        target.load_finetune(corrupt_path)
    assert_state_equal(target.pi.state_dict(), actor)
    assert_state_equal(target.wm.state_dict(), model)


def test_finetune_loads_once_before_prefill_and_updates_ac_from_iteration_zero(
        runners, side_checkpoint, monkeypatch):
    _, make = runners
    source, path = side_checkpoint
    target = make(finetune_checkpoint=str(path), wm_only_iters=0)
    events = stub_updates(target, monkeypatch)
    real_import = target.load_finetune

    def load(source_path):
        assert events == []
        result = real_import(source_path)
        assert_state_equal(target.pi.state_dict(), source.pi.state_dict())
        assert_state_equal(target.wm.state_dict(), source.wm.state_dict())
        events.append(("load", str(source_path)))
        return result

    monkeypatch.setattr(target, "load_finetune", load)
    target.learn(1)
    assert events == [("load", str(path)), ("collect", 4), ("collect", 1),
                      ("wm", 0), ("ac", 0)]
    target.learn(1)
    assert events[-4:] == [("collect", 4), ("collect", 1), ("wm", 1), ("ac", 1)]
    assert target.current_learning_iteration == 2


def test_finetuned_full_resume_does_not_reopen_either_initialization_source(
        runners, side_checkpoint, tmp_path, monkeypatch):
    _, make = runners
    _, path = side_checkpoint
    target = make(finetune_checkpoint=str(path))
    target.load_finetune(path)
    with torch.no_grad():
        next(target.pi.parameters()).add_(.3)
        next(target.wm.parameters()).add_(.5)
    target.current_learning_iteration = 7
    target.sim_steps, target.wall_time = 234, 12.5
    resume_path = tmp_path / "finetuned_model_7.pt"
    target.save(resume_path)
    saved = torch.load(resume_path, weights_only=False)
    assert saved["finetune_initialization"] == target.finetune_initialization
    assert saved["world_model_initialization"] == target.world_model_initialization
    path.unlink()

    resumed = make(finetune_checkpoint=str(path))
    resumed.load(resume_path)
    assert resumed.finetune_initialization == target.finetune_initialization
    assert resumed.world_model_initialization == target.world_model_initialization
    assert (resumed.current_learning_iteration, resumed.sim_steps, resumed.wall_time) == (7, 234, 12.5)
    for name in ("pi", "wm", "critic"):
        assert_state_equal(getattr(resumed, name).state_dict(), getattr(target, name).state_dict())

    def forbidden_import(*args):
        raise AssertionError("Resume reopened an initialization checkpoint")

    monkeypatch.setattr(resumed, "load_finetune", forbidden_import)
    monkeypatch.setattr(resumed, "load_world_model", forbidden_import)
    events = stub_updates(resumed, monkeypatch)
    resumed.learn(1)
    assert events[-2:] == [("wm", 7), ("ac", 7)]
    assert resumed.current_learning_iteration == 8


def test_finetune_cannot_overwrite_existing_initialization(runners, side_checkpoint):
    _, make = runners
    _, path = side_checkpoint
    target = make()
    target.load_finetune(path)
    with pytest.raises(ValueError):
        target.load_finetune(path)
    with pytest.raises(ValueError):
        target.load_world_model(path)


def test_finetune_and_wm_initialization_paths_are_mutually_exclusive(runners):
    _, make = runners
    with pytest.raises(ValueError):
        make(finetune_checkpoint="side.pt", wm_checkpoint="back.pt")
