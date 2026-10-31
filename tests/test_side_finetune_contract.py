"""Reward-only side finetuning is explicit; old evaluations retain old rewards."""

from copy import deepcopy
from dataclasses import replace

import pytest

from supermimic.scripts.evaluate import _restore_open_door_config
from supermimic.tasks.base import TrackingTask
from supermimic.tasks.open_door_side.env_cfgs import make_env_cfg
from supermimic.tasks.open_door_side.task import build_task
from tests.test_open_door_side_mdp import task


def test_legacy_side_checkpoint_preserves_zero_margin_for_evaluation_and_resume(task):
    checkpoint = {"task": task.checkpoint_metadata()}
    del checkpoint["task"]["door_reward"]["illegal_margin"]
    original = deepcopy(checkpoint)
    with pytest.raises(ValueError, match="mismatch"):
        task.validate_checkpoint(checkpoint)
    task.validate_finetune_checkpoint(checkpoint)

    cfg = make_env_cfg(play=True)
    _restore_open_door_config(cfg, checkpoint)
    assert cfg.commands["motion"].door_reward_cfg.illegal_margin == 0.
    restored = build_task(device="cpu", kin=task.mdp.kin, motion=task.motion, cfg=cfg)
    restored.validate_checkpoint(checkpoint)
    assert checkpoint == original


def test_changed_reward_requires_explicit_finetuning(task):
    checkpoint = {"task": task.checkpoint_metadata()}
    checkpoint["task"]["door_reward"]["illegal_margin"] = 0.
    checkpoint["task"]["robot_reward"]["w_root_pos"] *= .5
    original = deepcopy(checkpoint)
    with pytest.raises(ValueError, match="mismatch"):
        task.validate_checkpoint(checkpoint)
    task.validate_finetune_checkpoint(checkpoint)
    assert checkpoint == original


@pytest.mark.parametrize("key", ["contact_geometry", "contact_phase", "actor_observation", "actor_goal",
                                  "observation", "termination", "door_physics", "door_randomization",
                                  "dimensions", "door_asset", "control_dt", "model"])
def test_finetuning_preserves_every_nonreward_contract(task, key):
    checkpoint = {"task": task.checkpoint_metadata()}
    value = checkpoint["task"][key]
    if isinstance(value, dict):
        value.clear()
    elif isinstance(value, str):
        checkpoint["task"][key] = "changed"
    else:
        checkpoint["task"][key] *= 2
    with pytest.raises(ValueError):
        task.validate_finetune_checkpoint(checkpoint)


def test_back_actor_is_not_accepted_for_side_finetuning(task):
    checkpoint = {"task": task.checkpoint_metadata()}
    checkpoint["task"]["name"] = "g1_open_door"
    task.validate_world_model_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="mismatch"):
        task.validate_finetune_checkpoint(checkpoint)


@pytest.mark.parametrize("key", ["contact_weight", "penetration_scale"])
def test_legacy_restoration_only_supplies_missing_margin(task, key):
    checkpoint = {"task": task.checkpoint_metadata()}
    del checkpoint["task"]["door_reward"]["illegal_margin"]
    del checkpoint["task"]["door_reward"][key]
    with pytest.raises(ValueError, match="fields"):
        _restore_open_door_config(make_env_cfg(play=True), checkpoint)
    task.mdp.door_reward_cfg = replace(task.mdp.door_reward_cfg, illegal_margin=0.)
    with pytest.raises(ValueError, match="mismatch"):
        task.validate_checkpoint(checkpoint)


def test_other_tasks_do_not_implicitly_allow_actor_world_model_finetuning():
    task = TrackingTask("unsupported_task", None, None, None, None)
    with pytest.raises(ValueError, match="does not support"):
        task.validate_finetune_checkpoint({})
