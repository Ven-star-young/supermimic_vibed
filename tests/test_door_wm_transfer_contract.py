"""A reusable door WM must describe the same physical state and transition."""

from copy import deepcopy
import json

import pytest

from supermimic.tasks.base import TrackingTask
from supermimic.tasks.open_door.env_cfgs import make_env_cfg as make_back_env_cfg
from supermimic.tasks.open_door.task import build_task as build_back_task
from tests.test_open_door_side_mdp import task


@pytest.fixture
def back_checkpoint(task):
    back = build_back_task(device="cpu", kin=task.mdp.kin, motion=task.motion,
                           cfg=make_back_env_cfg(play=True))
    return {"task": back.checkpoint_metadata()}


def test_back_world_model_can_initialize_side_but_full_resume_stays_incompatible(task, back_checkpoint):
    original = deepcopy(back_checkpoint)
    task.validate_world_model_checkpoint(back_checkpoint)
    task.validate_world_model_checkpoint(json.loads(json.dumps(back_checkpoint)))
    assert back_checkpoint == original
    with pytest.raises(ValueError, match="mismatch"):
        task.validate_checkpoint(back_checkpoint)


def test_transfer_ignores_policy_reward_and_context_conditioned_physics(task, back_checkpoint):
    metadata = back_checkpoint["task"]
    metadata["door_reward"] = {"different_back_reward": 2.}
    metadata["actor_goal"] = "different policy goal"
    metadata["dimensions"]["observation"] = 1
    metadata["dimensions"]["goal_per_frame"] = 2
    metadata["model"]["actor_hidden"] = [8]
    metadata["door_physics"]["door_mass"] = 12.
    metadata["door_randomization"]["mass"] = [7., 11.]
    metadata["model"]["world_model_hidden"] = tuple(metadata["model"]["world_model_hidden"])
    task.validate_world_model_checkpoint(back_checkpoint)


@pytest.mark.parametrize("field,value", [
    ("dynamics_schema", "open-door-g1-hinge-v2"),
    ("state", "g1_p3_quat_xyzw4_v3_w3_q29_qd29,hinge_theta_omega"),
    ("context", "robot33,hinge_xyz3,door_dims3,door_physics7_rescaled"),
    ("control_dt", .01),
    ("geometry", {"point_cloud_size": 16}),
])
def test_same_size_world_models_with_different_semantics_are_rejected(task, back_checkpoint, field, value):
    back_checkpoint["task"][field] = value
    with pytest.raises(ValueError, match=field):
        task.validate_world_model_checkpoint(back_checkpoint)


@pytest.mark.parametrize("field", ["state", "action", "context"])
def test_transfer_requires_physical_dimensions(task, back_checkpoint, field):
    back_checkpoint["task"]["dimensions"][field] += 1
    with pytest.raises(ValueError, match="dimensions"):
        task.validate_world_model_checkpoint(back_checkpoint)


def test_transfer_rejects_other_hinge_axis_and_world_model_architecture(task, back_checkpoint):
    changed_axis = deepcopy(back_checkpoint)
    changed_axis["task"]["door_asset"]["joint_axis"][2] *= -1
    with pytest.raises(ValueError, match="door_asset"):
        task.validate_world_model_checkpoint(changed_axis)
    back_checkpoint["task"]["model"]["world_model_hidden"] = [1024, 1024, 1024]
    with pytest.raises(ValueError, match="world_model_hidden"):
        task.validate_world_model_checkpoint(back_checkpoint)


@pytest.mark.parametrize("metadata", [None, {}, {"name": "g1_sitting", "schema_version": 2},
                                     {"name": "g1_open_door", "schema_version": 1}])
def test_transfer_rejects_unknown_or_legacy_tasks(task, metadata):
    with pytest.raises(ValueError, match="schema-2"):
        task.validate_world_model_checkpoint({"task": metadata})


@pytest.mark.parametrize("missing", ["state", "geometry", "door_asset", "model"])
def test_transfer_requires_explicit_metadata(task, back_checkpoint, missing):
    del back_checkpoint["task"][missing]
    with pytest.raises(ValueError, match="metadata"):
        task.validate_world_model_checkpoint(back_checkpoint)


def test_transfer_rejects_nonfinite_metadata(task, back_checkpoint):
    back_checkpoint["task"]["control_dt"] = float("nan")
    with pytest.raises(ValueError, match="metadata"):
        task.validate_world_model_checkpoint(back_checkpoint)


def test_other_tasks_must_explicitly_define_transfer_compatibility():
    generic = TrackingTask("unsupported_task", None, None, None, None)
    with pytest.raises(ValueError, match="does not support"):
        generic.validate_world_model_checkpoint({"task": {"name": generic.name}})
