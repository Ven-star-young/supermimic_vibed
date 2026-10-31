"""Portable hinge-trajectory validation shared by conversion and training."""

from pathlib import Path

import numpy as np


DOOR_METADATA_SHAPES = {
    "hinge_pos": (3,), "base_yaw": (), "panel_half_size": (3,),
    "hinge_to_panel": (3,), "joint_axis": (3,), "joint_range": (2,),
}


def door_metadata(clip: dict) -> dict:
    """Validate a door clip and return its static metadata without prefixes."""
    n = len(clip["joint_pos"])
    shapes = {"door_angle": (n,), **{f"door_{key}": shape for key, shape in DOOR_METADATA_SHAPES.items()}}
    if "door_angular_velocity" in clip:
        shapes["door_angular_velocity"] = (n,)
    for key, shape in shapes.items():
        if key not in clip:
            raise ValueError(f"OpenDoor training requires {key}")
        value = np.asarray(clip[key])
        if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain finite numbers with shape {shape}")
    values = {key: np.asarray(clip[f"door_{key}"]) for key in DOOR_METADATA_SHAPES}
    if (values["panel_half_size"] <= 0).any():
        raise ValueError("door_panel_half_size must be positive")
    if not np.isclose(np.linalg.norm(values["joint_axis"]), 1.0, atol=1e-6):
        raise ValueError("door_joint_axis must have unit length")
    lo, hi = values["joint_range"]
    if lo >= hi:
        raise ValueError("door_joint_range must be increasing")
    angle = np.asarray(clip["door_angle"])
    if (angle < lo - 1e-5).any() or (angle > hi + 1e-5).any():
        raise ValueError("door_angle lies outside door_joint_range")
    if "door_contact_mask" in clip:
        mask = np.asarray(clip["door_contact_mask"])
        if mask.shape != (n,) or mask.dtype.kind != "b":
            raise ValueError(f"door_contact_mask must be boolean with shape ({n},)")
    if "door_opening_frame" in clip:
        frame = np.asarray(clip["door_opening_frame"])
        if frame.shape != () or frame.dtype.kind not in "iu" or not 0 <= int(frame) < n:
            raise ValueError("door_opening_frame must be an integer source-frame index")
    return values


def validate_door_asset(clip: dict, path: str | Path) -> None:
    """Check the paired passive-door MJCF against the trajectory's geometry."""
    import mujoco

    values = door_metadata(clip)
    model = mujoco.MjModel.from_xml_path(str(path))
    if model.nu or model.nq != 1 or model.nv != 1:
        raise ValueError(f"{path}: expected one passive hinge with no actuator")
    joint, panel = model.joint("door_hinge"), model.geom("door_panel")
    if joint.type[0] != mujoco.mjtJoint.mjJNT_HINGE or panel.type[0] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError(f"{path}: expected door_hinge and a box door_panel")
    if not model.jnt_limited[joint.id]:
        raise ValueError(f"{path}: door_hinge must have enabled joint limits")
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)
    rotation = data.geom_xmat[panel.id].reshape(3, 3)
    yaw = float(values["base_yaw"])
    c, s = np.cos(yaw), np.sin(yaw)
    expected_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    checks = {
        "hinge_pos": (data.xanchor[joint.id], values["hinge_pos"]),
        "joint_axis": (model.jnt_axis[joint.id], values["joint_axis"]),
        "joint_range": (model.jnt_range[joint.id], values["joint_range"]),
        "panel_half_size": (model.geom_size[panel.id], values["panel_half_size"]),
        "base_yaw": (rotation, expected_rotation),
        "hinge_to_panel": (rotation.T @ (data.geom_xpos[panel.id] - data.xanchor[joint.id]),
                           values["hinge_to_panel"]),
    }
    for key, (actual, expected) in checks.items():
        if not np.allclose(actual, expected, atol=1e-6, rtol=0):
            raise ValueError(f"{path}: XML {key} does not match door_{key} in the motion")
