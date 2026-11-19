"""HDMI pinhole depth on the native mjlab control clock, without image noise."""
import math

import torch
from mjlab.sensor import CameraSensorCfg

from .assets.g1.g1_constants import HDMI_CAMERA_METADATA, HDMI_CAMERA_NAME


def configure_depth_camera(cfg, *, rgb=False):
    """Wrap the G1 asset camera; native step/reset senses before observations."""
    if not math.isclose(cfg.sim.mujoco.timestep * cfg.decimation,
                        HDMI_CAMERA_METADATA["update_period"]):
        raise ValueError("Depth requires 50 Hz control and camera updates")
    if any(sensor.name == "depth_camera" for sensor in cfg.scene.sensors):
        raise ValueError("depth_camera is already registered")
    cfg.scene.sensors += (CameraSensorCfg(
        name="depth_camera", camera_name=f"robot/{HDMI_CAMERA_NAME}",
        width=HDMI_CAMERA_METADATA["width"], height=HDMI_CAMERA_METADATA["height"],
        data_types=("rgb", "depth") if rgb else ("depth",),
        enabled_geom_groups=(0, 1, 2), use_shadows=False,
    ),)
    return cfg


def preprocess_depth(raw: torch.Tensor) -> torch.Tensor:
    """Raw [B,H,W,1] planar metres -> finite float32 [B,1,H,W].

    Warp represents no return as zero. Keep it separate from a near obstacle:
    all nonpositive/nonfinite returns map to the far value before clipping.
    This function never modifies the renderer's shared buffer.
    """
    expected = (HDMI_CAMERA_METADATA["height"], HDMI_CAMERA_METADATA["width"], 1)
    if raw.ndim != 4 or tuple(raw.shape[1:]) != expected:
        raise ValueError(f"Expected depth [B,{expected[0]},{expected[1]},1], got {tuple(raw.shape)}")
    valid = torch.isfinite(raw) & (raw > 0)
    depth = torch.where(valid, raw.float(), HDMI_CAMERA_METADATA["max_depth"])
    return depth.clamp(HDMI_CAMERA_METADATA["min_depth"], HDMI_CAMERA_METADATA["max_depth"]).permute(0, 3, 1, 2).contiguous()


def depth_observation(env) -> torch.Tensor:
    """Read the current step's cached frame; never advance history or render."""
    return preprocess_depth(env.scene["depth_camera"].data.depth)
