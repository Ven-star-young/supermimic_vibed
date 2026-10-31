"""Load and verify the passive door paired with the converted motion files."""

from pathlib import Path

import mujoco
import numpy as np

from motion_data import load_motion
from motion_data.door import validate_door_asset
from ...assets import DATA_DIR


DEFAULT_MOTION_DIR = DATA_DIR / "motions" / "open_door_prepared"


def select_motion_files(cfg):
    path = Path(cfg.motion_dir)
    files = (list(map(Path, cfg.motion_files)) if cfg.motion_files else
             [path] if path.is_file() else sorted(path.glob("*.npz")))
    if not files:
        raise FileNotFoundError(f"No OpenDoor motions in {path}; run motion_data.converters.gmr first")
    return files


def door_xml_path(cfg):
    return Path(cfg.door_xml) if cfg.door_xml else select_motion_files(cfg)[0].with_suffix(".xml")


def make_door_spec(cfg):
    path = door_xml_path(cfg)
    for file in select_motion_files(cfg):
        validate_door_asset(load_motion(file), path)
    spec = mujoco.MjSpec.from_file(str(path))
    hinge, panel = spec.joint("door_hinge"), spec.geom("door_panel")
    hinge.damping[0] = cfg.door_damping
    hinge.frictionloss = cfg.door_friction
    hinge.stiffness[0] = cfg.door_stiffness
    hinge.springref = cfg.door_spring_reference
    panel.mass = cfg.door_mass
    panel.priority = 1
    panel.friction = [cfg.robot_door_friction, .005, .0001]
    return spec


def door_context(metadata, cfg):
    """Raw context after robot33: hinge3, dimensions3, physical parameters7."""
    half = metadata["panel_half_size"]
    floor_friction = (cfg.physical_cfg.evaluation_friction if cfg.play
                      else cfg.physical_cfg.friction[0])
    return np.asarray([*metadata["hinge_pos"], 2 * half[1], 2 * half[2], 2 * half[0],
                       cfg.door_mass, cfg.door_damping, cfg.door_friction, floor_friction,
                       cfg.robot_door_friction, cfg.door_stiffness, cfg.door_spring_reference],
                      dtype=np.float32)


def configure_scene(spec):
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            geom.contype, geom.conaffinity = 1, 0
    from mujoco_warp._src import collision_convex
    collision_convex.MJ_MAX_EPAHORIZON = max(collision_convex.MJ_MAX_EPAHORIZON, 64)
