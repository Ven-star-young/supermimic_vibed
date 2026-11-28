"""Reviewed video11 reference and its fixed wall/platform collision scene."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from motion_data import load_motion
from ...assets import DATA_DIR
from ...assets.g1.g1_constants import make_g1_spec


DEFAULT_MOTION = DATA_DIR / "motions/ledge_video11/video11_g1_ledge.npz"
PLATFORMS = ("start_platform", "narrow_ledge", "goal_platform")
WALLS = ("entry_wall", "central_wall", "exit_wall")


def selected_paths(command):
    motion = Path(command.motion_dir).resolve()
    if motion.is_dir():
        files = sorted(motion.glob("*.npz"))
        if len(files) != 1:
            raise ValueError("Ledge requires exactly one reviewed reference and matching static scene")
        motion = files[0]
    if not motion.is_file() or motion.suffix != ".npz":
        raise ValueError(f"Ledge reference must be a standard NPZ: {motion}")
    scene = Path(command.scene_xml).resolve() if command.scene_xml else motion.with_suffix(".xml")
    return motion, scene, motion.with_suffix(".json")


def selected_asset(command):
    paths = selected_paths(command)
    signatures = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
    return _load_asset(signatures)


@lru_cache(maxsize=4)
def _load_asset(signatures):
    motion_path, scene_path, metadata_path = (Path(item[0]) for item in signatures)
    clip = load_motion(motion_path)
    provenance = json.loads(metadata_path.read_text())
    scene_hash = hashlib.sha256(scene_path.read_bytes()).hexdigest()
    motion_hash = hashlib.sha256(motion_path.read_bytes()).hexdigest()
    if (str(clip.get("scene_xml_sha256", "")) != scene_hash
            or provenance.get("scene_xml_sha256") != scene_hash
            or provenance.get("reference_sha256") != motion_hash):
        raise ValueError("Ledge NPZ/XML/JSON hashes differ; use the matching reviewed asset set")
    count = len(clip["root_pos"])
    for key, shape in (("back_required", (count,)), ("back_alignment_weight", (count,)),
                       ("planned_stance", (count, 2))):
        if key not in clip or clip[key].shape != shape or not np.isfinite(clip[key]).all():
            raise ValueError(f"Ledge reference requires finite {key}{shape}")
    if (clip["back_required"].dtype != np.bool_ or clip["planned_stance"].dtype != np.bool_
            or not clip["planned_stance"].any(axis=-1).all()
            or np.any((clip["back_alignment_weight"] < 0) | (clip["back_alignment_weight"] > 1))):
        raise ValueError("Invalid planned ledge support/back phase labels")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    if model.nq or model.nv or model.nmocap:
        raise ValueError("Ledge scene must be entirely static")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    metadata = {"scene_sha256": scene_hash, "reference_sha256": motion_hash}
    for kind, names, collision_type in (("platform", PLATFORMS, 1), ("wall", WALLS, 2)):
        ids = [model.geom(name).id for name in names]
        if (np.any(model.geom_type[ids] != mujoco.mjtGeom.mjGEOM_BOX)
                or not np.allclose(data.geom_xmat[ids].reshape(-1, 3, 3), np.eye(3), atol=1e-7)
                or np.any(model.geom_contype[ids] != collision_type)
                or np.any(model.geom_conaffinity[ids] != 0)):
            raise ValueError(f"Ledge {kind}s require axis-aligned boxes with the reviewed collision roles")
        metadata[kind + "_center"] = data.geom_xpos[ids].copy()
        metadata[kind + "_half_size"] = model.geom_size[ids].copy()
    heights = metadata["platform_center"][:, 2] + metadata["platform_half_size"][:, 2]
    fronts = metadata["wall_center"][:, 1] + metadata["wall_half_size"][:, 1]
    floor = model.geom("lower_floor").id
    if (not np.allclose(heights, heights[0], atol=1e-6) or not np.allclose(fronts, 0., atol=1e-6)
            or model.geom_type[floor] != mujoco.mjtGeom.mjGEOM_PLANE
            or not np.allclose(data.geom_xmat[floor].reshape(3, 3), np.eye(3))):
        raise ValueError("Ledge requires a common platform height, wall front Y=0 and horizontal lower floor")
    metadata.update(platform_height=float(heights[0]), floor_height=float(data.geom_xpos[floor, 2]))
    return clip, metadata


def make_robot_spec():
    spec = make_g1_spec(interaction_collisions=True)
    # Retain the explicit left/right foot collision used to validate this reference.
    spec.add_pair(name="ledge_feet", geomname1="left_ankle_roll_link_object_collision",
                  geomname2="right_ankle_roll_link_object_collision")
    return spec


def configure_scene(spec, *, command):
    selected_asset(command)  # Validate before attaching any scene geometry.
    _, scene_path, _ = selected_paths(command)
    scene = mujoco.MjSpec.from_file(str(scene_path))
    physical = command.physical_cfg
    # Equal-priority contacts take max(robot, platform) friction. This lower
    # bound makes the sampled robot friction in context also the contact friction.
    ground_friction = physical.evaluation_friction if command.play else physical.friction[0]
    for name in (*PLATFORMS, "lower_floor"):
        scene.geom(name).friction[0] = ground_friction
    spec.attach(scene, prefix="ledge/", frame=spec.worldbody.add_frame())
    from mujoco_warp._src import collision_convex
    collision_convex.MJ_MAX_EPAHORIZON = max(collision_convex.MJ_MAX_EPAHORIZON, 64)
