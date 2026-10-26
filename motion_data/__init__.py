"""Portable kinematic clips: metres, radians, Z-up, world poses, wxyz quaternions.

Required: fps, root_pos (T,3), root_quat (T,4), joint_pos (T,J), joint_names (J,).
Optional: object_pos (T,3), object_quat (T,4), object_scale (3,),
object_vertices (V,3) in the unscaled object frame, object_faces (F,3).
Only numeric/string arrays are stored; loading never requires pickle.
"""

from pathlib import Path
import os
import tempfile

import numpy as np


def validate_motion(clip: dict) -> dict:
    """Validate a clip and return arrays with normalized wxyz quaternions."""
    clip = {key: np.asarray(value) for key, value in clip.items()}
    required = {"fps", "root_pos", "root_quat", "joint_pos", "joint_names"}
    if missing := required - clip.keys():
        raise ValueError(f"Missing motion fields: {sorted(missing)}")
    if clip["fps"].shape != () or not np.isfinite(clip["fps"]) or clip["fps"] <= 0:
        raise ValueError("fps must be a finite positive scalar")
    names = clip["joint_names"]
    if names.ndim != 1 or names.dtype.kind != "U" or len(set(names)) != len(names):
        raise ValueError("joint_names must be a unique list of strings")
    joints = clip["joint_pos"]
    if joints.ndim != 2 or joints.shape[1] != len(names) or not len(joints):
        raise ValueError("joint_pos must have shape (T,J), with T > 0")
    n = len(joints)
    shapes = {"root_pos": (n, 3), "root_quat": (n, 4), "joint_pos": joints.shape}
    object_keys = {"object_pos", "object_quat", "object_scale"}
    if object_keys & clip.keys():
        if not object_keys <= clip.keys():
            raise ValueError("Object poses require object_pos, object_quat and object_scale")
        shapes.update(object_pos=(n, 3), object_quat=(n, 4), object_scale=(3,))
    for key, shape in shapes.items():
        value = clip[key]
        if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain finite numbers with shape {shape}")
        if key.endswith("quat"):
            norm = np.linalg.norm(value, axis=-1, keepdims=True)
            if (norm < 1e-8).any():
                raise ValueError(f"{key} contains a zero quaternion")
            clip[key] = value / norm
    if "object_scale" in clip and (clip["object_scale"] <= 0).any():
        raise ValueError("object_scale must be positive")
    mesh_keys = {"object_vertices", "object_faces"}
    if mesh_keys & clip.keys():
        if not (mesh_keys | object_keys) <= clip.keys():
            raise ValueError("An object mesh requires vertices, faces, poses and scale")
        vertices, faces = clip["object_vertices"], clip["object_faces"]
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4
                or vertices.dtype.kind not in "fiu" or not np.isfinite(vertices).all()):
            raise ValueError("object_vertices must contain finite (V,3) coordinates")
        if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or faces.dtype.kind not in "iu" or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("object_faces must contain valid (F,3) vertex indices")
    if any(value.dtype.hasobject for value in clip.values()):
        raise ValueError("Object/pickle arrays are not supported")
    return clip


def load_motion(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return validate_motion(dict(archive))


def save_motion(path: str | Path, clip: dict) -> None:
    """Validate and atomically write the same format consumed by training/replay."""
    path = Path(path)
    clip = validate_motion(clip)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
            temporary = Path(f.name)
            np.savez_compressed(f, **clip)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
