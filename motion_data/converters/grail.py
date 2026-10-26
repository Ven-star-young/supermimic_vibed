"""Convert GRAIL robot/object motion directories to portable motion NPZs.

Usage: uv run --extra motion python -m motion_data.converters.grail [source] [output]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from motion_data import validate_motion
from ._common import ROOT, convert_batch, g1_joints


def read_entry(path: Path) -> tuple[str, dict]:
    import joblib

    data = joblib.load(path)
    if not isinstance(data, dict) or len(data) != 1:
        raise ValueError(f"{path}: expected exactly one motion entry")
    name, entry = next(iter(data.items()))
    if not isinstance(name, str) or not isinstance(entry, dict):
        raise ValueError(f"{path}: expected a named motion dictionary")
    return name, entry


def find_usd_dir(source: Path, explicit: Path | None) -> Path:
    candidates = [explicit] if explicit else [source / "object_usd"]
    manifest = source / "COPY_MANIFEST.json"
    if explicit is None and manifest.exists():
        original = json.loads(manifest.read_text()).get("source")
        if original:
            candidates.append(Path(original) / "object_usd")
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError("Missing object USDs; specify --object-usd-dir PATH")


def read_usd_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract an object's mesh in its original, unscaled local coordinates."""
    from pxr import Usd, UsdGeom

    if not path.is_file():
        raise FileNotFoundError(path)
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Cannot open USD: {path}")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
        raise ValueError(f"{path}: expected a Z-up USD authored in metres")
    cache = UsdGeom.XformCache()
    vertices, faces, offset = [], [], 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        transform = np.asarray(cache.GetLocalToWorldTransform(prim))
        # USD matrices use row vectors. Asset transforms are separate from the
        # trajectory's object pose/scale, which stay in their own NPZ fields.
        points = points @ transform[:3, :3] + transform[3, :3]
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        if not np.all(counts == 3) or len(indices) != 3 * len(counts):
            raise ValueError(f"{prim.GetPath()}: expected triangulated USD mesh")
        triangles = indices.reshape(-1, 3)
        if (mesh.GetOrientationAttr().Get() == "leftHanded") ^ (np.linalg.det(transform[:3, :3]) < 0):
            triangles = triangles[:, [0, 2, 1]]
        vertices.append(points)
        faces.append(triangles + offset)
        offset += len(points)
    if not vertices:
        raise ValueError(f"No meshes in {path}")
    return np.concatenate(vertices).astype(np.float32), np.concatenate(faces).astype(np.int32)


def convert_grail(path: Path, usd_dir: Path, names: np.ndarray, axes: np.ndarray) -> dict:
    import joblib

    source = path.parent.parent
    key, robot = read_entry(path)
    object_key, obj = read_entry(source / "objects" / path.name)
    meta = joblib.load(source / "meta" / path.name)
    if object_key != key:
        raise ValueError("Robot and object motion keys differ")
    if not isinstance(meta, dict) or float(meta.get("scene_scale", 1.0)) != 1.0:
        raise ValueError("Expected GRAIL metadata with scene_scale=1")
    if float(robot["fps"]) != float(obj["fps"]):
        raise ValueError("Robot and object frame rates differ")
    joints = np.asarray(robot["dof"])
    pose_aa = np.asarray(robot["pose_aa"])
    if joints.ndim != 2 or joints.shape[1] != len(names):
        raise ValueError("Expected G1 motion with 29 joint angles")
    if pose_aa.shape != (len(joints), len(names) + 1, 3) or not np.allclose(
        pose_aa[:, 1:], joints[..., None] * axes, atol=2e-4, rtol=1e-4
    ):
        raise ValueError("pose_aa does not match the G1 joint order and axes")
    object_pos, object_quat = np.asarray(obj["root_pos"]), np.asarray(obj["root_quat"])
    if object_pos.shape != (len(joints), 1, 3) or object_quat.shape != (len(joints), 1, 4):
        raise ValueError("Expected one object with the same frame count as the robot")
    vertices, faces = read_usd_mesh(usd_dir / f"{path.stem}.usd")
    return validate_motion(dict(
        fps=np.asarray(robot["fps"]), root_pos=np.asarray(robot["root_trans_offset"]),
        root_quat=np.asarray(robot["root_rot"])[:, [3, 0, 1, 2]],
        joint_pos=joints, joint_names=names, object_pos=object_pos[:, 0],
        # GRAIL writes robot xyzw but object wxyz; see export_successful_rollouts.py
        # https://github.com/NVlabs/GRAIL/blob/main/grail/data_export/export_successful_rollouts.py
        object_quat=object_quat[:, 0], object_scale=np.asarray(obj["scale"]).reshape(3),
        object_vertices=vertices, object_faces=faces, source_name=np.asarray(key),
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=ROOT / "data/grail/sitting")
    parser.add_argument("output", nargs="?", type=Path, default=ROOT / "data/motions/sitting")
    parser.add_argument("--object-usd-dir", type=Path, help="object_usd directory; otherwise discover from source/manifest")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing NPZs (otherwise skip them)")
    args = parser.parse_args()
    try:
        names, axes = g1_joints()
        usd_dir = find_usd_dir(args.source, args.object_usd_dir)
        files = sorted((args.source / "robot").glob("*.pkl"))
        if not files:
            raise ValueError(f"No GRAIL motions found in {args.source / 'robot'}")
        expected = {p.name for p in files}
        for folder in ("objects", "meta"):
            actual = {p.name for p in (args.source / folder).glob("*.pkl")}
            if actual != expected:
                raise ValueError(f"{folder}/ pairing differs: missing={sorted(expected - actual)[:3]}, extra={sorted(actual - expected)[:3]}")
        args.output.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")
    return convert_batch(
        files, args.output, lambda path: convert_grail(path, usd_dir, names, axes), overwrite=args.overwrite
    )


if __name__ == "__main__":
    sys.exit(main())
