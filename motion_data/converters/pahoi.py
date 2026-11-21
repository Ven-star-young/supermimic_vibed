"""Convert reviewed PA-HOI G1 episodes to portable robot/object motion NPZs.

Usage: uv run python -m motion_data.converters.pahoi [source] [output]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from motion_data import validate_motion
from ._common import ROOT, convert_batch, g1_joints


def convert_pahoi(path: Path, source: Path, entry: dict, assets: dict, names: np.ndarray) -> dict:
    motion_path, metadata_path = source / entry["motion"], source / entry["metadata"]
    metadata = json.loads(metadata_path.read_text())
    if metadata["episode_id"] != path.name:
        raise ValueError("Episode metadata does not match the manifest")
    asset = assets[metadata["object_asset_id"]]
    if not asset["scale_baked"] or float(asset["runtime_scale"]) != 1.0:
        raise ValueError("Expected an object mesh with its size already baked in")
    mesh_path = source / asset["mesh_npz"]
    with np.load(motion_path, allow_pickle=False) as archive:
        motion = dict(archive)
    with np.load(mesh_path, allow_pickle=False) as archive:
        mesh = dict(archive)
    with np.load(source / entry["contact_task"], allow_pickle=False) as archive:
        contact = dict(archive)

    source_names = motion["joint_names"]
    if (source_names.shape != names.shape or source_names.dtype.kind != "U"
            or len(set(source_names)) != len(names) or set(source_names) != set(names)):
        raise ValueError("Expected the complete 29 named G1 joints")
    joints = motion["joint_positions_rad"]
    n = int(metadata["frames"])
    if joints.shape != (n, len(names)) or int(entry["frames"]) != n:
        raise ValueError("Motion and metadata frame counts differ")
    order = [source_names.tolist().index(name) for name in names]
    fps = float(metadata["fps"])
    timestamps = motion["timestamps_s"]
    if (timestamps.shape != (n,) or not np.isfinite(timestamps).all()
            or not np.isfinite(fps) or fps <= 0 or n < 2
            or not np.allclose(np.diff(timestamps), 1.0 / fps, rtol=1e-5, atol=1e-8)):
        raise ValueError("Expected uniformly sampled source timestamps matching fps")
    if (contact["active"].shape != (n, 3) or contact["weight"].shape != (n, 3)
            or not np.array_equal(contact["timestamps_s"], timestamps)
            or contact["patch_ids"].tolist() != [
                "left_hand_contact", "right_hand_contact", "front_torso_contact"]):
        raise ValueError("Expected aligned left-hand/right-hand/front-torso contact tasks")

    # Source poses already use metres, radians, Z-up and wxyz. Mesh dimensions
    # and Stage2 ground alignment are baked in; neither should be applied again.
    clip = dict(
        fps=np.asarray(fps), timestamps_s=timestamps,
        root_pos=motion["root_position_world_m"], root_quat=motion["root_quaternion_wxyz"],
        joint_pos=joints[:, order], joint_names=names,
        object_pos=motion["object_position_world_m"], object_quat=motion["object_quaternion_wxyz"],
        object_scale=np.ones(3), object_vertices=mesh["vertices"], object_faces=mesh["faces"],
        source_name=np.asarray(path.name),
        source_episode=np.asarray(os.path.relpath(path.resolve(), ROOT)),
        source_sha256=np.asarray(hashlib.sha256(motion_path.read_bytes()).hexdigest()),
        source_metadata_sha256=np.asarray(hashlib.sha256(metadata_path.read_bytes()).hexdigest()),
        source_mesh_sha256=np.asarray(hashlib.sha256(mesh_path.read_bytes()).hexdigest()),
    )
    # These are optimization targets/diagnostics, not measured contact forces.
    # Keep the larger tactile arrays in the source episode identified above.
    clip.update({f"contact_{key}": value for key, value in contact.items()})
    return validate_motion(clip)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=ROOT / "data/pahoi/hold")
    parser.add_argument("output", nargs="?", type=Path, default=ROOT / "data/motions/hold")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing NPZs (otherwise skip them)")
    args = parser.parse_args()
    try:
        names, _ = g1_joints()
        manifest = json.loads((args.source / "manifest.json").read_text())
        expected = dict(world="right_handed_Z_up", length="m", angle="rad", time="s",
                        quaternion="wxyz", object_scale_baked=True, runtime_scale=1.0,
                        motion_already_grounded=True)
        if (manifest["schema"] != "wbt.motion_geometric_tactile_dataset/v1"
                or any(manifest["conventions"].get(key) != value for key, value in expected.items())):
            raise ValueError("Expected reviewed PA-HOI data with grounded G1 poses and baked object scale")
        entries = {entry["episode_id"]: entry for entry in manifest["episodes"]}
        if not entries or len(entries) != len(manifest["episodes"]):
            raise ValueError("Expected a nonempty manifest with unique episode IDs")
        files = [args.source / "episodes" / name for name in sorted(entries)]
        if any(not path.is_dir() for path in files):
            raise FileNotFoundError("A manifest episode directory is missing")
        args.output.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")
    return convert_batch(
        files, args.output,
        lambda path: convert_pahoi(path, args.source, entries[path.name], manifest["object_assets"], names),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    sys.exit(main())
