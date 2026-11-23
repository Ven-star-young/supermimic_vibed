"""Import Kimodo's native G1 CSV export with an authored door trajectory.

Usage: python -m motion_data.converters.kimodo SOURCE.csv OUTPUT \
    --metadata SOURCE.json --door-reference DOOR.npz

CSV columns are root xyz, root wxyz and 29 joint angles. Metadata supplies fps,
joint_names, world_offset (XY translation only), door_angle and optional provenance.
This importer does not retarget, rotate, resample or repair the generated motion.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from motion_data import load_motion, save_motion, validate_motion
from motion_data.door import DOOR_METADATA_SHAPES, validate_door_asset
from ._common import g1_joints


def convert_csv(path: Path, metadata_path: Path, door_reference: Path) -> dict:
    """Keep the exported wxyz and named joint order; translate into the door frame."""
    path, metadata_path, door_reference = map(Path, (path, metadata_path, door_reference))
    metadata = json.loads(metadata_path.read_text())
    required = {"fps", "joint_names", "world_offset", "door_angle"}
    if not isinstance(metadata, dict) or not required <= metadata.keys():
        raise ValueError(f"Metadata must be an object containing {sorted(required)}")
    names = metadata["joint_names"]
    expected, _ = g1_joints()
    if (not isinstance(names, list) or len(names) != 29
            or any(not isinstance(name, str) for name in names)
            or set(names) != set(expected.tolist())):
        raise ValueError("Metadata joint_names must list all 29 G1 joints exactly once")
    offset = np.asarray(metadata["world_offset"], dtype=np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all() or offset[2] != 0:
        raise ValueError("world_offset must be finite [x, y, 0]; only XY translation is supported")
    raw = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if raw.shape[1] != 36 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Kimodo G1 CSV must contain finite rows of 36 columns: xyz, wxyz, 29 joints")
    source_xml = door_reference.with_suffix(".xml")
    template = load_motion(door_reference)
    validate_door_asset(template, source_xml)
    clip = {
        "fps": np.asarray(metadata["fps"], dtype=np.float64),
        "root_pos": raw[:, :3] + offset,
        "root_quat": raw[:, 3:7],
        "joint_pos": raw[:, 7:],
        "joint_names": np.asarray(names),
        **{f"door_{key}": template[f"door_{key}"] for key in DOOR_METADATA_SHAPES},
        "door_angle": np.asarray(metadata["door_angle"], dtype=np.float64),
        "door_coordinate_frame": np.asarray("asset"),
        "door_contact_region": np.asarray("back"),
        "root_quat_order": np.asarray("wxyz"),
        "world_offset": offset,
        "kinematic_only": np.asarray(True),
        "door_angle_source": np.asarray("authored_kinematic"),
        "source_name": np.asarray(path.name),
        "source_sha256": np.asarray(hashlib.sha256(path.read_bytes()).hexdigest()),
        "source_metadata_sha256": np.asarray(hashlib.sha256(metadata_path.read_bytes()).hexdigest()),
        "source_xml_sha256": np.asarray(hashlib.sha256(source_xml.read_bytes()).hexdigest()),
        "door_reference_sha256": np.asarray(hashlib.sha256(door_reference.read_bytes()).hexdigest()),
    }
    if "provenance" in metadata:
        clip["provenance"] = np.asarray(json.dumps(metadata["provenance"], ensure_ascii=False,
                                                   sort_keys=True, allow_nan=False))
    clip = validate_motion(clip)
    validate_door_asset(clip, source_xml)
    return clip


def convert_pair(path: Path, output: Path, metadata_path: Path, door_reference: Path,
                 *, overwrite=False) -> Path:
    """Write a new NPZ/XML pair without replacing any input file."""
    path, output, metadata_path, door_reference = map(
        Path, (path, output, metadata_path, door_reference))
    target = output / f"{path.stem}.npz"
    target_xml, source_xml = target.with_suffix(".xml"), door_reference.with_suffix(".xml")
    inputs = {source.resolve() for source in (path, metadata_path, door_reference, source_xml)}
    if target.resolve() in inputs or target_xml.resolve() in inputs:
        raise ValueError("Kimodo conversion must not overwrite an input CSV, metadata or door reference")
    if not overwrite and (target.exists() or target_xml.exists()):
        raise ValueError("Output NPZ/XML already exists; use --overwrite to replace it")
    clip = convert_csv(path, metadata_path, door_reference)
    save_motion(target, clip)
    shutil.copy2(source_xml, target_xml)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Native Kimodo G1 CSV export (xyz, wxyz, 29 joints)")
    parser.add_argument("output", type=Path, help="Destination directory for the NPZ/XML pair")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--door-reference", type=Path, required=True, help="Existing NPZ with matching XML")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        target = convert_pair(args.source, args.output, args.metadata, args.door_reference,
                              overwrite=args.overwrite)
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(1, f"{args.source}: {exc}\n")
    print(f"Converted {args.source} -> {target} (kinematic reference only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
