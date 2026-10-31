"""Adapt GVHMR/GMR G1 + door NPZ/XML pairs without changing their coordinates.

Usage: uv run python -m motion_data.converters.gmr SOURCE OUTPUT
"""

import argparse
import hashlib
from pathlib import Path
import shutil
import sys

import numpy as np

from motion_data import load_motion, save_motion
from motion_data.door import validate_door_asset
from ._common import g1_joints


def convert_gmr(path: Path) -> dict:
    """Validate portable GMR arrays and the paired XML, preserving diagnostics."""
    clip = load_motion(path)
    names, _ = g1_joints()
    if set(clip["joint_names"].tolist()) != set(names.tolist()):
        raise ValueError(f"{path}: expected all 29 G1 joints")
    if "root_quat_order" in clip and str(clip["root_quat_order"]) != "wxyz":
        raise ValueError(f"{path}: expected root_quat_order wxyz")
    if "door_coordinate_frame" in clip and str(clip["door_coordinate_frame"]) != "asset":
        raise ValueError(f"{path}: use the GMR export in the door asset coordinate frame")
    validate_door_asset(clip, path.with_suffix(".xml"))
    # Keep source FPS, joint order, hinge sign and all diagnostic arrays intact.
    return {
        **clip,
        "source_name": np.asarray(path.name),
        "source_sha256": np.asarray(hashlib.sha256(path.read_bytes()).hexdigest()),
        "source_xml_sha256": np.asarray(hashlib.sha256(path.with_suffix(".xml").read_bytes()).hexdigest()),
    }


def convert_pair(path: Path, output: Path, *, overwrite=False) -> bool:
    """Write one validated pair; never overwrite any source file."""
    path, output = Path(path), Path(output)
    target, source_xml = output / path.name, path.with_suffix(".xml")
    target_xml = target.with_suffix(".xml")
    if target.resolve() == path.resolve() or target_xml.resolve() == source_xml.resolve():
        raise ValueError("GMR conversion output must be separate from the source directory")
    clip = convert_gmr(path)
    if not overwrite and (target.exists() or target_xml.exists()):
        if target.exists() and target_xml.exists():
            previous = load_motion(target)
            signatures = ("source_sha256", "source_xml_sha256")
            if any(str(previous.get(key, "")) != str(clip[key]) for key in signatures):
                raise ValueError("Source pair changed or output has no source hashes; use --overwrite to refresh it")
            if hashlib.sha256(target_xml.read_bytes()).hexdigest() != str(clip["source_xml_sha256"]):
                raise ValueError("Prepared XML changed; use --overwrite to refresh it")
            validate_door_asset(previous, target_xml)
            return False
        raise ValueError("Incomplete output pair exists; use --overwrite to replace it")
    save_motion(target, clip)
    shutil.copy2(source_xml, target_xml)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="GMR NPZ or directory of NPZ/XML pairs")
    parser.add_argument("output", type=Path, help="Separate destination for portable NPZ/XML pairs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    files = [args.source] if args.source.is_file() else sorted(args.source.glob("*.npz"))
    if not files:
        parser.error(f"No GMR NPZ clips found in {args.source}")
    converted = skipped = 0
    for path in files:
        try:
            if convert_pair(path, args.output, overwrite=args.overwrite):
                converted += 1
            else:
                skipped += 1
        except (ValueError, OSError) as exc:
            parser.exit(1, f"{path}: {exc}\n")
    print(f"Converted {converted}, skipped {skipped} -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
