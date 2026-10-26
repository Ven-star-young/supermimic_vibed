"""Small helpers shared by the source-specific conversion commands."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

from motion_data import save_motion

ROOT = Path(__file__).resolve().parents[2]


def g1_joints() -> tuple[np.ndarray, np.ndarray]:
    """Both sources use the G1's depth-first MJCF joint order."""
    xml = ET.parse(ROOT / "data/g1_description/g1_29dof.xml")
    joints = xml.findall("./worldbody//joint")
    if len(joints) != 29 or any(j.get("type", "hinge") != "hinge" for j in joints):
        raise ValueError("Expected the 29 hinge joints of the G1 robot")
    names = np.asarray([j.attrib["name"] for j in joints])
    axes = np.asarray([np.fromstring(j.get("axis", "0 0 1"), sep=" ") for j in joints])
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    return names, axes


def convert_batch(
    files: Sequence[Path], output_dir: Path, convert: Callable[[Path], dict], *, overwrite: bool
) -> int:
    """Convert independently, retaining completed files if a batch fails."""
    converted = skipped = failed = 0
    for index, path in enumerate(files, 1):
        output = output_dir / f"{path.stem}.npz"
        if output.exists() and not overwrite:
            skipped += 1
            continue
        try:
            save_motion(output, convert(path))
            converted += 1
        except Exception as exc:
            failed += 1
            print(f"FAILED {path.name}: {exc}", file=sys.stderr, flush=True)
        if index % 100 == 0 or index == len(files):
            print(f"{index}/{len(files)} processed", flush=True)
    print(f"Converted {converted}, skipped {skipped}, failed {failed} -> {output_dir}")
    return int(failed > 0)
