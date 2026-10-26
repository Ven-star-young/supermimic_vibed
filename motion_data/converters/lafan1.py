"""Convert LAFAN1 G1 CSVs to portable motion NPZs.

Usage: uv run --extra motion python -m motion_data.converters.lafan1 [source] [output] --fps 30
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from motion_data import validate_motion
from ._common import ROOT, convert_batch, g1_joints


def convert_csv(path: Path, fps: float, names: np.ndarray) -> dict:
    raw = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)
    if raw.shape[1] != 7 + len(names):
        raise ValueError(f"Expected {7 + len(names)} columns, got {raw.shape[1]}")
    return validate_motion(dict(
        fps=np.asarray(fps), root_pos=raw[:, :3], root_quat=raw[:, [6, 3, 4, 5]],
        joint_pos=raw[:, 7:], joint_names=names, source_name=np.asarray(path.name),
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=ROOT / "data/lafan1_g1")
    parser.add_argument("output", nargs="?", type=Path, default=ROOT / "data/motions/lafan1")
    parser.add_argument("--fps", type=float, default=30.0, help="Source CSV frame rate (default: 30)")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing NPZs (otherwise skip them)")
    args = parser.parse_args()
    try:
        names, _ = g1_joints()
        files = [args.source] if args.source.is_file() and args.source.suffix == ".csv" else sorted(args.source.glob("*.csv"))
        if not files:
            raise ValueError(f"No LAFAN1 CSVs found in {args.source}")
        args.output.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        parser.exit(1, f"Error: {exc}\n")
    return convert_batch(
        files, args.output, lambda path: convert_csv(path, args.fps, names), overwrite=args.overwrite
    )


if __name__ == "__main__":
    sys.exit(main())
