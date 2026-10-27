"""Create a deterministic chair-level split without copying motion data."""
import argparse
import json
from pathlib import Path

from motion_data.chair_split import cache_report, create_split
from motion_data.collision import DEFAULT_CACHE_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=Path("data/motions/sitting"))
    parser.add_argument("--output", type=Path, default=Path("data/splits/sitting_80_20_seed0.json"))
    parser.add_argument("--train-fraction", type=float, default=.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    manifest = create_split(args.source, args.output, fraction=args.train_fraction, seed=args.seed)
    report = cache_report(manifest, args.cache_dir)
    print(json.dumps({k: {f: v for f, v in r.items() if f != "missing"} for k, r in report.items()}, indent=2))
    print(f"Saved {args.output}; split fingerprint: {manifest['fingerprint']}")


if __name__ == "__main__":
    main()
