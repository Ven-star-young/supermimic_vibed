"""Connect final per-chair references to the existing multi-chair experiment.

Keeps each chair in its original train/test partition and replaces its motion
list with exactly the selected reference. No simulation or GPU is initialized.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from motion_data.chair_split import cache_report, fingerprint, load_split, verify_files
from motion_data.collision import DEFAULT_CACHE_DIR
from motion_data.reference_training import load_reference_source

ROOT = Path(__file__).resolve().parents[1]


def prepare_reference_split(references, base_split, output, *, data_root=None, selection_path=None):
    """Freeze the reference whitelist without changing existing chair membership.

    An identical existing output is reusable. A changed selection requires a
    different output path, keeping resume/checkpoint fingerprints meaningful.
    """
    root, selected_groups, provenance = load_reference_source(
        references, data_root, selection_path=selection_path)
    base, _ = load_split(base_split)
    by_shape = {group["shape_key"]: group for group in selected_groups}
    partitions = {partition: [] for partition in ("train", "test")}
    found = set()
    for partition in partitions:
        for original in base[partition]:
            key = original["shape_key"]
            if key not in by_shape:
                continue
            selected = by_shape[key]
            record, = selected["clips"]
            matching = next((item for item in original["clips"] if item["path"] == record["path"]), None)
            if matching is None or any(record[field] != matching[field]
                    for field in ("sha256", "geometry_key", "collision_key")):
                raise ValueError(f"Selected reference differs from the original split: {record['path']}")
            partitions[partition].append(selected)
            found.add(key)
    if missing := by_shape.keys() - found:
        raise ValueError(f"{len(missing)} selected chair shapes are absent from the original split")
    if not all(partitions.values()):
        raise ValueError("Selected references must retain at least one original training and test chair")
    output = Path(output).resolve()
    manifest = {**base, **partitions,
                "data_root": os.path.relpath(root, output.parent),
                "reference_protocol": "one_reference_per_chair_preserve_partition",
                "parent_split_fingerprint": base["fingerprint"],
                "reference_source": provenance}
    manifest["fingerprint"] = fingerprint(manifest)
    if output.exists():
        existing, existing_root = load_split(output)
        if existing["fingerprint"] != manifest["fingerprint"] or existing_root != root:
            raise ValueError(f"{output} belongs to different reference choices or data. Use a new --output path.")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    # Use the same validation called by training, including both partitions here.
    for partition in partitions:
        verify_files(manifest, root, partition)
    return manifest, root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=ROOT / "data/selections/sitting_back_references.json")
    parser.add_argument("--selection", type=Path, help="Initial keep/reject file for live reference choices")
    parser.add_argument("--data-root", type=Path, help="Override the selected NPZ location, e.g. on another host")
    parser.add_argument("--base-split", type=Path, default=ROOT / "data/splits/sitting_80_20_seed0.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/splits/sitting_back_references_seed0.json")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    try:
        manifest, root = prepare_reference_split(args.references, args.base_split, args.output,
                                                 data_root=args.data_root, selection_path=args.selection)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Cannot prepare selected-reference training: {exc}\n")
    report = cache_report(manifest, args.cache_dir)
    print(json.dumps({partition: {k: v for k, v in values.items() if k != "missing"}
                      for partition, values in report.items()}, indent=2))
    print(f"References: {sum(len(manifest[p]) for p in ('train', 'test'))}; one per chair")
    print(f"Data root: {root}\nSplit: {args.output.resolve()}\nFingerprint: {manifest['fingerprint']}")
    if any(values["missing_collision_scenes"] for values in report.values()):
        print("Some collision assets are missing locally. Prepare this split with tools/prepare_collisions.py before training.")


if __name__ == "__main__":
    main()
