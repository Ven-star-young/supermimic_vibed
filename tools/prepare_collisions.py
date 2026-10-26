"""Prepare reusable convex object collision assets from standard motion NPZs.

uv run --extra motion python tools/prepare_collisions.py data/motions/sitting
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from pathlib import Path
import sys

# Spawned workers inherit this before loading the CoACD/OpenMP shared library.
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

from motion_data.collision import DEFAULT_CACHE_DIR, collision_key, prepare_collision


def _read_mesh(path):
    with np.load(path, allow_pickle=False) as archive:
        if not {"object_vertices", "object_faces", "object_pos", "object_quat", "object_scale"} & set(archive.files):
            return None
        return {key: archive[key] for key in ("object_vertices", "object_faces")}


def _prepare_one(path, cache_dir):
    clip = _read_mesh(path)
    target = Path(cache_dir) / f"{collision_key(clip)}.npz"
    cached = target.exists()
    prepare_collision(clip, cache_dir)
    with np.load(target, allow_pickle=False) as data:
        count = len(data["vertex_offsets"]) - 1
    return target, cached, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path, help="Standard motion NPZ or directory")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Process at most this many unique object meshes")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    files = sorted(args.motion.glob("*.npz")) if args.motion.is_dir() else [args.motion]
    if not files or any(not path.is_file() for path in files):
        parser.error(f"No motion NPZ files at {args.motion}")
    unique = {}
    failed = skipped = 0
    for index, path in enumerate(files, 1):
        try:
            clip = _read_mesh(path)
            if clip is None:
                skipped += 1
                continue
            key = collision_key(clip)
            unique.setdefault(key, path)
        except Exception as exc:
            failed += 1
            print(f"FAILED {path.name}: {exc}", file=sys.stderr, flush=True)
        if args.limit is not None and len(unique) >= args.limit:
            break
        if index % 250 == 0:
            print(f"Scanned {index}/{len(files)} clips; {len(unique)} unique meshes", flush=True)
    print(f"Preparing {len(unique)} unique meshes with {args.workers} workers, "
          f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}; skipped {skipped} robot-only clips",
          flush=True)
    completed = reused = 0
    # spawn avoids inheriting initialized OpenMP state and only imports this
    # NumPy/CoACD utility in workers, without the training stack.
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(_prepare_one, path, args.cache_dir): path for path in unique.values()}
        for index, future in enumerate(as_completed(pending), 1):
            path = pending[future]
            try:
                target, cached, count = future.result()
                reused += int(cached)
                completed += int(not cached)
                print(f"[{index}/{len(unique)}] {'cached' if cached else 'created'} "
                      f"{path.name}: {count} parts -> {target.name}", flush=True)
            except Exception as exc:
                failed += 1
                print(f"FAILED {path.name}: {exc}", file=sys.stderr, flush=True)
    print(f"Created {completed}, reused {reused}, failed {failed} -> {args.cache_dir}", flush=True)
    return int(failed > 0)


if __name__ == "__main__":
    sys.exit(main())
