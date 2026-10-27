"""Reproducible, chair-disjoint experiments over standard motion files."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from .collision import collision_key


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def shape_key(clip):
    """Identity excludes scale: differently scaled copies cannot cross splits.

    This groups identical mesh arrays, not semantic chair styles or near-duplicates.
    """
    digest = hashlib.sha256(b"chair-shape-v1")
    for field, dtype in (("object_vertices", "<f8"), ("object_faces", "<i4")):
        value = np.ascontiguousarray(clip[field], dtype=dtype)
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def scaled_key(clip):
    scale = np.asarray(clip["object_scale"], dtype="<f8")
    return collision_key(clip) + ":" + hashlib.sha256(scale.tobytes()).hexdigest()


def fingerprint(manifest):
    # Locations are relocatable; experimental membership and content are not.
    payload = {k: v for k, v in manifest.items() if k not in ("data_root", "fingerprint")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def create_split(source, output, *, fraction=.8, seed=0):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Train fraction must lie strictly between 0 and 1")
    groups = {}
    files = sorted(source.glob("*.npz"))
    if not files:
        raise ValueError(f"No standard motion NPZs in {source}")
    for path in files:
        with np.load(path, allow_pickle=False) as clip:
            required = {"object_vertices", "object_faces", "object_scale", "object_pos", "object_quat"}
            if not required <= set(clip.files):
                raise ValueError(f"Missing object fields: {path}")
            key = shape_key(clip)
            group = groups.setdefault(key, dict(shape_key=key, clips=[]))
            group["clips"].append(dict(path=path.name, sha256=file_digest(path),
                                       geometry_key=scaled_key(clip), collision_key=collision_key(clip)))
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("A chair-disjoint experiment requires at least two shapes")
    count = max(1, min(len(keys) - 1, math.floor(len(keys) * fraction)))
    selected = set(np.random.default_rng(seed).permutation(keys)[:count])
    manifest = dict(version=1, grouping="exact_unscaled_mesh_arrays", seed=seed,
                    requested_train_fraction=fraction,
                    data_root=os.path.relpath(source, output.parent),
                    train=[groups[k] for k in keys if k in selected],
                    test=[groups[k] for k in keys if k not in selected])
    manifest["fingerprint"] = fingerprint(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    # A split is an experiment artifact: never silently overwrite its membership.
    with output.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_split(path, data_root=None):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("version") != 1 or manifest.get("fingerprint") != fingerprint(manifest):
        raise ValueError("Invalid or modified chair split manifest")
    seen_shapes, seen_files, seen_geometry = set(), set(), set()
    for partition in ("train", "test"):
        if not manifest.get(partition):
            raise ValueError(f"Empty {partition} partition")
        for group in manifest[partition]:
            if group["shape_key"] in seen_shapes or not group["clips"]:
                raise ValueError("Chair shapes overlap across groups or a group is empty")
            seen_shapes.add(group["shape_key"])
            group_geometry = set()
            for clip in group["clips"]:
                relative = Path(clip["path"])
                if relative.is_absolute() or ".." in relative.parts or clip["path"] in seen_files:
                    raise ValueError("Repeated or unsafe motion filename in split")
                if clip["geometry_key"] in seen_geometry:
                    raise ValueError("Geometry occurs in multiple chair groups")
                seen_files.add(clip["path"])
                group_geometry.add(clip["geometry_key"])
            seen_geometry.update(group_geometry)
    root = Path(data_root).resolve() if data_root else (path.parent / manifest["data_root"]).resolve()
    return manifest, root


def scenes(manifest, partition):
    """Separate scale variants physically, retaining their common split group."""
    result = []
    for group in manifest[partition]:
        variants = {}
        for clip in group["clips"]:
            variants.setdefault(clip["geometry_key"], []).append(clip)
        for key, clips in sorted(variants.items()):
            result.append(dict(shape_key=group["shape_key"], geometry_key=key, clips=clips))
    return result


def verify_files(manifest, root, partition):
    """Validate only the selected partition; training never opens held-out clips."""
    for group in manifest[partition]:
        for clip in group["clips"]:
            path = root / clip["path"]
            if not path.is_file() or file_digest(path) != clip["sha256"]:
                raise ValueError(f"Missing or changed reference file: {path}")
            with np.load(path, allow_pickle=False) as source:
                if shape_key(source) != group["shape_key"] or scaled_key(source) != clip["geometry_key"]:
                    raise ValueError(f"Reference geometry no longer matches split: {path}")


def cache_report(manifest, cache_dir):
    result = {}
    for partition in ("train", "test"):
        missing = []
        entries = scenes(manifest, partition)
        for scene in entries:
            clip = scene["clips"][0]
            if not (Path(cache_dir) / (clip["collision_key"] + ".npz")).is_file():
                missing.append(clip["path"])
        result[partition] = dict(chairs=len(manifest[partition]), scenes=len(entries),
                                 clips=sum(len(g["clips"]) for g in manifest[partition]),
                                 missing_collision_scenes=len(missing), missing=missing)
    return result
