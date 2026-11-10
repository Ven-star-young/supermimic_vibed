"""Frozen G1 skin points and the original 112-channel membership, independent of the simulator."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


_ASSETS = Path(__file__).with_suffix("")
_ZONES = ("chest", "back", "left_arm", "right_arm", "left_shoulder", "right_shoulder")


@dataclass(frozen=True)
class G1TactileLayout:
    name: str
    point_ids: tuple[str, ...]
    link_names: tuple[str, ...]
    positions: np.ndarray
    normals: np.ndarray
    radii: np.ndarray
    channel_members: tuple[np.ndarray, ...]
    channel_names: tuple[str, ...]
    channel_zones: tuple[str, ...]
    raw_indices: np.ndarray
    geometry_sha256: str
    channel_abi_sha256: str


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_tactile_layout(layout: str = "original_outer") -> G1TactileLayout:
    """Load 464 link-local points (metres) without changing IDs, order, or geometry.

    ``original_outer`` is the original physical-demo layout; ``inner_arms_v1``
    is the historical inward-facing arm layout used by the Genesis HOI replay.
    Raw indices are zero-based positions in the 256-byte JQ upper-body packet.
    """
    if layout not in ("original_outer", "inner_arms_v1"):
        raise ValueError(f"Unknown G1 tactile layout: {layout!r}")
    directory = _ASSETS / layout
    document = json.loads((directory / "points.json").read_text())
    mapping = json.loads((directory / "jq-channel-map.json").read_text())
    abi = json.loads((_ASSETS / "upper-body-channel-layout.json").read_text())
    if tuple(document.get(k) for k in ("schema", "units", "coordinate_frame", "surface")) != (
        "wbt.tactile-points/v1", "m", "link", "visual"
    ):
        raise ValueError("Expected the original link-local visual tactile layout in metres")
    points = [point for group in document["groups"] for point in group["points"]]
    point_ids = tuple(point["id"] for point in points)
    if len(point_ids) != 464 or len(set(point_ids)) != 464:
        raise ValueError("Expected 464 unique stable tactile point IDs")
    positions = np.asarray([point["pos"] for point in points], dtype=np.float64)
    normals = np.asarray([point["normal"] for point in points], dtype=np.float64)
    radii = np.asarray([point["radius_m"] for point in points], dtype=np.float64)
    if (positions.shape != (464, 3) or normals.shape != (464, 3) or radii.shape != (464,)
            or not all(np.isfinite(value).all() for value in (positions, normals, radii))
            or not np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)
            or np.any(radii <= 0)):
        raise ValueError("Invalid tactile positions, unit normals, or sensing radii")
    geometry_hash = _digest(sorted([
        {key: point[key] for key in ("id", "link_name", "pos", "normal")} for point in points
    ], key=lambda point: point["id"]))
    zones = abi["zones"]
    abi_hash = _digest(zones)
    raw = [index for zone in zones for index in zone["raw_indices_1based"]]
    if (tuple(zone["name"] for zone in zones) != _ZONES
            or tuple(zone["channel_count"] for zone in zones) != (48, 40, 8, 8, 4, 4)
            or len(raw) != 112 or len(set(raw)) != 112
            or any(type(index) is not int or not 1 <= index <= 256 for index in raw)
            or any(zone["channel_count"] != len(zone["raw_indices_1based"])
                   or np.prod(zone["display_grid_shape"]) != zone["channel_count"] for zone in zones)):
        raise ValueError("Invalid JQ upper-body channel ABI")
    if (mapping.get("schema") != "wbt.jq-channel-map/v1"
            or mapping.get("model") != document["model"]
            or mapping.get("geometry_sha256") != geometry_hash
            or mapping.get("channel_abi_sha256") != abi_hash):
        raise ValueError("Tactile geometry, source model, and channel ABI hashes do not match")

    owners = {}
    for group in document["groups"]:
        for point in group["points"]:
            zone = {"前胸": "chest", "后胸": "back", "左臂": "left_arm", "右臂": "right_arm"}.get(group["name"])
            if group["name"] == "肩部":
                zone = "left_shoulder" if point["link_name"].startswith("left_") else "right_shoulder"
            owners[point["id"]] = zone
    lookup = {point_id: index for index, point_id in enumerate(point_ids)}
    expected = [(zone["name"], local, raw_index)
                for zone in zones for local, raw_index in enumerate(zone["raw_indices_1based"])]
    channels = mapping["channels"]
    if len(channels) != 112:
        raise ValueError("Expected 112 tactile channels")
    used, members, channel_names, channel_zones = [], [], [], []
    for index, (channel, (zone, local, raw_index)) in enumerate(zip(channels, expected, strict=True)):
        if tuple(channel.get(key) for key in ("state_index", "zone", "zone_index", "raw_index_1based")) != (
            index, zone, local, raw_index
        ):
            raise ValueError("Tactile channel order differs from the original JQ ABI")
        ids = channel["point_ids"]
        if not ids or any(point_id not in lookup or owners[point_id] != zone for point_id in ids):
            raise ValueError("Tactile channel has missing points or points from another skin region")
        used.extend(ids)
        members.append(np.asarray([lookup[point_id] for point_id in ids], dtype=np.int64))
        channel_names.append(f"upper_body.{zone}.{local:02d}")
        channel_zones.append(zone)
    if len(used) != 464 or len(set(used)) != 464:
        raise ValueError("Each tactile point must belong to exactly one channel")
    raw_indices = np.asarray(raw, dtype=np.int64) - 1
    for value in (positions, normals, radii, raw_indices, *members):
        value.setflags(write=False)
    return G1TactileLayout(
        layout, point_ids, tuple(point["link_name"] for point in points), positions, normals, radii,
        tuple(members), tuple(channel_names), tuple(channel_zones), raw_indices, geometry_hash, abi_hash,
    )
