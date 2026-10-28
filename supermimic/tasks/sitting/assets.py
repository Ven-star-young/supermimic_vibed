"""One rigid object per scene, selected by collision shape and physical scale.

Assets are loaded lazily when the scene is built. Motion files and collision
caches remain read-only; missing decomposition is an actionable error.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import warnings

import mujoco
import numpy as np

from motion_data import load_motion
from motion_data.collision import add_object_geoms, collision_key, load_collision_parts


def geometry_key(clip, profile="default"):
    scale = np.asarray(clip["object_scale"], dtype="<f8")
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Object scale must be a positive finite 3-vector")
    return collision_key(clip, profile=profile) + ":" + hashlib.sha256(scale.tobytes()).hexdigest()


def _signature(path):
    path = Path(path).resolve()
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


def _read_geometry(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in ("object_vertices", "object_faces", "object_scale")}


@lru_cache(maxsize=8)
def _select(signatures, anchor):
    key = geometry_key(_read_geometry(anchor[0]))
    selected = tuple(Path(path) for path, _, _ in signatures
                     if geometry_key(_read_geometry(path)) == key)
    if not selected:
        raise ValueError("No motion in motion_dir matches object_motion_file's collision shape and scale")
    print(f"[sitting] Selected {len(selected)}/{len(signatures)} clips matching "
          f"{Path(anchor[0]).name}; excluded {len(signatures) - len(selected)} clips of other shapes/scales.")
    return selected


def select_motion_files(cfg):
    path = Path(cfg.motion_dir)
    explicit = getattr(cfg, "motion_files", ())
    files = [Path(file) for file in explicit] if explicit else ([path] if path.is_file() else sorted(path.glob("*.npz")))
    if not files:
        raise FileNotFoundError(f"No standard motion NPZ files found in {path}")
    anchor = Path(cfg.object_motion_file) if cfg.object_motion_file else files[0]
    return list(_select(tuple(_signature(file) for file in files), _signature(anchor)))


def mass_properties(clip, mass=10.0):
    """Uniform-density inertia from the original scaled mesh, about its COM.

    Convex parts can overlap, so they must not be summed for inertia. Trimesh's
    limited triangle/quad hole repair affects this calculation only. Open or
    invalid meshes fail instead of silently using a box or a whole-chair hull.
    """
    import trimesh

    mesh = trimesh.Trimesh(
        vertices=clip["object_vertices"] * clip["object_scale"],
        faces=clip["object_faces"], process=True,
    )
    before = len(mesh.faces)
    trimesh.repair.fill_holes(mesh)
    mesh.fix_normals(multibody=True)
    if not mesh.is_watertight or not np.isfinite(mesh.volume) or mesh.volume <= 0:
        raise ValueError("Object inertia requires a closed positive-volume mesh after limited hole repair")
    if len(mesh.faces) != before:
        warnings.warn(f"Closed {len(mesh.faces) - before} boundary triangles for object inertia only; "
                      "visual and collision geometry are unchanged.", stacklevel=2)
    mesh.density = mass / mesh.volume
    com, inertia = mesh.center_mass, mesh.moment_inertia
    values = np.linalg.eigvalsh(inertia)
    if not np.isfinite(com).all() or not np.isfinite(inertia).all() or values.min() <= 0:
        raise ValueError("Object mesh produced invalid center of mass or inertia")
    if values[-1] > values[:2].sum() * (1 + 1e-6):
        raise ValueError("Object mesh inertia violates the rigid-body triangle inequality")
    return np.asarray(com), np.asarray(inertia)


def sample_geometry(parts, scale, count=1024):
    """Deterministic area sampling of physical collision surfaces and normals."""
    triangles = np.concatenate([vertices[faces] * scale for vertices, faces in parts])
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice_area = np.linalg.norm(normals, axis=1)
    valid = np.isfinite(twice_area) & (twice_area > 1e-12)
    if not valid.any():
        raise ValueError("Collision geometry contains no finite positive-area triangles")
    triangles, normals, twice_area = triangles[valid], normals[valid], twice_area[valid]
    rng = np.random.default_rng(0)
    index = rng.choice(len(triangles), count, p=twice_area / twice_area.sum())
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, :1])
    weights = np.concatenate((1 - root, root * (1 - uv[:, 1:]), root * uv[:, 1:]), axis=1)
    xyz = (triangles[index] * weights[..., None]).sum(axis=1)
    normal = normals[index] / twice_area[index, None]
    points = np.concatenate((xyz, normal), axis=1).astype(np.float32)
    extents = np.ptp(triangles.reshape(-1, 3), axis=0).astype(np.float32)
    return points, extents


@dataclass(frozen=True)
class ObjectAsset:
    clip: dict
    parts: list
    key: str
    com: np.ndarray
    inertia: np.ndarray
    points: np.ndarray
    extent: np.ndarray


@lru_cache(maxsize=8)
def _load_asset(signature, cache_dir, profile="default"):
    clip = load_motion(signature[0])
    parts = load_collision_parts(clip, cache_dir, profile=profile)
    com, inertia = mass_properties(clip)
    points, extent = sample_geometry(parts, clip["object_scale"])
    return ObjectAsset(clip, parts, geometry_key(clip, profile), com, inertia, points, extent)


def selected_asset(cfg):
    return _load_asset(_signature(select_motion_files(cfg)[0]), str(Path(cfg.collision_cache_dir).resolve()),
                       getattr(cfg, "collision_profile", "default"))


def make_chair_spec(cfg):
    asset = selected_asset(cfg)
    spec = mujoco.MjSpec()
    # Reference world poses belong to reset, not the shared variant topology.
    body = spec.worldbody.add_body(name="chair")
    body.add_freejoint(name="chair_free")
    body.explicitinertial = True
    mass = cfg.object_physical_cfg.evaluation_mass
    body.mass, body.ipos = mass, asset.com
    inertia = asset.inertia * (mass / 10.0)
    body.fullinertia = inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    add_object_geoms(spec, body, asset.clip, asset.parts)
    for geom in list(spec.geoms):
        if geom.contype != 2:
            continue
        # G1-only copy wins MuJoCo's friction mixing by priority. A separate
        # ground-only copy keeps chair-ground friction fixed and avoids double
        # contacts; robot geoms have contype=0 and ground has contype=1.
        geom.priority = 1
        geom.friction = [cfg.object_physical_cfg.evaluation_robot_friction, .005, .0001]
        body.add_geom(
            name=geom.name + "_ground", type=geom.type, meshname=geom.meshname,
            pos=geom.pos, quat=geom.quat, group=3, contype=0, conaffinity=1,
            priority=0, mass=0, condim=3, friction=[cfg.object_physical_cfg.ground_friction, .005, .0001],
            rgba=[.95, .55, .12, .7],
        )
    return spec
