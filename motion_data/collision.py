"""Shared, offline convex object geometry for standard motion clips.

Collision assets retain the unscaled object frame. Runtime loading needs only
NumPy; CoACD is an optional preprocessing dependency, never a runtime fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "data/collisions/chairs"
COACD_VERSION = "1.0.14"
CACHE_VERSION = 1
COACD_PARAMS = dict(
    threshold=0.01, real_metric=True, max_convex_hull=-1,
    preprocess_mode="auto", preprocess_resolution=50, resolution=1000,
    mcts_nodes=10, mcts_iterations=30, mcts_max_depth=3, pca=False,
    merge=True, decimate=True, max_ch_vertex=64, extrude=False,
    extrude_margin=0.01, apx_mode="ch", seed=0,
)
COLLISION_PROFILES = ("default", "compact")


def collision_parameters(profile="default"):
    """Return the pinned recipe for a named collision approximation.

    ``default`` retains the original cache identity. ``compact`` trades fine
    collision detail for at most 32 convex hulls and 32 vertices per hull.
    Bounds are checked on preprocessing output and every subsequent load;
    neither profile drops parts or silently uses another approximation.
    """
    if profile not in COLLISION_PROFILES:
        raise ValueError(f"Unknown collision profile {profile!r}; choose {COLLISION_PROFILES}")
    params = dict(COACD_PARAMS)
    if profile == "compact":
        params.update(threshold=.03, max_convex_hull=32, max_ch_vertex=32)
    return params


def _mesh_arrays(vertices, faces, *, name="object"):
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4
            or vertices.dtype.kind not in "fiu" or not np.isfinite(vertices).all()):
        raise ValueError(f"{name} vertices must be finite (V,3), with V >= 4")
    if (faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4
            or faces.dtype.kind not in "iu" or faces.min() < 0
            or faces.max() >= len(vertices)):
        raise ValueError(f"{name} faces must contain valid (F,3) indices, with F >= 4")
    return (np.ascontiguousarray(vertices, dtype="<f8"),
            np.ascontiguousarray(faces, dtype="<i4"))


def _object_mesh(clip):
    if "object_vertices" not in clip or "object_faces" not in clip:
        raise ValueError("Object collision requires object_vertices and object_faces")
    return _mesh_arrays(clip["object_vertices"], clip["object_faces"])


def _parameters(profile="default"):
    return json.dumps(collision_parameters(profile), sort_keys=True, separators=(",", ":"))


def collision_key(clip, profile="default") -> str:
    """Hash unscaled mesh content and the complete pinned decomposition recipe.

    Trajectory, object pose and scale intentionally do not affect the key.
    """
    vertices, faces = _object_mesh(clip)
    digest = hashlib.sha256()
    digest.update(f"motion-collision:{CACHE_VERSION}:coacd:{COACD_VERSION}:".encode())
    digest.update(_parameters(profile).encode())
    for array in (vertices, faces):
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_parts(parts):
    if not parts:
        raise ValueError("Collision asset must contain at least one convex part")
    checked = []
    for index, (vertices, faces) in enumerate(parts):
        vertices, faces = _mesh_arrays(vertices, faces, name=f"part {index}")
        extent = float(np.ptp(vertices, axis=0).max())
        tolerance = max(1e-9, extent * 1e-6)
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0],
                           triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        if extent <= 0 or np.any(lengths <= max(1e-18, extent ** 2 * 1e-12)):
            raise ValueError(f"Collision part {index} has degenerate triangles")
        normals /= lengths[:, None]
        # Every face must support the whole convex hull, with outward winding.
        distance = normals @ vertices.T - np.einsum("ij,ij->i", normals, triangles[:, 0])[:, None]
        if np.any(distance > tolerance):
            raise ValueError(f"Collision part {index} is not convex or has inconsistent winding")
        edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                       faces[:, [2, 0]])), axis=1)
        if not np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2):
            raise ValueError(f"Collision part {index} is not a closed manifold")
        centered = triangles - vertices.mean(axis=0)
        volume = np.einsum("ij,ij->i", centered[:, 0],
                           np.cross(centered[:, 1], centered[:, 2])).sum() / 6
        if volume <= max(1e-21, extent ** 3 * 1e-12):
            raise ValueError(f"Collision part {index} has zero or negative volume")
        checked.append((vertices, faces))
    return checked


def _convex_parts(parts):
    """Normalize each part's hull topology without moving or merging vertices.

    CoACD decimation can leave slightly nonconvex triangulations. MuJoCo uses
    each part's convex hull anyway, so this preserves its collision geometry
    and leaves already-valid version-1 cache entries equivalent and reusable.
    """
    from scipy.spatial import ConvexHull

    normalized = []
    for index, (vertices, faces) in enumerate(parts):
        vertices, _ = _mesh_arrays(vertices, faces, name=f"part {index}")
        hull = ConvexHull(vertices)
        faces = hull.simplices.copy()
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0],
                           triangles[:, 2] - triangles[:, 0])
        flip = np.einsum("ij,ij->i", normals, hull.equations[:, :3]) < 0
        faces[flip] = faces[flip][:, [0, 2, 1]]
        normalized.append((vertices, faces))
    return normalized


def _validate_profile(parts, profile):
    params = collision_parameters(profile)
    # Version-1 default caches predate bounded-profile checks. Preserve their
    # loader behavior, while making compact limits a strict asset contract.
    if profile != "default":
        if len(parts) > params["max_convex_hull"]:
            raise ValueError(f"Collision profile {profile!r} allows at most "
                             f"{params['max_convex_hull']} parts; received {len(parts)}")
        for index, (vertices, _) in enumerate(parts):
            if len(vertices) > params["max_ch_vertex"]:
                raise ValueError(f"Collision profile {profile!r} allows at most "
                                 f"{params['max_ch_vertex']} vertices per part; "
                                 f"part {index} contains {len(vertices)}")
    return parts


def load_collision_parts(clip, cache_dir=DEFAULT_CACHE_DIR, profile="default"):
    """Load verified convex parts, failing explicitly if preprocessing is absent."""
    key = collision_key(clip, profile)
    path = Path(cache_dir) / f"{key}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing collision asset: {path}. Run uv run --extra motion python "
            "tools/prepare_collisions.py <motion.npz-or-directory> "
            f"--cache-dir {Path(cache_dir)} --profile {profile}"
        )
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = dict(archive)
        expected = {"vertices", "faces", "vertex_offsets", "face_offsets", "key",
                    "format_version", "coacd_version", "parameters"}
        if not expected <= arrays.keys() or any(a.dtype.hasobject for a in arrays.values()):
            raise ValueError("Missing fields or unsupported object arrays")
        for field, value in (("key", key), ("format_version", CACHE_VERSION),
                             ("coacd_version", COACD_VERSION), ("parameters", _parameters(profile))):
            if arrays[field].shape != () or arrays[field].item() != value:
                raise ValueError(f"Mismatched {field}")
        if profile != "default" or "profile" in arrays:
            if ("profile" not in arrays or arrays["profile"].shape != ()
                    or arrays["profile"].item() != profile):
                raise ValueError("Mismatched profile")
        vertices, faces = arrays["vertices"], arrays["faces"]
        vo, fo = arrays["vertex_offsets"], arrays["face_offsets"]
        for offsets, total in ((vo, len(vertices)), (fo, len(faces))):
            if (offsets.ndim != 1 or offsets.dtype.kind not in "iu" or len(offsets) < 2
                    or offsets[0] != 0 or offsets[-1] != total
                    or np.any(offsets[1:] <= offsets[:-1])):
                raise ValueError("Invalid part offsets")
        if len(vo) != len(fo):
            raise ValueError("Vertex and face part counts differ")
        parts = [(vertices[vo[i]:vo[i + 1]], faces[fo[i]:fo[i + 1]])
                 for i in range(len(vo) - 1)]
        return _validate_parts(_validate_profile(parts, profile))
    except (ValueError, KeyError, IndexError, TypeError, OSError) as exc:
        raise ValueError(f"Invalid collision asset {path}: {exc}") from exc


def prepare_collision(clip, cache_dir=DEFAULT_CACHE_DIR, profile="default") -> Path:
    """Create/reuse a validated asset atomically; return its cache path.

    An invalid existing cache is reported instead of silently replaced.
    Set OMP_NUM_THREADS before importing CoACD to control preprocessing threads.
    """
    path = Path(cache_dir) / f"{collision_key(clip, profile)}.npz"
    if path.exists():
        load_collision_parts(clip, cache_dir, profile)
        return path
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("coacd")
    except PackageNotFoundError as exc:
        raise ImportError("Collision preprocessing needs uv run --extra motion") from exc
    if installed != COACD_VERSION:
        raise RuntimeError(f"Collision recipe requires coacd=={COACD_VERSION}; found {installed}")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import coacd

    coacd.set_log_level("warn")
    vertices, faces = _object_mesh(clip)
    params = collision_parameters(profile)
    parts = _validate_parts(_validate_profile(
        _convex_parts(coacd.run_coacd(coacd.Mesh(vertices, faces), **params)), profile))
    arrays = dict(
        vertices=np.concatenate([v for v, _ in parts]),
        faces=np.concatenate([f for _, f in parts]),
        vertex_offsets=np.cumsum([0] + [len(v) for v, _ in parts], dtype=np.int64),
        face_offsets=np.cumsum([0] + [len(f) for _, f in parts], dtype=np.int64),
        key=np.asarray(path.stem), format_version=np.asarray(CACHE_VERSION),
        coacd_version=np.asarray(COACD_VERSION), parameters=np.asarray(_parameters(profile)),
        profile=np.asarray(profile),
        hull_normalization=np.asarray("scipy.spatial.ConvexHull"),
        scipy_version=np.asarray(version("scipy")),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def add_object_geoms(spec, body, clip, parts):
    """Attach visual mesh and convex collision parts to an existing object body.

    Chair bit 2 pairs with robot affinity 2; floor bit 1 remains independent.
    These massless geometry additions do not determine dynamic object inertia.
    """
    import mujoco

    vertices, faces = _object_mesh(clip)
    scale = np.asarray(clip["object_scale"], dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("object_scale must be a positive finite 3-vector")
    parts = _validate_parts(parts)
    prefix = body.name or "object"
    mesh_name = f"{prefix}_mesh"
    spec.add_mesh(name=mesh_name, uservert=vertices.ravel(), userface=faces.ravel(), scale=scale)
    body.add_geom(name=f"{prefix}_visual", type=mujoco.mjtGeom.mjGEOM_MESH,
                  meshname=mesh_name, group=1, contype=0, conaffinity=0, mass=0,
                  rgba=[.56, .36, .18, 1])
    for index, (part_vertices, part_faces) in enumerate(parts):
        mesh_name = f"{prefix}_collision_mesh_{index:03d}"
        spec.add_mesh(name=mesh_name, uservert=part_vertices.ravel(),
                      userface=part_faces.ravel(), scale=scale)
        body.add_geom(name=f"{prefix}_collision_{index:03d}", type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=mesh_name, group=3, contype=2, conaffinity=0,
                      margin=0, mass=0, condim=3, friction=[1., .005, .0001],
                      rgba=[.95, .55, .12, .7])
