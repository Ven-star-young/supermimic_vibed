"""Lazy physical scene assets for the shared multi-entity task family."""

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from motion_data.collision import add_object_geoms
from ..interaction.assets import selected_asset


@dataclass(frozen=True, kw_only=True)
class ObjectCfg:
    """Additional rigid body. ``size`` denotes box half-extents in metres.

    ``asset_file`` optionally selects a standard motion NPZ's collision mesh;
    its trajectory is not a task target. Position and orientation below always
    define the extra body's initial state. Dynamic bodies use the task's training
    mass/friction ranges unless ``randomize_physics`` is disabled; evaluation
    and static bodies use this object's configured mass and friction.
    """

    name: str = "box"
    size: tuple[float, float, float] = (.15, .15, .15)
    pos: tuple[float, float, float] = (1.5, 0., .15)
    quat: tuple[float, float, float, float] = (1., 0., 0., 0.)
    dynamic: bool = True
    mass: float = 10.
    friction: float = 1.
    ground_friction: float = 1.
    randomize_physics: bool = True
    asset_file: str | None = None

    def __post_init__(self):
        if not self.name or self.name in ("robot", "chair", "terrain") or "/" in self.name:
            raise ValueError("Extra object names must be unique entity names other than robot/chair/terrain")
        if (len(self.size) != 3 or not np.isfinite(self.size).all() or min(self.size) <= 0
                or len(self.pos) != 3 or not np.isfinite(self.pos).all()
                or len(self.quat) != 4 or not np.isfinite(self.quat).all()
                or not np.isclose(np.linalg.norm(self.quat), 1., atol=1e-5)):
            raise ValueError("Object size/position/quaternion must be finite, with positive size and unit quaternion")
        if not np.isfinite([self.mass, self.friction, self.ground_friction]).all() or self.mass <= 0:
            raise ValueError("Object physical parameters must be finite and mass must be positive")
        if min(self.friction, self.ground_friction) < 0:
            raise ValueError("Object friction must be nonnegative")


def object_names(cfg):
    names = (("chair",) if cfg.object_enabled else ()) + tuple(obj.name for obj in cfg.extra_objects)
    if len(names) > cfg.max_objects or len(set(names)) != len(names):
        raise ValueError("Scene objects must have unique names and fit max_objects")
    return names


def load_object_configs(path):
    """Read explicit scene configuration, without opening any motion or mesh."""
    path = Path(path)
    values = json.loads(path.read_text())
    if not isinstance(values, list) or any(not isinstance(item, dict) for item in values):
        raise ValueError("extra_objects_file must contain a JSON list of ObjectCfg records")
    result = []
    for record in values:
        record = dict(record)
        for key in ("size", "pos", "quat"):
            if key in record:
                record[key] = tuple(record[key])
        if record.get("asset_file") is not None:
            record["asset_file"] = str((path.parent / record["asset_file"]).resolve())
        result.append(ObjectCfg(**record))
    return tuple(result)


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
class EntityAsset:
    key: str
    points: np.ndarray
    extent: np.ndarray
    com: np.ndarray
    inertia: np.ndarray
    source: object | None = None


@lru_cache(maxsize=32)
def _box_asset(size):
    import trimesh
    mesh = trimesh.creation.box(extents=np.asarray(size) * 2)
    points, extent = sample_geometry([(mesh.vertices, mesh.faces)], np.ones(3))
    # Unit-mass inertia: callers scale this together with mass.
    inertia = np.diag((np.asarray(size)**2)[[1, 0, 0]] + (np.asarray(size)**2)[[2, 2, 1]]) / 3
    key = "box:" + hashlib.sha256(np.asarray(size, dtype="<f8").tobytes()).hexdigest()
    return EntityAsset(key, points, extent, np.zeros(3), inertia)


def entity_asset(cfg, index):
    if cfg.object_enabled and index == 0:
        asset = selected_asset(cfg)
    else:
        obj = cfg.extra_objects[index - int(cfg.object_enabled)]
        if obj.asset_file is None:
            return _box_asset(tuple(obj.size))
        asset = selected_asset(replace(cfg, object_enabled=True, motion_dir=obj.asset_file,
                                       object_motion_file=obj.asset_file))
    points, extent = sample_geometry(asset.parts, asset.clip["object_scale"])
    return EntityAsset(asset.key, points, extent, asset.com, asset.inertia / 10., asset)


def make_object_spec(cfg, index):
    """Distinct object-interaction and ground copies prevent mixed friction."""
    names = object_names(cfg)
    asset = entity_asset(cfg, index)
    target = cfg.object_enabled and index == 0
    obj = None if target else cfg.extra_objects[index - int(cfg.object_enabled)]
    dynamic = target or obj.dynamic
    mass = cfg.object_physical_cfg.evaluation_mass if target else obj.mass
    friction = cfg.object_physical_cfg.evaluation_robot_friction if target else obj.friction
    floor_mu = cfg.object_physical_cfg.ground_friction if target else obj.ground_friction
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=names[index])
    if dynamic:
        body.add_freejoint(name=names[index] + "_free")
    else:
        # mjlab otherwise adds a massless mocap wrapper, whose root-state origin
        # and inertial fields would not describe this object's body.
        body.mocap = True
    body.explicitinertial = True
    body.mass, body.ipos = mass, asset.com
    inertia = asset.inertia * mass
    body.fullinertia = inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    if asset.source is not None:
        add_object_geoms(spec, body, asset.source.clip, asset.source.parts)
    else:
        body.add_geom(name=names[index] + "_collision", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=obj.size, contype=2, conaffinity=2, group=3, mass=0,
                      condim=3, friction=[friction, .005, .0001], rgba=[.3, .5, .8, 1.])
        body.add_geom(name=names[index] + "_visual", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=obj.size, contype=0, conaffinity=0, group=1, mass=0,
                      rgba=[.3, .5, .8, 1.])
    for geom in list(spec.geoms):
        if geom.contype != 2:
            continue
        # The main copy collides with G1 and other main object copies. Object
        # copies have the same priority, hence MuJoCo takes max pair friction.
        geom.conaffinity, geom.priority, geom.friction = 2, 1, [friction, .005, .0001]
        kwargs = {"meshname": geom.meshname} if geom.type == mujoco.mjtGeom.mjGEOM_MESH else {"size": geom.size}
        body.add_geom(name=geom.name + "_ground", type=geom.type, **kwargs,
                      pos=geom.pos, quat=geom.quat, group=3, contype=0, conaffinity=1,
                      priority=0, mass=0, condim=3, friction=[floor_mu, .005, .0001],
                      rgba=[.95, .55, .12, .7])
    return spec
