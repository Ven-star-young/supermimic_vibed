"""One free yoga ball, using the reviewed mesh and existing convex parts."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from motion_data import load_motion
from motion_data.collision import add_object_geoms
from ...assets import DATA_DIR
from ..sitting.assets import ObjectAsset, mass_properties, sample_geometry


def _signature(path):
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=8)
def _load_asset(signatures):
    import trimesh

    clip = load_motion(signatures[0][0])
    metadata = json.loads(Path(signatures[1][0]).read_text())
    mesh_path = Path(signatures[2][0])
    mesh_sha = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    if (not np.array_equal(clip["object_scale"], np.ones(3))
            or str(clip["source_mesh_sha256"].item()) != mesh_sha):
        raise ValueError("YogaBall requires the unchanged source mesh with baked size and runtime scale 1")
    with np.load(mesh_path, allow_pickle=False) as mesh:
        if (not np.array_equal(clip["object_vertices"], mesh["vertices"])
                or not np.array_equal(clip["object_faces"], mesh["faces"])):
            raise ValueError("Motion object mesh differs from its reviewed source asset")
    parts = []
    for path, _, _ in signatures[4:]:
        mesh = trimesh.load_mesh(path, process=False)
        parts.append((np.asarray(mesh.vertices), np.asarray(mesh.faces)))
    com, inertia = mass_properties(clip)
    points, extent = sample_geometry(parts, clip["object_scale"])
    return ObjectAsset(clip, parts, metadata["object_asset_id"] + ":" + mesh_sha,
                       com, inertia, points, extent)


def selected_asset(cfg):
    path = Path(cfg.motion_dir).resolve()
    if not path.is_file() or path.suffix != ".npz":
        raise ValueError("YogaBall motion_dir must name one standard motion NPZ")
    with np.load(path, allow_pickle=False) as clip:
        episode = (DATA_DIR.parent / str(clip["source_episode"].item())).resolve()
    metadata_path = episode / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["object_label"] != "yoga ball" or metadata["episode_id"] != episode.name:
        raise ValueError("YogaBall requires a reviewed yoga ball episode")
    asset_dir = episode.parent.parent / "assets" / "objects" / metadata["object_asset_id"]
    mesh_path = asset_dir / "mesh.npz"
    urdf_path = asset_dir / "collision" / "genesis" / "object_contact_compound.urdf"
    collisions = ET.parse(urdf_path).findall("./link/collision")
    if not collisions:
        raise ValueError("Reviewed yoga ball collision parts are missing")
    parts = []
    for collision in collisions:
        origin, mesh = collision.find("origin"), collision.find("geometry/mesh")
        if (mesh is None or not np.array_equal(np.fromstring(mesh.get("scale", "1 1 1"), sep=" "), np.ones(3))
                or (origin is not None and any(
                    not np.array_equal(np.fromstring(origin.get(key, "0 0 0"), sep=" "), np.zeros(3))
                    for key in ("xyz", "rpy")))):
            raise ValueError("Reviewed collision parts must retain their baked object-local coordinates")
        parts.append((urdf_path.parent / mesh.attrib["filename"]).resolve())
    return _load_asset(tuple(_signature(p) for p in [path, metadata_path, mesh_path, urdf_path, *parts]))


def make_ball_spec(cfg):
    asset = selected_asset(cfg)
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=cfg.object_entity_name)
    body.add_freejoint(name=cfg.object_entity_name + "_free")
    body.explicitinertial = True
    mass = cfg.object_physical_cfg.evaluation_mass
    body.mass, body.ipos = mass, asset.com
    inertia = asset.inertia * (mass / 10.0)
    body.fullinertia = inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]
    add_object_geoms(spec, body, asset.clip, asset.parts)
    for geom in list(spec.geoms):
        if geom.contype != 2:
            continue
        # Separate robot and floor collision roles, as in the Sitting asset.
        geom.priority = 1
        geom.friction = [cfg.object_physical_cfg.evaluation_robot_friction, .005, .0001]
        body.add_geom(
            name=geom.name + "_ground", type=geom.type, meshname=geom.meshname,
            pos=geom.pos, quat=geom.quat, group=3, contype=0, conaffinity=1,
            priority=0, mass=0, condim=3, friction=[cfg.object_physical_cfg.ground_friction, .005, .0001],
            rgba=[.95, .55, .12, .7],
        )
    return spec
