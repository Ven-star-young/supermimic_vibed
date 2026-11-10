"""G1 geometric tactile sensing on original object surfaces with mjlab/Warp.

The frozen WBT point/channel order is independent of MuJoCo's body order.
Distances are in metres. The configurable 0-255 response is synthetic and
uncalibrated; neither distances nor counts are solver forces.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np
import torch
import trimesh
import warp as wp
from mjlab.sensor import Sensor, SensorCfg

from .assets.g1.tactile import load_tactile_layout


@wp.struct
class _PointData:
    positions: wp.array2d(dtype=wp.vec3)
    normals: wp.array2d(dtype=wp.vec3)
    distance: wp.array2d(dtype=float)
    signed_distance: wp.array2d(dtype=float)
    signed_valid: wp.array2d(dtype=bool)
    inside: wp.array2d(dtype=bool)
    near: wp.array2d(dtype=bool)
    contact: wp.array2d(dtype=bool)
    valid: wp.array2d(dtype=bool)
    nearest: wp.array2d(dtype=wp.vec3)
    normal: wp.array2d(dtype=wp.vec3)
    face: wp.array2d(dtype=int)


@wp.func
def _positive_sign(value: float):
    return wp.where(value < 0.0, -1.0, 1.0)


@wp.func
def _primitive_surface(kind: int, point: wp.vec3, size: wp.vec3):
    """Exact closest surface point and outward normal for MuJoCo primitives."""
    closest = wp.vec3(0.0)
    normal = wp.vec3(0.0, 0.0, 1.0)
    inside = False
    if kind == 0:  # plane: unsigned, as for an open triangle surface
        closest = wp.vec3(point[0], point[1], 0.0)
    elif kind == 6:  # box: size contains half extents
        closest = wp.vec3(wp.clamp(point[0], -size[0], size[0]),
                          wp.clamp(point[1], -size[1], size[1]),
                          wp.clamp(point[2], -size[2], size[2]))
        delta = point - closest
        if wp.length_sq(delta) > 0.0:
            normal = wp.normalize(delta)
        else:
            margin = size - wp.vec3(wp.abs(point[0]), wp.abs(point[1]), wp.abs(point[2]))
            axis = int(0)
            if margin[1] < margin[axis]:
                axis = 1
            if margin[2] < margin[axis]:
                axis = 2
            normal = wp.vec3(0.0)
            normal[axis] = _positive_sign(point[axis])
            closest[axis] = normal[axis] * size[axis]
            inside = margin[axis] > 0.0
    elif kind == 2 or kind == 3:  # sphere / capsule (local Z axis)
        center = wp.vec3(0.0)
        if kind == 3:
            center[2] = wp.clamp(point[2], -size[1], size[1])
        delta = point - center
        length = wp.length(delta)
        normal = wp.vec3(1.0, 0.0, 0.0)
        if length > 0.0:
            normal = delta / length
        closest = center + size[0] * normal
        inside = length < size[0]
    elif kind == 5:  # cylinder (radius, half height)
        radial = wp.sqrt(point[0] * point[0] + point[1] * point[1])
        radial_normal = wp.vec3(1.0, 0.0, 0.0)
        if radial > 0.0:
            radial_normal = wp.vec3(point[0] / radial, point[1] / radial, 0.0)
        closest = wp.min(radial, size[0]) * radial_normal
        closest[2] = wp.clamp(point[2], -size[1], size[1])
        delta = point - closest
        if radial > size[0] or wp.abs(point[2]) > size[1]:
            normal = wp.normalize(delta)
        elif size[0] - radial < size[1] - wp.abs(point[2]):
            closest = size[0] * radial_normal
            closest[2] = point[2]
            normal = radial_normal
            inside = radial < size[0]
        else:
            normal = wp.vec3(0.0, 0.0, _positive_sign(point[2]))
            closest[2] = normal[2] * size[1]
            inside = wp.abs(point[2]) < size[1]
    return closest, normal, inside


@wp.kernel
def _query_surface(
    body_ids: wp.array(dtype=int),
    local_points: wp.array(dtype=wp.vec3),
    local_normals: wp.array(dtype=wp.vec3),
    body_pos: wp.array2d(dtype=wp.vec3),
    body_mat: wp.array2d(dtype=wp.mat33),
    geom_pos: wp.array2d(dtype=wp.vec3),
    geom_mat: wp.array2d(dtype=wp.mat33),
    geom_dataid: wp.array2d(dtype=int),
    geom_size: wp.array2d(dtype=wp.vec3),
    mesh_ids: wp.array(dtype=wp.uint64),
    mesh_signed: wp.array(dtype=bool),
    mesh_radius: wp.array(dtype=float),
    geom_id: int,
    geom_type: int,
    threshold: float,
    out: _PointData,
):
    world, index = wp.tid()
    body = body_ids[index]
    point_w = body_pos[world, body] + body_mat[world, body] * local_points[index]
    out.positions[world, index] = point_w
    out.normals[world, index] = body_mat[world, body] * local_normals[index]
    rotation = geom_mat[world, geom_id]
    origin = geom_pos[world, geom_id]
    point = wp.transpose(rotation) * (point_w - origin)
    closest = point
    normal = wp.vec3(0.0)
    face = int(-1)
    valid = False
    signed_valid = False
    inside = False
    finite = wp.isfinite(point[0]) and wp.isfinite(point[1]) and wp.isfinite(point[2])
    if finite and geom_type == 7:  # mesh: per-world ids include disabled variant slots (-1)
        mesh_index = geom_dataid[world % geom_dataid.shape[0], geom_id]
        if mesh_index >= 0:
            mesh = mesh_ids[mesh_index]
            # A conservative radius enclosing every triangle; no distance clipping.
            radius = wp.length(point) + mesh_radius[mesh_index] + 1.0
            signed_valid = mesh_signed[mesh_index]
            if signed_valid:
                query = wp.mesh_query_point_sign_parity(mesh, point, radius, 3)
            else:
                query = wp.mesh_query_point_no_sign(mesh, point, radius)
            if query.result:
                valid = True
                face = query.face
                closest = wp.mesh_eval_position(mesh, face, query.u, query.v)
                v0 = wp.mesh_get_point(mesh, wp.mesh_get_index(mesh, face * 3))
                v1 = wp.mesh_get_point(mesh, wp.mesh_get_index(mesh, face * 3 + 1))
                v2 = wp.mesh_get_point(mesh, wp.mesh_get_index(mesh, face * 3 + 2))
                normal = wp.normalize(wp.cross(v1 - v0, v2 - v0))
                # Match the original near-zero surface convention.
                inside = signed_valid and query.sign < 0.0 and wp.length(point - closest) > 1.0e-8
    elif finite:
        closest, normal, inside = _primitive_surface(
            geom_type, point, geom_size[world % geom_size.shape[0], geom_id])
        valid = True
        signed_valid = geom_type != 0
        inside = inside and wp.length(point - closest) > 1.0e-8
    gap = wp.length(point - closest)
    near = valid and gap <= threshold
    out.distance[world, index] = wp.where(valid, gap, wp.inf)
    signed_gap = wp.where(inside, -gap, gap)
    out.signed_distance[world, index] = wp.where(valid and signed_valid, signed_gap, wp.nan)
    out.signed_valid[world, index] = valid and signed_valid
    out.inside[world, index] = inside
    out.near[world, index] = near
    out.contact[world, index] = inside or near
    out.valid[world, index] = valid
    out.nearest[world, index] = wp.where(valid, origin + rotation * closest, wp.vec3(wp.nan))
    out.normal[world, index] = rotation * normal
    out.face[world, index] = face


@dataclass
class G1TactileData:
    point_positions_w: torch.Tensor
    point_normals_w: torch.Tensor
    surface_distance_m: torch.Tensor
    signed_distance_m: torch.Tensor
    signed_distance_valid: torch.Tensor
    inside_object: torch.Tensor
    near_surface: torch.Tensor
    geometric_contact: torch.Tensor
    valid: torch.Tensor
    nearest_point_w: torch.Tensor
    nearest_normal_w: torch.Tensor
    nearest_face: torch.Tensor
    channel_surface_distance_m: torch.Tensor
    channel_signed_distance_m: torch.Tensor
    channel_signed_distance_valid: torch.Tensor
    channel_geometric_contact: torch.Tensor
    channel_contact_point_count: torch.Tensor
    channel_valid: torch.Tensor
    compression_depth_m: torch.Tensor
    point_counts: torch.Tensor
    channel_counts: torch.Tensor


@dataclass
class G1TactileSensorCfg(SensorCfg):
    target_entity: str = "chair"
    robot_entity: str = "robot"
    layout: str = "original_outer"
    contact_distance_m: float = 0.004
    geom_name: str | None = None
    counts_full_scale_depth_m: float = 0.010
    counts_exponent: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.contact_distance_m) or self.contact_distance_m < 0:
            raise ValueError("contact_distance_m must be finite and non-negative")
        if self.robot_entity == self.target_entity:
            raise ValueError("Tactile queries must target an object, not the robot itself")
        self.validate_counts_response()

    def validate_counts_response(self):
        limits = np.finfo(np.float32)
        for name in ("counts_full_scale_depth_m", "counts_exponent"):
            value = getattr(self, name)
            if not math.isfinite(value) or not limits.tiny <= value <= limits.max:
                raise ValueError(f"{name} must be positive and representable as a normal float32 value")

    def build(self):
        self.__post_init__()  # Also validate configs edited after construction.
        return G1TactileSensor(self)


class G1TactileSensor(Sensor[G1TactileData]):
    """Lazy, batched surface queries using the native Sensor update/reset cache."""

    def __init__(self, cfg: G1TactileSensorCfg):
        super().__init__()
        self.cfg = cfg
        self.layout = load_tactile_layout(cfg.layout)

    def edit_spec(self, scene_spec, entities):
        del scene_spec
        for name in (self.cfg.robot_entity, self.cfg.target_entity):
            if name not in entities:
                raise ValueError(f"Tactile entity {name!r} is missing from the scene")

    def initialize(self, mj_model, model, data, device):
        self._model, self._data, self._device = model.struct, data.struct, device
        prefix = self.cfg.target_entity + "/"
        if self.cfg.geom_name is not None:
            name = prefix + self.cfg.geom_name
            candidates = [i for i in range(mj_model.ngeom) if mj_model.geom(i).name == name]
        else:
            geoms = [i for i in range(mj_model.ngeom) if mj_model.geom(i).name.startswith(prefix)]
            visual = [i for i in geoms if mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
                      and not (mj_model.geom_contype[i] or mj_model.geom_conaffinity[i])]
            candidates = visual or [i for i in geoms if mj_model.geom_contype[i] or mj_model.geom_conaffinity[i]]
        if len(candidates) != 1:
            names = [mj_model.geom(i).name for i in candidates]
            raise ValueError(f"Tactile needs one object surface; select geom_name explicitly: {names}")
        self._geom_id = candidates[0]
        self._geom_type = int(mj_model.geom_type[self._geom_id])
        if self._geom_type not in (0, 2, 3, 5, 6, 7):
            raise ValueError("Tactile supports original meshes, planes, spheres, capsules, cylinders and boxes")
        body_ids = []
        for link in self.layout.link_names:
            name = f"{self.cfg.robot_entity}/{link}"
            index = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if index < 0:
                raise ValueError(f"Missing tactile body {name}; retain the source link frames")
            body_ids.append(index)
        self._body_ids = wp.array(body_ids, dtype=int, device=device)
        self._local_points = wp.array(self.layout.positions, dtype=wp.vec3, device=device)
        self._local_normals = wp.array(self.layout.normals, dtype=wp.vec3, device=device)
        mesh_ids = np.zeros(mj_model.nmesh, dtype=np.uint64)
        signed = np.zeros(mj_model.nmesh, dtype=bool)
        radii = np.zeros(mj_model.nmesh, dtype=np.float32)
        self._meshes = []
        if self._geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            selected = np.unique(self._model.geom_dataid.numpy()[:, self._geom_id])
            for mesh_index in selected[selected >= 0]:
                start = mj_model.mesh_vertadr[mesh_index]
                vertices = mj_model.mesh_vert[start:start + mj_model.mesh_vertnum[mesh_index]]
                start = mj_model.mesh_faceadr[mesh_index]
                faces = mj_model.mesh_face[start:start + mj_model.mesh_facenum[mesh_index]]
                surface = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                if not len(faces) or np.any(surface.area_faces <= 1e-16):
                    raise ValueError("Tactile object surface has empty or degenerate triangles")
                triangles = vertices[faces]
                e0, e1, e2 = (triangles[:, 1] - triangles[:, 0],
                              triangles[:, 2] - triangles[:, 0], triangles[:, 2] - triangles[:, 1])
                aspect = np.linalg.norm(np.cross(e0, e1), axis=1) / (
                    (e0 * e0).sum(1) + (e1 * e1).sum(1) + (e2 * e2).sum(1))
                # Warp skips these nonzero-area slivers; never silently omit source faces.
                if np.any(aspect < 1e-6):
                    raise ValueError("Tactile mesh contains thin triangles that Warp would skip; "
                                     "refusing an incomplete surface-distance query")
                signed[mesh_index] = surface.is_watertight and surface.is_winding_consistent
                mesh = wp.Mesh(points=wp.array(vertices, dtype=wp.vec3, device=device),
                               indices=wp.array(faces.reshape(-1), dtype=int, device=device))
                self._meshes.append(mesh)
                mesh_ids[mesh_index] = mesh.id
                radii[mesh_index] = np.linalg.norm(vertices, axis=1).max()
        self._mesh_ids = wp.array(mesh_ids, dtype=wp.uint64, device=device)
        self._mesh_signed = wp.array(signed, dtype=bool, device=device)
        self._mesh_radius = wp.array(radii, dtype=float, device=device)
        self._shape = (self._data.nworld, len(body_ids))
        self._point_data = _PointData()
        self._views = {}
        fields = {
            "positions": ("point_positions_w", wp.vec3), "normals": ("point_normals_w", wp.vec3),
            "distance": ("surface_distance_m", float), "signed_distance": ("signed_distance_m", float),
            "signed_valid": ("signed_distance_valid", bool), "inside": ("inside_object", bool),
            "near": ("near_surface", bool), "contact": ("geometric_contact", bool), "valid": ("valid", bool),
            "nearest": ("nearest_point_w", wp.vec3), "normal": ("nearest_normal_w", wp.vec3),
            "face": ("nearest_face", int),
        }
        for field, (name, dtype) in fields.items():
            array = wp.zeros(self._shape, dtype=dtype, device=device)
            setattr(self._point_data, field, array)
            self._views[name] = wp.to_torch(array)
        width = max(map(len, self.layout.channel_members))
        members = np.zeros((len(self.layout.channel_members), width), dtype=np.int64)
        mask = np.zeros_like(members, dtype=bool)
        for i, indices in enumerate(self.layout.channel_members):
            members[i, :len(indices)], mask[i, :len(indices)] = indices, True
        self._members = torch.as_tensor(members, device=device)
        self._member_mask = torch.as_tensor(mask, device=device)
        self.metadata = {
            "backend": "warp_original_surface_bvh_and_ray_parity",
            "layout": self.cfg.layout,
            "geometry_sha256": self.layout.geometry_sha256,
            "channel_abi_sha256": self.layout.channel_abi_sha256,
            "target_geom": mj_model.geom(self._geom_id).name,
            "query_scope": "selected object only; excludes floor and robot self-contact",
            "contact_distance_m": self.cfg.contact_distance_m,
            "signed_distance_convention": "negative inside; NaN for open or inconsistently wound meshes",
            "force_N": None, "hardware_counts": None,
            "counts_response": self._counts_metadata(),
        }

    def _counts_metadata(self):
        return {
            "units": "synthetic_counts", "calibration_status": "uncalibrated",
            "contact_distance_m": self.cfg.contact_distance_m,
            "full_scale_depth_m": self.cfg.counts_full_scale_depth_m,
            "exponent": self.cfg.counts_exponent,
            "distance": "signed when valid; unsigned surface proximity otherwise",
            "formula": "round_even(255 * clip(max(0, contact_distance - distance) / full_scale_depth, 0, 1) ** exponent)",
            "channel_pooling": "max before rounding",
            "invalid_counts": "zero placeholder; consult valid/channel_valid",
        }

    def _compute_data(self):
        self.cfg.validate_counts_response()
        m, d = self._model, self._data
        if wp.get_device(self._device).is_cuda:
            # Simulation uses Warp's stream; observations may use another Torch stream.
            wp.get_stream(self._device).wait_stream(wp.stream_from_torch(torch.cuda.current_stream(self._device)))
        wp.launch(_query_surface, dim=self._shape, inputs=[
            self._body_ids, self._local_points, self._local_normals, d.xpos, d.xmat,
            d.geom_xpos, d.geom_xmat, m.geom_dataid, m.geom_size,
            self._mesh_ids, self._mesh_signed, self._mesh_radius,
            self._geom_id, self._geom_type, self.cfg.contact_distance_m, self._point_data,
        ], device=self._device)
        if wp.get_device(self._device).is_cuda:
            torch.cuda.current_stream(self._device).wait_stream(wp.stream_to_torch(wp.get_stream(self._device)))
        values = self._views
        ids, mask = self._members, self._member_mask
        valid = (values["valid"][:, ids] | ~mask).all(-1)
        signed_valid = (values["signed_distance_valid"][:, ids] | ~mask).all(-1)
        distance = values["surface_distance_m"][:, ids].masked_fill(~mask, torch.inf).amin(-1)
        signed_distance = values["signed_distance_m"][:, ids].masked_fill(~mask, torch.inf).amin(-1)
        contacts = values["geometric_contact"][:, ids] & mask
        # Use the sign only when topology supports it: deep interior points must
        # not lose signal as their unsigned distance from the surface increases.
        effective_distance = torch.where(values["signed_distance_valid"],
                                         values["signed_distance_m"], values["surface_distance_m"])
        depth = (self.cfg.contact_distance_m - effective_distance).clamp_min(0)
        depth = depth.masked_fill(~values["valid"], 0)
        continuous = 255. * (depth / self.cfg.counts_full_scale_depth_m).clamp(0, 1).pow(self.cfg.counts_exponent)
        point_counts = continuous.round().to(torch.uint8)
        channel_counts = continuous[:, ids].masked_fill(~mask, 0).amax(-1).round().to(torch.uint8)
        channel_counts = channel_counts.masked_fill(~valid, 0)
        self.metadata["counts_response"] = self._counts_metadata()
        return G1TactileData(**values, channel_surface_distance_m=distance,
            channel_signed_distance_m=signed_distance.masked_fill(~signed_valid, torch.nan),
            channel_signed_distance_valid=signed_valid, channel_geometric_contact=contacts.any(-1),
            channel_contact_point_count=contacts.sum(-1), channel_valid=valid,
            compression_depth_m=depth, point_counts=point_counts, channel_counts=channel_counts)
