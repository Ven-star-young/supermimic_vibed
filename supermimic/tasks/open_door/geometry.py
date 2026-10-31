"""Asset-derived samples for forbidden G1/door contact in back-opening tasks.

Only construction uses MuJoCo/NumPy. Runtime users transform these fixed body-
frame points with differentiable FK and measure distance to the finite door.
The bounded sampling recipe is adapted from the sibling open_door project
(collision_surfaces.py). Samples approximate collision surfaces, not forces;
real and imagined rewards use exactly the same geometric approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import itertools

import mujoco
import numpy as np
import torch

from ...assets.g1.g1_constants import make_g1_spec


@dataclass(frozen=True)
class CollisionSurfaceSamples:
    """One padded group per physical geom, expressed in its FK owner's frame.

    ``points`` is [geoms, samples_per_geom, 3], ``valid`` masks padding, and
    ``body_names`` maps each group to a body present in G1Kinematics. Fixed
    logo/pelvis-contour transforms are composed into their owner's coordinates.
    ``source_bodies`` and ``geom_names`` are diagnostic metadata only.
    """

    body_names: tuple[str, ...]
    source_bodies: tuple[str, ...]
    geom_names: tuple[str, ...]
    points: np.ndarray
    valid: np.ndarray


def _rotation(quaternion):
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def _sphere_directions(count=96):
    index = np.arange(count, dtype=np.float64)
    z = 1. - 2. * (index + .5) / count
    angle = index * np.pi * (3. - np.sqrt(5.))
    radius = np.sqrt(1. - z * z)
    fibonacci = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))
    return np.concatenate((np.eye(3), -np.eye(3), fibonacci))


def _surface_candidates(model, geom_id):
    """Return geom-frame candidates on MuJoCo's convex collision boundary."""
    geom_type = model.geom_type[geom_id]
    size = model.geom_size[geom_id]
    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = int(model.geom_dataid[geom_id])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        vertices = model.mesh_vert[start:start + count].astype(np.float64)
        address = int(model.mesh_graphadr[mesh])
        if address >= 0:
            # This is the same convex graph layout used by MuJoCo-Warp GJK:
            # [nvert, nface, vert_edgeadr[nvert], vert_globalid[nvert], edges].
            graph = model.mesh_graph
            hull_count = int(graph[address])
            edge_address = graph[address + 2:address + 2 + hull_count]
            global_ids = graph[address + 2 + hull_count:address + 2 + 2 * hull_count]
            vertices = vertices[global_ids]
            edges_start = address + 2 + 2 * hull_count
            midpoints = []
            for i, offset in enumerate(edge_address):
                cursor = edges_start + int(offset)
                while graph[cursor] >= 0:
                    j = int(graph[cursor])
                    if j > i:
                        midpoints.append((vertices[i] + vertices[j]) / 2)
                    cursor += 1
            if midpoints:
                vertices = np.concatenate((vertices, np.asarray(midpoints)))
        return vertices
    directions = _sphere_directions()
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return directions * size[0]
    if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        points = directions * size[0]
        points[:, 2] += np.where(points[:, 2] >= 0., size[1], -size[1])
        return points
    if geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        angles = np.arange(64) * (2 * np.pi / 64)
        rings = [np.column_stack((size[0] * np.cos(angles), size[0] * np.sin(angles),
                                 np.full(len(angles), height)))
                 for height in (-size[1], 0., size[1])]
        return np.concatenate((*rings, [[0., 0., -size[1]], [0., 0., size[1]]]))
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return np.asarray([xyz for xyz in itertools.product((-1., 0., 1.), repeat=3)
                           if any(abs(x) == 1. for x in xyz)]) * size
    if geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return directions * size
    raise ValueError(f"Unsupported robot collision primitive {mujoco.mjtGeom(geom_type).name}")


def _select_surface_points(points, count, *, preserve_rear_heel=False):
    """Preserve cardinal/diagonal support extrema, then fill with farthest points.

    Axis extrema make the nominal rear-facing clearances exact for each geom;
    diagonal supports and farthest points cover rotated limbs and panel edges.
    Using minima over samples can still miss contacts between samples, so contact
    diagnostics should also report actual MuJoCo collision counts.
    """
    if len(points) <= count:
        return points
    directions = np.asarray([xyz for xyz in itertools.product((-1., 0., 1.), repeat=3)
                             if any(x != 0. for x in xyz)])
    # First retain six cardinal extrema even with a small requested sample count.
    cardinal = np.concatenate((np.eye(3), -np.eye(3)))
    candidates = (points @ cardinal.T).argmax(axis=0).tolist()
    if preserve_rear_heel:
        # The heel's furthest rear point is below the panel's bottom edge in a
        # normal stand. Axis support alone misses heel/panel corner contact.
        # Retain the actual rear silhouette in twelve height bands as well.
        boundaries = np.linspace(points[:, 2].min(), points[:, 2].max(), 13)
        for low, high in zip(boundaries[:-1], boundaries[1:]):
            ids = np.flatnonzero((points[:, 2] >= low) & (points[:, 2] <= high))
            if len(ids):
                candidates.append(int(ids[points[ids, 0].argmin()]))
    candidates.extend((points @ directions.T).argmax(axis=0).tolist())
    chosen = list(dict.fromkeys(candidates))[:count]
    distance = np.full(len(points), np.inf)
    for i in chosen:
        distance = np.minimum(distance, ((points - points[i]) ** 2).sum(-1))
    while len(chosen) < count:
        chosen.append(int(distance.argmax()))
        distance = np.minimum(distance, ((points - points[chosen[-1]]) ** 2).sum(-1))
    return points[chosen]


@lru_cache(maxsize=4)
def _surface_samples(body_names, samples_per_geom=32):
    """Build <=960 samples from all 30 door-collidable geoms in this G1 asset.

    Physical ground-only foot spheres are excluded, while the full foot meshes
    are included. On torso/logo geoms only the same rear patch accepted by
    ``contacts.classify_contact_points`` is excluded. All other door-collidable
    geometry (including shoulders, head, pelvis, hands and feet) is forbidden.
    No hand-tuned radius or safety inflation is applied to the physical asset.
    """
    if not isinstance(samples_per_geom, int) or not 6 <= samples_per_geom <= 32:
        raise ValueError('samples_per_geom must be an integer between 6 and 32')
    model = make_g1_spec(interaction_collisions=True).compile()
    torso = model.body('torso_link').id
    owners, sources, names, groups = [], [], [], []
    for geom_id in range(model.ngeom):
        if not model.geom_conaffinity[geom_id] & 2:
            continue
        body = int(model.geom_bodyid[geom_id])
        owner = body
        points = (_surface_candidates(model, geom_id) @ _rotation(model.geom_quat[geom_id]).T
                  + model.geom_pos[geom_id])
        # The FK tracks articulated bodies. Compose remaining
        # fixed children into the nearest such ancestor, including mesh offsets.
        while model.body(owner).name not in body_names:
            points = points @ _rotation(model.body_quat[owner]).T + model.body_pos[owner]
            owner = int(model.body_parentid[owner])
            if owner == 0:
                raise ValueError('Robot interaction geom is not attached to an FK body')
        if body in (torso, model.body('logo_link').id):
            if owner != torso:
                raise ValueError('Torso contact patch must be expressed in the torso frame')
            x, y, z = points.T
            allowed = (x < -.025) & (np.abs(y) < .13) & (z > .05) & (z < .38)
            points = points[~allowed]
        if not len(points):
            continue
        groups.append(_select_surface_points(
            points, samples_per_geom,
            preserve_rear_heel=model.body(body).name.endswith('ankle_roll_link')))
        owners.append(model.body(owner).name)
        sources.append(model.body(body).name)
        names.append(model.geom(geom_id).name or f'{model.body(body).name}_collision_{geom_id}')
    points = np.zeros((len(groups), samples_per_geom, 3), dtype=np.float32)
    valid = np.zeros((len(groups), samples_per_geom), dtype=bool)
    for i, group in enumerate(groups):
        points[i, :len(group)] = group
        valid[i, :len(group)] = True
    points.setflags(write=False)
    valid.setflags(write=False)
    return CollisionSurfaceSamples(tuple(owners), tuple(sources), tuple(names), points, valid)


class DoorGeometry:
    """Differentiable finite-panel distances from the same MJCF used by FK.

    The legal target is the panel's +X face. Torso +X must align with its
    normal: the outward back normal is torso -X, opposite the panel normal.
    Illegal clearance is the minimum sampled signed distance per physical
    geom, positive outside and negative inside the door box. This is a bounded
    surface approximation; a gap between samples can miss a small collision.
    """

    def __init__(self, kin, metadata, device):
        self.kin = kin
        self.samples = _surface_samples(tuple(kin.body_names))
        self.body_ids = torch.tensor([kin.body_names.index(name) for name in self.samples.body_names],
                                     device=device)
        self.points = torch.tensor(self.samples.points, device=device)
        self.valid = torch.tensor(self.samples.valid, device=device)
        self.torso = kin.body_names.index("torso_link")
        self.back_offset = torch.tensor([-.06861, .00763, .213], device=device)
        self.hinge = torch.as_tensor(metadata["hinge_pos"], dtype=torch.float32, device=device)
        self.half = torch.as_tensor(metadata["panel_half_size"], dtype=torch.float32, device=device)
        self.center = torch.as_tensor(metadata["hinge_to_panel"], dtype=torch.float32, device=device)
        axis = np.asarray(metadata["joint_axis"])
        if axis.shape != (3,) or not (np.allclose(axis, [0, 0, 1]) or np.allclose(axis, [0, 0, -1])):
            raise ValueError("DoorGeometry requires a vertical +Z or -Z hinge")
        self.hinge_sign, self.base_yaw = float(axis[2]), float(metadata["base_yaw"])

    def _local(self, points, yaw):
        # Broadcast over any number of point axes after the state's batch axes.
        extra = points.ndim - yaw.ndim - 1
        yaw = yaw.reshape((*yaw.shape, *([1] * extra)))
        delta = points - self.hinge
        c, s = yaw.cos(), yaw.sin()
        return torch.stack((c * delta[..., 0] + s * delta[..., 1],
                            -s * delta[..., 0] + c * delta[..., 1], delta[..., 2]), -1) - self.center

    def __call__(self, S):
        positions, rotations, _, _ = self.kin.forward(*S[:6], all_bodies=True)
        yaw = self.base_yaw + self.hinge_sign * S[6].squeeze(-1)
        torso_R = rotations[..., self.torso, :, :]
        back = positions[..., self.torso, :] + (torso_R @ self.back_offset)
        local_back = self._local(back, yaw)
        # Distance to a finite face, not an infinite plane or the nearer side
        # of the box; the opposite face does not grant legal-back shaping.
        distance_sq = ((local_back[..., 0] - self.half[0]).square()
                       + (local_back[..., 1:].abs() - self.half[1:]).clamp_min(0).square().sum(-1))
        normal = torch.stack((yaw.cos(), yaw.sin(), torch.zeros_like(yaw)), -1)
        alignment = (torso_R[..., :, 0] * normal).sum(-1)
        world = (torch.einsum("...gij,gsj->...gsi", rotations[..., self.body_ids, :, :], self.points)
                 + positions[..., self.body_ids, :].unsqueeze(-2))
        excess = self._local(world, yaw).abs() - self.half
        clearance = (torch.linalg.vector_norm(excess.clamp_min(0), dim=-1)
                     + excess.amax(-1).clamp_max(0))
        clearance = clearance.masked_fill(~self.valid, torch.inf).amin(-1)
        return {"back_distance_sq": distance_sq, "alignment": alignment,
                "illegal_clearance": clearance}
