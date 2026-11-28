"""Asset-derived sole support and finite-scene distances for the static ledge.

MuJoCo and NumPy are used only during construction. Runtime features and
clearances use differentiable tensor FK, including imagined world-model states.
Body clearances use the existing bounded collision-surface approximation.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from ...assets.g1.g1_constants import make_g1_spec
from ..open_door.geometry import _forbidden_back_points, _surface_samples


def _legal_back_points(source, owner, points):
    return ~_forbidden_back_points(source, owner, points)


def _rectangle_union(center, half, tolerance=1e-5):
    """Return snapped rectangles and only their exposed boundary segments.

    XML round trips leave micron-sized gaps between otherwise adjacent boxes.
    Merge only coordinates within ``tolerance`` before constructing the union;
    an internal box edge must never reduce the available sole support margin.
    """
    low, high = center[:, :2] - half[:, :2], center[:, :2] + half[:, :2]
    coordinates = []
    for axis in range(2):
        groups = []
        for value in np.unique(np.r_[low[:, axis], high[:, axis]]):
            if groups and value - groups[-1][0] <= tolerance:
                groups[-1].append(value)
            else:
                groups.append([value])
        snapped = {value: float(np.mean(group)) for group in groups for value in group}
        low[:, axis] = [snapped[value] for value in low[:, axis]]
        high[:, axis] = [snapped[value] for value in high[:, axis]]
        coordinates.append(np.asarray([np.mean(group) for group in groups]))
    if np.any(high <= low):
        raise ValueError("Ledge boxes must be wider than the XML rounding tolerance")
    x, y = coordinates
    mid_x, mid_y = np.meshgrid((x[:-1] + x[1:]) / 2, (y[:-1] + y[1:]) / 2,
                             indexing="ij")
    middle = np.stack((mid_x, mid_y), -1)
    occupied = ((middle[..., None, :] > low) & (middle[..., None, :] < high)).all(-1).any(-1)
    edges = []
    for i, j in np.argwhere(occupied):
        if i == 0 or not occupied[i - 1, j]:
            edges.append(((x[i], y[j]), (x[i], y[j + 1])))
        if i == len(x) - 2 or not occupied[i + 1, j]:
            edges.append(((x[i + 1], y[j]), (x[i + 1], y[j + 1])))
        if j == 0 or not occupied[i, j - 1]:
            edges.append(((x[i], y[j]), (x[i + 1], y[j])))
        if j == len(y) - 2 or not occupied[i, j + 1]:
            edges.append(((x[i], y[j + 1]), (x[i + 1], y[j + 1])))
    edges = np.asarray(edges)
    return low, high, edges[:, 0], edges[:, 1] - edges[:, 0]


def _union_signed_distance(points, outline):
    """Exact planar union distance: positive inside, negative outside."""
    low, high, start, delta = outline
    relative = points.unsqueeze(-2) - start
    fraction = ((relative * delta).sum(-1) / delta.square().sum(-1)).clamp(0., 1.)
    distance = torch.linalg.vector_norm(relative - fraction.unsqueeze(-1) * delta, dim=-1).amin(-1)
    inside = ((points.unsqueeze(-2) >= low) & (points.unsqueeze(-2) <= high)).all(-1).any(-1)
    return torch.where(inside, distance, -distance)


def _solid_clearance(points, outline, z_bounds):
    """Signed distance to an extrusion of the planar union, negative inside."""
    planar = -_union_signed_distance(points[..., :2], outline)
    vertical = torch.maximum(z_bounds[0] - points[..., 2], points[..., 2] - z_bounds[1])
    excess = torch.stack((planar, vertical), -1)
    return (torch.linalg.vector_norm(excess.clamp_min(0), dim=-1)
            + excess.amax(-1).clamp_max(0))


class LedgeGeometry:
    """Four-sphere sole support and legal rear-patch contact with finite walls.

    The scene frame has +X along the ledge, +Y away from the wall, +Z up.
    Optional ``body_state`` is the full ``kin.forward(..., all_bodies=True)``
    result and permits the MDP to share its existing FK computation.
    """

    def __init__(self, kin, metadata, device):
        self.kin = kin
        self.platform_height = float(metadata["platform_height"])
        self.floor_height = float(metadata["floor_height"])
        for name in ("platform", "wall"):
            center = np.asarray(metadata[f"{name}_center"], dtype=np.float64)
            half = np.asarray(metadata[f"{name}_half_size"], dtype=np.float64)
            if (center.ndim != 2 or center.shape[1:] != (3,) or len(center) == 0
                    or half.shape != center.shape or not np.isfinite(center).all()
                    or not np.isfinite(half).all() or np.any(half <= 0)):
                raise ValueError(f"{name} geometry requires finite axis-aligned boxes with positive sizes")
            bottom, top = center[:, 2] - half[:, 2], center[:, 2] + half[:, 2]
            if np.ptp(bottom) > 1e-5 or np.ptp(top) > 1e-5:
                raise ValueError(f"{name} boxes must share their bottom and top heights")
            outline = tuple(torch.tensor(value, dtype=torch.float32, device=device)
                            for value in _rectangle_union(center, half))
            setattr(self, f"{name}_outline", outline)
            setattr(self, f"{name}_z_bounds", (float(bottom.mean()), float(top.mean())))
        if (not np.isfinite([self.platform_height, self.floor_height]).all()
                or abs(self.platform_z_bounds[1] - self.platform_height) > 1e-5):
            raise ValueError("platform_height must match the top of the platform boxes")

        model = make_g1_spec(interaction_collisions=True).compile()
        feet = tuple(f"{side}_ankle_roll_link" for side in ("left", "right"))
        self.foot_ids = torch.tensor([kin.body_names.index(name) for name in feet], device=device)
        sole_groups = [[geom for geom in range(model.ngeom)
                        if model.body(int(model.geom_bodyid[geom])).name == foot
                        and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE
                        and model.geom_conaffinity[geom] & 1] for foot in feet]
        if any(len(group) != 4 for group in sole_groups):
            raise ValueError("Ledge support requires the four production sole spheres per G1 foot")
        self.sole_points = torch.tensor(model.geom_pos[sole_groups], dtype=torch.float32, device=device)
        self.sole_radius = torch.tensor(model.geom_size[sole_groups, 0], dtype=torch.float32, device=device)

        names = tuple(kin.body_names)
        legal = _surface_samples(names, point_filter=_legal_back_points)
        illegal = _surface_samples(names)
        self.legal_count = len(legal.body_names)
        self.body_ids = torch.tensor([kin.body_names.index(name)
                                     for name in legal.body_names + illegal.body_names], device=device)
        self.points = torch.tensor(np.concatenate((legal.points, illegal.points)), device=device)
        self.valid = torch.tensor(np.concatenate((legal.valid, illegal.valid)), device=device)
        ground_names = set()
        for geom in range(model.ngeom):
            body = model.body(int(model.geom_bodyid[geom])).name
            if model.geom_conaffinity[geom] & 1 and body not in feet:
                ground_names.add(model.geom(geom).name or f"{body}_collision_{geom}")
        ground = torch.tensor([name in ground_names for name in legal.geom_names + illegal.geom_names],
                              dtype=torch.bool, device=device)
        self.ground_valid = self.valid & ground[:, None]
        self.torso = kin.body_names.index("torso_link")

    def platform_signed_distance(self, points_xy):
        return _union_signed_distance(points_xy, self.platform_outline)

    def _feet(self, body_state):
        positions, rotations, velocities, angular_velocities = body_state
        offset = torch.einsum("...fij,fsj->...fsi", rotations[..., self.foot_ids, :, :], self.sole_points)
        centers = positions[..., self.foot_ids, :].unsqueeze(-2) + offset
        velocity = (velocities[..., self.foot_ids, :].unsqueeze(-2)
                    + torch.cross(angular_velocities[..., self.foot_ids, :].unsqueeze(-2).expand_as(offset),
                                  offset, dim=-1))
        margin = (self.platform_signed_distance(centers[..., :2]) - self.sole_radius).amin(-1)
        height = (centers[..., 2] - self.sole_radius - self.platform_height).amin(-1)
        speed_sq = velocity[..., :2].square().sum(-1).mean(-1)
        return margin, height, speed_sq, centers

    def foot_features(self, S, body_state=None):
        if body_state is None:
            body_state = self.kin.forward(*S[:6], all_bodies=True)
        margin, height, _, _ = self._feet(body_state)
        return torch.cat((margin, height), -1)

    def actor_features(self, S, body_state=None):
        position = torch.stack((S[0][..., 0], S[0][..., 1], S[0][..., 2] - self.platform_height), -1)
        forward_xy = S[1][..., :2, 0]
        heading = forward_xy / torch.linalg.vector_norm(forward_xy, dim=-1, keepdim=True).clamp_min(1e-8)
        yaw_sin_cos = torch.stack((heading[..., 1], heading[..., 0]), -1)
        return torch.cat((position, yaw_sin_cos, self.foot_features(S, body_state), S[2]), -1)

    def surface_world_points(self, positions, rotations):
        return (torch.einsum("...gij,gsj->...gsi", rotations[..., self.body_ids, :, :], self.points)
                + positions[..., self.body_ids, :].unsqueeze(-2))

    def face_distance_sq(self, points):
        """Distance to the finite +Y faces; the wall's rear cannot earn contact."""
        low, high = self.wall_outline[:2]
        x, y, z = points.unbind(-1)
        dx = torch.maximum(low[:, 0] - x.unsqueeze(-1), x.unsqueeze(-1) - high[:, 0]).clamp_min(0)
        dy = y.unsqueeze(-1) - high[:, 1]
        dz = torch.maximum(self.wall_z_bounds[0] - z, z - self.wall_z_bounds[1]).clamp_min(0)
        return (dx.square() + dy.square() + dz.square().unsqueeze(-1)).amin(-1)

    def __call__(self, S, body_state=None):
        if body_state is None:
            body_state = self.kin.forward(*S[:6], all_bodies=True)
        positions, rotations, _, _ = body_state
        margin, height, speed_sq, sole_centers = self._feet(body_state)
        world = self.surface_world_points(positions, rotations)
        wall = _solid_clearance(world, self.wall_outline, self.wall_z_bounds).masked_fill(~self.valid, torch.inf)
        legal_distance = self.face_distance_sq(world[..., :self.legal_count, :, :])
        back_distance = (legal_distance.masked_fill(~self.valid[:self.legal_count], torch.inf).amin((-1, -2))
                         if self.legal_count else S[0].new_full(S[0].shape[:-1], torch.inf))
        illegal = (wall[..., self.legal_count:, :].amin((-1, -2))
                   if self.legal_count < len(self.points) else S[0].new_full(S[0].shape[:-1], torch.inf))
        platform = _solid_clearance(world, self.platform_outline, self.platform_z_bounds)
        platform = torch.minimum(platform, world[..., 2] - self.floor_height)
        platform = platform.masked_fill(~self.ground_valid, torch.inf).amin((-1, -2))
        # Ground contact includes the physical sole spheres, but not the
        # object-only toe meshes: toe overhang is allowed, sole tunnelling is not.
        sole_clearance = _solid_clearance(sole_centers, self.platform_outline, self.platform_z_bounds)
        sole_clearance = torch.minimum(sole_clearance, sole_centers[..., 2] - self.floor_height)
        sole_clearance = (sole_clearance - self.sole_radius).amin((-1, -2))
        platform = torch.minimum(platform, sole_clearance)
        return {"foot_margin": margin, "foot_height": height, "foot_speed_sq": speed_sq,
                "back_distance_sq": back_distance, "alignment": rotations[..., self.torso, 1, 0],
                "illegal_clearance": illegal, "wall_clearance": wall.amin((-1, -2)),
                "platform_clearance": platform}
