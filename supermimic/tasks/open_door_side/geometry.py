"""FK-driven right torso, shoulder and forearm distances to the finite door."""

import numpy as np
import torch

from ..open_door.geometry import CollisionSurfaceSamples, DoorGeometry, _surface_samples
from .contacts import forbidden_surface_points, legal_surface_points


class SideDoorGeometry(DoorGeometry):
    def __init__(self, kin, metadata, device):
        legal = _surface_samples(tuple(kin.body_names), point_filter=legal_surface_points)
        forbidden = _surface_samples(tuple(kin.body_names), point_filter=forbidden_surface_points)
        combined = CollisionSurfaceSamples(
            legal.body_names + forbidden.body_names, legal.source_bodies + forbidden.source_bodies,
            legal.geom_names + forbidden.geom_names, np.concatenate((legal.points, forbidden.points)),
            np.concatenate((legal.valid, forbidden.valid)))
        super().__init__(kin, metadata, device, samples=combined)
        self.legal_count = len(legal.body_names)

    def __call__(self, S):
        positions, rotations, _, _ = self.kin.forward(*S[:6], all_bodies=True)
        yaw = self.base_yaw + self.hinge_sign * S[6].squeeze(-1)
        world = self.surface_world_points(positions, rotations)
        clearance = self.panel_clearance(world, yaw).masked_fill(~self.valid, torch.inf).amin(-1)
        distance = self.face_distance_sq(world[..., :self.legal_count, :, :], yaw)
        distance = distance.masked_fill(~self.valid[:self.legal_count], torch.inf).amin((-1, -2))
        normal = torch.stack((yaw.cos(), yaw.sin(), torch.zeros_like(yaw)), -1)
        alignment = (rotations[..., self.torso, :, 1] * normal).sum(-1)
        return {"contact_distance_sq": distance, "alignment": alignment,
                "contact_clearance": clearance[..., :self.legal_count],
                "illegal_clearance": clearance[..., self.legal_count:]}
