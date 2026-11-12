"""The same G1/door tactile channels from physical or imagined tensor states."""

from __future__ import annotations

import torch

from ...assets.g1.tactile import load_tactile_layout
from ...tactile import G1TactileSensorCfg


class DoorTactile:
    """FK and finite-panel SDF with integer-valued counts and surrogate gradients.

    Forward values match the geometric sensor's channel pooling and rounding.
    Backward uses the continuous counts response, so imagined actions can receive
    gradients through the subsequent tactile observations. Invalid state frames
    produce zero placeholders and false validity masks.
    """

    def __init__(self, geometry, cfg: G1TactileSensorCfg):
        cfg.__post_init__()
        if cfg.target_entity != "door" or cfg.geom_name not in (None, "door_panel"):
            raise ValueError("DoorTactile requires the door/door_panel box surface")
        self.geometry = geometry
        self.layout = load_tactile_layout(cfg.layout)
        device = geometry.points.device
        self.body_ids = torch.tensor(
            [geometry.kin.body_names.index(name) for name in self.layout.link_names], device=device,
        )
        self.points = torch.tensor(self.layout.positions, dtype=torch.float32, device=device)
        width = max(map(len, self.layout.channel_members))
        self.members = torch.zeros((len(self.layout.channel_members), width), dtype=torch.long, device=device)
        self.member_mask = torch.zeros_like(self.members, dtype=torch.bool)
        for index, members in enumerate(self.layout.channel_members):
            self.members[index, :len(members)] = torch.tensor(members, device=device)
            self.member_mask[index, :len(members)] = True
        self.contact_distance_m = cfg.contact_distance_m
        self.full_scale_depth_m = cfg.counts_full_scale_depth_m
        self.exponent = cfg.counts_exponent
        self.metadata = {
            "layout": cfg.layout,
            "geometry_sha256": self.layout.geometry_sha256,
            "channel_abi_sha256": self.layout.channel_abi_sha256,
            "contact_distance_m": self.contact_distance_m,
            "full_scale_depth_m": self.full_scale_depth_m,
            "exponent": self.exponent,
            "backend": "torch_fk_box_sdf",
            "target_geom": "door/door_panel",
            "quantization": "rounded_forward_straight_through_backward",
            "history_order": "oldest_to_newest",
            "channel_pooling": "max_before_rounding",
            "units": "synthetic_counts",
            "calibration_status": "uncalibrated",
        }

    def __call__(self, S):
        batch_shape = S[0].shape[:-1]
        finite = [torch.isfinite(value) for value in S]
        state_valid = torch.stack(
            [mask.reshape(*batch_shape, -1).all(-1) for mask in finite], -1,
        ).all(-1)
        # Sanitize before FK, not only after distance queries: masked NaNs can
        # otherwise survive multiplication by zero during backpropagation.
        clean = tuple(torch.where(mask, value, 0.) for value, mask in zip(S, finite, strict=True))
        positions, rotations, _, _ = self.geometry.kin.forward(*clean[:6], all_bodies=True)
        world = (positions[..., self.body_ids, :]
                 + (rotations[..., self.body_ids, :, :] @ self.points[..., None]).squeeze(-1))
        point_valid = torch.isfinite(world).all(-1) & state_valid[..., None]
        world = torch.where(point_valid[..., None], world, 0.)
        yaw = self.geometry.base_yaw + self.geometry.hinge_sign * clean[6].squeeze(-1)
        distance = self.geometry.panel_clearance(world, yaw)
        point_valid = point_valid & torch.isfinite(distance)
        distance = torch.where(point_valid, distance, self.contact_distance_m)
        fraction = ((self.contact_distance_m - distance) / self.full_scale_depth_m).clamp(0., 1.)
        # At zero, x**exponent has an infinite derivative when exponent<1.
        # Evaluate only positive arguments and define the inactive slope as zero.
        active = fraction > 0
        response = torch.where(active, fraction, 1.).pow(self.exponent)
        continuous = 255. * torch.where(active & point_valid, response, 0.)
        grouped = continuous[..., self.members].masked_fill(~self.member_mask, 0.)
        channel_valid = (point_valid[..., self.members] | ~self.member_mask).all(-1)
        pooled = grouped.amax(-1)
        counts = pooled + (pooled.round() - pooled).detach()
        return counts.masked_fill(~channel_valid, 0.), channel_valid
