"""Shared right-side contact definition for tensor rewards and force diagnostics."""

import numpy as np


# G1's elbow and wrist-roll shells cover the forearm; wrist-pitch/yaw and
# rubber hand are not included. Shoulder-yaw is the upper arm, not the shoulder.
ALLOWED_BODIES = ("right_shoulder_pitch_link", "right_shoulder_roll_link",
                  "right_elbow_link", "right_wrist_roll_link")
TORSO_BODIES = ("torso_link", "logo_link")
SIDE_CONTACT_SCHEMA = {
    "side": "right",
    "allowed_bodies": list(ALLOWED_BODIES),
    "torso_bodies": list(TORSO_BODIES),
    "torso_patch_m": {"y_max": -.04, "abs_x_max": .12, "z_min": .02, "z_max": .35},
    "samples_per_geom": 32,
    "alignment": "torso_+Y_dot_panel_+X",
}


def torso_patch(points):
    """Works on NumPy construction samples and differentiable Torch tensors."""
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    return (y < -.04) & (abs(x) < .12) & (z > .02) & (z < .35)


def legal_surface_points(source, owner, points):
    if source in TORSO_BODIES:
        if owner != "torso_link":
            raise ValueError("Side torso samples must use the torso frame")
        return torso_patch(points)
    return np.full(len(points), source in ALLOWED_BODIES, dtype=bool)


def forbidden_surface_points(source, owner, points):
    return ~legal_surface_points(source, owner, points)


def classify_contact_points(torso_local_points, torso_geom, allowed_limb_geom):
    return allowed_limb_geom | (torso_geom & torso_patch(torso_local_points))
