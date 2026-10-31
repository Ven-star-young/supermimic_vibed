"""Right-side collision surfaces follow FK and target only the finite front face."""

import numpy as np
import pytest
import torch

from supermimic.assets.g1.g1_constants import G1_JOINT_NAMES
from supermimic.rotations import axis_angle_to_mat
from supermimic.tasks.open_door_side.contacts import ALLOWED_BODIES, legal_surface_points, torso_patch
from supermimic.tasks.open_door_side.geometry import SideDoorGeometry


@pytest.fixture
def geometry(cpu_kin):
    return SideDoorGeometry(cpu_kin, dict(
        hinge_pos=[0., 0., 1.], panel_half_size=[.02, .5, 1.],
        hinge_to_panel=[0., -.5, 0.], base_yaw=0., joint_axis=[0., 0., -1.],
        joint_range=[0., 1.8]), "cpu")


def state(batch=1):
    return (torch.zeros(batch, 3), torch.eye(3).expand(batch, 3, 3).clone(),
            torch.zeros(batch, 3), torch.zeros(batch, 3),
            torch.zeros(batch, 29), torch.zeros(batch, 29),
            torch.zeros(batch, 1), torch.zeros(batch, 1))


def test_asset_regions_allow_shoulder_forearm_and_only_right_torso_patch(geometry):
    samples, count = geometry.samples, geometry.legal_count
    legal, forbidden = set(samples.source_bodies[:count]), set(samples.source_bodies[count:])
    assert set(ALLOWED_BODIES) <= legal
    assert not set(ALLOWED_BODIES) & forbidden
    for source in ("right_shoulder_yaw_link", "right_wrist_pitch_link",
                   "right_wrist_yaw_link", "head_link", "left_elbow_link"):
        assert source in forbidden and source not in legal
    for group, source in enumerate(samples.source_bodies):
        if source in ("torso_link", "logo_link"):
            points = samples.points[group, samples.valid[group]]
            assert (torso_patch(points) == (group < count)).all()
    # Central back and left side stay forbidden, even at a legal torso height.
    points = np.array([[0., -.1, .2], [-.07, 0., .2], [0., .1, .2], [0., -.1, .01]])
    np.testing.assert_array_equal(legal_surface_points("torso_link", "torso_link", points),
                                  [True, False, False, False])


def test_forearm_samples_move_with_elbow_instead_of_a_fixed_torso_point(geometry):
    S = state(2)
    elbow = G1_JOINT_NAMES.index("right_elbow_joint")
    S[4][1, elbow] = .8
    S[4].requires_grad_()
    positions, rotations, _, _ = geometry.kin.forward(*S[:6], all_bodies=True)
    world = geometry.surface_world_points(positions, rotations)
    forearm = geometry.samples.source_bodies.index("right_wrist_roll_link")
    shoulder = geometry.samples.source_bodies.index("right_shoulder_roll_link")
    assert (world[1, forearm] - world[0, forearm]).abs().max() > .05
    torch.testing.assert_close(world[0, shoulder], world[1, shoulder])
    gradient, = torch.autograd.grad(world[1, forearm, 0].sum(), S[4])
    assert torch.isfinite(gradient).all() and gradient[1, elbow].abs() > .01


def test_alignment_uses_torso_positive_y(geometry):
    S = state(3)
    S[1][:] = axis_angle_to_mat(torch.tensor([0., 0., 1.]),
                              torch.tensor([0., -torch.pi / 2, torch.pi / 2]))
    torch.testing.assert_close(geometry(S)["alignment"], torch.tensor([0., 1., -1.]), atol=1e-6, rtol=0)


def test_contact_target_is_finite_positive_x_face(geometry):
    S = state(3)
    positions, rotations, _, _ = geometry.kin.forward(*S[:6], all_bodies=True)
    world = geometry.surface_world_points(positions, rotations)
    # Isolate one actual legal asset sample to exercise each target boundary.
    geometry.valid[:geometry.legal_count] = False
    geometry.valid[0, 0] = True
    target = geometry.hinge + geometry.center + torch.tensor([.02, 0., 0.])
    S[0][:] = target - world[:, 0, 0]
    S[0][1, 0] -= .04
    S[0][2, 1] += .54
    torch.testing.assert_close(geometry(S)["contact_distance_sq"],
                               torch.tensor([0., .04**2, .04**2]), atol=1e-7, rtol=1e-5)


def test_side_geometry_has_finite_pose_joint_and_hinge_gradients(geometry):
    S = state(2)
    S[0][:] = torch.tensor([[.2, -.4, .8], [.3, -.3, .7]])
    S[6][:] = torch.tensor([[.1], [.4]])
    S = tuple(value.unsqueeze(0).clone().requires_grad_() for value in S)
    values = geometry(S)
    assert values["contact_clearance"].shape == (1, 2, geometry.legal_count)
    assert values["illegal_clearance"].shape[:2] == (1, 2)
    loss = sum(value.square().sum() for value in values.values())
    gradients = torch.autograd.grad(loss, (S[0], S[4], S[6]))
    assert all(torch.isfinite(gradient).all() and gradient.abs().max() > 0 for gradient in gradients)
