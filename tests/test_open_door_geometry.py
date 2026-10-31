"""Finite door geometry, actual G1 collision surfaces, and policy gradients."""

import mujoco
import numpy as np
import pytest
import torch

from supermimic.assets.g1.g1_constants import make_g1_spec
from supermimic.rotations import axis_angle_to_mat
from supermimic.tasks.open_door.geometry import DoorGeometry


@pytest.fixture
def geometry(cpu_kin):
    return DoorGeometry(cpu_kin, dict(
        hinge_pos=[0., 0., 1.], panel_half_size=[.02, .5, 1.],
        hinge_to_panel=[0., -.5, 0.], base_yaw=0., joint_axis=[0., 0., -1.],
        joint_range=[0., 1.8]), "cpu")


def _state(batch=1):
    return (torch.zeros(batch, 3), torch.eye(3).expand(batch, 3, 3).clone(),
            torch.zeros(batch, 3), torch.zeros(batch, 3),
            torch.zeros(batch, 29), torch.zeros(batch, 29),
            torch.zeros(batch, 1), torch.zeros(batch, 1))


def test_back_targets_only_finite_front_face(geometry):
    S = _state(3)
    P, R, _, _ = geometry.kin.forward(*S[:6], all_bodies=True)
    back = P[:, geometry.torso] + R[:, geometry.torso] @ geometry.back_offset
    target = geometry.hinge + geometry.center + torch.tensor([.02, 0., 0.])
    S[0][:] = target - back
    S[0][1, 0] -= .04  # The opposite face is not the legal contact face.
    S[0][2, 1] += .54  # Four cm beyond the finite panel edge.
    result = geometry(S)
    torch.testing.assert_close(result["back_distance_sq"], torch.tensor([0., .04**2, .04**2]),
                               atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(result["alignment"], torch.ones(3))
    S[1][0] = axis_angle_to_mat(torch.tensor([0., 0., 1.]), torch.tensor(torch.pi))
    assert geometry(S)["alignment"][0] == pytest.approx(-1.)


def test_only_legal_torso_patch_is_excluded(geometry):
    samples = geometry.samples
    assert len(samples.body_names) == 30
    assert samples.points.shape == (30, 32, 3)
    for source in ("pelvis_contour_link", "head_link", "torso_link", "logo_link",
                   "left_wrist_yaw_link", "left_ankle_roll_link"):
        assert source in samples.source_bodies
    assert samples.body_names[samples.source_bodies.index("head_link")] == "torso_link"
    assert samples.body_names[samples.source_bodies.index("pelvis_contour_link")] == "pelvis"
    for i, source in enumerate(samples.source_bodies):
        if source in ("torso_link", "logo_link"):
            x, y, z = samples.points[i, samples.valid[i]].T
            assert not ((x < -.025) & (abs(y) < .13) & (z > .05) & (z < .38)).any()
    # Full foot meshes collide with objects; ground-only support spheres do not.
    assert sum(name.endswith("_object_collision") for name in samples.geom_names) == 2


def test_sampling_preserves_mesh_extrema_in_fk_owner_frames(geometry):
    model = make_g1_spec(interaction_collisions=True).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for source in ("head_link", "pelvis_contour_link", "left_ankle_roll_link"):
        group = geometry.samples.source_bodies.index(source)
        source_id = model.body(source).id
        owner_id = model.body(geometry.samples.body_names[group]).id
        geom_id = next(g for g in range(model.ngeom)
                       if model.geom_bodyid[g] == source_id and model.geom_conaffinity[g] & 2)
        mesh = model.geom_dataid[geom_id]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start:start + count]
        world = vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
        local = (world - data.xpos[owner_id]) @ data.xmat[owner_id].reshape(3, 3)
        points = geometry.samples.points[group, geometry.samples.valid[group]]
        np.testing.assert_allclose(points.min(0), local.min(0), atol=1e-6, rtol=0)
        np.testing.assert_allclose(points.max(0), local.max(0), atol=1e-6, rtol=0)


@pytest.mark.parametrize("source", ["head_link", "pelvis_contour_link", "left_wrist_yaw_link",
                                   "left_ankle_roll_link"])
def test_mesh_surface_detects_contact_when_link_origin_is_outside(geometry, source):
    S = _state()
    P, R, _, _ = geometry.kin.forward(*S[:6], all_bodies=True)
    group = geometry.samples.source_bodies.index(source)
    owner = geometry.body_ids[group]
    points = geometry.points[group, geometry.valid[group]]
    world = points @ R[0, owner].T + P[0, owner]
    # Put the rear support extremum just inside the panel's front face.
    sample = world[world[:, 0].argmin()]
    target = geometry.hinge + geometry.center + torch.tensor([.015, 0., 0.])
    S[0][0] = target - sample
    result = geometry(S)
    assert result["illegal_clearance"][0, group] < -.0049

    # Independent MuJoCo collision oracle with the same actual mesh and pose.
    spec = make_g1_spec(interaction_collisions=True)
    spec.worldbody.add_geom(name="test_door", type=mujoco.mjtGeom.mjGEOM_BOX,
                           size=geometry.half.numpy(),
                           pos=(geometry.hinge + geometry.center).numpy(),
                           contype=2, conaffinity=0)
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:3] = S[0][0].numpy()
    mujoco.mj_forward(model, data)
    door = model.geom("test_door").id
    source_body = model.body(source).id
    assert any(door in (c.geom1, c.geom2) and c.dist < 0
               and model.geom_bodyid[c.geom2 if c.geom1 == door else c.geom1] == source_body
               for c in data.contact)
    origin = data.xpos[source_body] - (geometry.hinge + geometry.center).numpy()
    assert np.max(np.abs(origin) - geometry.half.numpy()) > 0


@pytest.mark.parametrize("sign", [-1., 1.])
def test_geometry_is_rigid_transform_invariant_and_differentiable(cpu_kin, sign):
    metadata = dict(hinge_pos=[0., 0., 1.], panel_half_size=[.02, .5, 1.],
                    hinge_to_panel=[0., -.5, 0.], base_yaw=0., joint_axis=[0., 0., sign],
                    joint_range=[0., 1.8])
    original = DoorGeometry(cpu_kin, metadata, "cpu")
    S = _state(2)
    S[0][:] = torch.tensor([[.2, -.4, .8], [.3, -.3, .7]])
    S[6][:] = torch.tensor([[.1], [.4]])
    angle, shift = torch.tensor(.7), torch.tensor([1.2, -.8, .15])
    rotation = axis_angle_to_mat(torch.tensor([0., 0., 1.]), angle)
    metadata["hinge_pos"] = (rotation @ original.hinge + shift).numpy()
    metadata["base_yaw"] = float(angle)
    moved = DoorGeometry(cpu_kin, metadata, "cpu")
    moved_state = (S[0] @ rotation.T + shift, rotation @ S[1], *S[2:])
    for key, value in original(S).items():
        torch.testing.assert_close(value, moved(moved_state)[key], atol=2e-6, rtol=2e-5)

    # Two leading dimensions represent e.g. environment and imagined time.
    S = tuple(value.unsqueeze(0).clone().requires_grad_() for value in S)
    values = original(S)
    assert values["illegal_clearance"].shape == (1, 2, 30)
    loss = sum(value.square().sum() for value in values.values())
    grads = torch.autograd.grad(loss, (S[0], S[4], S[6]))
    assert all(torch.isfinite(gradient).all() and gradient.abs().max() > 0 for gradient in grads)
