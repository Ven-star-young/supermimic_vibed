"""Actual MuJoCo contacts for the opt-in G1 object-interaction profile."""

import mujoco
import numpy as np
import pytest

from supermimic.assets.g1.g1_constants import (
    G1_JOINT_NAMES,
    default_joint_pos_list,
    make_g1_robot_cfg,
    make_g1_spec,
)


def _active(model):
    return (model.geom_contype != 0) | (model.geom_conaffinity != 0)


def test_profile_preserves_kinematics_inertia_and_default_configuration(asset_path):
    baseline = make_g1_spec().compile()
    interaction = make_g1_robot_cfg(interaction_collisions=True).spec_fn().compile()
    default = make_g1_robot_cfg().spec_fn().compile()
    assert (baseline.nq, baseline.nv, baseline.nbody, baseline.njnt) == (
        interaction.nq, interaction.nv, interaction.nbody, interaction.njnt
    )
    assert [baseline.joint(i).name for i in range(baseline.njnt)] == [
        interaction.joint(i).name for i in range(interaction.njnt)
    ]
    for field in (
        "body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos",
        "body_iquat", "jnt_pos", "jnt_axis", "jnt_range", "qpos0",
        "mesh_vert", "mesh_face",
    ):
        np.testing.assert_array_equal(getattr(baseline, field), getattr(interaction, field))
    assert interaction.ngeom == baseline.ngeom + 2
    assert default.ngeom == baseline.ngeom
    np.testing.assert_array_equal(default.geom_conaffinity, baseline.geom_conaffinity)
    np.testing.assert_array_equal(default.geom_group, baseline.geom_group)
    assert np.all(default.geom_conaffinity[_active(default)] == 1)
    # Both training profiles and direct MJCF consumers (including replay) omit
    # hand geometry, while retaining wrist coordinates and fixed-link inertias.
    for model in (mujoco.MjModel.from_xml_path(str(asset_path)), baseline, interaction):
        assert model.nq == 36 and model.nv == 35
        for side in ("left", "right"):
            hand = model.body(f"{side}_rubber_hand")
            assert hand.geomnum[0] == 0
            assert hand.mass[0] == pytest.approx(.17)
            assert not any(model.mesh(i).name == f"{side}_rubber_hand" for i in range(model.nmesh))
            assert model.joint(f"{side}_wrist_yaw_joint").id >= 0


def test_profile_filters_objects_ground_and_self_collisions(asset_path):
    model = make_g1_spec(interaction_collisions=True).compile()
    active = _active(model)
    assert np.all(model.geom_contype == 0)
    assert np.all(model.geom_group[active] == 3)
    for side in ("left", "right"):
        foot = model.geom(f"{side}_ankle_roll_link_object_collision")
        assert model.geom_conaffinity[foot.id] == 2
        foot_id = model.body(f"{side}_ankle_roll_link").id
        spheres = (model.geom_bodyid == foot_id) & (
            model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE
        )
        assert spheres.sum() == 4
        assert np.all(model.geom_conaffinity[spheres] == 1)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0


def _floor_contacts(model, data):
    floor_id = model.geom("test_floor").id
    contacts = []
    for contact in data.contact:
        if floor_id not in (contact.geom1, contact.geom2):
            continue
        robot_geom = contact.geom2 if contact.geom1 == floor_id else contact.geom1
        body = model.body(model.geom_bodyid[robot_geom]).name
        assert body in {"left_ankle_roll_link", "right_ankle_roll_link"}
        assert model.geom_type[robot_geom] == mujoco.mjtGeom.mjGEOM_SPHERE
        contacts.append((body, *contact.pos, contact.dist))
    return sorted(contacts)


def test_original_foot_ground_contacts_are_unchanged(asset_path):
    models = []
    for enabled in (False, True):
        spec = make_g1_spec(interaction_collisions=enabled)
        spec.worldbody.add_geom(
            name="test_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.1], contype=1, conaffinity=0,
        )
        models.append(spec.compile())
    baseline, interaction = models
    data = mujoco.MjData(baseline)
    data.qpos[2] = 0.76
    for name, value in zip(G1_JOINT_NAMES, default_joint_pos_list()):
        data.joint(name).qpos[0] = value
    mujoco.mj_forward(baseline, data)
    spheres = baseline.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE
    foot_bottom = np.min(data.geom_xpos[spheres, 2] - baseline.geom_size[spheres, 0])
    data.qpos[2] += -0.001 - foot_bottom
    mujoco.mj_forward(baseline, data)
    other_data = mujoco.MjData(interaction)
    other_data.qpos[:] = data.qpos
    mujoco.mj_forward(interaction, other_data)
    before = _floor_contacts(baseline, data)
    after = _floor_contacts(interaction, other_data)
    assert len(before) >= 4
    assert [row[0] for row in before] == [row[0] for row in after]
    np.testing.assert_allclose(
        [row[1:] for row in before], [row[1:] for row in after], atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("side", ["left", "right"])
def test_foot_center_contacts_objects(asset_path, side):
    body_name = f"{side}_ankle_roll_link"
    point = [0.04, 0, -0.03]
    counts = []
    for enabled in (False, True):
        spec = make_g1_spec(interaction_collisions=enabled)
        if not enabled:
            # Permit object contacts in the old geometry too, proving missing
            # shape coverage rather than merely a difference in filter masks.
            for geom in spec.geoms:
                if geom.conaffinity:
                    geom.conaffinity = 3
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        body = model.body(body_name).id
        position = data.xpos[body] + data.xmat[body].reshape(3, 3) @ point
        spec.worldbody.add_geom(
            name="test_object", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.005, 0, 0], pos=position, contype=2, conaffinity=0,
        )
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        probe = model.geom("test_object").id
        contacts = [c for c in data.contact if probe in (c.geom1, c.geom2)]
        if enabled:
            assert all(c.dist < 0 for c in contacts)
            assert all(model.geom_bodyid[c.geom1 if c.geom2 == probe else c.geom2]
                       == model.body(body_name).id for c in contacts)
        counts.append(len(contacts))
    assert counts[0] == 0
    assert counts[1] > 0
