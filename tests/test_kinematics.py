"""Independent CPU MuJoCo oracle for named-joint FK and velocity conventions."""

import mujoco
import numpy as np
import pytest
import torch

from supermimic.assets.g1.g1_constants import G1_JOINT_NAMES, TRACKED_BODIES
from supermimic.rotations import mat_to_quat


@pytest.mark.parametrize("frame", [0, 37, 113])
def test_fk_matches_mujoco_link_jacobians(asset_path, cpu_kin, cpu_motion, frame):
    model = mujoco.MjModel.from_xml_path(str(asset_path))
    data = mujoco.MjData(model)
    state = cpu_motion.ref_state(torch.tensor([frame]))
    p, rotation, velocity, angular_velocity, q, qd = [value[0].numpy() for value in state]
    root_joint = model.joint("floating_base_joint")
    qpos_adr = int(root_joint.qposadr[0])
    dof_adr = int(root_joint.dofadr[0])
    data.qpos[qpos_adr:qpos_adr + 3] = p
    data.qpos[qpos_adr + 3:qpos_adr + 7] = mat_to_quat(state[1])[0].numpy()
    # MuJoCo free-joint linear velocity is global; angular velocity is local.
    data.qvel[dof_adr:dof_adr + 3] = velocity
    data.qvel[dof_adr + 3:dof_adr + 6] = rotation.T @ angular_velocity
    for i, name in enumerate(G1_JOINT_NAMES):
        joint = model.joint(name)
        data.qpos[int(joint.qposadr[0])] = q[i]
        data.qvel[int(joint.dofadr[0])] = qd[i]
    mujoco.mj_forward(model, data)
    actual_pos, actual_rot, actual_vel, actual_angvel = [x[0].numpy() for x in cpu_kin.forward(*state)]
    for i, name in enumerate(TRACKED_BODIES):
        body = model.body(name)
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, body.id)
        np.testing.assert_allclose(actual_pos[i], data.xpos[body.id], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(actual_rot[i], data.xmat[body.id].reshape(3, 3), atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(actual_vel[i], jacp @ data.qvel, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(actual_angvel[i], jacr @ data.qvel, atol=1e-6, rtol=1e-6)
