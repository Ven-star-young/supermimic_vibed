"""Differentiable forward kinematics of the G1 in torch, built from the compiled MJCF.

The same FK computes tracked-body poses and velocities for the reference motion, for logged
simulator states, and for world-model predictions, so the analytic reward head matches the
reward the simulator transitions are labeled with.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .g1_constants import G1_JOINT_NAMES, TRACKED_BODIES, make_g1_spec
from ...rotations import axis_angle_to_mat, quat_to_mat


class G1Kinematics:
    def __init__(self, device: str, tracked: list[str] = TRACKED_BODIES):
        m = make_g1_spec().compile()
        self.device = device
        jname_to_idx = {n: i for i, n in enumerate(G1_JOINT_NAMES)}
        actual_joints = {m.joint(j).name for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE}
        if actual_joints != set(G1_JOINT_NAMES):
            raise ValueError(f"G1 MJCF joints differ from retargeting order: {actual_joints ^ set(G1_JOINT_NAMES)}")
        # Include all moving/tracked bodies and any fixed ancestors. Omit purely
        # visual leaf bodies so the differentiable rollout keeps the original cost.
        selected = {b for b in range(1, m.nbody) if m.body_jntnum[b] or m.body(b).name in tracked}
        for b in list(selected):
            pb = m.body_parentid[b]
            while pb:
                selected.add(int(pb))
                pb = m.body_parentid[pb]
        bodies, parent, joint_idx, axes, offs_p, offs_R, anchors, refs = [], [], [], [], [], [], [], []
        mj_to_local = {}
        for b in range(1, m.nbody):
            if b not in selected:
                continue
            name = m.body(b).name
            if m.body_jntnum[b] > 1:
                raise ValueError(f"Expected at most one joint on G1 body {name}")
            jid = int(m.body_jntadr[b]) if m.body_jntnum[b] else -1
            mj_to_local[b] = len(bodies)
            bodies.append(name)
            if jid >= 0 and m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                if name != "pelvis" or m.body_parentid[b] != 0:
                    raise ValueError("Expected the free root joint on the pelvis")
                parent.append(-1)
                joint_idx.append(-1)
            else:
                if m.body_parentid[b] not in mj_to_local:
                    raise ValueError(f"G1 body {name} is not descended from the free pelvis")
                parent.append(mj_to_local[m.body_parentid[b]])
                if jid >= 0 and m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError(f"Expected a hinge joint on G1 body {name}")
                joint_idx.append(jname_to_idx[m.joint(jid).name] if jid >= 0 else -1)
            axes.append(m.jnt_axis[jid].copy() if jid >= 0 else np.zeros(3))
            anchors.append(m.jnt_pos[jid].copy() if jid >= 0 else np.zeros(3))
            refs.append(float(m.qpos0[m.jnt_qposadr[jid]]) if jid >= 0 and joint_idx[-1] >= 0 else 0.0)
            offs_p.append(m.body_pos[b].copy())
            offs_R.append(m.body_quat[b].copy())
        self.body_names = bodies
        self.parent = parent
        self.joint_idx = joint_idx
        self.axes = torch.tensor(np.array(axes), dtype=torch.float32, device=device)
        self.off_p = torch.tensor(np.array(offs_p), dtype=torch.float32, device=device)
        self.off_R = quat_to_mat(torch.tensor(np.array(offs_R), dtype=torch.float32, device=device))
        self.anchors = torch.tensor(np.array(anchors), dtype=torch.float32, device=device)
        self.has_anchor_offset = [bool(np.any(anchor)) for anchor in anchors]
        self.joint_ref = torch.tensor(refs, dtype=torch.float32, device=device)
        self.tracked_idx = [bodies.index(n) for n in tracked]
        self.num_bodies = len(bodies)
        # joint limits (in G1_JOINT_NAMES order)
        lo, hi = np.zeros(29), np.zeros(29)
        for j in range(m.njnt):
            n = m.joint(j).name
            if n in jname_to_idx:
                lo[jname_to_idx[n]], hi[jname_to_idx[n]] = m.jnt_range[j]
        self.q_lo = torch.tensor(lo, dtype=torch.float32, device=device)
        self.q_hi = torch.tensor(hi, dtype=torch.float32, device=device)

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Tracked-body world positions, rotations, linear (link-origin) and angular velocities.

        Args:
            p, R, v, w: root (pelvis) position (B,3), rotation (B,3,3), world lin/ang velocity (B,3).
            q, qd: joint positions/velocities (B,29) in G1_JOINT_NAMES order.
        """
        P, Rs, V, W = [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies, [None] * self.num_bodies
        for i in range(self.num_bodies):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint_idx[i]
            Rp = Rs[par]
            pi = P[par] + (Rp @ self.off_p[i].unsqueeze(-1)).squeeze(-1)
            R_off = Rp @ self.off_R[i]
            if j < 0:  # fixed link
                Ri, wi = R_off, W[par]
                vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1)
            else:
                Ri = R_off @ axis_angle_to_mat(self.axes[i], q[..., j] - self.joint_ref[i])
                axis_w = (R_off @ self.axes[i].unsqueeze(-1)).squeeze(-1)
                joint_w = axis_w * qd[..., j : j + 1]
                wi = W[par] + joint_w
                if self.has_anchor_offset[i]:
                    joint_p = pi + (R_off @ self.anchors[i].unsqueeze(-1)).squeeze(-1)
                    pi = joint_p - (Ri @ self.anchors[i].unsqueeze(-1)).squeeze(-1)
                    vi = (V[par] + torch.cross(W[par], pi - P[par], dim=-1)
                          + torch.cross(joint_w, pi - joint_p, dim=-1))
                else:
                    vi = V[par] + torch.cross(W[par], pi - P[par], dim=-1)
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = range(self.num_bodies) if all_bodies else self.tracked_idx
        return (
            torch.stack([P[i] for i in idx], -2),
            torch.stack([Rs[i] for i in idx], -3),
            torch.stack([V[i] for i in idx], -2),
            torch.stack([W[i] for i in idx], -2),
        )
