"""Motion tracking on mjlab's native Scene, Entity and MuJoCo-Warp Simulation.

The simulator preserves SuperMimic's replay transition contract and analytic MDP
heads. Every control step is four 5 ms physics steps; terminal states are copied
before resetting only the finished worlds. No application launcher is required.
"""

from __future__ import annotations

import mujoco
import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from .mdp import MAX_STEP_REWARD, MDP, flatten_state
from .motion import CONTROL_DT, MotionLib
from .robot import G1_JOINT_NAMES, action_scale, make_g1_robot_cfg
from .rotations import mat_to_quat, quat_to_mat, so3_exp

DECIMATION = 4
PHYSICS_DT = CONTROL_DT / DECIMATION


def make_sim_cfg() -> SimulationCfg:
    return SimulationCfg(
        njmax=500,
        nconmax=128,
        mujoco=MujocoCfg(timestep=PHYSICS_DT, integrator="implicitfast", cone="pyramidal"),
    )


class TrackingSim:
    """Batched simulator, with joint arrays always in the motion library's order.

    Use ``with TrackingSim(num_envs, motion, mdp, device="cuda:0") as env``.
    MuJoCo quaternions are wxyz. Root velocities are measured at the pelvis link
    origin in world coordinates, matching the differentiable FK and motion data.
    """

    def __init__(self, num_envs: int, motion: MotionLib, mdp: MDP, device: str = "cuda:0",
                 gamma: float = 0.99, eval_mode: bool = False,
                 max_episode_steps: int = 500, reset_noise: bool = True):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.device = device
        self.N = num_envs
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.eval_mode = eval_mode
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        self.scene = Scene(
            SceneCfg(
                num_envs=num_envs,
                env_spacing=0.0,
                terrain=TerrainEntityCfg(terrain_type="plane"),
                entities={"robot": make_g1_robot_cfg()},
            ),
            device=device,
        )
        self.sim = Simulation(
            num_envs=num_envs, cfg=make_sim_cfg(), spec=self.scene.spec, device=device,
        )
        self.scene.initialize(mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data)
        self.robot = self.scene["robot"]
        if set(self.robot.joint_names) != set(G1_JOINT_NAMES):
            raise ValueError(f"Unexpected robot joints: {self.robot.joint_names}")
        self.joint_ids = torch.tensor(
            [self.robot.joint_names.index(n) for n in G1_JOINT_NAMES], device=device,
        )
        self.default_q = self.robot.data.default_joint_pos[0, self.joint_ids].clone()
        self.alpha = torch.tensor([action_scale(n) for n in G1_JOINT_NAMES], device=device)
        self.torso_id = self.robot.body_names.index("torso_link")

        # Physical context: friction, 29 joint offsets, torso COM offset.
        self.xi = torch.zeros(num_envs, 33, device=device)
        self.sim.expand_model_fields(self._randomize_physics.model_fields)
        self._randomize_physics()
        self.sim.recompute_constants(RecomputeLevel.set_const)

        self.f = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ep_step = torch.zeros_like(self.f)
        self.a_prev = torch.zeros(num_envs, 29, device=device)
        self.start_bin = torch.zeros_like(self.f)
        self.ep_disc_ret = torch.zeros(num_envs, device=device)
        self.ep_ret = torch.zeros(num_envs, device=device)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = torch.zeros(motion.num_bins, device=device)
        self.bin_w_avg = torch.full((motion.num_bins,), 1.0 / motion.num_bins, device=device)
        self.done_eval = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.failed_eval = torch.zeros_like(self.done_eval)
        self.reset(torch.arange(num_envs, device=device))

    @requires_model_fields("geom_friction", "body_ipos", recompute=RecomputeLevel.set_const)
    def _randomize_physics(self):
        """Allocate and write actual per-world model parameters, including derived fields."""
        if self.eval_mode:
            self.xi[:, 0] = 1.0
        else:
            self.xi[:, 0].uniform_(0.3, 1.2)
            self.xi[:, 30].uniform_(-0.025, 0.025)
            self.xi[:, 31:33].uniform_(-0.05, 0.05)
        geoms = self.robot.indexing.geom_ids
        self.sim.model.geom_friction[:, geoms, 0] = self.xi[:, :1]
        # MuJoCo combines equal-priority geom friction using max. Keep the ground
        # at the range minimum so each world's sampled robot friction is effective.
        planes = torch.as_tensor(
            self.sim.mj_model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE, device=self.device,
        )
        self.sim.model.geom_friction[:, planes, 0] = 1.0 if self.eval_mode else 0.3
        torso = int(self.robot.indexing.body_ids[self.torso_id])
        self.sim.model.body_ipos[:, torso] = (
            self.sim.get_default_field("body_ipos")[torso] + self.xi[:, 30:33]
        )

    def state(self):
        d = self.robot.data
        return (
            d.root_link_pos_w.clone() - self.scene.env_origins,
            quat_to_mat(d.root_link_quat_w.clone()),
            d.root_link_lin_vel_w.clone(),
            d.root_link_ang_vel_w.clone(),
            d.joint_pos[:, self.joint_ids].clone(),
            d.joint_vel[:, self.joint_ids].clone(),
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        # Native mjlab owns no global application. Drop the simulation and scene
        # together so CUDA graph/model buffers can be reclaimed between runs.
        self.robot = None
        self.scene = None
        self.sim = None

    # ------------------------------------------------------------------ reset
    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.motion
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probs = 0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + 50, m.clip_end[m.bin_clip[bins]])  # keep >= 1 step before the clip end
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        f = torch.minimum(f, m.clip_end[m.bin_clip[bins]] - 1)
        return f, bins

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            frames, bins = self.sample_frames(n)
        else:
            bins = self.motion.frame_bin[frames]
        m = self.motion
        p, R, v, w, q, qd = (x.clone() for x in m.ref_state(frames))
        if self.reset_noise:
            p = p + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.05, 0.05), torch.empty(n, device=dev).uniform_(-0.05, 0.05),
                 torch.empty(n, device=dev).uniform_(-0.01, 0.01)), -1)  # fmt: skip
            rv = torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.1, 0.1), torch.empty(n, device=dev).uniform_(-0.1, 0.1),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            R = so3_exp(rv) @ R
            v = v + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.5, 0.5), torch.empty(n, device=dev).uniform_(-0.5, 0.5),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            w = w + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.52, 0.52), torch.empty(n, device=dev).uniform_(-0.52, 0.52),
                 torch.empty(n, device=dev).uniform_(-0.78, 0.78)), -1)  # fmt: skip
            q = q + torch.empty_like(q).uniform_(-0.1, 0.1)
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        quat = mat_to_quat(R)
        # Clear MuJoCo solver state and actuator buffers before writing RSI.
        self.sim.reset(env_ids)
        self.scene.reset(env_ids)
        pose = torch.cat((p + self.scene.env_origins[env_ids], quat), -1)
        self.robot.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim(torch.cat((v, w), -1), env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q, qd, joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.set_joint_position_target(q, joint_ids=self.joint_ids, env_ids=env_ids)
        self.f[env_ids] = frames
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        if not self.eval_mode:
            self.xi[env_ids, 1:30] = torch.empty(n, 29, device=dev).uniform_(-0.01, 0.01)
        self.refresh()

    def refresh(self):
        """Propagate written states so that ``state()`` is consistent after a reset."""
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.scene.update(0.0)

    # ------------------------------------------------------------------ step
    def step(self, a: torch.Tensor) -> dict:
        S = self.state()
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        target = self.default_q + xi[:, 1:30] + self.alpha * a
        self.robot.set_joint_position_target(target, joint_ids=self.joint_ids)
        for _ in range(DECIMATION):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(PHYSICS_DT)
        # MuJoCo step leaves derived kinematics at its pre-integration state.
        # Rewards and stored s1 must describe the actual integrated next state.
        self.sim.forward()
        S1 = self.state()
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        f1 = torch.minimum(f + 1, clip_end)
        r, terms = self.mdp.reward(S1, f1, a, a_prev, terms=True)
        term = self.mdp.termination(S1, f1)
        timeout = (f1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        out = {
            "s": flatten_state(S), "a": a.clone(), "a_prev": a_prev, "f": f, "s1": flatten_state(S1), "r": r,
            "term": term, "done": done, "ep_step": ep_step, "xi": xi, "terms": terms,
        }  # fmt: skip
        # bookkeeping
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = a.clone()
        if self.eval_mode:
            self.failed_eval |= term & ~self.done_eval
            self.done_eval |= done
            return out
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            # bin return estimate; truncated episodes bootstrap with their mean reward
            T = self.ep_step[ids].float()
            boot = torch.where(term[ids], 0.0, (self.gamma**T) * (self.ep_ret[ids] / T) / (1 - self.gamma))
            G = self.ep_disc_ret[ids] + boot
            b = self.start_bin[ids]
            self.bin_G.index_put_((b,), 0.8 * self.bin_G[b] + 0.2 * G)
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * MAX_STEP_REWARD)
            self.reset(ids)
        return out
