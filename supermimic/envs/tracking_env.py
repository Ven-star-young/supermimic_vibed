"""Native mjlab execution and replay transitions for configured tracking tasks.

Scene/physics and event execution live here; action mapping, reference progress,
RSI, physical randomization and differentiable MDP heads live in mdp/.
"""
from __future__ import annotations

import copy
import math

import torch
from mjlab.managers.event_manager import EventManager
from mjlab.scene import Scene
from mjlab.sim import Simulation

from ..mdp import G1Dynamics
from ..mdp.actions import JointPositionAction
from ..mdp.commands import ReferenceMotionCommand
from .config import TrackingEnvCfg


class TrackingSim:
    def __init__(self, num_envs, motion, mdp, device="cuda:0", gamma=0.99,
                 eval_mode=False, max_episode_steps=None, reset_noise=None,
                 *, cfg: TrackingEnvCfg | None = None, dynamics=None):
        if cfg is None:
            from ..tasks.g1_tracking.env_cfgs import make_env_cfg
            cfg = make_env_cfg(play=eval_mode)
        self.cfg = copy.deepcopy(cfg)
        if num_envs < 1 or self.cfg.decimation < 1:
            raise ValueError("num_envs and decimation must be positive")
        if not math.isclose(self.cfg.step_dt, motion.control_dt):
            raise ValueError("Motion sample rate must match the environment control period")
        if not math.isclose(self.cfg.step_dt, mdp.control_dt):
            raise ValueError("MDP reward scaling must match the environment control period")
        self.device, self.N = device, num_envs
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.dynamics = G1Dynamics() if dynamics is None else dynamics
        if self.cfg.context_dim != self.dynamics.context_dim:
            raise ValueError("Environment context dimension differs from dynamics schema")
        self.eval_mode = eval_mode
        self.reset_noise = (self.cfg.reset_noise if reset_noise is None else reset_noise) and not eval_mode
        steps = max_episode_steps if max_episode_steps is not None else math.ceil(self.cfg.episode_length_s / self.step_dt)
        self.max_episode_steps = steps if not eval_mode else 10**9
        self.cfg.scene.num_envs = num_envs
        self.scene = Scene(self.cfg.scene, device=device)
        self.sim = Simulation(num_envs=num_envs, cfg=self.cfg.sim, spec=self.scene.spec,
                              variant_info=self.scene.collect_variant_info(), device=device)
        self.scene.initialize(mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data)
        if self.scene.sensor_context is not None:
            self.sim.set_sensor_context(self.scene.sensor_context)
        self.action = JointPositionAction(self.cfg.actions, self.scene, device)
        if self.action.action_dim != self.dynamics.action_dim:
            raise ValueError("Action dimension differs from the task dynamics schema")
        self.robot = self.action.robot
        self.joint_ids = self.action.joint_ids
        self.default_q, self.alpha = self.action.offset, self.action.scale
        self.xi = torch.zeros(num_envs, self.cfg.context_dim, device=device)
        self.common_step_counter = 0
        self.ep_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.a_prev = torch.zeros(num_envs, self.action.action_dim, device=device)
        self.ep_disc_ret = torch.zeros(num_envs, device=device)
        self.ep_ret = torch.zeros_like(self.ep_disc_ret)
        self.G_max = mdp.max_step_reward / (1 - gamma)
        self.command = ReferenceMotionCommand(self.cfg.commands, motion, num_envs, device, self.G_max)
        self.done_eval = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.failed_eval = torch.zeros_like(self.done_eval)
        self.event_manager = EventManager(self.cfg.events, self)
        self.sim.expand_model_fields(self.event_manager.domain_randomization_fields)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(torch.arange(num_envs, device=device))

    @property
    def num_envs(self):
        return self.N

    @property
    def step_dt(self):
        return self.cfg.step_dt

    @property
    def physics_dt(self):
        return self.cfg.sim.mujoco.timestep

    # Preserve the old replay/evaluation accessors while command owns progress.
    @property
    def f(self):
        return self.command.time_steps

    @f.setter
    def f(self, value):
        self.command.time_steps = value

    @property
    def start_bin(self):
        return self.command.start_bin

    @property
    def bin_G(self):
        return self.command.bin_returns

    @property
    def bin_w_avg(self):
        return self.command.bin_history

    @property
    def torso_id(self):
        return self.robot.body_names.index("torso_link")

    def state(self):
        """G1 state in reference joint order; velocities at the pelvis link origin.

        An object task must extend this state reader together with its dynamics
        adapter and reference schema; merely spawning an object is insufficient.
        """
        return self.dynamics.read_sim_state(self)

    def sample_frames(self, count):
        return self.command.sample(count)

    def reset(self, env_ids, frames=None):
        if not len(env_ids):
            return
        self.sim.reset(env_ids)
        self.scene.reset(env_ids)
        self.command.reset(env_ids, frames)
        self.event_manager.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(mode="reset", env_ids=env_ids,
                                     global_env_step_count=self.common_step_counter)
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0
        self.ep_disc_ret[env_ids] = 0
        self.ep_ret[env_ids] = 0
        self.refresh()

    def refresh(self):
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.sim.sense()
        # scene.reset(env_ids) already invalidates sensor caches. Calling a global
        # scene.update(0) here would shift contact histories in untouched worlds.

    def step(self, action):
        state = self.state()
        frame, context = self.f.clone(), self.xi.clone()
        previous, ep_step = self.a_prev.clone(), self.ep_step.clone()
        self.action.apply(action, context)
        for _ in range(self.cfg.decimation):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.physics_dt)
        self.common_step_counter += 1
        for mode in ("step", "interval"):
            if mode in self.event_manager.available_modes:
                self.event_manager.apply(mode=mode, dt=self.step_dt)
        # Derived body kinematics must describe integrated qpos/qvel, not the
        # pre-integration quantities left by mj_step.
        self.sim.forward()
        self.sim.sense()
        next_state, next_frame = self.state(), self.command.next_frames()
        reward, terms = self.mdp.reward(next_state, next_frame, action, previous, terms=True)
        terminated = self.mdp.termination(next_state, next_frame)
        last = self.motion.clip_end[self.motion.clip_id[frame]]
        done = terminated | (next_frame >= last) | (ep_step + 1 >= self.max_episode_steps)
        out = dict(s=self.dynamics.pack(state), a=action.clone(), a_prev=previous, f=frame,
                   s1=self.dynamics.pack(next_state), r=reward, term=terminated, done=done,
                   ep_step=ep_step, xi=context, terms=terms)
        self.ep_disc_ret += self.gamma ** ep_step.float() * reward
        self.ep_ret += reward
        self.f, self.ep_step, self.a_prev = next_frame, ep_step + 1, action.clone()
        if self.eval_mode:
            self.failed_eval |= terminated & ~self.done_eval
            self.done_eval |= done
            return out
        ids = done.nonzero().flatten()
        if len(ids):
            length = self.ep_step[ids].float()
            bootstrap = torch.where(terminated[ids], 0.0,
                                    self.gamma ** length * (self.ep_ret[ids] / length) / (1 - self.gamma))
            self.command.update_returns(ids, self.ep_disc_ret[ids] + bootstrap)
            out["ep_len"] = length
            out["ep_ret_norm"] = self.ep_ret[ids] / (length * self.mdp.max_step_reward)
            self.reset(ids)
        return out

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.event_manager = None
        self.action = None
        self.robot = None
        self.scene = None
        self.sim = None
