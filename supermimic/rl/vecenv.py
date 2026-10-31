"""Replay transitions from the same native environment created by mjlab.

The adapter never constructs or replaces a simulator. Training disables native
auto-reset so terminal next states can be recorded before resetting finished worlds.
"""
import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv


class ReplayEnvAdapter:
    def __init__(self, env, gamma=0.99):
        self.wrapped_env = env
        self.env = env.unwrapped
        if not isinstance(self.env, ManagerBasedRlEnv):
            raise TypeError("SuperMimic requires mjlab.ManagerBasedRlEnv")
        if self.env.cfg.auto_reset:
            raise ValueError("Replay collection requires auto_reset=False to preserve terminal states")
        self.command = self.env.command_manager.get_term("motion")
        self.task = self.command.task
        self.task.validate_model_coverage()
        self.motion, self.mdp, self.dynamics = self.task.motion, self.task.mdp, self.task.dynamics
        self.N = self.num_envs = self.env.num_envs
        self.device = self.env.device
        self.gamma = gamma
        self.command.curriculum.max_return = self.mdp.max_step_reward / (1 - gamma)
        self.ep_disc_ret = torch.zeros(self.N, device=self.device)
        self.ep_ret = torch.zeros_like(self.ep_disc_ret)

    @property
    def xi(self):
        return self.command.xi

    @property
    def f(self):
        return self.command.time_steps

    @property
    def a_prev(self):
        return self.env.action_manager.action

    @property
    def ep_step(self):
        return self.env.episode_length_buf

    def state(self):
        return self.command.state()

    def reset(self, ids=None, frames=None):
        if ids is None:
            ids = torch.arange(self.N, device=self.device)
        if frames is not None:
            self.command.set_reset_frames(ids, frames)
        self.ep_disc_ret[ids] = 0
        self.ep_ret[ids] = 0
        return self.env.reset(env_ids=ids)

    def step(self, action):
        s = self.dynamics.pack(self.state())
        if not bool(torch.isfinite(s).all() & torch.isfinite(action).all()):
            raise FloatingPointError("Nonfinite initial state or policy action before physics step")
        f, xi = self.f.clone(), self.xi.clone()
        previous, ep_step = self.a_prev.clone(), self.ep_step.clone()
        # Calling the existing wrapped chain preserves native VideoRecorder behavior.
        # RslRlVecEnvWrapper merges term/trunc; retrieve the separate native flags
        # from the same underlying env after its step.
        self.wrapped_env.step(action)
        # Warp keeps overflow flags across substeps until reset. Reject truncated
        # physics before recording it or resetting an episode (which clears flags).
        overflow = self.env.sim.data.overflow
        if bool(overflow.any()):
            from mujoco_warp._src.types import OverflowType
            bits = 0
            for value in overflow.unique().tolist():
                bits |= value
            names = [flag.name for flag in OverflowType if bits & int(flag)]
            raise RuntimeError(
                f"MuJoCo-Warp physics buffer overflow ({','.join(names)}; bits={bits}). "
                "Increase the affected simulation capacity; reduce num-envs if VRAM is insufficient. "
                "The transition was not recorded in Replay."
            )
        s1 = self.dynamics.pack(self.state())
        reward = self.env.reward_buf.clone()
        terminated = self.env.reset_terminated.clone()
        # Warp does not automatically reset numerically divergent worlds. NaN
        # comparisons in task terminations can also leave these worlds alive.
        # Use MuJoCo's numerical bound, not a task/contact termination threshold.
        valid = ((s1.abs() < mujoco.mjMAXVAL).all(-1)
                 & (self.env.sim.data.qacc.abs() < mujoco.mjMAXVAL).all(-1)
                 & torch.isfinite(reward))
        invalid = ~valid
        # Finite placeholders keep batched storage safe. These transitions are
        # excluded from WM fitting/normalization and policy starting states.
        s1[invalid] = s[invalid]
        reward[invalid] = 0
        terminated |= invalid
        done = terminated | self.env.reset_time_outs.clone()
        out = dict(s=s, s1=s1, a=action.clone(), a_prev=previous, f=f, xi=xi,
                   r=reward, term=terminated, done=done, ep_step=ep_step, valid=valid)
        self.ep_disc_ret += self.gamma ** ep_step.float() * reward
        self.ep_ret += reward
        ids = done.nonzero().flatten()
        if len(ids):
            length = (ep_step[ids] + 1).float()
            bootstrap = torch.where(terminated[ids], 0.0,
                                    self.gamma ** length * (self.ep_ret[ids] / length) / (1 - self.gamma))
            self.command.curriculum.update_returns(ids, self.ep_disc_ret[ids] + bootstrap)
            out["ep_len"] = length
            out["ep_ret_norm"] = self.ep_ret[ids] / (length * self.task.max_step_reward)
            self.reset(ids)
        return out
