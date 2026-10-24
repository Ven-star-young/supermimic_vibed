"""mjlab manager terms backed by the same tensor heads used for imagination."""
from __future__ import annotations

import torch
from mjlab.managers.manager_base import ManagerTermBase


def motion_command(env, command_name: str = "motion"):
    return env.command_manager.get_term(command_name)


def goal(env, command_name: str = "motion"):
    return motion_command(env, command_name).command


def state(env, command_name: str = "motion"):
    cmd = motion_command(env, command_name)
    return cmd.dynamics.pack(cmd.state())


def context(env, command_name: str = "motion"):
    return motion_command(env, command_name).xi


def frame(env, command_name: str = "motion"):
    return motion_command(env, command_name).time_steps[:, None]


def step_count(env):
    return env.episode_length_buf[:, None]


def previous_action(env):
    # Observations are evaluated after the just-executed action and after reset.
    return env.action_manager.action


class PolicyHistory(ManagerTermBase):
    """Oldest-to-newest history, with idempotent reads and scoped reset backfill.

    Buffering here makes the policy input identical in training, stock mjlab
    play and evaluation. mjlab may read observations repeatedly in one step;
    repeated reads must not shift history or resample sensor noise.
    """

    def __init__(self, cfg, env):
        super().__init__(env)
        self.command_name = cfg.params.get("command_name", "motion")
        cmd = motion_command(env, self.command_name)
        self.buffer = torch.zeros(env.num_envs, cmd.cfg.history_length, cmd.mdp.observation_dim,
                                  device=env.device)
        self.pending_reset = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        self.last_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.pending_reset[ids] = True
        self.last_step[ids] = -1

    def __call__(self, env, command_name: str = "motion"):
        changed = self.last_step != env.common_step_counter
        if changed.any():
            cmd = motion_command(env, command_name)
            value = cmd.mdp.obs(cmd.state(), env.action_manager.action, cmd.xi, cmd.time_steps)
            if not cmd.cfg.play:
                value = cmd.mdp.add_noise(value)
            self.buffer[changed] = self.buffer[changed].roll(-1, dims=1)
            self.buffer[changed, -1] = value[changed]
            pending = self.pending_reset & changed
            self.buffer[pending] = value[pending, None].expand(-1, self.buffer.shape[1], -1)
            self.pending_reset[pending] = False
            self.last_step[changed] = env.common_step_counter
        return self.buffer.flatten(1)


def tracking_reward(env, command_name: str = "motion"):
    cmd = motion_command(env, command_name)
    return cmd.mdp.reward(cmd.state(), cmd.transition_frame, env.action_manager.action,
                          env.action_manager.prev_action)


def tracking_failure(env, command_name: str = "motion"):
    cmd = motion_command(env, command_name)
    return cmd.mdp.termination(cmd.state(), cmd.transition_frame)


def motion_complete(env, command_name: str = "motion"):
    cmd = motion_command(env, command_name)
    last = cmd.motion.clip_end[cmd.motion.clip_id[cmd.time_steps]]
    return cmd.transition_frame >= last
