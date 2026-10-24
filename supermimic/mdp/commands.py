"""Reference frame progression and return-based reset curriculum."""
from dataclasses import dataclass, field
import math
from typing import Literal

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from .observations import ObservationCfg
from .rewards import RewardCfg
from .terminations import TerminationCfg
from .events import PhysicalContextCfg, ReferenceResetCfg, reset_robot_to_reference


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
    motion_dir: str = "data/lafan1_g1"
    play: bool = False
    goal_frames: int = 10
    history_length: int = 10
    gamma: float = 0.99
    reset_noise: bool = True
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    observation_cfg: ObservationCfg = field(default_factory=ObservationCfg)
    reward_cfg: RewardCfg = field(default_factory=RewardCfg)
    termination_cfg: TerminationCfg = field(default_factory=TerminationCfg)
    physical_cfg: PhysicalContextCfg = field(default_factory=PhysicalContextCfg)
    reset_cfg: ReferenceResetCfg = field(default_factory=ReferenceResetCfg)
    sampling_mode: Literal["adaptive", "start"] = "adaptive"
    uniform_weight: float = 0.5
    current_weight: float = 0.4
    history_weight: float = 0.1
    temperature: float = 25.0
    history_update: float = 0.001
    return_update: float = 0.2

    def build(self, env):
        return MotionCommand(self, env)


class ReferenceMotionCommand:
    def __init__(self, cfg: MotionCommandCfg, motion, num_envs: int, device: str, max_return: float):
        if cfg.sampling_mode not in ("adaptive", "start"):
            raise ValueError(f"Unknown motion sampling mode: {cfg.sampling_mode}")
        self.cfg, self.motion, self.device = cfg, motion, device
        self.max_return = max_return
        self.time_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.start_bin = torch.zeros_like(self.time_steps)
        self.bin_returns = torch.zeros(motion.num_bins, device=device)
        self.bin_history = torch.full((motion.num_bins,), 1.0 / motion.num_bins, device=device)

    def sample(self, count):
        m, cfg = self.motion, self.cfg
        if cfg.sampling_mode == "start":
            clips = torch.arange(count, device=self.device) % len(m.names)
            frames = m.clip_start[clips]
            return frames, m.frame_bin[frames]
        soft = torch.softmax(-cfg.temperature * self.bin_returns / self.max_return, 0)
        self.bin_history.lerp_(soft, cfg.history_update)
        probs = (cfg.uniform_weight / m.num_bins + cfg.current_weight * soft
                 + cfg.history_weight * self.bin_history / self.bin_history.sum())
        bins = torch.multinomial(probs, count, replacement=True)
        start = m.bin_start[bins]
        last = m.clip_end[m.bin_clip[bins]]
        end = torch.minimum(start + m.bin_frames, last)
        frames = start + (torch.rand(count, device=self.device) * (end - start)).long()
        return torch.minimum(frames, last - 1), bins

    def reset(self, ids, frames=None):
        if frames is None:
            frames, bins = self.sample(len(ids))
        else:
            bins = self.motion.frame_bin[frames]
        self.time_steps[ids] = frames
        self.start_bin[ids] = bins
        return frames

    def next_frames(self):
        last = self.motion.clip_end[self.motion.clip_id[self.time_steps]]
        return torch.minimum(self.time_steps + 1, last)

    def update_returns(self, ids, returns):
        bins = self.start_bin[ids]
        # Several worlds may finish from the same bin in one step. Aggregate them
        # before the EMA instead of issuing undefined duplicate indexed writes.
        unique, inverse = bins.unique(return_inverse=True)
        total = torch.zeros(len(unique), device=self.device).scatter_add_(0, inverse, returns)
        count = torch.zeros_like(total).scatter_add_(0, inverse, torch.ones_like(returns))
        rate = self.cfg.return_update
        self.bin_returns[unique] = (1 - rate) * self.bin_returns[unique] + rate * total / count


class MotionCommand(CommandTerm):
    """Reference progress, physical context and RSI in mjlab's lifecycle.

    While stepping from f, reward and termination compare with f+1. The command
    manager then advances to f+1 before observations. Reset marks the current
    control step so compute(dt=0) cannot skip the sampled starting frame.
    """

    def __init__(self, cfg: MotionCommandCfg, env):
        super().__init__(cfg, env)
        from ..tasks.g1_tracking.task import build_task

        if cfg.goal_frames < 1 or cfg.history_length < 1 or not 0 < cfg.gamma < 1:
            raise ValueError("Goal/history lengths must be positive and gamma must lie in (0, 1)")
        self.task = build_task(cfg.motion_dir, device=self.device, play=cfg.play, cfg=env.cfg)
        self.motion, self.mdp, self.dynamics = self.task.motion, self.task.mdp, self.task.dynamics
        if not math.isclose(env.step_dt, self.motion.control_dt):
            raise ValueError("Motion sample rate must match the environment control period")
        self.scene = env.scene
        action_cfg = env.cfg.actions["joint_position"]
        self.robot = self.scene[action_cfg.entity_name]
        self.joint_ids = torch.tensor([self.robot.joint_names.index(name) for name in action_cfg.joint_names],
                                      device=self.device)
        self.xi = torch.zeros(self.num_envs, self.dynamics.context_dim, device=self.device)
        self.curriculum = ReferenceMotionCommand(cfg, self.motion, self.num_envs, self.device,
                                                 self.mdp.max_step_reward / (1 - cfg.gamma))
        self.time_steps = self.curriculum.time_steps
        self._last_step = torch.full_like(self.time_steps, env.common_step_counter)
        self._reset_frames = torch.full_like(self.time_steps, -1)
        self._forward_step = -1

    @property
    def command(self):
        return self.motion.goal(self.time_steps, self.cfg.goal_frames)

    @property
    def transition_frame(self):
        return self.curriculum.next_frames()

    def state(self):
        # mj_step leaves derived root velocities one substep behind qpos/qvel.
        # Refresh once before the first reward/termination read each control step.
        # Do not scene.update() here: that would advance contact sensor histories.
        if self._forward_step != self._env.common_step_counter:
            self._env.sim.forward()
            self._forward_step = self._env.common_step_counter
        return self.dynamics.read_sim_state(self)

    def set_reset_frames(self, env_ids, frames):
        frames = torch.as_tensor(frames, device=self.device, dtype=torch.long)
        if frames.shape != env_ids.shape or torch.any(frames < 0) or torch.any(frames >= self.motion.num_frames):
            raise ValueError("Reset frames must supply a valid frame for each environment")
        self._reset_frames[env_ids] = frames

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        queued = self._reset_frames[env_ids]
        explicit = queued >= 0
        if self.cfg.play or self.cfg.sampling_mode == "start":
            frames = self.motion.clip_start[env_ids % len(self.motion.names)].clone()
        else:
            frames, _ = self.curriculum.sample(len(env_ids))
        frames[explicit] = queued[explicit]
        self.curriculum.reset(env_ids, frames)
        self._reset_frames[env_ids] = -1
        reset_robot_to_reference(self._env, env_ids, self.cfg.reset_cfg, command=self)
        self._last_step[env_ids] = self._env.common_step_counter
        self._forward_step = -1

    def _update_command(self, env_ids):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        changed = ids[self._last_step[ids] != self._env.common_step_counter]
        if len(changed):
            self.time_steps[changed] = self.transition_frame[changed]
            self._last_step[changed] = self._env.common_step_counter
