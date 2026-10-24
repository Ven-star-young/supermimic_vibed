"""Assemble the same analytic MDP heads for simulator and imagined states."""

from __future__ import annotations

import torch

from ..assets.g1.kinematics import G1Kinematics
from ..motion import CONTROL_DT, MotionLib
from .observations import (
    ObservationCfg, add_observation_noise, critic_feature_dimension, critic_features, proprioception,
)
from .rewards import RewardCfg, tracking_reward
from .terminations import TerminationCfg, soft_tracking_termination, tracking_termination


class MDP:
    """Configured G1 motion tracking heads, with no dependency on a simulator instance."""

    def __init__(self, kin: G1Kinematics, motion: MotionLib, default_q: torch.Tensor,
                 cfg: RewardCfg | None = None, *, observation_cfg: ObservationCfg | None = None,
                 termination_cfg: TerminationCfg | None = None, control_dt: float = CONTROL_DT):
        cfg = RewardCfg() if cfg is None else cfg
        self.kin, self.motion, self.cfg = kin, motion, cfg
        self.default_q = default_q
        self.observation_cfg = ObservationCfg() if observation_cfg is None else observation_cfg
        self.termination_cfg = (
            TerminationCfg(height=cfg.term_height, tilt=cfg.term_tilt)
            if termination_cfg is None else termination_cfg
        )
        self.control_dt = control_dt
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo, self.soft_hi = mid - half * cfg.soft_limit_factor, mid + half * cfg.soft_limit_factor
        self.obs_noise = self.observation_cfg.noise_tensor(default_q.device)
        self.obs_scale = self.observation_cfg.scale_tensor(default_q.device)

    @property
    def observation_dim(self) -> int:
        return self.observation_cfg.dimension

    @property
    def critic_feature_dim(self) -> int:
        return critic_feature_dimension(len(self.kin.tracked_idx))

    @property
    def max_step_reward(self) -> float:
        return self.cfg.max_step_reward(self.control_dt)

    def obs(self, S, a_prev, xi, f) -> torch.Tensor:
        # Keep the task-head call signature shared with other tasks, but the
        # actor's proprioception must not depend on reference frame or attitude.
        del f
        return proprioception(S, a_prev, xi, self.default_q)

    def add_noise(self, o: torch.Tensor) -> torch.Tensor:
        return add_observation_noise(o, self.obs_noise)

    def reward(self, S, f, a, a_prev, terms: bool = False):
        m = self.motion
        return tracking_reward(
            S, self.kin.forward(*S), (m.root_p[f], m.root_R[f]),
            (m.body_p[f], m.body_R[f], m.body_v[f], m.body_w[f]), a, a_prev,
            self.soft_lo, self.soft_hi, self.cfg, self.control_dt, terms,
        )

    def termination(self, S, f) -> torch.Tensor:
        return tracking_termination(S, self.motion.root_p[f], self.motion.root_R[f], self.termination_cfg)

    def soft_termination(self, S, f, tz: float | None = None, tt: float | None = None) -> torch.Tensor:
        return soft_tracking_termination(
            S, self.motion.root_p[f], self.motion.root_R[f], self.termination_cfg, tz, tt,
        )

    def critic_features(self, S, f, xi) -> torch.Tensor:
        m = self.motion
        return critic_features(
            S, self.kin.forward(*S)[0], m.root_p[f], m.root_R[f], m.q[f], m.body_p[f], xi,
        )
