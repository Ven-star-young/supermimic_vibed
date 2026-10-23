"""Reference frame progression and return-based reset curriculum."""
from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(kw_only=True)
class MotionCommandCfg:
    sampling_mode: Literal["adaptive", "start"] = "adaptive"
    uniform_weight: float = 0.5
    current_weight: float = 0.4
    history_weight: float = 0.1
    temperature: float = 25.0
    history_update: float = 0.001
    return_update: float = 0.2


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
