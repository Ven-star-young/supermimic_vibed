"""Bounded CPU replay storage for sitting depth DAgger samples."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .models import ACTION_DIM, DEPTH_HEIGHT, DEPTH_WIDTH, GOAL_DIM, HISTORY_LENGTH, PROPRIO_DIM


@dataclass
class DaggerBatch:
    proprio_history: torch.Tensor
    goal: torch.Tensor
    expert_action: torch.Tensor
    done: torch.Tensor
    depth: torch.Tensor

    def to(self, device: torch.device | str) -> "DaggerBatch":
        return DaggerBatch(self.proprio_history.to(device), self.goal.to(device),
                           self.expert_action.to(device), self.done.to(device),
                           self.depth.to(device=device, dtype=torch.float32).clamp(0.1, 4.0))


class SittingDaggerDataset:
    """Ring buffer of pre-action snapshots; metric depth is stored as float16.

    Sampled depth is restored to float32 and encoded again with current CNN
    parameters. Capacity 16384 requires about 0.69 GiB for the complete buffer.
    """

    def __init__(self, capacity: int = 16_384):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._size = 0
        self._next = 0
        self._proprio = torch.empty(0)
        self._goal = torch.empty(0)
        self._action = torch.empty(0)
        self._done = torch.empty(0, dtype=torch.bool)
        self._depth = torch.empty(0, dtype=torch.float16)

    def __len__(self) -> int:
        return self._size

    def _valid_indices(self) -> torch.Tensor:
        indices = torch.arange(self._size, device="cpu")
        return (indices + self._next) % self.capacity if self._size == self.capacity else indices

    def _allocate(self) -> None:
        self._proprio = torch.empty(self.capacity, HISTORY_LENGTH, PROPRIO_DIM,
                                    dtype=torch.float32, device="cpu")
        self._goal = torch.empty(self.capacity, GOAL_DIM, dtype=torch.float32, device="cpu")
        self._action = torch.empty(self.capacity, ACTION_DIM, dtype=torch.float32, device="cpu")
        self._done = torch.empty(self.capacity, dtype=torch.bool, device="cpu")
        self._depth = torch.empty(self.capacity, 1, DEPTH_HEIGHT, DEPTH_WIDTH,
                                  dtype=torch.float16, device="cpu")

    def add(self, proprio_history: torch.Tensor, goal: torch.Tensor,
            expert_action: torch.Tensor, done: torch.Tensor | None = None, *,
            depth: torch.Tensor) -> None:
        for name, value in (("proprio_history", proprio_history), ("goal", goal),
                            ("expert_action", expert_action), ("depth", depth)):
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                raise TypeError(f"{name} must be a floating-point tensor")
        if proprio_history.ndim != 3 or proprio_history.shape[-2:] != (HISTORY_LENGTH, PROPRIO_DIM):
            raise ValueError("proprio_history must be [B,10,93]")
        batch = proprio_history.shape[0]
        if goal.shape != (batch, GOAL_DIM) or expert_action.shape != (batch, ACTION_DIM):
            raise ValueError("DAgger goal/action dimensions do not match the sitting contract")
        if depth.shape != (batch, 1, DEPTH_HEIGHT, DEPTH_WIDTH):
            raise ValueError("depth must be [B,1,120,160]")
        done = torch.zeros(batch, dtype=torch.bool, device=proprio_history.device) if done is None else done
        if not isinstance(done, torch.Tensor) or done.dtype != torch.bool or done.shape != (batch,):
            raise ValueError("done must have dtype bool and shape [B]")
        if batch == 0:
            return
        if self._proprio.numel() == 0:
            self._allocate()
        if batch >= self.capacity:
            proprio_history = proprio_history[-self.capacity:]
            goal, expert_action, done = goal[-self.capacity:], expert_action[-self.capacity:], done[-self.capacity:]
            depth = depth[-self.capacity:]
            batch = self.capacity
        indices = (torch.arange(batch, device="cpu") + self._next) % self.capacity
        self._proprio[indices] = proprio_history.detach().float().cpu()
        self._goal[indices] = goal.detach().float().cpu()
        self._action[indices] = expert_action.detach().float().cpu()
        self._done[indices] = done.detach().cpu()
        self._depth[indices] = depth.detach().to(device="cpu", dtype=torch.float16)
        self._next = (self._next + batch) % self.capacity
        self._size = min(self.capacity, self._size + batch)

    def _batch(self, indices: torch.Tensor, device: torch.device | str | None) -> DaggerBatch:
        batch = DaggerBatch(self._proprio[indices], self._goal[indices], self._action[indices],
                            self._done[indices], self._depth[indices])
        return batch.to("cpu" if device is None else device)

    def sample(self, batch_size: int, *, device: torch.device | str | None = None,
               generator: torch.Generator | None = None) -> DaggerBatch:
        if self._size == 0:
            raise RuntimeError("Cannot sample an empty DAgger dataset")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        # Every allocated slot below _size is valid, regardless of ring order.
        indices = torch.randint(self._size, (batch_size,), device="cpu", generator=generator)
        return self._batch(indices, device)

    def normalization_batches(self, batch_size: int):
        """Only proprioception and robot goals enter running normalization."""
        if self._size == 0 or batch_size < 1:
            raise ValueError("Normalization requires a nonempty dataset and positive batch_size")
        for indices in self._valid_indices().split(batch_size):
            yield self._proprio[indices], self._goal[indices]

    def all(self, *, device: torch.device | str | None = None) -> DaggerBatch:
        """Return retained samples oldest-to-newest; intended for small diagnostics."""
        if self._size == 0:
            raise RuntimeError("Cannot read an empty DAgger dataset")
        return self._batch(self._valid_indices(), device)
