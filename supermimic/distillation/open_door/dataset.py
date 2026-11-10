"""CPU replay storage for OpenDoor proprioception and tactile DAgger samples."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .models import ACTION_DIM, GOAL_DIM, HISTORY_LENGTH, PROPRIO_DIM, TACTILE_CHANNELS


@dataclass
class DaggerBatch:
    proprio_history: torch.Tensor
    goal: torch.Tensor
    expert_action: torch.Tensor
    done: torch.Tensor
    tactile_history: torch.Tensor
    tactile_valid: torch.Tensor

    def to(self, device: torch.device | str) -> "DaggerBatch":
        return DaggerBatch(
            self.proprio_history.to(device),
            self.goal.to(device),
            self.expert_action.to(device),
            self.done.to(device),
            self.tactile_history.to(device),
            self.tactile_valid.to(device),
        )


class OpenDoorDaggerDataset:
    """Bounded ring buffer; samples are copied to CPU on insertion.

    Storing the already assembled history makes the first version independent
    of camera code and prevents an episode reset from changing old labels.
    """

    def __init__(self, capacity: int = 200_000, *, history_length: int = HISTORY_LENGTH,
                 proprio_dim: int = PROPRIO_DIM, goal_dim: int = GOAL_DIM,
                 action_dim: int = ACTION_DIM):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.history_length = history_length
        self.proprio_dim = proprio_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self._size = 0
        self._next = 0
        self._proprio = torch.empty(0)
        self._goal = torch.empty(0)
        self._action = torch.empty(0)
        self._done = torch.empty(0, dtype=torch.bool)
        self._tactile = torch.empty(0, dtype=torch.uint8)
        self._tactile_valid = torch.empty(0, dtype=torch.bool)

    def __len__(self) -> int:
        return self._size

    def _valid_indices(self) -> torch.Tensor:
        if self._size < self.capacity:
            return torch.arange(self._size, device="cpu")
        return (torch.arange(self._size, device="cpu") + self._next) % self.capacity

    def _allocate(self) -> None:
        self._proprio = torch.empty(self.capacity, self.history_length, self.proprio_dim,
                                    dtype=torch.float32, device="cpu")
        self._goal = torch.empty(self.capacity, self.goal_dim, dtype=torch.float32, device="cpu")
        self._action = torch.empty(self.capacity, self.action_dim, dtype=torch.float32, device="cpu")
        self._done = torch.empty(self.capacity, dtype=torch.bool, device="cpu")
        self._tactile = torch.empty(self.capacity, self.history_length, TACTILE_CHANNELS,
                                    dtype=torch.uint8, device="cpu")
        self._tactile_valid = torch.empty(self.capacity, self.history_length, TACTILE_CHANNELS,
                                          dtype=torch.bool, device="cpu")

    def add(self, proprio_history: torch.Tensor, goal: torch.Tensor,
            expert_action: torch.Tensor, done: torch.Tensor | None = None, *,
            tactile_history: torch.Tensor, tactile_valid: torch.Tensor) -> None:
        for name, value in (("proprio_history", proprio_history), ("goal", goal),
                            ("expert_action", expert_action)):
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                raise TypeError(f"{name} must be a floating-point tensor")
        for name, value, dtype in (("tactile_history", tactile_history, torch.uint8),
                                   ("tactile_valid", tactile_valid, torch.bool)):
            if not isinstance(value, torch.Tensor) or value.dtype != dtype:
                raise TypeError(f"{name} must have dtype {dtype}")
        if proprio_history.ndim != 3 or proprio_history.shape[-2:] != (self.history_length, self.proprio_dim):
            raise ValueError(f"proprio_history must be [B,{self.history_length},{self.proprio_dim}]")
        batch = proprio_history.shape[0]
        if goal.shape != (batch, self.goal_dim) or expert_action.shape != (batch, self.action_dim):
            raise ValueError("DAgger batch dimensions do not match the OpenDoor contract")
        tactile_shape = (batch, self.history_length, TACTILE_CHANNELS)
        if tactile_history.shape != tactile_shape or tactile_valid.shape != tactile_shape:
            raise ValueError(f"tactile_history and tactile_valid must have shape {tactile_shape}")
        done = torch.zeros(batch, dtype=torch.bool, device=proprio_history.device) if done is None else done
        if not isinstance(done, torch.Tensor) or done.dtype != torch.bool:
            raise TypeError("done must have dtype torch.bool")
        if done.shape != (batch,):
            raise ValueError("done must have shape [B]")
        if batch == 0:
            return
        if self._proprio.numel() == 0:
            self._allocate()
        # A single vectorized step can contain more samples than capacity.
        if batch >= self.capacity:
            proprio_history = proprio_history[-self.capacity:]
            goal, expert_action, done = goal[-self.capacity:], expert_action[-self.capacity:], done[-self.capacity:]
            tactile_history = tactile_history[-self.capacity:]
            tactile_valid = tactile_valid[-self.capacity:]
            batch = self.capacity
        indices = (torch.arange(batch, device="cpu") + self._next) % self.capacity
        self._proprio[indices] = proprio_history.detach().float().cpu()
        self._goal[indices] = goal.detach().float().cpu()
        self._action[indices] = expert_action.detach().float().cpu()
        self._done[indices] = done.detach().cpu()
        self._tactile[indices] = tactile_history.detach().cpu()
        self._tactile_valid[indices] = tactile_valid.detach().cpu()
        self._next = int((self._next + batch) % self.capacity)
        self._size = min(self.capacity, self._size + batch)

    def sample(self, batch_size: int, *, device: torch.device | str | None = None,
               generator: torch.Generator | None = None) -> DaggerBatch:
        if self._size == 0:
            raise RuntimeError("Cannot sample an empty DAgger dataset")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        indices = self._valid_indices()
        picks = torch.randint(len(indices), (batch_size,), device="cpu", generator=generator)
        indices = indices[picks]
        batch = DaggerBatch(self._proprio[indices], self._goal[indices], self._action[indices],
                            self._done[indices], self._tactile[indices], self._tactile_valid[indices])
        return batch if device is None else batch.to(device)

    def normalization_batches(self, batch_size: int):
        """Yield bounded CPU batches of only the two normalized input fields."""
        if self._size == 0:
            raise RuntimeError("Cannot read an empty DAgger dataset")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for indices in self._valid_indices().split(batch_size):
            yield self._proprio[indices], self._goal[indices]

    def all(self, *, device: torch.device | str | None = None) -> DaggerBatch:
        """Return the currently retained samples in ring-buffer order."""

        if self._size == 0:
            raise RuntimeError("Cannot read an empty DAgger dataset")
        indices = self._valid_indices()
        batch = DaggerBatch(self._proprio[indices], self._goal[indices],
                            self._action[indices], self._done[indices],
                            self._tactile[indices], self._tactile_valid[indices])
        return batch if device is None else batch.to(device)
