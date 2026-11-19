"""DAgger collection and supervised updates for the sitting student."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import SittingDaggerConfig
from .dataset import SittingDaggerDataset
from .models import FrozenSittingExpert, SittingStudentPolicy, split_sitting_history


@dataclass
class DaggerStep:
    proprio_history: torch.Tensor
    goal: torch.Tensor
    expert_action: torch.Tensor
    student_action: torch.Tensor
    executed_action: torch.Tensor
    expert_mask: torch.Tensor
    depth: torch.Tensor | None


def mix_actions(expert_action: torch.Tensor, student_action: torch.Tensor,
                beta: float, *, generator: torch.Generator | None = None):
    """Choose the teacher or student independently per environment."""
    if expert_action.shape != student_action.shape:
        raise ValueError("expert and student actions must have the same shape")
    if not 0 <= beta <= 1:
        raise ValueError("beta must lie in [0, 1]")
    mask = torch.rand(expert_action.shape[0], 1, device=expert_action.device, generator=generator) < beta
    return torch.where(mask, expert_action, student_action), mask.squeeze(-1)


class DaggerCoordinator:
    """Keep collection, immutable teacher labels and student optimization explicit."""

    def __init__(self, expert: FrozenSittingExpert, student: SittingStudentPolicy,
                 dataset: SittingDaggerDataset | None = None,
                 config: SittingDaggerConfig | None = None, *, device=None):
        self.config = config or SittingDaggerConfig()
        if student.use_depth != self.config.use_depth:
            raise ValueError("Student depth input must match the DAgger configuration")
        self.expert = expert.eval()
        self.student = student
        self.dataset = dataset if dataset is not None else SittingDaggerDataset(
            self.config.capacity, use_depth=self.config.use_depth)
        self.device = device or next(student.parameters()).device
        self.optimizer = torch.optim.Adam(student.parameters(), lr=self.config.learning_rate,
                                          weight_decay=self.config.weight_decay)
        self._normalizers_initialized = False

    def beta(self, round_index: int) -> float:
        if round_index < 0:
            raise ValueError("round_index must be nonnegative")
        alpha = min(round_index / self.config.beta_decay_rounds, 1.0)
        return self.config.beta_start + alpha * (self.config.beta_end - self.config.beta_start)

    @torch.no_grad()
    def label_and_select(self, policy_history: torch.Tensor, goal: torch.Tensor,
                         beta: float, *, depth: torch.Tensor | None, geometry_ids: torch.Tensor,
                         generator: torch.Generator | None = None) -> DaggerStep:
        """Snapshot one pre-action state; privileged chair data stays in the teacher."""
        policy_history = policy_history.detach().clone()
        proprio = split_sitting_history(policy_history).clone()
        goal = goal.detach().clone()
        if self.config.use_depth:
            if depth is None:
                raise ValueError("Depth configuration requires a depth observation")
            depth = depth.detach().clone()
        elif depth is not None:
            raise ValueError("Proprioception configuration does not accept depth")
        geometry_ids = geometry_ids.detach().clone()
        self.student.eval()
        expert_action = self.expert(policy_history, goal, geometry_ids)
        student_action = self.student(proprio, goal, depth)
        executed, mask = mix_actions(expert_action, student_action, beta, generator=generator)
        return DaggerStep(proprio, goal, expert_action, student_action, executed, mask, depth)

    def record(self, step: DaggerStep, done: torch.Tensor | None = None) -> None:
        self.dataset.add(step.proprio_history, step.goal, step.expert_action, done, depth=step.depth)

    def train_epoch(self, batches: int | None = None) -> float:
        if len(self.dataset) == 0:
            raise RuntimeError("Collect DAgger samples before training the student")
        if not self._normalizers_initialized:
            with torch.no_grad():
                for proprio, goal in self.dataset.normalization_batches(self.config.batch_size):
                    self.student.update_normalizers(proprio.to(self.device), goal.to(self.device))
            self._normalizers_initialized = True
        batches = max(1, len(self.dataset) // self.config.batch_size) if batches is None else batches
        if batches < 1:
            raise ValueError("batches must be positive")
        self.student.train()
        losses = []
        for _ in range(batches):
            batch = self.dataset.sample(self.config.batch_size, device=self.device)
            prediction = self.student(batch.proprio_history, batch.goal, batch.depth)
            loss = (prediction - batch.expert_action).square().mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.student.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            losses.append(loss.detach())
        return torch.stack(losses).mean().item()

    def train_round(self, round_index: int, *, batches: int | None = None) -> dict[str, float]:
        loss = sum(self.train_epoch(batches=batches) for _ in range(self.config.train_epochs)) / self.config.train_epochs
        return {"round": float(round_index), "beta": self.beta(round_index),
                "dataset_size": float(len(self.dataset)), "action_mse": loss}

    def pretrain_bc(self, *, epochs: int | None = None, batches: int | None = None) -> dict:
        """Initialize on teacher-only rollouts before the first student-controlled round."""
        epochs = self.config.bc_train_epochs if epochs is None else epochs
        batches = self.config.bc_batches if batches is None else batches
        if epochs < 1 or batches < 1:
            raise ValueError("BC epochs and batches must be positive")
        losses = [self.train_epoch(batches=batches) for _ in range(epochs)]
        return {"phase": "bc", "dataset_size": float(len(self.dataset)),
                "action_mse": sum(losses) / len(losses)}
