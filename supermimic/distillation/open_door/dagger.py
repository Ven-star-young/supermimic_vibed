"""DAgger collection and supervised updates for the OpenDoor student."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import OpenDoorDaggerConfig
from .dataset import DaggerBatch, OpenDoorDaggerDataset
from .models import FrozenOpenDoorExpert, OpenDoorStudentPolicy, split_open_door_history


@dataclass
class DaggerStep:
    proprio_history: torch.Tensor
    goal: torch.Tensor
    expert_action: torch.Tensor
    student_action: torch.Tensor
    executed_action: torch.Tensor
    expert_mask: torch.Tensor
    tactile_history: torch.Tensor
    tactile_valid: torch.Tensor


def mix_actions(expert_action: torch.Tensor, student_action: torch.Tensor,
                beta: float, *, generator: torch.Generator | None = None):
    """Choose teacher/student per environment; never average joint actions."""

    if expert_action.shape != student_action.shape:
        raise ValueError("expert and student actions must have the same shape")
    if not 0 <= beta <= 1:
        raise ValueError("beta must lie in [0, 1]")
    mask = torch.rand(expert_action.shape[0], 1, device=expert_action.device, generator=generator) < beta
    return torch.where(mask, expert_action, student_action), mask.squeeze(-1)


class DaggerCoordinator:
    """One explicit DAgger coordinator.

    The caller owns the simulator.  At every control step it calls
    :meth:`label_and_select`, stores/executes the returned action, then resets
    any visual/recurrent state after the simulator reports ``done``.
    """

    def __init__(self, expert: FrozenOpenDoorExpert, student: OpenDoorStudentPolicy,
                 dataset: OpenDoorDaggerDataset | None = None,
                 config: OpenDoorDaggerConfig | None = None, *, device=None):
        self.config = config or OpenDoorDaggerConfig()
        if student.use_tactile != self.config.use_tactile:
            raise ValueError("Student tactile input must match the DAgger configuration")
        self.expert = expert.eval()
        self.student = student
        self.dataset = dataset if dataset is not None else OpenDoorDaggerDataset(
            self.config.capacity,
            history_length=self.config.history_length,
            proprio_dim=self.config.proprio_dim,
            goal_dim=self.config.goal_dim,
            action_dim=self.config.action_dim,
        )
        self.device = device or next(student.parameters()).device
        self.optimizer = torch.optim.Adam(
            self.student.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self._normalizers_initialized = False

    def beta(self, round_index: int) -> float:
        if round_index < 0:
            raise ValueError("round_index must be nonnegative")
        alpha = min(round_index / self.config.beta_decay_rounds, 1.0)
        return self.config.beta_start + alpha * (self.config.beta_end - self.config.beta_start)

    @torch.no_grad()
    def label_and_select(self, policy_history: torch.Tensor, goal: torch.Tensor,
                         beta: float, *, tactile_history: torch.Tensor,
                         tactile_valid: torch.Tensor,
                         generator: torch.Generator | None = None) -> DaggerStep:
        """Label current student-observed states before executing the next action."""

        policy_history = policy_history.detach().clone()
        proprio = split_open_door_history(
            policy_history,
            history_length=self.config.history_length,
            expert_observation_dim=self.config.expert_observation_dim,
            proprio_dim=self.config.proprio_dim,
        ).clone()
        goal = goal.detach().clone()
        tactile_history = tactile_history.detach().clone()
        tactile_valid = tactile_valid.detach().clone()
        self.student.eval()
        expert_action = self.expert(policy_history, goal)
        student_action = self.student(proprio, goal, tactile_history, tactile_valid)
        executed, mask = mix_actions(expert_action, student_action, beta, generator=generator)
        return DaggerStep(proprio, goal, expert_action, student_action, executed, mask,
                          tactile_history, tactile_valid)

    def record(self, step: DaggerStep, done: torch.Tensor | None = None) -> None:
        """Store the pre-step snapshot with the executed transition's done flag."""

        self.dataset.add(step.proprio_history, step.goal, step.expert_action, done,
                         tactile_history=step.tactile_history, tactile_valid=step.tactile_valid)

    def train_epoch(self, batches: int | None = None) -> float:
        if len(self.dataset) == 0:
            raise RuntimeError("Collect DAgger samples before training the student")
        if not self._normalizers_initialized:
            with torch.no_grad():
                for proprio, goal in self.dataset.normalization_batches(self.config.batch_size):
                    self.student.update_normalizers(proprio.to(self.device), goal.to(self.device))
            self._normalizers_initialized = True
        batches = batches or max(1, len(self.dataset) // self.config.batch_size)
        self.student.train()
        losses = []
        for _ in range(batches):
            batch: DaggerBatch = self.dataset.sample(self.config.batch_size, device=self.device)
            prediction = self.student(batch.proprio_history, batch.goal,
                                      batch.tactile_history, batch.tactile_valid)
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

    def pretrain_bc(self, *, epochs: int | None = None, batches: int | None = None) -> dict[str, float]:
        """BC-initialize the student from expert-only samples.

        The caller should collect these samples with ``beta=1.0`` before the
        first DAgger round.  Keeping this as a separate method makes the
        initialization phase visible in logs and prevents an untrained student
        from entering the environment prematurely.
        """

        epochs = self.config.bc_train_epochs if epochs is None else epochs
        batches = self.config.bc_batches if batches is None else batches
        if epochs < 1 or batches < 1:
            raise ValueError("BC epochs and batches must be positive")
        losses = [self.train_epoch(batches=batches) for _ in range(epochs)]
        return {"phase": "bc", "dataset_size": float(len(self.dataset)),
                "action_mse": sum(losses) / len(losses)}
