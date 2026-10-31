"""SuperMimic learner plugged into mjlab's train/play lifecycle.

The CLI creates and owns the native environment. This runner only learns from that
instance: real transitions fit the world model; imagined trajectories train policy
and critic through BPTT. Inference allocates only the policy.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import torch.nn as nn

from . import models as networks
from .config import SuperMimicRunnerCfg
from .models import TensorDictPolicy
from .storage import ReplayBuffer
from . import distributed


class SuperMimicRunner:
    """The runner interface expected by stock mjlab train and play scripts."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env, self.device = env, distributed.initialize(device)
        self.train_cfg = asdict(SuperMimicRunnerCfg()) | dict(train_cfg)
        self.cfg = SimpleNamespace(**self.train_cfg)
        if self.cfg.clip_actions is not None:
            raise ValueError("SuperMimic requires clip_actions=None so replay stores the executed action")
        command = env.unwrapped.command_manager.get_term("motion")
        self.task = command.task
        self.motion, self.mdp, self.dynamics = self.task.motion, self.task.mdp, self.task.dynamics
        self.dims = self.task.dimensions
        self.history_length, self.goal_frames = command.cfg.history_length, command.cfg.goal_frames
        self.goal_dim = self.task.goal_dim(self.goal_frames)
        networks.AMP_DTYPE = torch.bfloat16 if self.cfg.amp == "bf16" else None
        self.geometry_bank = getattr(command, "geometry_bank", None)
        self.pi = self.task.make_policy(self.history_length, self.goal_frames,
                                        self.geometry_bank).to(self.device)
        self.wm = self.critic = None
        self.buffer = self.replay_env = None
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.current_learning_iteration, self.wall_time, self.sim_steps = 0, 0.0, 0
        self._clock_start = None
        self._git_repos = [__file__]
        self._writer = self._wandb_run = None
        self._validate_config()

    def _validate_config(self):
        positive = ("buffer_steps", "horizon", "wm_window", "wm_batches", "wm_batch", "pi_batch",
                    "critic_epochs", "critic_minibatches", "lr_decay_iters", "log_every", "save_interval")
        if any(getattr(self.cfg, name) < 1 for name in positive):
            raise ValueError("Training sizes and intervals must be positive")
        if self.cfg.buffer_steps < self.cfg.wm_window:
            raise ValueError("buffer_steps must be at least wm_window")
        if self.cfg.utd <= 0 or not 0 < self.cfg.gamma < 1 or not 0 <= self.cfg.lam <= 1:
            raise ValueError("Require utd > 0, 0 < gamma < 1, and 0 <= lam <= 1")
        if self.cfg.horizon * self.cfg.pi_batch < self.cfg.critic_minibatches:
            raise ValueError("Imagined transitions must cover all critic minibatches")
        if self.cfg.amp not in ("bf16", "none"):
            raise ValueError("amp must be 'bf16' or 'none'")

    def _init_training_models(self):
        if self.wm is not None:
            return
        self.task.validate_model_coverage()
        encoder = getattr(self.pi, "geometry_encoder", None)
        self.wm = self.task.make_world_model(self.geometry_bank, encoder).to(self.device)
        self.critic = self.task.make_critic(self.goal_frames).to(self.device)
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=self.cfg.lr)
        shared = set() if encoder is None else {id(p) for p in encoder.parameters()}
        self.policy_parameters = [p for p in self.pi.parameters() if id(p) not in shared]
        self.opt_pi = torch.optim.Adam(self.policy_parameters, lr=self.cfg.lr, betas=(0.9, 0.95))
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)
        if distributed.active():
            settings = {k: v for k, v in self.train_cfg.items() if k != "seed"}
            signature = dict(task=self.task.checkpoint_metadata(), settings=settings,
                             envs=self.env.num_envs, history=self.history_length,
                             goals=self.goal_frames)
            if hasattr(self.wm, "geometry_bank"):
                signature["geometry"] = distributed.bank_digest(self.wm.geometry_bank)
            signature["parameters"] = [
                [(name, tuple(p.shape)) for name, p in module.named_parameters()]
                for module in (self.pi, self.wm, self.critic)]
            distributed.assert_same(signature, "training schema, per-rank batch sizes and geometry bank")
            distributed.broadcast_modules(self.pi, self.wm, self.critic)
        self.error_floor = self.dynamics.error_floor(self.device)
        compile_fn = (lambda fn: fn) if self.cfg.no_compile else torch.compile
        self.wm_rollout = compile_fn(self._wm_rollout)
        self.wm_error = compile_fn(self._wm_error)
        self.imag_step = compile_fn(self._imag_step)
        self.reward_term = compile_fn(self._reward_term)
        self.critic_input = compile_fn(self._critic_input)

    def _wm_rollout(self, states, actions, context):
        state = self.dynamics.unpack(states[:, 0])
        predictions = []
        for step in range(self.cfg.wm_window):
            state = self.wm.step(state, actions[:, step], self.dynamics.normalize_context(context[:, step]))
            predictions.append(state)
        return tuple(torch.stack([pred[i] for pred in predictions], 1) for i in range(len(state)))

    def _wm_error(self, predicted, actual, mask, context=None):
        scale = torch.maximum(self.wm.out_norm.std(), self.error_floor)
        error = (self.wm.state_error(predicted, self.dynamics.unpack(actual)) / scale).square().mean(-1)
        return (error * mask).sum() / actual.shape[0], error[:, -1]

    def _imag_step(self, state, history, frame, context, normalized_context, goal):
        action = self.pi(history, goal, self.task.geometry_ids(context))
        state = self.wm.step(state, action, normalized_context)
        frame = frame + 1
        observation = self.mdp.add_noise(self.mdp.obs(state, action, context, frame))
        history = torch.cat((history[:, 1:], observation.unsqueeze(1)), 1)
        return state, action, history, frame, self.motion.goal(frame, self.goal_frames)

    def _reward_term(self, state, frame, action, previous_action):
        return self.mdp.reward(state, frame, action, previous_action), self.mdp.soft_termination(state, frame)

    def _critic_input(self, state, frame, normalized_context, goal):
        geometry = self.wm.encode_geometry(normalized_context)
        if geometry is not None:
            geometry = geometry.detach()
        return self.task.critic_input(state, frame, normalized_context, goal,
                                      self.goal_frames, geometry)

    def _learning_rate(self, iteration):
        phase = min(iteration / self.cfg.lr_decay_iters, 1.0)
        return self.cfg.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * phase)))

    def make_replay_env(self):
        from .vecenv import ReplayEnvAdapter
        return ReplayEnvAdapter(self.env, gamma=self.cfg.gamma)

    def prepare_collection(self):
        """Optional experiment hook; ordinary tasks keep their existing scene."""

    @torch.no_grad()
    def collect(self, num_steps, stats):
        if num_steps == 0:
            return
        with distributed.local_phase(self.device, "scene preparation"):
            self.prepare_collection()
        env = self.replay_env
        for _ in range(num_steps):
            observations = self.env.get_observations()
            history = observations["policy_history"].reshape(env.N, self.history_length, self.dims.observation)
            goal = observations["goal"]
            self.pi.update_normalizers(history[:, -1:], goal)
            action = self.pi(history, goal, self.task.geometry_ids(env.xi))
            action = (action + self.cfg.action_noise * torch.randn_like(action)).clamp(-20, 20)
            with distributed.local_phase(self.device, "physics step"):
                out = env.step(action)
            self.buffer.add(out)
            valid = out["valid"]
            initial = self.dynamics.unpack(out["s"][valid])
            final = self.dynamics.unpack(out["s1"][valid])
            # Empty ranks still participate in global normalizer reductions.
            self.wm.update_normalizers(initial, out["a"][valid], out["xi"][valid], final)
            self.sim_steps += env.N * distributed.world_size()
            stats["sim_rew"].append(out["r"].mean() / self.task.max_step_reward)
            stats["sim_invalid"].append((~valid).float().mean())
            if "ep_len" in out:
                stats["ep_len"].append(out["ep_len"].mean())
                stats["ep_ret_norm"].append(out["ep_ret_norm"].mean())
                stats["terminated"].append(out["term"].float().sum())
                stats["done"].append(out["done"].float().sum())

    def wm_update(self):
        window = self.buffer.sample_windows(self.cfg.wm_batch, self.cfg.wm_window)
        predicted = self.wm_rollout(window["s"], window["a"], window["xi"])
        loss, last_error = self.wm_error(predicted, window["s1"], window["mask"], window["xi"])
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        if not distributed.average_gradients(self.opt_wm):
            raise FloatingPointError("Nonfinite world-model gradients on at least one rank")
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt_wm.step()
        # Shared geometry leaves must have no stale WM gradients during actor clipping.
        self.opt_wm.zero_grad(set_to_none=True)
        mask = window["mask"][:, -1]
        return loss.detach() / self.cfg.wm_window, (last_error.detach() * mask).sum() / mask.sum().clamp(min=1)

    def _sample_valid_starts(self):
        batch, starts, histories, count = self.cfg.pi_batch, [], [], 0
        for _ in range(128):
            start, history = self.buffer.sample_starts(2 * batch, self.history_length)
            frame = start["f"]
            valid = start["valid"] & (frame + self.cfg.horizon <= self.motion.clip_end[self.motion.clip_id[frame]])
            indices = valid.nonzero(as_tuple=False).flatten()[:batch - count]
            starts.append({key: value[indices] for key, value in start.items()})
            histories.append({key: value[indices] for key, value in history.items()})
            count += len(indices)
            if count == batch:
                concatenate = lambda records: {key: torch.cat([record[key] for record in records]) for key in records[0]}
                return concatenate(starts), concatenate(histories)
        raise RuntimeError("Replay has too few starts with a full imagination horizon; collect more data or shorten horizon")

    def policy_update(self):
        cfg, batch, horizon = self.cfg, self.cfg.pi_batch, self.cfg.horizon
        sampling_error = None
        try:
            start, historical = self._sample_valid_starts()
        except RuntimeError as exc:
            sampling_error = str(exc)
        distributed.require_all(sampling_error is None,
            "Replay on at least one rank lacks valid imagination starts; collect more data or shorten horizon",
            self.device)
        state = self.dynamics.unpack(start["s"])
        context, frame, previous = start["xi"], start["f"], start["a_prev"]
        normalized_context = self.dynamics.normalize_context(context)
        with torch.no_grad():
            history = self.mdp.obs(self.dynamics.unpack(historical["s"]), historical["a_prev"],
                                   historical["xi"], historical["f"])
        history = self.mdp.add_noise(history)
        self.wm.requires_grad_(False)
        self.critic.requires_grad_(False)
        goal = initial_goal = self.motion.goal(frame, self.goal_frames)
        initial_state = state
        states, actions, previous_actions, frames, goals = [], [], [], [], []
        for _ in range(horizon):
            previous_actions.append(previous)
            state, previous, history, frame, goal = self.imag_step(
                state, history, frame, context, normalized_context, goal)
            states.append(state)
            actions.append(previous)
            frames.append(frame)
            goals.append(goal)
        flatten = lambda values: torch.stack(values).flatten(0, 1)
        all_states = tuple(flatten([step[i] for step in states]) for i in range(len(state)))
        rewards, terminated = self.reward_term(all_states, flatten(frames), flatten(actions), flatten(previous_actions))
        rewards, terminated = rewards.view(horizon, batch), terminated.view(horizon, batch)
        survival = torch.cumprod(1 - terminated, 0)
        alive_before = torch.cat((torch.ones(1, batch, device=self.device), survival[:-1]), 0)
        discounts = cfg.gamma ** torch.arange(horizon, device=self.device, dtype=torch.float32).unsqueeze(-1)
        returns = (discounts * alive_before * rewards).sum(0)
        if not cfg.no_critic:
            returns = returns + cfg.gamma ** horizon * survival[-1] * self.critic(
                self._critic_input(state, frame, normalized_context, goal))
        loss = -returns.mean()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        finite = distributed.average_gradients(self.opt_pi)
        gradnorm = nn.utils.clip_grad_norm_(self.policy_parameters, 1.0) if finite else loss.new_tensor(float('nan'))
        if finite:
            nn.utils.clip_grad_value_(self.policy_parameters, 0.1)
        if finite and torch.isfinite(gradnorm):
            self.opt_pi.step()
        self.wm.requires_grad_(True)
        self.critic.requires_grad_(True)
        value_loss = torch.zeros((), device=self.device)
        if not cfg.no_critic:
            value_loss = self._critic_update(start, initial_state, initial_goal, states, frames, goals,
                                             normalized_context, rewards.detach(), terminated.detach())
        imagined_reward = (alive_before * rewards).detach().sum(0).mean() / horizon / self.task.max_step_reward
        return loss.detach(), value_loss.detach(), imagined_reward, gradnorm.detach(), survival[-1].mean().detach()

    def _critic_update(self, start, initial_state, initial_goal, states, frames, goals, context, rewards, terminated):
        horizon, batch = self.cfg.horizon, self.cfg.pi_batch
        with torch.no_grad():
            states = tuple(torch.cat((initial_state[i].unsqueeze(0), torch.stack([step[i].detach() for step in states])), 0)
                           .flatten(0, 1) for i in range(len(initial_state)))
            frames = torch.cat((start["f"].unsqueeze(0), torch.stack(frames)), 0).flatten()
            goals = torch.cat((initial_goal.unsqueeze(0), torch.stack(goals)), 0).flatten(0, 1)
            context = context.unsqueeze(0).expand(horizon + 1, -1, -1).flatten(0, 1)
            features = self.critic_input(states, frames, context, goals).view(horizon + 1, batch, -1)
            self.critic.update_normalizers(features)
            values = self.critic(features)
            target, targets = values[-1], []
            for step in range(horizon - 1, -1, -1):
                target = rewards[step] + self.cfg.gamma * (1 - terminated[step]) * (
                    (1 - self.cfg.lam) * values[step + 1] + self.cfg.lam * target)
                targets.append(target)
            targets = torch.stack(targets[::-1]).reshape(-1)
        features = features[:-1].reshape(-1, features.shape[-1])
        for _ in range(self.cfg.critic_epochs):
            indices = torch.randperm(len(features), device=self.device)
            for minibatch in indices.tensor_split(self.cfg.critic_minibatches):
                loss = (self.critic(features[minibatch]) - targets[minibatch]).square().mean()
                self.opt_v.zero_grad(set_to_none=True)
                loss.backward()
                if not distributed.average_gradients(self.opt_v):
                    raise FloatingPointError("Nonfinite critic gradients on at least one rank")
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.opt_v.step()
        return loss

    @staticmethod
    def _new_stats():
        names = ("wm_loss", "wm_err8", "pi_loss", "v_loss", "imag_rew", "pi_gradnorm", "imag_survival",
                 "sim_rew", "sim_invalid", "ep_len", "ep_ret_norm", "terminated", "done")
        return {name: [] for name in names}

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # RSI already chooses reference starting frames; randomizing episode age
        # would misalign curriculum returns and observation-history padding.
        del init_at_random_ep_len
        if num_learning_iterations < 1:
            return
        distributed.assert_same((num_learning_iterations, self.current_learning_iteration),
                                "learning iteration range")
        self._init_training_models()
        self.replay_env = self.make_replay_env()
        self.buffer = ReplayBuffer(self.cfg.buffer_steps, self.replay_env.N, self.device,
                                   state_dim=self.dims.state, action_dim=self.dims.action, context_dim=self.dims.context)
        self.pi.train()
        self._open_logging()
        self._clock_start = time.monotonic() - self.wall_time
        stats = self._new_stats()
        step_accumulator = 0.0
        start_iteration = self.current_learning_iteration
        end_iteration = start_iteration + num_learning_iterations
        last_log_iteration, last_log_time = start_iteration, time.monotonic()
        try:
            # Replay is intentionally not checkpointed: every fresh or resumed
            # run must refill enough real transitions before sampling windows.
            self.collect(max(self.cfg.warmup_steps, self.cfg.wm_window), stats)
            for iteration in range(start_iteration, end_iteration):
                lr = self._learning_rate(iteration)
                for optimizer in (self.opt_wm, self.opt_pi, self.opt_v):
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                step_accumulator += 1 / self.cfg.utd
                steps = int(step_accumulator)
                step_accumulator -= steps
                self.collect(steps, stats)
                for _ in range(self.cfg.wm_batches):
                    model_loss, last_error = self.wm_update()
                stats["wm_loss"].append(model_loss)
                stats["wm_err8"].append(last_error)
                if iteration >= self.cfg.wm_only_iters:
                    results = self.policy_update()
                    for name, value in zip(("pi_loss", "v_loss", "imag_rew", "pi_gradnorm", "imag_survival"), results):
                        stats[name].append(value)
                self.current_learning_iteration = iteration + 1
                self.wall_time = time.monotonic() - self._clock_start
                if self.current_learning_iteration % self.cfg.log_every == 0 or self.current_learning_iteration == end_iteration:
                    now = time.monotonic()
                    rate = (self.current_learning_iteration - last_log_iteration) / (now - last_log_time)
                    self._log(stats, lr, rate)
                    stats = self._new_stats()
                    last_log_iteration, last_log_time = self.current_learning_iteration, now
                if self.log_dir is not None and (self.current_learning_iteration % self.cfg.save_interval == 0
                                                or self.current_learning_iteration == end_iteration):
                    self.save(self.log_dir / f"model_{self.current_learning_iteration}.pt")
        finally:
            self.wall_time = time.monotonic() - self._clock_start
            self._clock_start = None
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            if self._wandb_run is not None:
                self._wandb_run.finish()
                self._wandb_run = None

    def save(self, path, infos=None):
        self._init_training_models()
        if not distributed.primary():
            return
        checkpoint = {"pi": self.pi.state_dict(), "wm": self.wm.state_dict(), "critic": self.critic.state_dict(),
                      "opt_wm": self.opt_wm.state_dict(), "opt_pi": self.opt_pi.state_dict(), "opt_v": self.opt_v.state_dict(),
                      "iteration": self.current_learning_iteration, "wall_time": self.wall_time, "sim_steps": self.sim_steps,
                      "train_cfg": self.train_cfg, "policy_config": self.policy_config,
                      "task": self.task.checkpoint_metadata(), "infos": infos,
                      "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "distributed": {"world_size": distributed.world_size(),
                                      "batch_sizes": "per_rank", "sim_steps": "global"}}
        if self.geometry_bank is not None:
            checkpoint["geometry_bank"] = self.geometry_bank.to_dict()
        checkpoint.update(self.checkpoint_extras())
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
        if self._wandb_run is not None and self.cfg.upload_model:
            self._wandb_run.save(str(path), base_path=str(self.log_dir))

    @property
    def policy_config(self):
        return {"history_length": self.history_length, "goal_frames": self.goal_frames, "amp": self.cfg.amp}

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        actor_only = load_cfg == {"actor": True}
        if load_cfg is not None and not actor_only:
            raise ValueError("Supported load_cfg values: None (resume all) or {'actor': True} (inference)")
        # Load on CPU for inference: a full training checkpoint also contains large
        # world-model/optimizer tensors that should not occupy the play GPU.
        location = "cpu" if actor_only else (map_location or "cpu")
        checkpoint = torch.load(path, map_location=location, weights_only=False)
        self.task.validate_checkpoint(checkpoint)
        self.validate_checkpoint_extras(checkpoint, actor_only=actor_only)
        if self.geometry_bank is not None:
            encoder = lambda weights: {k: v for k, v in weights.items() if k.startswith("geometry_encoder.")}
            actor_encoder, wm_encoder = encoder(checkpoint["pi"]), encoder(checkpoint["wm"])
            if not actor_encoder or actor_encoder.keys() != wm_encoder.keys() or any(
                not torch.equal(v, wm_encoder[k]) for k, v in actor_encoder.items()
            ):
                raise ValueError("Checkpoint has inconsistent shared geometry encoder weights")
            if not actor_only:
                saved_bank = type(self.geometry_bank).from_dict(checkpoint["geometry_bank"])
                if distributed.bank_digest(saved_bank) != distributed.bank_digest(self.geometry_bank):
                    raise ValueError("Resume geometry bank differs from the active training scenes")
        legacy_args = checkpoint.get("args", {})
        policy_cfg = checkpoint.get("policy_config", {"history_length": legacy_args.get("hist", 10),
                                                       "goal_frames": legacy_args.get("goal_frames", 10),
                                                       "amp": legacy_args.get("amp", "bf16")})
        for key in ("history_length", "goal_frames"):
            if policy_cfg[key] != self.policy_config[key]:
                raise ValueError(f"Checkpoint {key}={policy_cfg[key]} does not match environment {self.policy_config[key]}")
        self.pi.load_state_dict(checkpoint["pi"], strict=strict)
        amp = policy_cfg.get("amp") if actor_only else self.cfg.amp
        networks.AMP_DTYPE = torch.bfloat16 if amp == "bf16" else None
        if not actor_only:
            self._init_training_models()
            for key, module in (("wm", self.wm), ("critic", self.critic), ("opt_wm", self.opt_wm),
                                ("opt_pi", self.opt_pi), ("opt_v", self.opt_v)):
                if key not in checkpoint:
                    raise ValueError(f"Checkpoint lacks {key!r}; use a full training checkpoint to resume")
                if isinstance(module, nn.Module):
                    module.load_state_dict(checkpoint[key], strict=strict)
                else:
                    module.load_state_dict(checkpoint[key])
            self.current_learning_iteration = checkpoint.get("iteration", 0)
            self.wall_time = checkpoint.get("wall_time", 0.0)
            self.sim_steps = checkpoint.get("sim_steps", 0)
        return checkpoint.get("infos")

    def checkpoint_extras(self):
        """Experiment metadata, separate from a task's network schema."""
        return {}

    def validate_checkpoint_extras(self, checkpoint, *, actor_only=False):
        """Experiments may enforce data identity for full training resume."""

    def get_inference_policy(self, device=None):
        self.pi.eval()
        if device is not None:
            self.pi.to(device)
        return TensorDictPolicy(self.pi)

    def add_git_repo_to_log(self, repo_file_path):
        self._git_repos.append(repo_file_path)

    def _open_logging(self):
        if self.log_dir is None or not distributed.primary():
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(str(self.log_dir))
        elif self.cfg.logger == "wandb":
            import wandb
            self._wandb_run = wandb.init(project=self.cfg.wandb_project, dir=str(self.log_dir),
                                         name=self.log_dir.name, config=self.train_cfg,
                                         tags=list(self.cfg.wandb_tags))
        else:
            raise ValueError(f"Unsupported logger: {self.cfg.logger}")
        import git
        for source in self._git_repos:
            try:
                repo = git.Repo(source, search_parent_directories=True)
            except (git.InvalidGitRepositoryError, git.NoSuchPathError):
                continue
            destination = self.log_dir / "git" / f"{Path(repo.working_dir).name}.diff"
            destination.parent.mkdir(exist_ok=True)
            destination.write_text(f"commit: {repo.head.commit.hexsha}\n{repo.git.status()}\n{repo.git.diff('HEAD')}")

    def _log(self, stats, lr, rate):
        if distributed.active():
            keys = list(stats)
            totals = torch.tensor([[sum(float(v) for v in stats[k]), len(stats[k])] for k in keys],
                                  device=self.device, dtype=torch.float64)
            torch.distributed.all_reduce(totals)
            stats = {k: ([float(totals[i, 0])] if k in ("terminated", "done") else
                         [float(totals[i, 0] / totals[i, 1].clamp_min(1))]) for i, k in enumerate(keys)}
            timing = torch.tensor([self.wall_time, 1 / max(rate, 1e-12)], device=self.device)
            torch.distributed.all_reduce(timing, op=torch.distributed.ReduceOp.MAX)
            rate = 1 / float(timing[1])
            if not distributed.primary():
                return
            wall_time = float(timing[0])
        else:
            wall_time = self.wall_time
        mean = lambda values: sum(float(value) for value in values) / max(len(values), 1)
        row = {"iteration": self.current_learning_iteration, "wall_h": wall_time / 3600,
               "sim_steps": self.sim_steps}
        row.update({name: mean(values) for name, values in stats.items() if name not in ("terminated", "done")})
        row.update(term_frac=sum(float(x) for x in stats["terminated"]) / max(sum(float(x) for x in stats["done"]), 1),
                   it_per_s=rate, lr=lr)
        if self.log_dir is not None:
            path = self.log_dir / "train_log.csv"
            write_header = not path.exists() or path.stat().st_size == 0
            with path.open("a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        if self._writer is not None:
            for key, value in row.items():
                self._writer.add_scalar(key, value, self.current_learning_iteration)
            self._writer.flush()
        if self._wandb_run is not None:
            self._wandb_run.log(row, step=self.current_learning_iteration)
        print(f"it {self.current_learning_iteration} | {row['wall_h']:.3f} h | wm {row['wm_loss']:.4f} "
              f"e8 {row['wm_err8']:.4f} | pi {row['pi_loss']:.3f} v {row['v_loss']:.4f} "
              f"imag_r {row['imag_rew']:.3f} surv {row['imag_survival']:.3f} | sim_r {row['sim_rew']:.3f} "
              f"invalid {row['sim_invalid']:.4%} "
              f"ep_len {row['ep_len']:.1f} ep_ret {row['ep_ret_norm']:.3f} | {rate:.2f} it/s", flush=True)
