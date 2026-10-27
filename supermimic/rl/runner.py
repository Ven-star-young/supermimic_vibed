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
from .models import Critic, Policy, TensorDictPolicy, WorldModel
from .storage import ReplayBuffer
from .shared_world_model import SharedDynamics, SharedWorldModel
from .entity_state import EntityDynamics
from .entity_world_model import EntityWorldModel
from .entity_policy import EntityPolicy, EntityCritic


class SuperMimicRunner:
    """The runner interface expected by stock mjlab train and play scripts."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env, self.device = env, torch.device(device)
        self.train_cfg = asdict(SuperMimicRunnerCfg()) | dict(train_cfg)
        self.cfg = SimpleNamespace(**self.train_cfg)
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise ValueError("SuperMimicRunner currently supports one GPU; select a single --gpu-ids value")
        if self.cfg.clip_actions is not None:
            raise ValueError("SuperMimic requires clip_actions=None so replay stores the executed action")
        command = env.unwrapped.command_manager.get_term("motion")
        self.task = command.task
        self.motion, self.mdp, self.dynamics = self.task.motion, self.task.mdp, self.task.dynamics
        self.dims = self.task.dimensions
        self.history_length, self.goal_frames = command.cfg.history_length, command.cfg.goal_frames
        self.goal_dim = self.task.goal_dim(self.goal_frames)
        networks.AMP_DTYPE = torch.bfloat16 if self.cfg.amp == "bf16" else None
        if isinstance(self.dynamics, EntityDynamics):
            self.pi = EntityPolicy(command.geometry_bank, max_objects=self.dynamics.max_objects,
                                   hist=self.history_length, goal_dim=self.goal_dim,
                                   action_dim=self.dims.action).to(self.device)
        else:
            self.pi = Policy(self.dims.observation, self.goal_dim, hist=self.history_length,
                             action_dim=self.dims.action).to(self.device)
        self.wm = self.critic = None
        self.buffer = self.replay_env = None
        self.shared_replay = None
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
        if not 0 <= self.cfg.world_model_replay_fraction <= 1:
            raise ValueError("world_model_replay_fraction must be in [0,1]")

    def _init_training_models(self):
        if self.wm is not None:
            return
        self.task.validate_model_coverage()
        if isinstance(self.dynamics, EntityDynamics):
            bank = self.env.unwrapped.command_manager.get_term("motion").geometry_bank
            self.wm = EntityWorldModel(dynamics=self.dynamics, geometry_bank=bank,
                                       geometry_encoder=self.pi.geometry_encoder).to(self.device)
            if self.cfg.world_model_replays:
                from .entity_replay import EntityReplay
                self.shared_replay = EntityReplay(self.cfg.world_model_replays, bank, self.dynamics)
        elif isinstance(self.dynamics, SharedDynamics):
            bank = self.env.unwrapped.command_manager.get_term("motion").geometry_bank
            self.wm = SharedWorldModel(dynamics=self.dynamics, geometry_bank=bank).to(self.device)
            if self.cfg.world_model_replays:
                from .shared_replay import SharedReplay
                self.shared_replay = SharedReplay(self.cfg.world_model_replays, bank)
        else:
            if self.cfg.world_model_file or self.cfg.world_model_replays or self.cfg.export_world_model_replay:
                raise ValueError("World-model transfer requires a shared-dynamics task")
            self.wm = WorldModel(dynamics=self.dynamics).to(self.device)
        critic_goal_dim = (self.task.critic_goal_dim(self.goal_frames)
                           if hasattr(self.task, "critic_goal_dim") else self.goal_dim)
        if isinstance(self.dynamics, EntityDynamics):
            self.critic = EntityCritic(bank, max_objects=self.dynamics.max_objects,
                                       goal_frames=self.goal_frames, goal_dim=self.goal_dim,
                                       geometry_encoder=self.pi.geometry_encoder).to(self.device)
            if self.critic.in_dim != self.dims.critic_features + critic_goal_dim:
                raise ValueError("Entity critic features and future-reference layout disagree")
        else:
            self.critic = Critic(self.dims.critic_features + critic_goal_dim).to(self.device)
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=self.cfg.lr)
        shared = ({id(p) for p in self.wm.geometry_encoder.parameters()}
                  if isinstance(self.wm, EntityWorldModel) else set())
        self.opt_pi = torch.optim.Adam((p for p in self.pi.parameters() if id(p) not in shared),
                                       lr=self.cfg.lr, betas=(0.9, 0.95))
        self.opt_v = torch.optim.Adam((p for p in self.critic.parameters() if id(p) not in shared), lr=self.cfg.lr)
        if self.cfg.world_model_file:
            self.load_world_model(self.cfg.world_model_file)
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
        if isinstance(self.wm, EntityWorldModel):
            error = self.wm.error_per_sample(predicted, self.dynamics.unpack(actual), context, self.error_floor)
            return (error * mask).sum() / actual.shape[0], error[:, -1]
        scale = torch.maximum(self.wm.out_norm.std(), self.error_floor)
        error = (self.wm.state_error(predicted, self.dynamics.unpack(actual)) / scale).square()
        if isinstance(self.wm, SharedWorldModel):
            valid = self.wm.loss_mask(context)
            error = (error * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        else:
            error = error.mean(-1)
        return (error * mask).sum() / actual.shape[0], error[:, -1]

    def _imag_step(self, state, history, frame, context, normalized_context, goal):
        action = self.pi(history, goal)
        state = self.wm.step(state, action, normalized_context)
        frame = frame + 1
        observation = self.mdp.add_noise(self.mdp.obs(state, action, context, frame))
        history = torch.cat((history[:, 1:], observation.unsqueeze(1)), 1)
        return state, action, history, frame, self.motion.goal(frame, self.goal_frames)

    def _reward_term(self, state, frame, action, previous_action):
        return self.mdp.reward(state, frame, action, previous_action), self.mdp.soft_termination(state, frame)

    def _critic_input(self, state, frame, normalized_context, goal):
        if isinstance(self.wm, EntityWorldModel):
            features = self.mdp.critic_features(state, frame, normalized_context)
            return torch.cat((features, goal, self.task.critic_goal(state, frame, self.goal_frames)), -1)
        if isinstance(self.wm, SharedWorldModel):
            geometry = self.wm.encode_geometry(normalized_context).detach()
            features = self.mdp.critic_features(state, frame, normalized_context, geometry=geometry)
            return torch.cat((features, goal, self.task.critic_goal(state, frame, self.goal_frames)), -1)
        return torch.cat((self.mdp.critic_features(state, frame, normalized_context), goal), -1)

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
        self.prepare_collection()
        env = self.replay_env
        for _ in range(num_steps):
            observations = self.env.get_observations()
            history = observations["policy_history"].reshape(env.N, self.history_length, self.dims.observation)
            goal = observations["goal"]
            if isinstance(self.pi, EntityPolicy):
                self.pi.update_normalizers(history[:, -1:], goal)
            else:
                self.pi.obs_norm.update(history[:, -1])
                self.pi.goal_norm.update(goal)
            action = self.pi(history, goal)
            action = (action + self.cfg.action_noise * torch.randn_like(action)).clamp(-20, 20)
            out = env.step(action)
            self.buffer.add(out)
            initial, final = self.dynamics.unpack(out["s"]), self.dynamics.unpack(out["s1"])
            if isinstance(self.wm, (SharedWorldModel, EntityWorldModel)):
                self.wm.update_normalizers(initial, out["a"], out["xi"], final)
            else:
                self.wm.in_norm.update(self.wm.inputs(initial, out["a"], self.dynamics.normalize_context(out["xi"])))
                self.wm.out_norm.update(self.wm.targets(initial, final))
            self.sim_steps += env.N
            stats["sim_rew"].append(out["r"].mean() / self.task.max_step_reward)
            if "ep_len" in out:
                stats["ep_len"].append(out["ep_len"].mean())
                stats["ep_ret_norm"].append(out["ep_ret_norm"].mean())
                stats["terminated"].append(out["term"].float().sum())
                stats["done"].append(out["done"].float().sum())

    def wm_update(self):
        window = self.buffer.sample_windows(self.cfg.wm_batch, self.cfg.wm_window)
        if self.shared_replay is not None:
            count = int(self.cfg.wm_batch * self.cfg.world_model_replay_fraction)
            if count:
                prior = self.shared_replay.sample_windows(count, self.cfg.wm_window, self.device)
                for key in window:
                    window[key] = torch.cat((window[key][count:], prior[key]), 0)
        if isinstance(self.wm, (SharedWorldModel, EntityWorldModel)):
            valid = window["mask"].bool()
            self.wm.update_normalizers(self.dynamics.unpack(window["s"][valid]), window["a"][valid],
                                       window["xi"][valid], self.dynamics.unpack(window["s1"][valid]))
        predicted = self.wm_rollout(window["s"], window["a"], window["xi"])
        loss, last_error = self.wm_error(predicted, window["s1"], window["mask"], window["xi"])
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt_wm.step()
        if isinstance(self.wm, EntityWorldModel):
            # Shared leaves also occur in pi/critic.parameters(). Clear WM
            # gradients before task gradient clipping/diagnostics see them.
            self.opt_wm.zero_grad(set_to_none=True)
        mask = window["mask"][:, -1]
        return loss.detach() / self.cfg.wm_window, (last_error.detach() * mask).sum() / mask.sum().clamp(min=1)

    def _sample_valid_starts(self):
        batch, starts, histories, count = self.cfg.pi_batch, [], [], 0
        for _ in range(128):
            start, history = self.buffer.sample_starts(2 * batch, self.history_length)
            frame = start["f"]
            valid = frame + self.cfg.horizon <= self.motion.clip_end[self.motion.clip_id[frame]]
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
        start, historical = self._sample_valid_starts()
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
        gradnorm = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        if torch.isfinite(gradnorm):
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
            if isinstance(self.critic, EntityCritic):
                self.critic.update_normalizers(features)
            else:
                self.critic.norm.update(features)
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
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.opt_v.step()
        return loss

    @staticmethod
    def _new_stats():
        names = ("wm_loss", "wm_err8", "pi_loss", "v_loss", "imag_rew", "pi_gradnorm", "imag_survival",
                 "sim_rew", "ep_len", "ep_ret_norm", "terminated", "done")
        return {name: [] for name in names}

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # RSI already chooses reference starting frames; randomizing episode age
        # would misalign curriculum returns and observation-history padding.
        del init_at_random_ep_len
        if num_learning_iterations < 1:
            return
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
        checkpoint = {"pi": self.pi.state_dict(), "wm": self.wm.state_dict(), "critic": self.critic.state_dict(),
                      "opt_wm": self.opt_wm.state_dict(), "opt_pi": self.opt_pi.state_dict(), "opt_v": self.opt_v.state_dict(),
                      "iteration": self.current_learning_iteration, "wall_time": self.wall_time, "sim_steps": self.sim_steps,
                      "train_cfg": self.train_cfg, "policy_config": self.policy_config,
                      "task": self.task.checkpoint_metadata(), "infos": infos,
                      "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        if isinstance(self.dynamics, EntityDynamics):
            checkpoint["entity_geometry"] = self.pi.geometry_bank.state_dict()
            checkpoint["entity_network"] = self.wm.network_schema()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
        if isinstance(self.wm, (SharedWorldModel, EntityWorldModel)):
            self.save_world_model(path.parent / "world_models" / path.name)
            if self.cfg.export_world_model_replay and self.buffer is not None:
                self.save_world_model_replay(path.parent / "replays" / path.name)
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
        if isinstance(self.dynamics, EntityDynamics):
            # Full checkpoints expose encoder tensors through all three module
            # paths. Require agreement before any load can overwrite shared leaves.
            encoders = [{k: v for k, v in checkpoint[name].items() if k.startswith("geometry_encoder.")}
                        for name in ("pi", "wm", "critic")]
            if not encoders[0] or any(
                set(encoder) != set(encoders[0]) or any(
                    not torch.equal(value, encoders[0][key]) for key, value in encoder.items())
                for encoder in encoders[1:]
            ):
                raise ValueError("Checkpoint has inconsistent shared geometry encoder weights")
            geometry = checkpoint.get("entity_geometry")
            if geometry is None:
                raise ValueError("Entity checkpoint lacks geometry assets")
            bank = self.pi.geometry_bank
            if geometry["keys"][:len(bank.keys)] != bank.keys:
                raise ValueError("Entity checkpoint geometry mapping differs from the active scene")
            # Equal keys must also mean equal raw geometry; keep the bank object
            # shared by the command, actor, critic and world model.
            bank.merge(geometry)
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
            if isinstance(self.wm, EntityWorldModel) and checkpoint.get("entity_network") != self.wm.network_schema():
                raise ValueError("Incompatible entity network configuration for full resume")
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

    def save_world_model(self, path):
        """Save task-independent weights, geometry encoder and normalization."""
        self._init_training_models()
        if not isinstance(self.wm, (SharedWorldModel, EntityWorldModel)):
            raise ValueError("This task does not use the shared world-model schema")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        payload = dict(schema=self.dynamics.schema(), wm=self.wm.state_dict())
        if isinstance(self.wm, EntityWorldModel):
            payload["network"] = self.wm.network_schema()
        torch.save(payload, temporary)
        temporary.replace(path)

    def load_world_model(self, path):
        """Transfer physics without requiring matching actor or reward schemas."""
        self._init_training_models()
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        compatible = (self.dynamics.compatible_physics_schema(checkpoint.get("schema"))
                      if isinstance(self.wm, EntityWorldModel)
                      else checkpoint.get("schema") == self.dynamics.schema())
        if not isinstance(self.wm, (SharedWorldModel, EntityWorldModel)) or not compatible:
            raise ValueError("Incompatible shared world-model checkpoint")
        if isinstance(self.wm, EntityWorldModel) and checkpoint.get("network") != self.wm.network_schema():
            raise ValueError("Incompatible entity world-model network configuration")
        self.wm.load_state_dict(checkpoint["wm"], strict=True)

    def save_world_model_replay(self, path):
        if isinstance(self.wm, EntityWorldModel):
            from .entity_replay import save_entity_replay
            if self.buffer is None:
                raise ValueError("Collect entity transitions before exporting replay")
            save_entity_replay(path, self.buffer, self.wm.geometry_bank, self.dynamics)
            return
        from .shared_replay import save_replay
        if not isinstance(self.wm, SharedWorldModel) or self.buffer is None:
            raise ValueError("Collect shared-dynamics transitions before exporting replay")
        save_replay(path, self.buffer, self.wm.geometry_bank)

    def get_inference_policy(self, device=None):
        self.pi.eval()
        if device is not None:
            self.pi.to(device)
        return TensorDictPolicy(self.pi)

    def add_git_repo_to_log(self, repo_file_path):
        self._git_repos.append(repo_file_path)

    def _open_logging(self):
        if self.log_dir is None:
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
        mean = lambda values: sum(float(value) for value in values) / max(len(values), 1)
        row = {"iteration": self.current_learning_iteration, "wall_h": self.wall_time / 3600,
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
              f"ep_len {row['ep_len']:.1f} ep_ret {row['ep_ret_norm']:.3f} | {rate:.2f} it/s", flush=True)
