"""Two-process Gloo regression: real networks/BPTT, analytic fixture transitions.

Launched by test_distributed_training.py, not a training entry point.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from supermimic.rl import distributed
from supermimic.rl.models import RunningNorm
from supermimic.rl.shared_world_model import MaskedRunningNorm
from supermimic.rl.entity_world_model import EntityRunningNorm, EntityGeometryBank, EntityWorldModel
from supermimic.rl.entity_policy import EntityPolicy, EntityCritic, _update_valid
from supermimic.rl.runner import SuperMimicRunner
from supermimic.rl.config import SuperMimicRunnerCfg
from supermimic.rl.chair_experiment import ChairExperimentRunner


def digest(module):
    result = hashlib.sha256()
    for key, value in module.state_dict().items():
        result.update(key.encode())
        result.update(value.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def test_collectives():
    rank = distributed.rank()
    # Unequal sample counts, including an entirely empty rank: expect moments
    # of concatenated real samples, not an unweighted average of local means.
    ordinary = RunningNorm(2)
    local = torch.empty(0, 2) if rank == 0 else torch.tensor([[1., 3.], [5., 7.]])
    _update_valid(ordinary, local, torch.ones(len(local), dtype=torch.bool))
    torch.testing.assert_close(ordinary.mean, torch.tensor([3., 5.]))
    torch.testing.assert_close(ordinary.var, torch.tensor([4., 4.]))
    assert ordinary.count == 2
    local = torch.tensor([[9., 11.]]) if rank == 0 else torch.tensor([[13., 15.], [17., 19.]])
    ordinary.update(local)
    global_values = torch.tensor([[1., 3.], [5., 7.], [9., 11.], [13., 15.], [17., 19.]])
    torch.testing.assert_close(ordinary.mean, global_values.mean(0))
    torch.testing.assert_close(ordinary.var, global_values.var(0, unbiased=False))
    assert ordinary.count == 5
    for cls in (MaskedRunningNorm, EntityRunningNorm):
        norm = cls(2)
        values = torch.tensor([[1., 100.]]) if rank == 0 else torch.tensor([[5., 3.], [9., 7.]])
        mask = torch.tensor([[True, False]]) if rank == 0 else torch.ones(2, 2, dtype=torch.bool)
        norm.update(values, mask)
        torch.testing.assert_close(norm.mean, torch.tensor([5., 5.]))
        torch.testing.assert_close(norm.count, torch.tensor([3., 2.]))
        torch.testing.assert_close(norm.var, torch.tensor([32 / 3, 4.]))
    # A branch unused on rank0 still learns its mean gradient from rank1;
    # globally unused parameters must not acquire optimizer moments.
    parameters = [torch.nn.Parameter(torch.tensor([float(rank + 1)])) for _ in range(3)]
    opt = torch.optim.Adam(parameters, lr=.01)
    distributed.broadcast_modules(torch.nn.ParameterList(parameters))
    (parameters[0].sum() * (rank + 1) + (parameters[1].sum() * 4 if rank else 0)).backward()
    assert distributed.average_gradients(opt, bucket_elements=1)
    torch.testing.assert_close(parameters[0].grad, torch.tensor([1.5]))
    torch.testing.assert_close(parameters[1].grad, torch.tensor([2.]))
    assert parameters[2].grad is None
    opt.step()
    assert len(opt.state) == 2
    opt.zero_grad(set_to_none=True)
    parameters[0].grad = torch.tensor([float("nan") if rank else 1.])
    assert not distributed.average_gradients(opt)
    assert all(p.grad is None for p in parameters)
    try:
        distributed.require_all(rank == 0, "all ranks reject", "cpu")
    except RuntimeError:
        pass
    else:
        raise AssertionError("A failed rank must reject the update everywhere")
    schedule = object.__new__(ChairExperimentRunner)
    schedule.scenes, schedule.schedule_seed = list(range(5)), 0
    visits = [schedule.scene_for_visit(i) for i in range(5)]
    all_visits = distributed.gather_objects(visits)
    assert set(all_visits[0] + all_visits[1]) == set(range(5))
    assert all(a != b for a, b in zip(*all_visits))
    # Checkpoint exposure is the union of all ranks; resume retains each rank's
    # own scene-visit history rather than substituting rank0's schedule.
    schedule.manifest = dict(fingerprint="fixture", train=[dict(shape_key=f"chair{i}") for i in range(2)])
    schedule.visited_shapes = {f"chair{rank}"}
    schedule.visit_log = [dict(visit=0, shape=f"chair{rank}")]
    captured = {}
    with patch.object(SuperMimicRunner, "save", lambda self, path, infos=None: captured.update(infos)):
        schedule.save("unused.pt")
    details = captured["chair_experiment"]
    assert details["visited_train_shapes"] == ["chair0", "chair1"]
    assert details["scene_visits_by_rank"][rank] == schedule.visit_log
    schedule.visit_log = []
    with patch.object(SuperMimicRunner, "load", return_value=captured):
        schedule.load("unused.pt")
    assert schedule.visited_shapes == {"chair0", "chair1"}
    assert schedule.visit_log == [dict(visit=0, shape=f"chair{rank}")]


class FixtureEnv:
    """Analytic transition fixture. No MuJoCo or CPU physics emulation claim."""

    def __init__(self, directory):
        from supermimic.assets.g1.kinematics import G1Kinematics
        from supermimic.tasks.entities.env_cfgs import make_env_cfg
        from supermimic.tasks.entities.assets import ObjectCfg, entity_asset
        from supermimic.tasks.entities.task import build_task
        self.num_envs, self.device = 2, "cpu"
        self.cfg = make_env_cfg(object_enabled=False, max_objects=2,
                               extra_objects=(ObjectCfg(name="box"),))
        self.cfg.commands["motion"].motion_dir = str(directory)
        task = build_task(cfg=self.cfg, device="cpu", kin=G1Kinematics("cpu"))
        # The fixture supplies a box target to exercise all critic reference paths.
        task.motion.target_mask[0] = 1
        bank = EntityGeometryBank()
        asset = entity_asset(self.cfg.commands["motion"], 0)
        bank.add(asset.key, asset.points, asset.extent)
        self.command = SimpleNamespace(task=task, cfg=self.cfg.commands["motion"], geometry_bank=bank)
        self.command_manager = SimpleNamespace(get_term=lambda _: self.command)
        self.unwrapped = self
        self.frames = torch.tensor([10, 20])
        self.state = task.motion.full_ref_state(self.frames)
        self.xi = task.dynamics.empty_context(2)
        self.xi[:, 0] = .5 + .2 * distributed.rank()
        ctx = task.dynamics.object_context(self.xi)
        ctx[:, 0, 0] = 8 + 4 * distributed.rank()
        ctx[:, 0, 4:7] = .1
        ctx[:, 0, 10:13] = 1
        task.dynamics.pair_friction(self.xi)[..., 0] = .7
        self.previous = torch.zeros(2, 29)
        self.ep_step = torch.zeros(2, dtype=torch.int32)
        self.history = None

    def get_observations(self):
        task = self.command.task
        obs = task.mdp.obs(self.state, self.previous, self.xi, self.frames)
        if self.history is None:
            self.history = obs[:, None].expand(-1, 10, -1).clone()
        return dict(policy_history=self.history.flatten(1), goal=task.motion.goal(self.frames, 10))


class FixtureReplay:
    def __init__(self, env):
        self.env = env
        self.N = self.num_envs = env.num_envs

    def step(self, action):
        e, task = self.env, self.env.command.task
        before = task.dynamics.pack(e.state)
        frames, previous, steps = e.frames.clone(), e.previous.clone(), e.ep_step.clone()
        delta = action.new_zeros(2, task.dynamics.output_dim)
        delta[:, :3] = .001 * action[:, :3]
        delta[:, 12:41] = .002 * action
        delta[:, 70:73] = .001 * action[:, :3] + .0001 * (distributed.rank() + 1)
        e.state = task.dynamics.integrate(e.state, delta)
        e.frames += 1
        e.ep_step += 1
        reward = task.mdp.reward(e.state, e.frames, action, previous)
        obs = task.mdp.obs(e.state, action, e.xi, e.frames)
        e.history = torch.cat((e.history[:, 1:], obs[:, None]), 1)
        e.previous = action.clone()
        return dict(s=before, s1=task.dynamics.pack(e.state), a=action, a_prev=previous,
                    f=frames, xi=e.xi.clone(), r=reward, ep_step=steps,
                    done=torch.zeros(2, dtype=torch.bool), term=torch.zeros(2, dtype=torch.bool))


def run_training(output, directory):
    wm, actor, critic = EntityWorldModel.__init__, EntityPolicy.__init__, EntityCritic.__init__
    cfg = asdict(SuperMimicRunnerCfg(buffer_steps=8, utd=1., horizon=2, wm_window=2,
        wm_batches=1, wm_batch=2, pi_batch=2, warmup_steps=0, wm_only_iters=0,
        critic_epochs=1, critic_minibatches=1, no_compile=True, amp="none",
        log_every=1, save_interval=1, export_world_model_replay=True))
    cfg["seed"] += distributed.rank()
    with patch.object(EntityWorldModel, "__init__", lambda self, **kw: wm(self, width=32, layers=2, **kw)), \
         patch.object(EntityPolicy, "__init__", lambda self, *a, **kw: actor(self, *a, hidden=(32, 32), **kw)), \
         patch.object(EntityCritic, "__init__", lambda self, *a, **kw: critic(self, *a, hidden=(32, 32), **kw)):
        torch.manual_seed(100 + distributed.rank())
        env = FixtureEnv(directory)
        runner = SuperMimicRunner(env, cfg, log_dir=output, device="cpu")
        runner.make_replay_env = lambda: FixtureReplay(env)
        runner.learn(2)
        assert runner.sim_steps == 16  # 4 steps x 2 environments x 2 ranks.
        assert runner.pi.robot_norm.count == 16
        assert runner.pi.geometry_encoder is runner.wm.geometry_encoder is runner.critic.geometry_encoder
        snapshot = [digest(m) for m in (runner.pi, runner.wm, runner.critic)]
        assert distributed.gather_objects(snapshot) == [snapshot, snapshot]
        shared_ids = {id(p) for p in runner.wm.geometry_encoder.parameters()}
        for opt in (runner.opt_pi, runner.opt_v):
            assert not shared_ids & {id(p) for group in opt.param_groups for p in group["params"]}
        dist.barrier()
        restored_env = FixtureEnv(directory)
        restored = SuperMimicRunner(restored_env, cfg, log_dir=output, device="cpu")
        restored.make_replay_env = lambda: FixtureReplay(restored_env)
        restored.load(output / "model_2.pt")
        assert [digest(m) for m in (restored.pi, restored.wm, restored.critic)] == snapshot
        restored.learn(1)
        final = [digest(m) for m in (restored.pi, restored.wm, restored.critic)]
        assert distributed.gather_objects(final) == [final, final]
        assert restored.sim_steps == 28
        (output / f"rank{distributed.rank()}.json").write_text(json.dumps(dict(
            rank=distributed.rank(), iterations=restored.current_learning_iteration,
            hashes=final, sim_steps=restored.sim_steps, normalizer_count=float(restored.pi.robot_norm.count))))
        # A single rank's physics exception must reach every rank before any
        # worker enters the next normalizer collective or records a transition.
        original_step = restored.replay_env.step
        def failing_step(action):
            if distributed.rank() == 1:
                raise RuntimeError("physics buffer overflow EPA_HORIZON")
            return original_step(action)
        restored.replay_env.step = failing_step
        before_size, before_ptr = restored.buffer.size, restored.buffer.ptr
        try:
            restored.collect(1, restored._new_stats())
        except RuntimeError as exc:
            assert "rank 1" in str(exc) and "EPA_HORIZON" in str(exc)
        else:
            raise AssertionError("Physics failure on rank1 was not propagated")
        assert (restored.buffer.size, restored.buffer.ptr) == (before_size, before_ptr)
        assert restored.sim_steps == 28
        # The process group remains usable for orderly, symmetric cleanup.
        assert distributed.gather_objects("caught") == ["caught", "caught"]
        def failing_scene():
            if distributed.rank() == 0:
                raise ValueError("scene asset unavailable")
        restored.prepare_collection = failing_scene
        try:
            restored.collect(1, restored._new_stats())
        except RuntimeError as exc:
            assert "scene preparation" in str(exc) and "rank 0" in str(exc)
        else:
            raise AssertionError("Scene failure on rank0 was not propagated")
        dist.barrier()


def main():
    torch.set_num_threads(1)
    distributed.initialize("cpu")
    try:
        test_collectives()
        run_training(Path(sys.argv[1]), Path(sys.argv[2]))
    finally:
        distributed.shutdown()


if __name__ == "__main__":
    main()
