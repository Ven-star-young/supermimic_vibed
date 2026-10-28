"""CPU collective regression worker. Invoked by pytest through torchrun."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

from supermimic.rl import distributed
from supermimic.rl.config import SuperMimicRunnerCfg
from supermimic.rl.geometry import GeometryBank
from supermimic.rl.models import RunningNorm
from supermimic.rl.runner import SuperMimicRunner
from supermimic.tasks.base import TaskDimensions
from supermimic.tasks.sitting.dynamics import SittingDynamics
from supermimic.tasks.sitting.models import SittingCritic, SittingPolicy, SittingWorldModel


def digest(module):
    result = hashlib.sha256()
    for key, value in module.state_dict().items():
        result.update(key.encode())
        result.update(value.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def check_collectives():
    rank = distributed.rank()
    norm = RunningNorm(2)
    first = torch.tensor([[1., 3.]]) if rank == 0 else torch.tensor([[5., 7.], [9., 11.], [13., 15.]])
    norm.update(first)
    expected = torch.tensor([[1., 3.], [5., 7.], [9., 11.], [13., 15.]])
    torch.testing.assert_close(norm.mean, expected.mean(0))
    torch.testing.assert_close(norm.var, expected.var(0, unbiased=False))
    assert norm.count == 4
    # A rank with no samples must still enter both moment reductions.
    norm.update(torch.empty(0, 2) if rank == 0 else torch.tensor([[17., 19.]]))
    expected = torch.cat((expected, torch.tensor([[17., 19.]])))
    torch.testing.assert_close(norm.mean, expected.mean(0))
    torch.testing.assert_close(norm.var, expected.var(0, unbiased=False))
    assert norm.count == 5
    values = torch.tensor([[1., float("nan")]]) if rank == 0 else torch.tensor([[5., 3.], [9., 7.]])
    mask = torch.tensor([[True, False]]) if rank == 0 else torch.ones(2, 2, dtype=torch.bool)
    count, mean, variance = distributed.global_moments(values, mask)
    torch.testing.assert_close(count, torch.tensor([3., 2.]))
    torch.testing.assert_close(mean, torch.tensor([5., 5.]))
    torch.testing.assert_close(variance, torch.tensor([32 / 3, 4.]))

    parameters = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([float(rank + 1)])) for _ in range(3)])
    distributed.broadcast_modules(parameters)
    optimizer = torch.optim.Adam(parameters, lr=.01)
    (parameters[0].sum() * (rank + 1) + (parameters[1].sum() * 4 if rank else 0)).backward()
    assert distributed.average_gradients(optimizer, bucket_elements=1)
    torch.testing.assert_close(parameters[0].grad, torch.tensor([1.5]))
    torch.testing.assert_close(parameters[1].grad, torch.tensor([2.]))
    assert parameters[2].grad is None
    optimizer.step()
    assert len(optimizer.state) == 2
    assert distributed.gather_objects(digest(parameters)) == [digest(parameters)] * 2

    # Even an existing Adam momentum must not move a globally unused parameter.
    optimizer.zero_grad(set_to_none=True)
    before = parameters[1].detach().clone()
    parameters[0].sum().backward()
    assert distributed.average_gradients(optimizer)
    assert parameters[1].grad is None
    optimizer.step()
    torch.testing.assert_close(parameters[1], before, atol=0, rtol=0)
    rejected = 0
    for invalid in (float("nan"), float("inf")):
        optimizer.zero_grad(set_to_none=True)
        parameters[0].grad = torch.tensor([invalid if rank else 1.])
        before = digest(parameters)
        assert not distributed.average_gradients(optimizer)
        assert all(parameter.grad is None for parameter in parameters)
        assert digest(parameters) == before
        rejected += 1

    try:
        with distributed.local_phase("cpu", "simulation"):
            if rank == 1:
                raise ValueError("fixture physics overflow")
    except RuntimeError as exc:
        phase_error = str(exc)
        assert "rank 1: ValueError: fixture physics overflow" in phase_error
    else:
        raise AssertionError("Rank-local failure did not reach every process")
    assert distributed.gather_objects(phase_error) == [phase_error] * 2
    with distributed.local_phase("cpu", "healthy cleanup"):
        pass
    return dict(normalizer_count=float(norm.count), rejected_nonfinite=rejected, phase_error=phase_error)


class SmallSittingTask:
    """Keep real Sitting model interfaces while avoiding large MLPs in a CPU test."""

    dynamics = SittingDynamics()
    motion = mdp = None
    dimensions = TaskDimensions(84, 29, 46, 102, 73, 281)

    @staticmethod
    def goal_dim(frames):
        return 73 * frames

    @staticmethod
    def make_policy(history_length, goal_frames, geometry_bank):
        return SittingPolicy(geometry_bank, history_length=history_length,
                             goal_frames=goal_frames, hidden=(16,))

    @staticmethod
    def make_world_model(geometry_bank, geometry_encoder):
        return SittingWorldModel(geometry_bank, geometry_encoder=geometry_encoder, hidden=(16,))

    @staticmethod
    def make_critic(goal_frames):
        return SittingCritic(goal_frames)

    @staticmethod
    def validate_model_coverage():
        pass

    @staticmethod
    def checkpoint_metadata():
        return {"name": "small_sitting_cpu_fixture", "dimensions": asdict(SmallSittingTask.dimensions)}

    def validate_checkpoint(self, checkpoint):
        assert checkpoint["task"] == self.checkpoint_metadata()


def geometry_bank(key="chair", size=1.):
    bank = GeometryBank()
    xyz = torch.linspace(-.5, .5, 1024 * 3).reshape(1024, 3) * size
    points = torch.cat((xyz, torch.tensor([0., 0., 1.]).expand(1024, -1)), -1)
    bank.add(key, points, torch.full((3,), size))
    return bank


def make_runner(bank):
    command = SimpleNamespace(task=SmallSittingTask(), geometry_bank=bank,
                              cfg=SimpleNamespace(history_length=10, goal_frames=10))
    env = SimpleNamespace(num_envs=2, command_manager=SimpleNamespace(get_term=lambda _: command))
    env.unwrapped = env
    cfg = asdict(SuperMimicRunnerCfg(no_compile=True, amp="none"))
    cfg["seed"] += distributed.rank()
    return SuperMimicRunner(env, cfg, device="cpu")


def check_shared_model_ownership(output):
    torch.manual_seed(100 + distributed.rank())
    runner = make_runner(geometry_bank())
    runner._init_training_models()
    encoder = runner.pi.geometry_encoder
    assert encoder is runner.wm.geometry_encoder
    shared = {id(parameter) for parameter in encoder.parameters()}
    optimizer_ids = lambda optimizer: {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert shared <= optimizer_ids(runner.opt_wm)
    assert not shared & optimizer_ids(runner.opt_pi)
    assert not shared & optimizer_ids(runner.opt_v)
    initial = [digest(module) for module in (runner.pi, runner.wm, runner.critic)]
    assert distributed.gather_objects(initial) == [initial] * 2

    # First update the shared representation through WM, then run a policy update:
    # encoder weights stay frozen despite the WM optimizer's existing Adam state.
    loss = sum(parameter.square().sum() for parameter in encoder.parameters())
    loss.backward()
    assert distributed.average_gradients(runner.opt_wm)
    runner.opt_wm.step()
    runner.opt_wm.zero_grad(set_to_none=True)
    after_wm = digest(encoder)
    runner.wm.requires_grad_(False)
    runner.critic.requires_grad_(False)
    history = torch.randn(2, 10, 102)
    goal = torch.randn(2, 730)
    runner.pi(history, goal).square().sum().backward()
    assert distributed.average_gradients(runner.opt_pi)
    runner.opt_pi.step()
    assert digest(encoder) == after_wm
    runner.wm.requires_grad_(True)
    runner.critic.requires_grad_(True)
    updated = [digest(module) for module in (runner.pi, runner.wm, runner.critic)]
    assert distributed.gather_objects(updated) == [updated] * 2

    path = output / "model.pt"
    runner.save(path)
    dist.barrier()
    restored = make_runner(geometry_bank())
    restored.load(path)
    assert [digest(module) for module in (restored.pi, restored.wm, restored.critic)] == updated
    assert restored.pi.geometry_encoder is restored.wm.geometry_encoder

    scene_bank = geometry_bank("heldout-chair", size=2.)
    inference = make_runner(scene_bank)
    inference.load(path, load_cfg={"actor": True})
    assert inference.geometry_bank is scene_bank and inference.pi.geometry_bank is scene_bank
    assert inference.wm is None and inference.critic is None
    assert digest(inference.pi) == digest(runner.pi)
    mismatch = make_runner(scene_bank)
    try:
        mismatch.load(path)
    except ValueError as exc:
        assert "Resume geometry bank differs" in str(exc)
    else:
        raise AssertionError("Full resume accepted different assets")
    return dict(shared_encoder_wm_only=True, checkpoint_restored=True,
                actor_only_keeps_scene_geometry=True, model_hashes=updated)


def main():
    torch.set_num_threads(1)
    distributed.initialize("cpu")
    try:
        report = check_collectives()
        output = Path(sys.argv[1])
        output.mkdir(parents=True, exist_ok=True)
        report.update(check_shared_model_ownership(output))
        (output / f"rank{distributed.rank()}.json").write_text(json.dumps(report))
    finally:
        distributed.shutdown()


if __name__ == "__main__":
    main()
