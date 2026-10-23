"""Task boundaries that must remain valid for real and imagined rollouts."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


def test_importing_entrypoints_does_not_start_a_run(tmp_path):
    """Library imports must not parse the caller's argv or allocate a simulator."""
    root = Path(__file__).resolve().parents[1]
    script = """
import sys
import torch
from pathlib import Path

sys.argv = ['embedding-application', '--unrelated-argument']
torch.manual_seed(1234)
rng = torch.random.get_rng_state().clone()
def forbidden_cuda_init(*args, **kwargs):
    raise AssertionError('import initialized CUDA')
torch.cuda._lazy_init = forbidden_cuda_init
import supermimic.train
import supermimic.evaluate
assert torch.equal(rng, torch.random.get_rng_state()), 'import changed caller RNG'
assert not list(Path.cwd().iterdir()), 'import wrote files'
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root), env.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_task_configs_are_independent_and_preserve_control_frequency(asset_path):
    from supermimic.tasks.g1_tracking.env_cfgs import make_env_cfg

    first, second = make_env_cfg(), make_env_cfg()
    assert first is not second
    assert first.scene is not second.scene
    assert first.scene.entities["robot"] is not second.scene.entities["robot"]
    first.scene.entities.pop("robot")
    first.sim.mujoco.timestep *= 2
    first.rewards.w_body_pos = 3.0
    first.events["physical_context"].params["cfg"].friction = (0.9, 0.9)
    assert "robot" in second.scene.entities
    assert abs(second.sim.mujoco.timestep * second.decimation - 0.02) < 1e-9
    assert second.rewards.w_body_pos == 1.0
    assert second.events["physical_context"].params["cfg"].friction == (0.3, 1.2)


def test_reward_remains_differentiable_through_world_model(cpu_kin, cpu_motion):
    """An analytic reward refactor must preserve SuperMimic's policy gradient path."""
    from supermimic.mdp import MDP
    from supermimic.networks import WorldModel
    from supermimic.robot import default_joint_pos_list

    torch.manual_seed(41)
    mdp = MDP(cpu_kin, cpu_motion, torch.tensor(default_joint_pos_list()))
    wm = WorldModel(hidden=(32, 32))
    # Production models start with zero deltas. Represent a fitted model here:
    # a zero initialized output layer would intentionally block action gradients.
    torch.nn.init.normal_(wm.net[-1].weight, std=0.01)
    wm.requires_grad_(False)
    frame = torch.tensor([37, 62])
    state = cpu_motion.ref_state(frame)
    action = torch.full((2, 29), 0.1, requires_grad=True)
    predicted = wm.step(state, action, torch.zeros(2, 33))
    # Keep the action-rate penalty constant to isolate gradients through dynamics.
    zero_action = torch.zeros_like(action)
    reward = mdp.reward(predicted, frame + 1, zero_action, zero_action)
    gradient, = torch.autograd.grad(reward.sum(), action)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 1e-7


def test_observation_and_termination_overrides_reach_shared_heads(cpu_kin, cpu_motion):
    from supermimic.mdp import MDP
    from supermimic.mdp.observations import ObservationCfg
    from supermimic.mdp.terminations import TerminationCfg
    from supermimic.robot import default_joint_pos_list

    default_q = torch.tensor(default_joint_pos_list())
    default = MDP(cpu_kin, cpu_motion, default_q)
    custom = MDP(
        cpu_kin, cpu_motion, default_q,
        observation_cfg=ObservationCfg(include_reference_orientation=False),
        termination_cfg=TerminationCfg(height=0.1),
    )
    frame = torch.tensor([37])
    state = list(cpu_motion.ref_state(frame))
    state[0] = state[0] + torch.tensor([[0.0, 0.0, 0.15]])
    state = tuple(state)
    action, context = torch.zeros(1, 29), torch.zeros(1, 33)
    assert default.obs(state, action, context, frame).shape[-1] == 99
    assert custom.obs(state, action, context, frame).shape[-1] == 93
    assert custom.observation_dim == 93
    assert not default.termination(state, frame).item()
    assert custom.termination(state, frame).item()
    assert custom.soft_termination(state, frame).item() > 0.9
    assert default.soft_termination(state, frame).item() < 0.1


def test_replay_uses_task_dimensions_and_masks_reset_after_wrap():
    from supermimic.rl.storage import ReplayBuffer

    replay = ReplayBuffer(3, 2, "cpu", state_dim=5, action_dim=2, context_dim=1)
    for step in range(5):
        transition = {
            key: torch.full_like(value[0], step)
            for key, value in replay.data.items()
        }
        transition["done"].fill_(step == 3)
        replay.add(transition)
    sampled = replay.sample_windows(4, 3)
    assert sampled["s"].shape == (4, 3, 5)
    assert sampled["a"].shape == (4, 3, 2)
    assert sampled["xi"].shape == (4, 3, 1)
    torch.testing.assert_close(sampled["s"][..., 0], torch.tensor([[2., 3., 4.]]).expand(4, -1))
    torch.testing.assert_close(sampled["mask"], torch.tensor([[1., 1., 0.]]).expand(4, -1))


def test_models_use_supplied_dimensions_and_dynamics():
    from supermimic.rl.models import Policy, WorldModel

    class SmallDynamics:
        input_dim = 7  # Four state, two action, one context.
        output_dim = 4

        @staticmethod
        def features(state):
            return state

        @staticmethod
        def integrate(state, delta):
            return state + delta

        @staticmethod
        def targets(state, next_state):
            return next_state - state

        @staticmethod
        def state_error(predicted, actual):
            return predicted - actual

    model = WorldModel(hidden=(8,), dynamics=SmallDynamics())
    state, next_state = torch.zeros(3, 4), torch.ones(3, 4)
    predicted = model.step(state, torch.zeros(3, 2), torch.zeros(3, 1))
    assert model.in_dim == 7
    assert predicted.shape == (3, 4)
    torch.testing.assert_close(model.targets(state, next_state), next_state)
    torch.testing.assert_close(model.state_error(predicted, next_state), -next_state)
    policy = Policy(obs_dim=5, goal_dim=7, hist=2, action_dim=2)
    assert policy(torch.zeros(3, 2, 5), torch.zeros(3, 7)).shape == (3, 2)


def test_legacy_models_import_preserves_shared_amp_state():
    import supermimic.networks as legacy
    from supermimic.rl import models

    assert legacy is models


def test_task_dimensions_and_checkpoint_layout_follow_configuration(cpu_kin, cpu_motion):
    from supermimic.tasks import load_task
    from supermimic.tasks.g1_tracking.env_cfgs import make_env_cfg

    cfg = make_env_cfg()
    cfg.observations.include_reference_orientation = False
    task = load_task("g1_tracking", device="cpu", motion=cpu_motion, kin=cpu_kin, cfg=cfg)
    dims = task.dimensions
    assert (dims.state, dims.action, dims.context, dims.observation) == (71, 29, 33, 93)
    assert task.goal_dim(3) == 219
    frame = torch.tensor([37])
    state = cpu_motion.ref_state(frame)
    assert task.mdp.critic_features(state, frame, torch.zeros(1, 33)).shape[-1] == dims.critic_features
    metadata = task.checkpoint_metadata()
    assert metadata["dimensions"]["observation"] == 93
    task.validate_checkpoint({"task": metadata})
    metadata["dimensions"]["observation"] = 99
    with pytest.raises(ValueError, match="mismatch"):
        task.validate_checkpoint({"task": metadata})
    with pytest.raises(ValueError, match="Legacy checkpoint"):
        task.validate_checkpoint({})
    default = load_task("g1_tracking", device="cpu", motion=cpu_motion, kin=cpu_kin)
    default.validate_checkpoint({})


def test_robot_only_task_rejects_unmodeled_object_before_simulator_creation(cpu_kin, cpu_motion):
    from mjlab.entity import EntityCfg

    from supermimic.tasks import load_task
    from supermimic.tasks.g1_tracking.env_cfgs import make_env_cfg

    cfg = make_env_cfg()
    cfg.scene.entities["box"] = EntityCfg()
    task = load_task("g1_tracking", device="cpu", motion=cpu_motion, kin=cpu_kin, cfg=cfg)
    with pytest.raises(ValueError, match="do not model scene entities.*box"):
        task.make_env(1, device="cpu")


def test_reset_curriculum_averages_simultaneous_returns_in_same_bin(cpu_motion):
    from supermimic.mdp.commands import MotionCommandCfg, ReferenceMotionCommand

    command = ReferenceMotionCommand(MotionCommandCfg(return_update=0.2), cpu_motion, 3, "cpu", 10.0)
    command.start_bin[:] = torch.tensor([0, 0, 1])
    command.bin_returns[:2] = torch.tensor([5.0, 7.0])
    command.update_returns(torch.tensor([0, 1, 2]), torch.tensor([2.0, 8.0, 17.0]))
    # The first bin sees mean(2,8)=5 once, independently of indexed-write order.
    torch.testing.assert_close(command.bin_returns[:2], torch.tensor([5.0, 9.0]))


def test_g1_task_rejects_joint_order_mismatch_before_loading_simulator(cpu_kin, cpu_motion):
    from supermimic.tasks import load_task
    from supermimic.tasks.g1_tracking.env_cfgs import make_env_cfg

    cfg = make_env_cfg()
    cfg.actions.joint_names = tuple(reversed(cfg.actions.joint_names))
    with pytest.raises(ValueError, match="joint names/order"):
        load_task("g1_tracking", device="cpu", motion=cpu_motion, kin=cpu_kin, cfg=cfg)
    task = load_task("g1_tracking", device="cpu", motion=cpu_motion, kin=cpu_kin)
    task.env_cfg.actions.joint_names = tuple(reversed(task.env_cfg.actions.joint_names))
    with pytest.raises(ValueError, match="joint names/order"):
        task.make_env(1, device="cpu")


def test_g1_task_observation_uses_configured_default_joint_positions(cpu_kin, cpu_motion):
    from supermimic.assets.g1.g1_constants import G1_JOINT_NAMES, default_joint_pos_list
    from supermimic.tasks import load_task
    from supermimic.tasks.g1_tracking.env_cfgs import make_env_cfg

    expected = torch.tensor(default_joint_pos_list()) + 0.025
    cfg = make_env_cfg()
    cfg.scene.entities["robot"].init_state.joint_pos = dict(zip(G1_JOINT_NAMES, expected.tolist()))
    task = load_task("g1_tracking", device="cpu", motion=cpu_motion, kin=cpu_kin, cfg=cfg)
    torch.testing.assert_close(task.mdp.default_q, expected)
    frame = torch.tensor([37])
    state = cpu_motion.ref_state(frame)
    observation = task.mdp.obs(state, torch.zeros(1, 29), torch.zeros(1, 33), frame)
    torch.testing.assert_close(observation[:, :29], state[4] - expected)
