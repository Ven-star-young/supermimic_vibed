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
import supermimic.rl.runner
import supermimic.scripts.evaluate
import supermimic.tasks
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
    first.commands["motion"].reward_cfg.w_body_pos = 3.0
    first.commands["motion"].physical_cfg.friction = (0.9, 0.9)
    assert "robot" in second.scene.entities
    assert abs(second.sim.mujoco.timestep * second.decimation - 0.02) < 1e-9
    assert second.commands["motion"].reward_cfg.w_body_pos == 1.0
    assert second.commands["motion"].physical_cfg.friction == (0.3, 1.2)


def test_reward_remains_differentiable_through_world_model(cpu_kin, cpu_motion):
    """An analytic reward refactor must preserve SuperMimic's policy gradient path."""
    from supermimic.mdp import MDP
    from supermimic.rl.models import WorldModel
    from supermimic.assets.g1.g1_constants import default_joint_pos_list

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
    from supermimic.assets.g1.g1_constants import default_joint_pos_list

    default_q = torch.tensor(default_joint_pos_list())
    default = MDP(cpu_kin, cpu_motion, default_q)
    custom = MDP(
        cpu_kin, cpu_motion, default_q,
        observation_cfg=ObservationCfg(joint_position_noise=0.03, gravity_noise=0.0),
        termination_cfg=TerminationCfg(height=0.1),
    )
    frame = torch.tensor([37])
    state = list(cpu_motion.ref_state(frame))
    state[0] = state[0] + torch.tensor([[0.0, 0.0, 0.15]])
    state = tuple(state)
    action, context = torch.zeros(1, 29), torch.zeros(1, 33)
    assert default.obs(state, action, context, frame).shape[-1] == 93
    assert custom.obs(state, action, context, frame).shape[-1] == 93
    assert custom.observation_dim == 93
    torch.testing.assert_close(custom.obs_noise[:29], torch.full((29,), 0.03))
    torch.testing.assert_close(custom.obs_noise[61:64], torch.zeros(3))
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



def test_task_dimensions_and_checkpoint_layout_follow_configuration(cpu_kin, cpu_motion):
    from copy import deepcopy

    from supermimic.rl.models import Policy, TensorDictPolicy
    from supermimic.tasks.g1_tracking.task import build_task
    from supermimic.tasks.g1_tracking.env_cfgs import make_env_cfg

    cfg = make_env_cfg()
    task = build_task(device="cpu", motion=cpu_motion, kin=cpu_kin, cfg=cfg)
    dims = task.dimensions
    assert (dims.state, dims.action, dims.context, dims.observation) == (71, 29, 33, 93)
    assert dims.critic_features == 184
    assert task.goal_dim(3) == 219
    assert task.goal_dim(10) == 730
    policy = Policy(obs_dim=dims.observation, goal_dim=task.goal_dim(10), action_dim=dims.action)
    assert policy.dec[0].in_features == 962  # Ten 93-D observations and a 32-D goal latent.
    with torch.no_grad():
        action = TensorDictPolicy(policy)({
            "policy_history": torch.zeros(1, 930),
            "goal": torch.zeros(1, 730),
        })
    assert action.shape == (1, 29)
    frame = torch.tensor([37])
    state = cpu_motion.ref_state(frame)
    assert task.mdp.critic_features(state, frame, torch.zeros(1, 33)).shape[-1] == dims.critic_features
    metadata = task.checkpoint_metadata()
    assert metadata["schema_version"] == 2
    assert metadata["dimensions"]["observation"] == 93
    task.validate_checkpoint({"task": metadata})
    old_schema = deepcopy(metadata)
    old_schema["schema_version"] = 1
    with pytest.raises(ValueError, match="mismatch"):
        task.validate_checkpoint({"task": old_schema})
    metadata["dimensions"]["observation"] = 99
    with pytest.raises(ValueError, match="mismatch"):
        task.validate_checkpoint({"task": metadata})
    with pytest.raises(ValueError, match="Legacy checkpoint"):
        task.validate_checkpoint({})
    default = build_task(device="cpu", motion=cpu_motion, kin=cpu_kin)
    with pytest.raises(ValueError, match="Legacy checkpoint"):
        default.validate_checkpoint({})


def test_robot_only_task_rejects_unmodeled_object_before_simulator_creation(cpu_kin, cpu_motion):
    from mjlab.entity import EntityCfg

    from supermimic.tasks.g1_tracking.task import build_task
    from supermimic.tasks.g1_tracking.env_cfgs import make_env_cfg

    cfg = make_env_cfg()
    cfg.scene.entities["box"] = EntityCfg()
    task = build_task(device="cpu", motion=cpu_motion, kin=cpu_kin, cfg=cfg)
    with pytest.raises(ValueError, match="do not model scene entities.*box"):
        task.validate_model_coverage()


def test_reset_curriculum_averages_simultaneous_returns_in_same_bin(cpu_motion):
    from supermimic.mdp.commands import MotionCommandCfg, ReferenceMotionCommand

    command = ReferenceMotionCommand(MotionCommandCfg(return_update=0.2), cpu_motion, 3, "cpu", 10.0)
    command.start_bin[:] = torch.tensor([0, 0, 1])
    command.bin_returns[:2] = torch.tensor([5.0, 7.0])
    command.update_returns(torch.tensor([0, 1, 2]), torch.tensor([2.0, 8.0, 17.0]))
    # The first bin sees mean(2,8)=5 once, independently of indexed-write order.
    torch.testing.assert_close(command.bin_returns[:2], torch.tensor([5.0, 9.0]))


def test_g1_task_rejects_joint_order_mismatch_before_loading_simulator(cpu_kin, cpu_motion):
    from supermimic.tasks.g1_tracking.task import build_task
    from supermimic.tasks.g1_tracking.env_cfgs import make_env_cfg

    cfg = make_env_cfg()
    cfg.actions["joint_position"].joint_names = tuple(reversed(cfg.actions["joint_position"].joint_names))
    with pytest.raises(ValueError, match="joint names/order"):
        build_task(device="cpu", motion=cpu_motion, kin=cpu_kin, cfg=cfg)
    task = build_task(device="cpu", motion=cpu_motion, kin=cpu_kin)
    task.env_cfg.actions["joint_position"].joint_names = tuple(reversed(task.env_cfg.actions["joint_position"].joint_names))
    with pytest.raises(ValueError, match="joint names/order"):
        from supermimic.tasks.g1_tracking.task import validate_env_cfg
        validate_env_cfg(task.env_cfg)


def test_g1_task_observation_uses_configured_default_joint_positions(cpu_kin, cpu_motion):
    from supermimic.assets.g1.g1_constants import G1_JOINT_NAMES, default_joint_pos_list
    from supermimic.tasks.g1_tracking.task import build_task
    from supermimic.tasks.g1_tracking.env_cfgs import make_env_cfg

    expected = torch.tensor(default_joint_pos_list()) + 0.025
    cfg = make_env_cfg()
    cfg.scene.entities["robot"].init_state.joint_pos = dict(zip(G1_JOINT_NAMES, expected.tolist()))
    task = build_task(device="cpu", motion=cpu_motion, kin=cpu_kin, cfg=cfg)
    torch.testing.assert_close(task.mdp.default_q, expected)
    frame = torch.tensor([37])
    state = cpu_motion.ref_state(frame)
    observation = task.mdp.obs(state, torch.zeros(1, 29), torch.zeros(1, 33), frame)
    torch.testing.assert_close(observation[:, :29], state[4] - expected)


def test_task_is_discovered_by_stock_mjlab_entrypoints_without_motion_data(tmp_path):
    import importlib.metadata
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.tasks.registry import list_tasks, load_env_cfg, load_runner_cls
    from supermimic.rl.runner import SuperMimicRunner
    from supermimic.tasks import TASK_ID

    assert TASK_ID in list_tasks()
    assert isinstance(load_env_cfg(TASK_ID), ManagerBasedRlEnvCfg)
    assert load_runner_cls(TASK_ID) is SuperMimicRunner
    scripts = {entry.name: entry.value for entry in importlib.metadata.entry_points(group="console_scripts")}
    assert scripts["train"] == "mjlab.scripts.train:main"
    assert scripts["play"] == "mjlab.scripts.play:main"
    environment = os.environ.copy()
    environment["SUPERMIMIC_MOTION_DIR"] = str(tmp_path / "missing-motion-data")
    for module in ("mjlab.scripts.train", "mjlab.scripts.play"):
        result = subprocess.run(
            [sys.executable, "-m", module, TASK_ID, "--help"],
            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert TASK_ID in result.stdout


def test_inference_module_uses_supplied_observations_and_preserves_normalizer():
    from supermimic.rl.models import Policy, TensorDictPolicy

    torch.manual_seed(23)
    policy = Policy(obs_dim=5, goal_dim=7, hist=2, action_dim=2)
    policy.eval()
    module = TensorDictPolicy(policy)
    history, goal = torch.randn(3, 2, 5), torch.randn(3, 7)
    before = {key: value.clone() for key, value in policy.state_dict().items()}
    actual = module({"policy_history": history.flatten(1), "goal": goal})
    torch.testing.assert_close(actual, policy(history, goal))
    changed = module({"policy_history": history.flatten(1) + 2, "goal": goal + 1})
    assert not torch.allclose(actual, changed)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_training_rejects_physics_only_rewards_but_accepts_shared_configuration(cpu_kin, cpu_motion):
    from supermimic.tasks.g1_tracking.env_cfgs import make_env_cfg
    from supermimic.tasks.g1_tracking.task import build_task

    cfg = make_env_cfg()
    cfg.commands["motion"].reward_cfg.w_body_pos = 2.0
    task = build_task(device="cpu", motion=cpu_motion, kin=cpu_kin, cfg=cfg)
    task.validate_model_coverage()
    state, frame = cpu_motion.ref_state(torch.tensor([37])), torch.tensor([37])
    action = torch.zeros(1, 29)
    torch.testing.assert_close(task.mdp.reward(state, frame, action, action), torch.tensor([0.12]))
    cfg.rewards["tracking"].weight = 2.0
    with pytest.raises(ValueError, match="Simulator rewards differ"):
        task.validate_model_coverage()
    cfg.rewards["tracking"].weight = 1.0
    cfg.scale_rewards_by_dt = True
    with pytest.raises(ValueError, match="Simulator rewards differ"):
        task.validate_model_coverage()
