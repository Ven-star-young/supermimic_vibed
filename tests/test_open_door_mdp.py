"""Shared state-only reward, failure boundaries and differentiable policy paths."""

from dataclasses import replace
import inspect

import numpy as np
import pytest
import torch

from motion_data import save_motion
from supermimic.assets.g1.g1_constants import default_joint_pos_list
from supermimic.mdp import MDP
from supermimic.tasks.open_door.assets import door_context
from supermimic.tasks.open_door.env_cfgs import make_env_cfg
from supermimic.tasks.open_door.mdp import DoorRewardCfg
from supermimic.tasks.open_door.models import WorldModel
from supermimic.tasks.open_door.motion import DoorMotionLib
from supermimic.tasks.open_door.task import build_task
from tests.test_open_door_motion import DOOR_CLIP, linear_door_clip


@pytest.fixture
def task(tmp_path, cpu_kin):
    clip = linear_door_clip(fps=50., frames=10)
    clip['root_pos'][:] = [2., 0., 1.8]
    clip['joint_pos'][:] = default_joint_pos_list()
    path = tmp_path / 'door.npz'
    save_motion(path, clip)
    motion = DoorMotionLib([path], cpu_kin, 'cpu')
    cfg = make_env_cfg(play=True)
    return build_task(device='cpu', kin=cpu_kin, motion=motion, cfg=cfg)


def inputs(task):
    frames = torch.tensor([1, 4])
    state = task.motion.full_ref_state(frames)
    context = torch.zeros(2, 46)
    context[:, 0] = 1.
    context[:, 33:] = torch.as_tensor(door_context(task.motion.door_metadata, task.env_cfg.commands['motion']))
    return state, frames, torch.zeros(2, 29), context


def test_reward_preserves_robot_terms_and_scales_new_terms_once(task):
    state, frames, action, _ = inputs(task)
    robot = MDP(task.mdp.kin, task.motion, task.mdp.default_q)
    expected = robot.reward(state[:6], frames, action, action)
    reward, terms = task.mdp.reward(state, frames, action, action, terms=True)
    torch.testing.assert_close(task.mdp.reward(state, frames, action, action), reward)
    additions = sum(terms[k] for k in ('door_angle','door_velocity','back_state','illegal_state'))
    torch.testing.assert_close(reward, expected + additions)
    torch.testing.assert_close(terms['door_angle'], torch.full((2,), .02))
    torch.testing.assert_close(terms['door_velocity'], torch.full((2,), .005))
    assert not terms['illegal_state'].any()
    task.mdp.control_dt *= 2
    torch.testing.assert_close(task.mdp.reward(state, frames, action, action), 2 * reward)


def test_angle_and_velocity_tracking_have_independent_restoring_gradients(task):
    state, frames, action, _ = inputs(task)
    theta = (state[6] + .05).requires_grad_()
    omega = (state[7] - .2).requires_grad_()
    state = (*state[:6], theta, omega)
    _, terms = task.mdp.reward(state, frames, action, action, terms=True)
    gradients = torch.autograd.grad((terms['door_angle'] + terms['door_velocity']).sum(), (theta, omega))
    assert (gradients[0] < 0).all() and (gradients[1] > 0).all()


def test_observation_critic_and_checkpoint_contract(task):
    state, frames, action, context = inputs(task)
    observation = task.mdp.obs(state, action, context, frames)
    assert observation.shape == (2, 102)
    changed_context = context.clone(); changed_context[:, 36:] += 5
    torch.testing.assert_close(observation, task.mdp.obs(state, action, changed_context, frames + 1000))
    features = task.critic_input(state, frames, task.dynamics.normalize_context(context),
                                task.motion.goal(frames, 10), 10)
    assert features.shape == (2, 955)
    assert task.dimensions.goal_per_frame == 75
    task.validate_checkpoint({'task': task.checkpoint_metadata()})
    metadata = task.checkpoint_metadata(); metadata['door_physics']['door_damping'] += 1
    with pytest.raises(ValueError, match='mismatch'):
        task.validate_checkpoint({'task': metadata})


def test_state_heads_do_not_read_simulator_and_keep_frozen_wm_action_gradient(task, monkeypatch):
    import mujoco
    state, frames, action, context = inputs(task)
    def forbidden(*args, **kwargs):
        raise AssertionError('Reward queried the simulator')
    monkeypatch.setattr(mujoco, 'mj_forward', forbidden)
    model = WorldModel(dynamics=task.dynamics, hidden=(16,))
    torch.nn.init.normal_(model.net[-1].weight, std=.01)
    model.requires_grad_(False)
    action = action.requires_grad_()
    following = model.step(state, action, task.dynamics.normalize_context(context))
    reward = task.mdp.reward(following, frames + 1, action, action * 0)
    gradient, = torch.autograd.grad(reward.sum(), action)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 1e-8
    assert 'contacts' not in inspect.signature(task.mdp.reward).parameters
    assert all(p.grad is None for p in model.parameters())


def test_hard_and_soft_hinge_boundaries_use_the_same_threshold(task):
    state, frames, _, _ = inputs(task)
    state = (*state[:6], torch.tensor([[-.06], [-.05]]), state[7])
    assert task.mdp.termination(state, frames).all()
    probability = task.mdp.soft_termination(state, frames)
    assert probability[0] > probability[1] >= .5
    # Penetration termination is opt-in for raw video references.
    assert task.mdp.door_reward_cfg.penetration_limit is None


@pytest.mark.skipif(not DOOR_CLIP.is_file(), reason='Transferred GMR data is not installed')
def test_overlapping_video_references_keep_a_penetration_escape_gradient(cpu_kin):
    task = build_task(motion_dir=DOOR_CLIP, device='cpu', kin=cpu_kin)
    reference_frames = torch.arange(task.motion.num_frames)
    with torch.no_grad():
        clearance = task.mdp.geometry(task.motion.full_ref_state(reference_frames))['illegal_clearance'].amin(-1)
    overlapping = reference_frames[clearance < -.01]
    assert overlapping.numel() > 0
    frames = overlapping[[0, -1]]
    state = task.motion.full_ref_state(frames)
    position = state[0].detach().requires_grad_()
    state = (position, *state[1:])
    action = torch.zeros(2, 29)
    assert (task.mdp.geometry(state)['illegal_clearance'].amin(-1) < -.01).all()
    _, terms = task.mdp.reward(state, frames, action, action, terms=True)
    gradient, = torch.autograd.grad(terms['illegal_state'].sum(), position)
    assert torch.isfinite(gradient).all() and (gradient.norm(dim=-1) > 1e-4).all()

    clear_frames = reference_frames[clearance >= 0]
    assert clear_frames.numel() > 0
    action = torch.zeros(len(clear_frames), 29)
    _, terms = task.mdp.reward(task.motion.full_ref_state(clear_frames), clear_frames,
                               action, action, terms=True)
    torch.testing.assert_close(terms['illegal_state'], torch.zeros(len(clear_frames)))


@pytest.mark.parametrize('kwargs', [{'angle_weight': -1}, {'angle_std': 0},
                                   {'back_weight': float('nan')}, {'penetration_limit': -.1}])
def test_invalid_reward_parameters(kwargs):
    with pytest.raises(ValueError):
        DoorRewardCfg(**kwargs)
