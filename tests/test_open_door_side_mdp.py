"""Independent side reward shares one differentiable head with physical training."""

import pytest
import torch
from dataclasses import replace
from types import SimpleNamespace

from motion_data import save_motion
from supermimic.assets.g1.g1_constants import default_joint_pos_list
from supermimic.mdp import MDP
from supermimic.tasks.open_door.assets import door_context
from supermimic.tasks.open_door.models import WorldModel
from supermimic.tasks.open_door.motion import DoorMotionLib
from supermimic.tasks.open_door_side.env_cfgs import make_env_cfg
from supermimic.tasks.open_door_side.geometry import SideDoorGeometry
from supermimic.tasks.open_door_side.mdp import SideDoorRewardCfg
from supermimic.tasks.open_door_side.task import build_task
from tests.test_open_door_motion import linear_door_clip


@pytest.fixture
def task(tmp_path, cpu_kin):
    clip = linear_door_clip(fps=50., frames=10)
    clip["root_pos"][:] = [2., 0., 1.8]
    clip["joint_pos"][:] = default_joint_pos_list()
    path = tmp_path / "side.npz"
    save_motion(path, clip)
    motion = DoorMotionLib([path], cpu_kin, "cpu")
    return build_task(device="cpu", kin=cpu_kin, motion=motion, cfg=make_env_cfg(play=True))


def inputs(task):
    frames = torch.tensor([1, 4])
    state = task.motion.full_ref_state(frames)
    context = torch.zeros(2, 46)
    context[:, 0] = 1.
    context[:, 33:] = torch.as_tensor(door_context(task.motion.door_metadata, task.env_cfg.commands["motion"]))
    return state, frames, torch.zeros(2, 29), context


def test_robot_reward_preserved_and_side_terms_scaled_by_dt_once(task):
    state, frames, action, _ = inputs(task)
    robot = MDP.reward(task.mdp, state[:6], frames, action, action)
    reward, terms = task.mdp.reward(state, frames, action, action, terms=True)
    additions = sum(terms[key] for key in ("door_angle", "door_velocity", "contact_state",
                                         "illegal_state", "contact_penetration"))
    torch.testing.assert_close(reward, robot + additions)
    torch.testing.assert_close(task.mdp.reward(state, frames, action, action), reward)
    torch.testing.assert_close(terms["door_angle"], torch.full((2,), .02))
    torch.testing.assert_close(terms["door_velocity"], torch.full((2,), .005))
    assert not terms["illegal_state"].any() and not terms["contact_penetration"].any()
    task.mdp.control_dt *= 2
    torch.testing.assert_close(task.mdp.reward(state, frames, action, action), 2 * reward)


def test_reference_proximity_enables_contact_with_closed_door_and_stops_on_departure(task):
    motion = task.motion
    motion.door_angle.zero_()
    motion.door_omega.zero_()
    motion.root_R[:] = torch.tensor([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    motion.root_p.zero_()
    geometry = SideDoorGeometry(task.mdp.kin, motion.door_metadata, "cpu")
    state = motion.full_ref_state(torch.tensor([0]))
    positions, rotations, _, _ = geometry.kin.forward(*state[:6], all_bodies=True)
    world = geometry.surface_world_points(positions, rotations)[0, :geometry.legal_count]
    samples = world[geometry.valid[:geometry.legal_count]]
    nearest = samples[samples[:, 0].argmin()]
    target = geometry.hinge + geometry.center + torch.tensor([float(geometry.half[0]), 0., 0.])
    motion.root_p[:] = target - nearest
    motion.root_p[5:, 0] += .3
    task = build_task(device="cpu", kin=task.mdp.kin, motion=motion, cfg=task.env_cfg)
    phase = task.mdp.reference_contact_phase
    assert phase[1] > .999 and phase[7] < 1e-10
    assert motion.door_angle[1] == motion.door_angle[7] == 0
    state = motion.full_ref_state(torch.tensor([1, 1]))
    _, terms = task.mdp.reward(state, torch.tensor([1, 7]), torch.zeros(2, 29), torch.zeros(2, 29), terms=True)
    assert terms["contact_state"][0] > .0049
    assert terms["contact_state"][1] < 1e-12


def test_only_allowed_regions_receive_three_mm_penetration_tolerance(task, monkeypatch):
    state, frames, action, _ = inputs(task)
    clearances = torch.tensor([[-.003], [-.004]], requires_grad=True)
    monkeypatch.setattr(task.mdp, "geometry", lambda S: {
        "contact_distance_sq": torch.zeros(2), "alignment": torch.ones(2),
        "contact_clearance": clearances, "illegal_clearance": clearances,
    })
    _, terms = task.mdp.reward(state, frames, action, action, terms=True)
    assert terms["contact_penetration"][0] == 0 and terms["contact_penetration"][1] < 0
    assert (terms["illegal_state"] < 0).all()
    gradient, = torch.autograd.grad((terms["contact_penetration"] + terms["illegal_state"]).sum(), clearances)
    assert torch.isfinite(gradient).all() and (gradient > 0).all()


@pytest.mark.parametrize("clearance,penalized", [(.0015, True), (.01, False), (.02, False)])
def test_illegal_clearance_margin_shared_by_simulation_and_imagination(task, monkeypatch, clearance, penalized):
    from supermimic.mdp.native import tracking_reward

    state, frames, action, _ = inputs(task)
    distances = torch.full((2, 1), clearance, requires_grad=True)
    monkeypatch.setattr(task.mdp, "geometry", lambda S: {
        "contact_distance_sq": torch.zeros(2), "alignment": torch.ones(2),
        "contact_clearance": distances, "illegal_clearance": distances,
    })
    reward, terms = task.mdp.reward(state, frames, action, action, terms=True)
    assert bool((terms["illegal_state"] < 0).all()) is penalized
    assert not terms["contact_penetration"].any()
    gradient, = torch.autograd.grad(terms["illegal_state"].sum(), distances)
    assert torch.isfinite(gradient).all()
    assert bool((gradient > 0).all()) is penalized

    command = SimpleNamespace(mdp=task.mdp, state=lambda: state, transition_frame=frames)
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command),
                          action_manager=SimpleNamespace(action=action, prev_action=action))
    torch.testing.assert_close(tracking_reward(env), reward)

    task.mdp.door_reward_cfg = replace(task.mdp.door_reward_cfg, illegal_margin=0.)
    _, legacy = task.mdp.reward(state, frames, action, action, terms=True)
    assert not legacy["illegal_state"].any()
    torch.testing.assert_close(legacy["contact_penetration"], terms["contact_penetration"])


def test_state_reward_keeps_gradients_through_frozen_world_model(task, monkeypatch):
    import mujoco
    state, frames, action, context = inputs(task)
    def no_simulator(*args, **kwargs):
        raise AssertionError("Side reward queried the simulator")
    monkeypatch.setattr(mujoco, "mj_forward", no_simulator)
    model = WorldModel(dynamics=task.dynamics, hidden=(16,))
    torch.nn.init.normal_(model.net[-1].weight, std=.01)
    model.requires_grad_(False)
    action.requires_grad_()
    imagined = model.step(state, action, task.dynamics.normalize_context(context))
    reward = task.mdp.reward(imagined, frames + 1, action, action * 0)
    gradient, = torch.autograd.grad(reward.sum(), action)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 1e-8
    assert all(parameter.grad is None for parameter in model.parameters())


def test_side_task_preserves_network_dimensions_but_rejects_back_checkpoint(task):
    from supermimic.tasks.open_door.env_cfgs import make_env_cfg as make_back_env_cfg
    from supermimic.tasks.open_door.task import build_task as build_back_task
    state, frames, action, context = inputs(task)
    assert task.name == "g1_open_door_side"
    assert task.dimensions.state == 73 and task.dimensions.context == 46
    assert task.dimensions.observation == 102 and task.dimensions.goal_per_frame == 75
    assert task.mdp.obs(state, action, context, frames).shape == (2, 102)
    features = task.critic_input(state, frames, task.dynamics.normalize_context(context),
                                task.motion.goal(frames, 10), 10)
    assert features.shape == (2, 955)
    metadata = task.checkpoint_metadata()
    assert metadata["contact_phase"] == "reference-proximity-v1"
    assert metadata["contact_geometry"]["side"] == "right"
    task.validate_checkpoint({"task": metadata})
    back = build_back_task(device="cpu", kin=task.mdp.kin, motion=task.motion, cfg=make_back_env_cfg(play=True))
    with pytest.raises(ValueError, match="mismatch"):
        task.validate_checkpoint({"task": back.checkpoint_metadata()})
    with pytest.raises(ValueError, match="mismatch"):
        back.validate_checkpoint({"task": metadata})
    metadata["contact_geometry"]["allowed_bodies"].clear()
    assert task.checkpoint_metadata()["contact_geometry"]["allowed_bodies"]


@pytest.mark.parametrize("region", [None, "back"])
def test_side_motion_selection_rejects_unlabelled_or_mixed_contact_regions(tmp_path, region):
    from supermimic.tasks.open_door_side.assets import select_motion_files
    command = make_env_cfg(play=True).commands["motion"]
    command.motion_dir = str(tmp_path)
    clip = linear_door_clip()
    right = tmp_path / "right.npz"
    save_motion(right, {**clip, "door_contact_region": "right_side"})
    assert select_motion_files(command) == [right]
    other = {**clip, "door_contact_region": region} if region is not None else clip
    save_motion(tmp_path / "other.npz", other)
    with pytest.raises(ValueError, match="door_contact_region='right_side'"):
        select_motion_files(command)
    command.motion_files = (str(right),)
    assert select_motion_files(command) == [right]


@pytest.mark.parametrize("kwargs", [{"contact_weight": -1}, {"reference_contact_std": 0},
                                   {"contact_penetration_tolerance": -.001},
                                   {"illegal_margin": -.001}, {"illegal_margin": float("inf")},
                                   {"contact_penetration_weight": float("nan")}])
def test_invalid_side_reward_parameters(kwargs):
    with pytest.raises(ValueError):
        SideDoorRewardCfg(**kwargs)
