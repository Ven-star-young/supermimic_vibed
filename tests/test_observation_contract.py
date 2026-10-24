"""The actor observes proprioception while reference motion enters through its goal."""

import torch


def test_actor_observation_does_not_read_reference_motion(cpu_kin, cpu_motion, monkeypatch):
    from supermimic.assets.g1.g1_constants import default_joint_pos_list
    from supermimic.mdp import MDP

    default_q = torch.tensor(default_joint_pos_list())
    mdp = MDP(cpu_kin, cpu_motion, default_q)
    frame = torch.tensor([37, 62])
    state = cpu_motion.ref_state(frame)
    previous_action = torch.linspace(-0.2, 0.2, 58).reshape(2, 29)
    context = torch.zeros(2, 33)
    context[:, 1:30] = 0.005

    class ForbiddenReference:
        def __getitem__(self, index):
            raise AssertionError("Actor proprioception read reference rotation")

    monkeypatch.setattr(cpu_motion, "root_R", ForbiddenReference())
    actual = mdp.obs(state, previous_action, context, frame)
    # The shared task API still carries the frame, but this observation head
    # must not use it, even when no corresponding reference frame exists.
    unrelated_frame = torch.full_like(frame, 10**9)
    torch.testing.assert_close(actual, mdp.obs(state, previous_action, context, unrelated_frame))

    _, rotation, _, angular_velocity, joint_position, joint_velocity = state
    expected = torch.cat((
        joint_position - default_q - context[:, 1:30],
        joint_velocity,
        (rotation.transpose(-1, -2) @ angular_velocity.unsqueeze(-1)).squeeze(-1),
        -rotation[:, 2, :],
        previous_action,
    ), dim=-1)
    assert actual.shape == (2, 93)
    assert mdp.obs_noise.shape == (93,)
    torch.testing.assert_close(actual, expected)


def test_actor_proprioception_does_not_encode_global_heading(cpu_kin, cpu_motion):
    from supermimic.assets.g1.g1_constants import default_joint_pos_list
    from supermimic.mdp import MDP

    mdp = MDP(cpu_kin, cpu_motion, torch.tensor(default_joint_pos_list()))
    frame = torch.tensor([37])
    state = cpu_motion.ref_state(frame)
    action, context = torch.zeros(1, 29), torch.zeros(1, 33)
    # Change the actual robot's global yaw by 90 degrees, keeping all body-frame
    # quantities and the reference fixed. Pure proprioception is unchanged.
    yaw = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    position, rotation, velocity, angular_velocity, joints, joint_velocities = state
    rotated = (
        (yaw @ position.unsqueeze(-1)).squeeze(-1),
        yaw @ rotation,
        (yaw @ velocity.unsqueeze(-1)).squeeze(-1),
        (yaw @ angular_velocity.unsqueeze(-1)).squeeze(-1),
        joints,
        joint_velocities,
    )
    torch.testing.assert_close(
        mdp.obs(rotated, action, context, frame),
        mdp.obs(state, action, context, frame),
    )


def test_root_orientation_reward_remains_when_actor_reference_input_is_removed(cpu_kin, cpu_motion):
    """Paper Table A.2 retains this reward despite the 93-D actor observation."""
    from supermimic.assets.g1.g1_constants import default_joint_pos_list
    from supermimic.mdp import MDP
    from supermimic.mdp.rewards import RewardCfg

    # Isolate the paper's root orientation term from other tracking rewards.
    cfg = RewardCfg(
        w_body_pos=0.0, w_body_ori=0.0, w_body_linvel=0.0,
        w_body_angvel=0.0, w_root_pos=0.0,
    )
    mdp = MDP(cpu_kin, cpu_motion, torch.tensor(default_joint_pos_list()), cfg)
    frame = torch.tensor([37])
    state = cpu_motion.ref_state(frame)
    yaw = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    position, rotation, velocity, angular_velocity, joints, joint_velocities = state
    rotated = (
        position, yaw @ rotation,
        (yaw @ velocity.unsqueeze(-1)).squeeze(-1),
        (yaw @ angular_velocity.unsqueeze(-1)).squeeze(-1),
        joints, joint_velocities,
    )
    action, context = torch.zeros(1, 29), torch.zeros(1, 33)
    torch.testing.assert_close(
        mdp.obs(rotated, action, context, frame),
        mdp.obs(state, action, context, frame),
    )
    aligned_reward = mdp.reward(state, frame, action, action)
    rotated_reward = mdp.reward(rotated, frame, action, action)
    assert aligned_reward.item() > 0.009
    assert rotated_reward.item() < 1e-8
    assert abs(RewardCfg().max_step_reward() - 0.1) < 1e-9
