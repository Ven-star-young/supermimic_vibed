"""Source-independent reference clips preserve physical time and policy inputs."""

import numpy as np
import pytest
import torch

from motion_data import load_motion, save_motion
from supermimic.assets.g1.g1_constants import G1_JOINT_NAMES, default_joint_pos_list
from supermimic.motion import MotionLib


ROOT_ORIGIN = np.array([.1, -.2, .8])
ROOT_VELOCITY = np.array([1.2, -.6, .1])
JOINT_ORIGIN = np.linspace(-.2, .2, 29)
JOINT_VELOCITY = np.linspace(-.4, .7, 29)
YAW_RATE = 2.5


def linear_clip(fps=25., frames=3):
    """A trajectory with known positions and constant world/joint velocities."""
    time = np.arange(frames) / fps
    yaw = .3 + YAW_RATE * time
    return dict(
        fps=fps,
        root_pos=ROOT_ORIGIN + time[:, None] * ROOT_VELOCITY,
        root_quat=np.column_stack((np.cos(yaw / 2), np.zeros((frames, 2)), np.sin(yaw / 2))),
        joint_names=np.array(G1_JOINT_NAMES),
        joint_pos=JOINT_ORIGIN + time[:, None] * JOINT_VELOCITY,
    )


def write_clip(directory, clip, name="clip"):
    path = directory / f"{name}.npz"
    save_motion(path, clip)
    return path


@pytest.mark.parametrize("fps,frames,expected_frames", [
    (25., 3, 5),  # 0, 20, 40, 60, 80 ms; duration must not halve.
    (30., 4, 6),  # Noninteger rate ratio with an exactly representable end time.
    (29.97, 4, 6),
    (60., 3, 2),  # The last source frame is beyond the last 50 Hz sample.
])
def test_per_clip_fps_preserves_time_and_analytic_velocities(
    tmp_path, cpu_kin, fps, frames, expected_frames,
):
    path = write_clip(tmp_path, linear_clip(fps, frames))
    motion = MotionLib([str(path)], cpu_kin, "cpu")
    assert motion.control_dt == pytest.approx(.02)
    assert motion.num_frames == expected_frames
    assert motion.clip_len.tolist() == [expected_frames]
    time = np.arange(expected_frames) * .02
    assert time[-1] <= (frames - 1) / fps + 1e-12
    assert time[-1] + .02 > (frames - 1) / fps

    np.testing.assert_allclose(motion.root_p, ROOT_ORIGIN + time[:, None] * ROOT_VELOCITY, atol=1e-7)
    np.testing.assert_allclose(motion.q, JOINT_ORIGIN + time[:, None] * JOINT_VELOCITY, atol=1e-7)
    yaw = .3 + YAW_RATE * time
    expected_rotation = np.zeros((expected_frames, 3, 3))
    expected_rotation[:, 0, 0] = expected_rotation[:, 1, 1] = np.cos(yaw)
    expected_rotation[:, 1, 0], expected_rotation[:, 0, 1] = np.sin(yaw), -np.sin(yaw)
    expected_rotation[:, 2, 2] = 1.
    np.testing.assert_allclose(motion.root_R, expected_rotation, atol=2e-7)
    np.testing.assert_allclose(motion.root_v, np.tile(ROOT_VELOCITY, (expected_frames, 1)), atol=4e-6)
    np.testing.assert_allclose(motion.qd, np.tile(JOINT_VELOCITY, (expected_frames, 1)), atol=2e-6)
    np.testing.assert_allclose(motion.root_w, np.tile([0., 0., YAW_RATE], (expected_frames, 1)), atol=2e-5)
    assert torch.isfinite(motion.body_v).all()


def test_joint_names_reorder_columns_into_robot_policy_order(tmp_path, cpu_kin):
    clip = linear_clip()
    clip["joint_names"] = clip["joint_names"][::-1]
    clip["joint_pos"] = clip["joint_pos"][:, ::-1]
    path = write_clip(tmp_path, clip)
    motion = MotionLib([str(path)], cpu_kin, "cpu")
    np.testing.assert_allclose(motion.q[0], JOINT_ORIGIN, atol=1e-7)
    np.testing.assert_allclose(motion.q[-1], JOINT_ORIGIN + .08 * JOINT_VELOCITY, atol=1e-7)
    np.testing.assert_allclose(motion.qd[2], JOINT_VELOCITY, atol=2e-6)


def test_antipodal_quaternions_do_not_introduce_spin(tmp_path, cpu_kin):
    clip = linear_clip()
    original = write_clip(tmp_path, clip, "original")
    clip["root_quat"][1::2] *= -1
    flipped = write_clip(tmp_path, clip, "flipped")
    a = MotionLib([str(original)], cpu_kin, "cpu")
    b = MotionLib([str(flipped)], cpu_kin, "cpu")
    torch.testing.assert_close(a.root_R, b.root_R)
    torch.testing.assert_close(a.root_w, b.root_w)
    torch.testing.assert_close(a.body_v, b.body_v)


def test_mixed_fps_clip_boundaries_never_leak_into_goals_or_velocities(tmp_path, cpu_kin):
    first = linear_clip(25., 3)
    second = linear_clip(30., 4)
    # A large discontinuity makes accidental inter-clip finite differences obvious.
    second["root_pos"] += np.array([10., -10., 0.])
    second["joint_pos"] += .5
    write_clip(tmp_path, first, "a")
    write_clip(tmp_path, second, "b")
    (tmp_path / "unconverted.csv").write_text("This is not a portable motion clip.\n")
    motion = MotionLib.from_dir(str(tmp_path), cpu_kin, "cpu")
    assert motion.names == ["a", "b"]
    assert motion.clip_len.tolist() == [5, 6]
    assert motion.clip_start.tolist() == [0, 5]
    assert motion.clip_end.tolist() == [4, 10]
    assert motion.clip_id.tolist() == [0] * 5 + [1] * 6
    frames = torch.tensor([3, 4, 9, 10])
    expected_future = torch.tensor([[4] * 10, [4] * 10, [10] * 10, [10] * 10])
    torch.testing.assert_close(motion.future_frames(frames, 10), expected_future)
    # Check both sides of the concatenation boundary independently.
    np.testing.assert_allclose(motion.root_v[[4, 5]], np.tile(ROOT_VELOCITY, (2, 1)), atol=1e-4)
    np.testing.assert_allclose(motion.qd[[4, 5]], np.tile(JOINT_VELOCITY, (2, 1)), atol=5e-6)
    goal = motion.goal(torch.tensor([4, 10]), 10).reshape(2, 10, 73)
    torch.testing.assert_close(goal[:, :, :3], torch.zeros(2, 10, 3))
    torch.testing.assert_close(goal[:, :, 15:44], motion.q[[4, 10], None].expand(-1, 10, -1))
    torch.testing.assert_close(goal[:, :, 44:], .1 * motion.qd[[4, 10], None].expand(-1, 10, -1))


@pytest.mark.parametrize("kind", ["missing", "extra", "unknown"])
def test_robot_joint_contract_rejects_incompatible_clips(tmp_path, cpu_kin, kind):
    clip = linear_clip()
    names = clip["joint_names"].tolist()
    if kind == "missing":
        clip["joint_names"], clip["joint_pos"] = names[:-1], clip["joint_pos"][:, :-1]
    elif kind == "extra":
        clip["joint_names"] = names + ["extra_joint"]
        clip["joint_pos"] = np.column_stack((clip["joint_pos"], np.zeros(3)))
    else:
        clip["joint_names"] = ["unknown_joint"] + names[1:]
    path = write_clip(tmp_path, clip)
    with pytest.raises(ValueError, match="[Jj]oint"):
        MotionLib([str(path)], cpu_kin, "cpu")


@pytest.mark.parametrize("fps,frames", [(25., 1), (100., 2)])
def test_training_rejects_clips_without_two_control_frames(tmp_path, cpu_kin, fps, frames):
    path = write_clip(tmp_path, linear_clip(fps, frames))
    assert len(load_motion(path)["root_pos"]) == frames
    with pytest.raises(ValueError, match="(?:two|2|short|frames)"):
        MotionLib([str(path)], cpu_kin, "cpu")


def test_single_frame_is_valid_for_kinematic_replay(tmp_path, asset_path):
    from tools import replay_motion

    path = write_clip(tmp_path, linear_clip(frames=1))
    clip = load_motion(path)
    model, data, root, addresses = replay_motion.make_scene(asset_path, clip)
    replay_motion.set_frame(model, data, clip, 0, root, addresses)
    np.testing.assert_allclose(data.body("pelvis").xpos, ROOT_ORIGIN)


def test_object_geometry_preserves_existing_robot_input_contract(tmp_path, cpu_kin):
    from supermimic.mdp import MDP

    clip = linear_clip()
    robot_path = write_clip(tmp_path, clip, "robot")
    clip.update(
        object_pos=np.tile([.4, 0., .5], (3, 1)),
        object_quat=np.tile([1., 0., 0., 0.], (3, 1)),
        object_scale=np.array([.8, .6, .4]),
        object_vertices=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
        object_faces=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
    )
    object_path = write_clip(tmp_path, clip, "with_chair")
    loaded = load_motion(object_path)
    np.testing.assert_array_equal(loaded["object_faces"], clip["object_faces"])
    robot = MotionLib([str(robot_path)], cpu_kin, "cpu")
    chair = MotionLib([str(object_path)], cpu_kin, "cpu")
    frame = torch.tensor([1])
    for expected, actual in zip(robot.ref_state(frame), chair.ref_state(frame)):
        torch.testing.assert_close(actual, expected)
    assert chair.goal_feature_dim == 73
    assert chair.goal(frame, 10).shape == (1, 730)
    torch.testing.assert_close(chair.goal(frame, 10), robot.goal(frame, 10))
    observations = []
    for motion in (robot, chair):
        mdp = MDP(cpu_kin, motion, torch.tensor(default_joint_pos_list()))
        observations.append(mdp.obs(motion.ref_state(frame), torch.zeros(1, 29), torch.zeros(1, 33), frame))
    assert observations[1].shape == (1, 93)
    torch.testing.assert_close(*observations)


def test_training_directory_requires_converted_npz(tmp_path, cpu_kin):
    (tmp_path / "raw.csv").write_text("0,0,0\n")
    with pytest.raises(ValueError, match="(?:NPZ|npz)"):
        MotionLib.from_dir(str(tmp_path), cpu_kin, "cpu")
