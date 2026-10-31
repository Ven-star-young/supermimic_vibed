"""CPU contracts for the tracking-based hinged-door reference loader."""

from pathlib import Path

import numpy as np
import pytest
import torch

from motion_data import load_motion, save_motion
from motion_data.converters.gmr import convert_pair
from motion_data.door import door_metadata, validate_door_asset
from supermimic.assets.g1.g1_constants import G1_JOINT_NAMES
from supermimic.tasks.open_door.motion import OpenDoorMotionLib


DOOR_CLIP = Path(__file__).parents[1] / "data/motions/open_door/video1_g1_door_video45_centered_asset.npz"
DOOR_XML = '''<mujoco><compiler angle="radian"/>
  <worldbody><body name="door" pos="0 -.275 .92">
    <joint name="door_hinge" type="hinge" axis="0 0 -1" pos="0 .275 0"
           limited="true" range="0 1.8"/>
    <geom name="door_panel" type="box" size=".02 .275 .9" mass="8"/>
  </body></worldbody></mujoco>'''


@pytest.mark.skipif(not DOOR_CLIP.is_file(), reason="Transferred GMR door data is not installed")
def test_centered_open_door_clip_uses_robot_clock_and_75d_goals(cpu_kin):
    motion = OpenDoorMotionLib([DOOR_CLIP], cpu_kin, "cpu")
    assert motion.clip_len.tolist() == [407]  # 245 frames at 30 Hz, sampled at 50 Hz
    assert motion.goal_feature_dim == 75
    assert motion.door_angle.shape == (407,)
    assert motion.door_omega.shape == (407,)
    assert motion.door_opening_frames.tolist() == [120]

    frame = torch.tensor([0, 406])
    goal = motion.goal(frame, 10).reshape(2, 10, 75)
    assert goal.shape == (2, 10, 75)
    torch.testing.assert_close(goal[..., -2], motion.door_angle[motion.future_frames(frame, 10)])
    torch.testing.assert_close(
        goal[..., -1], motion.door_omega[motion.future_frames(frame, 10)] * 0.1
    )
    assert torch.isfinite(motion.door_omega).all()
    assert all(value.shape == (2, 1) for value in motion.full_ref_state(frame)[6:])
    validate_door_asset(load_motion(DOOR_CLIP), DOOR_CLIP.with_suffix(".xml"))


@pytest.mark.skipif(not DOOR_CLIP.is_file(), reason="Transferred GMR door data is not installed")
def test_aligned_prior_conversion_preserves_heading_and_tracking_goals(tmp_path, cpu_kin):
    clip = load_motion(DOOR_CLIP)
    if "door_alignment_reference" not in clip:
        pytest.skip("Initial-heading-aligned GMR prior is not installed")
    source = OpenDoorMotionLib([DOOR_CLIP], cpu_kin, "cpu")
    output = tmp_path / "prepared"
    assert convert_pair(DOOR_CLIP, output)
    prepared = OpenDoorMotionLib.from_dir(output, cpu_kin, "cpu")

    assert prepared.clip_len.tolist() == [407]
    heading = torch.atan2(prepared.root_R[0, 1, 0], prepared.root_R[0, 0, 0])
    assert heading.item() == pytest.approx(float(clip["door_base_yaw"]), abs=1e-6)
    np.testing.assert_allclose(prepared.root_p[0], clip["root_pos"][0], atol=1e-7)
    frames = torch.arange(407)
    for actual, expected in zip(prepared.full_ref_state(frames), source.full_ref_state(frames)):
        torch.testing.assert_close(actual, expected)
    goal = prepared.goal(frames, 10)
    assert goal.shape == (407, 750)
    torch.testing.assert_close(goal, source.goal(frames, 10))


def linear_door_clip(fps=25.0, frames=3, angle_offset=0.0):
    t = np.arange(frames) / fps
    return dict(
        fps=fps,
        root_pos=np.column_stack((t, np.zeros((frames, 2)))),
        root_quat=np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)),
        joint_pos=np.zeros((frames, 29)),
        joint_names=np.asarray(G1_JOINT_NAMES),
        door_angle=angle_offset + 5 * t,
        door_angular_velocity=np.full(frames, 999.0),  # must be ignored by resampling
        door_hinge_pos=np.array([0.0, 0.0, 0.92]),
        door_base_yaw=0.0,
        door_panel_half_size=np.array([0.02, 0.275, 0.9]),
        door_hinge_to_panel=np.array([0.0, -0.275, 0.0]),
        door_joint_axis=np.array([0.0, 0.0, -1.0]),
        door_joint_range=np.array([0.0, 1.8]),
    )


def test_door_omega_is_recomputed_after_resampling(tmp_path, cpu_kin):
    path = tmp_path / "door.npz"
    save_motion(path, linear_door_clip())
    motion = OpenDoorMotionLib([path], cpu_kin, "cpu")
    np.testing.assert_allclose(motion.door_angle, np.arange(5) * 0.1, atol=1e-7)
    np.testing.assert_allclose(motion.door_omega, 5.0, atol=1e-6)


def test_mixed_fps_clip_boundaries_do_not_leak_into_door_goals_or_velocity(tmp_path, cpu_kin):
    save_motion(tmp_path / "a.npz", linear_door_clip())
    save_motion(tmp_path / "b.npz", linear_door_clip(30.0, 4, angle_offset=1.0))
    motion = OpenDoorMotionLib.from_dir(tmp_path, cpu_kin, "cpu")
    assert motion.clip_len.tolist() == [5, 6]
    np.testing.assert_allclose(motion.door_omega[[4, 5]], 5.0, atol=4e-6)
    final_frames = torch.tensor([4, 10])
    goal = motion.goal(final_frames, 10).reshape(2, 10, 75)
    torch.testing.assert_close(goal[..., -2], motion.door_angle[final_frames, None].expand(2, 10))


@pytest.mark.parametrize("key,value", [
    ("door_angle", np.array([0.0, 0.1])),
    ("door_angle", np.array([0.0, np.nan, 0.2])),
    ("door_angle", np.array([0.0, 0.1, 2.0])),
    ("door_joint_axis", np.array([0.0, 0.0, -2.0])),
    ("door_joint_range", np.array([1.8, 0.0])),
    ("door_panel_half_size", np.array([-0.02, 0.275, 0.9])),
])
def test_door_schema_rejects_invalid_trajectory_and_metadata(key, value):
    clip = linear_door_clip()
    clip[key] = value
    with pytest.raises(ValueError, match="door_"):
        door_metadata(clip)


def test_task_rejects_incompatible_hinge_axis(tmp_path, cpu_kin):
    clip = linear_door_clip()
    clip["door_joint_axis"] *= -1
    path = tmp_path / "wrong_axis.npz"
    save_motion(path, clip)
    with pytest.raises(ValueError, match="door_joint_axis"):
        OpenDoorMotionLib([path], cpu_kin, "cpu")


def test_gmr_conversion_preserves_source_arrays_and_paired_asset(tmp_path):
    source = tmp_path / "source" / "clip.npz"
    clip = linear_door_clip()
    clip["door_contact_mask"] = np.array([False, True, True])
    save_motion(source, clip)
    source.with_suffix(".xml").write_text(DOOR_XML)
    original_bytes = source.read_bytes()
    output = tmp_path / "prepared"
    assert convert_pair(source, output)
    assert not convert_pair(source, output)
    converted = load_motion(output / source.name)
    for key, value in clip.items():
        np.testing.assert_array_equal(converted[key], value)
    assert source.read_bytes() == original_bytes
    assert (output / "clip.xml").read_bytes() == source.with_suffix(".xml").read_bytes()
    with pytest.raises(ValueError, match="separate"):
        convert_pair(source, source.parent, overwrite=True)
    mismatched = {**clip, "door_panel_half_size": np.array([.02, .3, .9])}
    with pytest.raises(ValueError, match="panel_half_size"):
        validate_door_asset(mismatched, source.with_suffix(".xml"))


def test_asset_rejects_disabled_hinge_limits_despite_matching_range(tmp_path):
    path = tmp_path / "unlimited.xml"
    path.write_text(DOOR_XML.replace('limited="true"', 'limited="false"'))
    with pytest.raises(ValueError, match="enabled joint limits"):
        validate_door_asset(linear_door_clip(), path)


def test_door_angle_is_required(tmp_path, cpu_kin):
    path = tmp_path / "robot_only.npz"
    save_motion(path, dict(
        fps=50.0,
        root_pos=np.zeros((2, 3)),
        root_quat=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
        joint_pos=np.zeros((2, 29)),
        joint_names=np.asarray(G1_JOINT_NAMES),
    ))
    with pytest.raises(ValueError, match="door_angle"):
        OpenDoorMotionLib([path], cpu_kin, "cpu")
