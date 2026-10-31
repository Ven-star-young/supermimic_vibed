"""Policy videos must use measured door state and stop at the episode boundary."""

import mujoco
import numpy as np
import pytest

from motion_data import save_motion
from tests.test_open_door_motion import DOOR_XML, linear_door_clip
from tools import render


@pytest.fixture
def recording():
    n = 6
    return dict(names=np.array(["door"]), root_p=np.tile([1., .2, .8], (n, 1, 1)),
                root_quat=np.tile([1., 0., 0., 0.], (n, 1, 1)), q=np.zeros((n, 1, 29)),
                ref_root_p=np.tile([0., 0., .8], (n, 1)),
                ref_root_R=np.tile(np.eye(3), (n, 1, 1)), ref_q=np.zeros((n, 29)),
                f=np.arange(n)[:, None], alive=np.array([1, 1, 1, 1, 0, 0], bool)[:, None],
                theta=np.array([0., .2, .4, .6, .8, 1.])[:, None],
                ref_theta=np.zeros(n), clip_start=np.array([0]), clip_len=np.array([n]))


@pytest.fixture
def door_files(tmp_path):
    motion = tmp_path / "door.npz"
    save_motion(motion, linear_door_clip())
    door = tmp_path / "door.xml"
    door.write_text(DOOR_XML)
    return door, motion


def test_policy_angle_and_global_reference_are_rendered(recording, door_files, asset_path, monkeypatch):
    model = render.make_model(recording, *door_files)
    data = mujoco.MjData(model)
    monkeypatch.setattr(mujoco, "mj_step", lambda *_: pytest.fail("Physics stepped"))
    monkeypatch.setattr(mujoco, "mj_forward", lambda *_: pytest.fail("Dynamics evaluated"))
    render.set_frame(model, data, recording, 0, 3)
    assert model.nq == 73
    assert data.joint("policy_door/door_hinge").qpos[0] == pytest.approx(.6)
    np.testing.assert_allclose(data.body("pelvis").xpos, [1., .2, .8])
    np.testing.assert_allclose(data.body("ref_pelvis").xpos, [0., 0., .8])
    rotation = data.geom("policy_door/door_panel").xmat.reshape(3, 3)
    np.testing.assert_allclose(rotation[:, 0], [np.cos(.6), -np.sin(.6), 0.], atol=1e-7)
    assert data.time == pytest.approx(.06)


def test_door_geometry_required_and_checked(recording, door_files, asset_path):
    with pytest.raises(ValueError, match="--door-xml and --motion-file"):
        render.make_model(recording)
    door, motion = door_files
    door.write_text(DOOR_XML.replace('size=".02 .275 .9"', 'size=".03 .275 .9"'))
    with pytest.raises(ValueError, match="panel_half_size"):
        render.make_model(recording, door, motion)


def test_terminal_frame_survives_downsampling_without_later_world_states(recording):
    frames, last = render.frame_indices(recording, 0, 0., 10., 25)
    np.testing.assert_array_equal(frames, [0, 2, 3])
    assert last == 3
    with pytest.raises(ValueError, match="No recorded episode frames"):
        render.frame_indices(recording, 0, .1, 10., 25)


def test_single_finished_clip_stops_at_its_last_recorded_state(recording):
    recording["alive"][:] = True
    frames, last = render.frame_indices(recording, 0, 0., 10., 25)
    np.testing.assert_array_equal(frames, [0, 2, 4, 5])
    assert last == 5


def test_robot_only_recording_keeps_anchored_ghost(recording, asset_path):
    recording.pop("theta")
    model = render.make_model(recording)
    data = mujoco.MjData(model)
    render.set_frame(model, data, recording, 0, 0)
    assert model.nq == 72
    np.testing.assert_allclose(data.body("ref_pelvis").xpos, [1., .2, .8])
