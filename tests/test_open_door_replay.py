"""Replay the supplied robot/door reference without physics or training imports."""

import subprocess
import sys

import mujoco
import numpy as np
import pytest

from motion_data import load_motion
from tools import replay_motion


@pytest.fixture
def door_clip():
    files = sorted((replay_motion.ROOT / "data/motions/open_door_prepared").glob("*.npz"))
    if not files or not files[0].with_suffix(".xml").is_file():
        pytest.skip("Convert the supplied OpenDoor NPZ/XML pair first")
    return files[0], load_motion(files[0])


def test_all_frames_replay_named_robot_and_negative_axis_hinge(door_clip, asset_path, monkeypatch):
    path, clip = door_clip
    monkeypatch.setattr(mujoco, "mj_step", lambda *_: pytest.fail("Physics stepped"))
    model, data, root, addresses = replay_motion.make_scene(asset_path, clip, path.with_suffix(".xml"))
    monkeypatch.setattr(mujoco, "mj_forward", lambda *_: pytest.fail("Replay evaluated dynamics"))
    joint, panel = model.joint("replay_door/door_hinge"), model.geom("replay_door/door_panel")
    np.testing.assert_array_equal(model.jnt_axis[joint.id], [0., 0., -1.])
    np.testing.assert_allclose(model.geom_size[panel.id], clip["door_panel_half_size"])
    assert model.nu == 0 and not model.geom_contype.any() and not model.geom_conaffinity.any()
    assert model.body("pelvis").id >= 0  # Attaching the door must not rename G1.
    assert np.ptp(clip["door_angle"]) > .5
    for frame, theta in enumerate(clip["door_angle"]):
        replay_motion.set_frame(model, data, clip, frame, root, addresses)
        np.testing.assert_allclose(data.qpos[root:root + 7],
                                   np.r_[clip["root_pos"][frame], clip["root_quat"][frame]])
        np.testing.assert_allclose(data.qpos[addresses], clip["joint_pos"][frame])
        assert data.joint("replay_door/door_hinge").qpos[0] == theta
        yaw = float(clip["door_base_yaw"]) - float(theta)
        c, s = np.cos(yaw), np.sin(yaw)
        expected_rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        np.testing.assert_allclose(data.xanchor[joint.id], clip["door_hinge_pos"], atol=1e-6, rtol=0)
        np.testing.assert_allclose(data.geom_xmat[panel.id].reshape(3, 3), expected_rotation,
                                   atol=1e-6, rtol=0)
        np.testing.assert_allclose(data.geom_xpos[panel.id],
                                   clip["door_hinge_pos"] + expected_rotation @ clip["door_hinge_to_panel"],
                                   atol=1e-6, rtol=0)
        assert data.time == pytest.approx(frame / float(clip["fps"]))


def test_door_trajectory_requires_matching_xml(door_clip, asset_path, tmp_path):
    path, clip = door_clip
    with pytest.raises(ValueError, match="door_angle.*door_xml"):
        replay_motion.make_scene(asset_path, clip)
    with pytest.raises(FileNotFoundError, match="paired door XML"):
        replay_motion.make_scene(asset_path, clip, tmp_path / "missing.xml")
    changed = {**clip, "door_hinge_pos": clip["door_hinge_pos"] + [.01, 0., 0.]}
    with pytest.raises(ValueError, match="hinge_pos.*does not match"):
        replay_motion.make_scene(asset_path, changed, path.with_suffix(".xml"))


def test_headless_cli_uses_paired_door_without_training_dependencies(door_clip):
    path, clip = door_clip
    code = """
import importlib.abc, runpy, sys
class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'mjlab', 'mujoco_warp'}:
            raise AssertionError('Unexpected training dependency: ' + fullname)
sys.meta_path.insert(0, BlockTrainingImports())
import mujoco
def no_physics(*args, **kwargs):
    raise AssertionError('Physics stepped during kinematic replay')
mujoco.mj_step = no_physics
sys.argv = ['replay_motion', sys.argv[1], '--headless']
runpy.run_module('tools.replay_motion', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", code, str(path)],
                            cwd=replay_motion.ROOT, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert f"Checked {len(clip['door_angle'])} frames; no physics steps" in result.stdout
