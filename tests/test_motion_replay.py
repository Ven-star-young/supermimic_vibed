"""Portable motion conventions and physics-free replay against the real G1 model."""

import subprocess
import sys

import mujoco
import numpy as np
import pytest

import motion_data as motion_io
from motion_data.converters import grail
from motion_data.converters._common import g1_joints
from tools import replay_motion


@pytest.fixture
def clip(asset_path):
    names, _ = g1_joints()
    return dict(
        fps=25., root_pos=np.array([[.1, -.2, .8], [.3, .4, .7]]),
        root_quat=np.array([[1., 0, 0, 0], [np.sqrt(.5), 0, 0, np.sqrt(.5)]]),
        joint_names=names[::-1], joint_pos=np.arange(58).reshape(2, 29) / 100.,
        object_pos=np.array([[0., 0, .4], [.2, -.1, .5]]),
        object_quat=np.array([[1., 0, 0, 0], [0., 1, 0, 0]]),
        object_scale=np.array([.8, .6, .4]),
        object_vertices=np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        object_faces=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
    )


def test_named_joints_root_and_object_use_only_fk(asset_path, clip, monkeypatch):
    model, data, root, addresses = replay_motion.make_scene(asset_path, clip)
    monkeypatch.setattr(mujoco, "mj_step", lambda *_: pytest.fail("Physics stepped"))
    monkeypatch.setattr(mujoco, "mj_forward", lambda *_: pytest.fail("Dynamics evaluated"))
    replay_motion.set_frame(model, data, clip, 1, root, addresses)
    np.testing.assert_allclose(data.qpos[root:root + 7],
                               np.r_[clip["root_pos"][1], clip["root_quat"][1]])
    for name, value in zip(clip["joint_names"], clip["joint_pos"][1]):
        assert data.joint(str(name)).qpos[0] == pytest.approx(value)
    np.testing.assert_allclose(data.body("pelvis").xpos, clip["root_pos"][1])
    np.testing.assert_allclose(data.body("pelvis").xmat.reshape(3, 3),
                               [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-7)
    body = model.body("replay_object")
    np.testing.assert_allclose(data.mocap_pos[body.mocapid[0]], clip["object_pos"][1])
    np.testing.assert_allclose(data.body("replay_object").xquat, clip["object_quat"][1])
    mesh = model.mesh("replay_object_mesh").id
    np.testing.assert_allclose(model.mesh_scale[mesh], clip["object_scale"])
    assert data.time == pytest.approx(1 / 25)
    assert model.nu == 0
    np.testing.assert_allclose(data.light_xpos[0], [0, 0, 5])
    assert (model.geom_group == 3).any()  # G1 collision geometry is hidden.


def test_npz_load_normalizes_quaternions(tmp_path, clip):
    clip["root_quat"] *= 2
    path = tmp_path / "clip.npz"
    np.savez(path, **clip)
    loaded = motion_io.load_motion(path)
    np.testing.assert_allclose(np.linalg.norm(loaded["root_quat"], axis=1), 1.)
    np.testing.assert_array_equal(loaded["joint_names"], clip["joint_names"])


@pytest.mark.parametrize("field,value,message", [
    ("root_pos", np.zeros((1, 3)), "root_pos"),
    ("root_quat", np.zeros((2, 4)), "zero quaternion"),
    ("fps", 0., "fps"),
    ("joint_names", np.array(["same"] * 29), "unique"),
    ("object_scale", np.array([1., 1., -1.]), "positive"),
    ("object_faces", np.array([[0, 1, 4]]), "indices"),
])
def test_invalid_motion_is_rejected(tmp_path, clip, field, value, message):
    clip[field] = value
    path = tmp_path / "invalid.npz"
    np.savez(path, **clip)
    with pytest.raises(ValueError, match=message):
        motion_io.load_motion(path)


def test_headless_cli_needs_no_torch_or_mjlab(tmp_path, clip):
    path = tmp_path / "clip.npz"
    np.savez(path, **clip)
    code = """
import importlib.abc, runpy, sys
class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'mjlab', 'mujoco_warp'}:
            raise AssertionError('Unexpected training dependency: ' + fullname)
sys.meta_path.insert(0, BlockTrainingImports())
sys.argv = ['replay_motion', sys.argv[1], '--headless']
runpy.run_module('tools.replay_motion', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", code, str(path)],
                            cwd=replay_motion.ROOT, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "Checked 2 frames; no physics steps" in result.stdout


def test_grail_robot_xyzw_and_object_wxyz_are_distinct(tmp_path, clip, monkeypatch):
    joblib = pytest.importorskip("joblib", reason="GRAIL conversion needs --extra motion")

    names, axes = g1_joints()
    joints = clip["joint_pos"][:, ::-1]
    pose_aa = np.zeros((2, 30, 3))
    pose_aa[:, 1:] = joints[..., None] * axes
    robot = dict(fps=25., dof=joints, pose_aa=pose_aa,
                 root_trans_offset=clip["root_pos"], root_rot=clip["root_quat"][:, [1, 2, 3, 0]])
    obj = dict(fps=25., root_pos=clip["object_pos"][:, None],
               root_quat=clip["object_quat"][:, None], scale=clip["object_scale"][:, None])
    for directory, payload in (("robot", {"motion": robot}), ("objects", {"motion": obj}),
                               ("meta", {"scene_scale": 1.})):
        (tmp_path / directory).mkdir()
        joblib.dump(payload, tmp_path / directory / "motion.pkl")
    monkeypatch.setattr(grail, "read_usd_mesh",
                        lambda _: (clip["object_vertices"], clip["object_faces"]))
    converted = grail.convert_grail(tmp_path / "robot/motion.pkl", tmp_path, names, axes)
    for field in ("root_pos", "root_quat", "object_pos", "object_quat", "object_scale"):
        np.testing.assert_allclose(converted[field], clip[field])
    np.testing.assert_array_equal(converted["joint_names"], names)
    np.testing.assert_allclose(converted["joint_pos"], joints)


def test_robot_only_clip_preserves_existing_mocap(tmp_path):
    path = tmp_path / "robot.xml"
    path.write_text('''<mujoco><worldbody>
      <body name="robot"><freejoint/><geom size=".1" mass="1"/>
        <body><joint name="hinge"/><geom size=".1" mass="1"/></body>
      </body><body name="target" mocap="true" pos="1 2 3"/>
    </worldbody></mujoco>''')
    clip = motion_io.validate_motion(dict(fps=25., root_pos=[[0, 0, 1]],
        root_quat=[[1., 0, 0, 0]], joint_pos=[[.2]], joint_names=["hinge"]))
    model, data, root, addresses = replay_motion.make_scene(path, clip)
    replay_motion.set_frame(model, data, clip, 0, root, addresses)
    np.testing.assert_allclose(data.body("target").xpos, [1, 2, 3])
