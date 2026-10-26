"""Collision cache integrity and concave object contact geometry."""

import importlib.metadata
import subprocess
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from motion_data import collision


@pytest.fixture
def tetra():
    return dict(
        object_vertices=np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        object_faces=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
        object_scale=np.array([.8, .6, .4]),
    )


@pytest.fixture
def cache(tmp_path, tetra, monkeypatch):
    # Exercise cache/storage behavior independently of the optional C++ tool.
    version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda name: collision.COACD_VERSION if name == "coacd" else version(name))
    fake = SimpleNamespace(
        set_log_level=lambda _: None,
        Mesh=lambda vertices, faces: (vertices, faces),
        run_coacd=lambda mesh, **_: [mesh],
    )
    monkeypatch.setitem(sys.modules, "coacd", fake)
    path = collision.prepare_collision(tetra, tmp_path)
    return path


def test_cache_identity_uses_geometry_and_recipe_not_pose(tetra, monkeypatch):
    key = collision.collision_key(tetra)
    other_motion = dict(tetra, object_scale=np.ones(3), object_pos=np.ones((5, 3)))
    assert collision.collision_key(other_motion) == key
    changed_mesh = dict(tetra, object_vertices=tetra["object_vertices"] * 2)
    assert collision.collision_key(changed_mesh) != key
    monkeypatch.setitem(collision.COACD_PARAMS, "threshold", .02)
    assert collision.collision_key(tetra) != key


def test_cache_roundtrip_reuses_valid_file_without_decomposer(cache, tetra, monkeypatch):
    parts = collision.load_collision_parts(tetra, cache.parent)
    np.testing.assert_array_equal(parts[0][0], tetra["object_vertices"])
    assert {tuple(sorted(face)) for face in parts[0][1]} == {
        tuple(sorted(face)) for face in tetra["object_faces"]}
    monkeypatch.setattr(sys.modules["coacd"], "run_coacd",
                        lambda *_, **__: pytest.fail("Existing cache recomputed"))
    assert collision.prepare_collision(tetra, cache.parent) == cache
    assert list(cache.parent.glob("*.npz")) == [cache]


@pytest.mark.parametrize("field,value", [
    ("key", np.asarray("wrong-key")),
    ("vertex_offsets", np.array([1, 4])),
    ("face_offsets", np.array([0, 3])),
    ("faces", np.array([[0, 2, 7], [0, 1, 3], [0, 3, 2], [1, 2, 3]])),
    ("faces", np.array([[0, 1, 2], [0, 1, 3], [0, 3, 2], [1, 2, 3]])),
])
def test_corrupt_cache_is_rejected_without_fallback(cache, tetra, field, value):
    with np.load(cache, allow_pickle=False) as archive:
        arrays = dict(archive)
    arrays[field] = value
    np.savez(cache, **arrays)
    with pytest.raises(ValueError, match="Invalid collision asset"):
        collision.load_collision_parts(tetra, cache.parent)
    with pytest.raises(ValueError, match="Invalid collision asset"):
        collision.prepare_collision(tetra, cache.parent)


def test_missing_cache_requires_explicit_preparation(tmp_path, tetra):
    with pytest.raises(FileNotFoundError, match="prepare_collisions.py"):
        collision.load_collision_parts(tetra, tmp_path)
    assert not list(tmp_path.iterdir())


def test_cache_load_has_no_simulator_or_preprocessor_dependency(cache, tetra, tmp_path):
    motion = tmp_path / "motion.npz"
    np.savez(motion, **tetra)
    script = """
import importlib.abc, sys
class BlockOptionalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'coacd', 'mujoco', 'mjlab', 'torch', 'mujoco_warp'}:
            raise AssertionError('Unexpected dependency: ' + fullname)
sys.meta_path.insert(0, BlockOptionalImports())
import numpy as np
from motion_data.collision import load_collision_parts
with np.load(sys.argv[1], allow_pickle=False) as clip:
    assert len(load_collision_parts(clip, sys.argv[2])) == 1
"""
    result = subprocess.run([sys.executable, "-c", script, str(motion), str(cache.parent)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def _box(center, half_size):
    vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    faces = np.array([[0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
                      [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                      [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6]])
    return vertices * half_size + center, faces


def test_preparation_repairs_part_triangulation_without_moving_vertices(cache, tetra, tmp_path, monkeypatch):
    vertices, faces = _box(np.zeros(3), np.ones(3))
    vertices[7, 2] -= 1e-4  # Old top-face diagonal is now slightly concave.
    with pytest.raises(ValueError, match="not convex"):
        collision._validate_parts([(vertices, faces)])
    monkeypatch.setattr(sys.modules["coacd"], "run_coacd", lambda *_, **__: [(vertices, faces)])
    output = tmp_path / "normalized"
    collision.prepare_collision(tetra, output)
    actual_vertices, actual_faces = collision.load_collision_parts(tetra, output)[0]
    np.testing.assert_array_equal(actual_vertices, vertices)
    assert len(actual_faces) == len(faces)


def test_separate_parts_preserve_under_seat_space_and_scale_once():
    parts = [_box(np.array([0, 0, 1.]), np.array([1., .5, .1])),
             _box(np.array([-.9, 0, .45]), np.array([.1, .5, .45])),
             _box(np.array([.9, 0, .45]), np.array([.1, .5, .45]))]
    clip = dict(object_vertices=np.concatenate([v for v, _ in parts]),
                object_faces=np.concatenate([f + i * 8 for i, (_, f) in enumerate(parts)]),
                object_scale=np.array([.8, .6, .4]))
    spec = mujoco.MjSpec()
    chair = spec.worldbody.add_body(name="chair")
    collision.add_object_geoms(spec, chair, clip, parts)
    probe = spec.worldbody.add_body(name="probe")
    probe.add_freejoint()
    probe.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.05, 0, 0], mass=1,
                   contype=0, conaffinity=2)
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:3] = [0, 0, .2]
    mujoco.mj_fwdPosition(model, data)
    assert data.ncon == 0  # Whole-chair convex hull would incorrectly fill this space.
    data.qpos[2] = .34  # Seat bottom = .9 * .4 = .36, sphere radius = .05.
    mujoco.mj_fwdPosition(model, data)
    assert data.ncon > 0
    assert min(c.dist for c in data.contact) == pytest.approx(-.03, abs=1e-6)
    for i in range(3):
        geom = model.geom(f"chair_collision_{i:03d}")
        assert geom.contype[0] == 2 and geom.conaffinity[0] == 0
        assert geom.group[0] == 3 and geom.margin[0] == 0
    assert model.geom("chair_visual").contype[0] == 0
    assert model.geom("chair_visual").conaffinity[0] == 0
    np.testing.assert_allclose(model.mesh_scale, np.tile(clip["object_scale"], (4, 1)))


def test_cli_deduplicates_geometry_and_reports_invalid_source(cache, tetra, tmp_path):
    motions = tmp_path / "motions"
    motions.mkdir()
    np.savez(motions / "a.npz", **tetra)
    np.savez(motions / "b.npz", **dict(tetra, object_scale=np.ones(3)))
    command = [sys.executable, "tools/prepare_collisions.py", str(motions),
               "--cache-dir", str(cache.parent), "--workers", "2"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "Preparing 1 unique meshes" in result.stdout
    assert "Created 0, reused 1, failed 0" in result.stdout
    np.savez(motions / "broken.npz", object_vertices=tetra["object_vertices"])
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    assert "FAILED broken.npz" in result.stderr
