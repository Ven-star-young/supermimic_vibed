"""Geometric regression checks, independent of the downloaded G1/chair assets."""

import mujoco
import numpy as np
import pytest
from scipy.spatial import cKDTree

from motion_data.back_proximity import BackProximity, summary


@pytest.fixture
def analyzer():
    # An offset mesh with a compensating geom pose catches failure to undo
    # MuJoCo's compiled mesh recentering. Its torso-space rear face is x=-0.1.
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ]) * [0.1, 0.2, 0.25] + [0.3, 0.02, 0.04]
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ])
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <asset><mesh name="torso" vertex="{' '.join(map(str, vertices.ravel()))}"
                   face="{' '.join(map(str, faces.ravel()))}"/></asset>
      <worldbody><body name="pelvis"><freejoint/>
        <inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/>
        <body name="torso_link"><joint name="waist" type="hinge" axis="0 0 1"/>
          <inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/>
          <geom type="mesh" mesh="torso" group="1" pos="-0.3 -0.02 -0.04"
                contype="0" conaffinity="0" mass="0"/>
        </body>
      </body></worldbody>
    </mujoco>""")
    return BackProximity(model, back_samples=256, chair_samples=8_000)


def plane_clip(x=-0.15, frames=3):
    return {
        "fps": np.array(25.0),
        "root_pos": np.tile([0.0, 0.0, 1.0], (frames, 1)),
        "root_quat": np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)),
        "joint_pos": np.zeros((frames, 1)),
        "joint_names": np.array(["waist"]),
        "object_pos": np.tile([0.0, 0.0, 1.0], (frames, 1)),
        "object_quat": np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)),
        "object_scale": np.ones(3),
        "object_vertices": np.array([[x, -0.5, -0.5], [x, 0.5, -0.5],
                                     [x, 0.5, 0.5], [x, -0.5, 0.5]]),
        "object_faces": np.array([[0, 1, 2], [0, 2, 3]]),
    }


def test_back_patch_and_known_plane_distance(analyzer):
    np.testing.assert_allclose(analyzer.back_points[:, 0], -0.1, atol=1e-7)
    result = analyzer.analyze(plane_clip())
    np.testing.assert_allclose(result["distances"], 0.05, atol=5e-4)
    np.testing.assert_allclose(
        np.linalg.norm(np.diff(result["points_world"], axis=1)[:, 0], axis=1),
        result["distances"], atol=1e-12,
    )
    # A near chest surface must not be classified as near back.
    front = analyzer.analyze(plane_clip(x=0.15))
    np.testing.assert_allclose(front["distances"], 0.25, atol=5e-4)


def test_world_rigid_transform_does_not_change_distances(analyzer):
    clip = plane_clip()
    before = analyzer.analyze(clip)
    angle = 0.87
    quat = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    translation = np.array([2.4, -1.7, 0.31])
    for prefix in ("root", "object"):
        clip[prefix + "_pos"] = clip[prefix + "_pos"] @ rotation.T + translation
        clip[prefix + "_quat"][:] = quat
    after = analyzer.analyze(clip)
    np.testing.assert_allclose(after["distances"], before["distances"], atol=1e-12)
    np.testing.assert_allclose(after["points_world"],
                               before["points_world"] @ rotation.T + translation, atol=1e-12)


def test_anisotropic_scale_and_moving_chair_are_applied_once(analyzer):
    clip = plane_clip()
    clip["object_scale"] = np.array([2.0, 0.7, 1.5])
    clip["object_pos"][:, 0] = [0.0, -0.1, 0.1]
    result = analyzer.analyze(clip)
    np.testing.assert_allclose(result["distances"], [0.2, 0.3, 0.1], atol=5e-4)
    np.testing.assert_allclose(result["points_world"][:, 1, 0], [-0.3, -0.4, -0.2])


def test_joint_mapping_rotates_the_back_not_the_chair(analyzer):
    clip = plane_clip()
    clip["joint_pos"][:, 0] = np.pi
    result = analyzer.analyze(clip)
    np.testing.assert_allclose(result["distances"], 0.25, atol=5e-4)
    np.testing.assert_allclose(result["points_world"][:, 0, 0], 0.1, atol=1e-7)


def test_deterministic_and_missing_chair_are_explicit(analyzer):
    clip = plane_clip()
    first, second = analyzer.analyze(clip), analyzer.analyze(clip)
    np.testing.assert_array_equal(first["distances"], second["distances"])
    np.testing.assert_array_equal(first["points_world"], second["points_world"])
    del clip["object_faces"]
    with pytest.raises(ValueError, match="chair mesh"):
        analyzer.analyze(clip)


def test_pruned_search_matches_exhaustive_surface_pair_distances(analyzer):
    rng = np.random.default_rng(123)
    cloud = rng.normal(size=(300, 3))
    queries = []
    for angle in (0.0, 0.4, 1.9, 3.0):
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        queries.append(analyzer.back_points @ rotation.T + rng.normal(size=3))
    queries = np.asarray(queries)
    distances, back_indices, chair_indices = analyzer._nearest_pairs(cKDTree(cloud), queries)
    exhaustive = np.linalg.norm(queries[:, :, None] - cloud[None, None], axis=-1)
    np.testing.assert_allclose(distances, exhaustive.min(axis=(1, 2)), atol=1e-12)
    np.testing.assert_allclose(distances, np.linalg.norm(
        queries[np.arange(len(queries)), back_indices] - cloud[chair_indices], axis=1))


def test_summary_tracks_consecutive_near_frames_and_validates_input():
    result = summary([0.1, 0.02, 0.03, 0.04, 0.01, 0.2], fps=10, threshold=0.03)
    assert result == {"fraction": 0.5, "peak_frame": 4,
                      "longest_seconds": 0.2, "minimum_distance": 0.01}
    assert summary([0.1, 0.2], 10, 0.03)["longest_seconds"] == 0
    assert summary([0.01, 0.02], 10, 0.03)["longest_seconds"] == 0.2
    for bad in ([], [np.nan], [-1.0], [[0.1]]):
        with pytest.raises(ValueError):
            summary(bad, 10, 0.03)
    with pytest.raises(ValueError, match="fps"):
        summary([0.1], 0, 0.03)
    with pytest.raises(ValueError, match="threshold"):
        summary([0.1], 10, -0.03)
