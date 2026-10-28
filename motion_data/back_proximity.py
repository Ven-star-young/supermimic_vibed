"""Approximate torso-back/chair surface proximity for reviewing kinematic clips.

This is an unsigned distance between sampled surfaces, not a contact detector:
sampling can overestimate separation, penetration is not detected, and no force
or support is inferred. The G1 faces +X in its torso frame. Its back patch is the
rear 45% of the torso visual mesh's X extent, restricted to faces whose outward
normal has an X component below -0.25. Pelvis, buttocks, arms and hands are absent.
"""

from __future__ import annotations

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from motion_data import validate_motion


def _rotation(quat: np.ndarray) -> np.ndarray:
    result = np.empty(9, dtype=np.float64)
    quat = np.asarray(quat, dtype=np.float64)
    mujoco.mju_quat2Mat(result, quat / np.linalg.norm(quat))
    return result.reshape(3, 3)


def _sample_surface(triangles: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Sample physical triangle area deterministically, omitting zero-area faces."""
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1)
    total = areas.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Surface mesh has no finite positive area")
    rng = np.random.default_rng(seed)
    selected = triangles[rng.choice(len(triangles), size=count, p=areas / total)]
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, 0])
    weights = np.column_stack((1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]))
    return np.einsum("ni,nij->nj", weights, selected)


class BackProximity:
    """Reuse one robot model and sampled back patch across clips; FK only.

    ``back_points`` are in ``body_id`` (``torso_link``) local coordinates, so the
    same samples can be attached to the robot in an interactive viewer. The
    supplied model is never compiled or mutated; this instance owns its MjData.
    Instances are not thread safe; use one per worker or serialize ``analyze``.
    """

    def __init__(self, model: mujoco.MjModel, *, back_samples: int = 512,
                 chair_samples: int = 20_000):
        if (not isinstance(back_samples, int) or back_samples <= 0
                or not isinstance(chair_samples, int) or chair_samples <= 0):
            raise ValueError("Surface sample counts must be positive integers")
        self.model = model
        self.data = mujoco.MjData(model)
        self.body_id = int(model.body("torso_link").id)
        self.chair_samples = chair_samples
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise ValueError("Back proximity expects exactly one floating robot root")
        self.root_adr = int(model.jnt_qposadr[free[0]])

        patches = []
        for geom in np.flatnonzero((model.geom_bodyid == self.body_id)
                                   & (model.geom_group == 1)
                                   & (model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)):
            mesh = int(model.geom_dataid[geom])
            va, fa = int(model.mesh_vertadr[mesh]), int(model.mesh_faceadr[mesh])
            vertices = model.mesh_vert[va:va + model.mesh_vertnum[mesh]]
            # MuJoCo recentres/reorients mesh vertices while compiling. Applying
            # the compiled geom pose recovers the actual torso-local surface.
            vertices = vertices @ _rotation(model.geom_quat[geom]).T + model.geom_pos[geom]
            faces = model.mesh_face[fa:fa + model.mesh_facenum[mesh]]
            triangles = vertices[faces]
            normals = np.cross(triangles[:, 1] - triangles[:, 0],
                               triangles[:, 2] - triangles[:, 0])
            lengths = np.linalg.norm(normals, axis=1)
            rear_limit = vertices[:, 0].min() + 0.45 * np.ptp(vertices[:, 0])
            selected = ((triangles.mean(axis=1)[:, 0] <= rear_limit)
                        & (normals[:, 0] < -0.25 * lengths) & (lengths > 0))
            patches.append(triangles[selected])
        if not patches or not any(len(patch) for patch in patches):
            raise ValueError("No posterior torso_link visual-mesh surface found in group 1")
        self.back_points = _sample_surface(np.concatenate(patches), back_samples, seed=0)
        # A small, well-spread subset gives rigorous bounds for the other
        # samples: distance to any fixed point cloud is 1-Lipschitz. Distances
        # between back samples remain constant under every robot pose.
        seeds = []
        closest_seed = np.full(back_samples, np.inf)
        candidate = int(self.back_points[:, 2].argmin())
        for _ in range(min(32, back_samples)):
            seeds.append(candidate)
            spacing = np.linalg.norm(self.back_points - self.back_points[candidate], axis=1)
            closest_seed = np.minimum(closest_seed, spacing)
            candidate = int(closest_seed.argmax())
        self._seeds = np.asarray(seeds)
        self._seed_spacing = np.linalg.norm(
            self.back_points[:, None] - self.back_points[self._seeds][None], axis=2)

    def _nearest_pairs(self, tree: cKDTree, queries: np.ndarray):
        """Find exact minima of the sampled clouds while pruning far samples."""
        seed_distances, seed_nearest = tree.query(queries[:, self._seeds], workers=1)
        best_seed = seed_distances.argmin(axis=1)
        rows = np.arange(len(queries))
        distances = seed_distances[rows, best_seed]
        back_indices = self._seeds[best_seed].copy()
        chair_indices = seed_nearest[rows, best_seed].copy()
        # Roundoff tolerance protects the lower bound after rigid transforms;
        # it never relaxes the final distance measurement itself.
        tolerance = 1e-9
        lower = (seed_distances[:, None] - self._seed_spacing[None]).max(axis=2)
        candidates = lower <= distances[:, None] + tolerance
        candidates[:, self._seeds] = False
        for frame, mask in enumerate(candidates):
            indices = np.flatnonzero(mask)
            if not len(indices):
                continue
            extra_distances, extra_nearest = tree.query(
                queries[frame, indices], distance_upper_bound=distances[frame] + tolerance,
                workers=1)
            closest = int(extra_distances.argmin())
            if extra_distances[closest] < distances[frame]:
                distances[frame] = extra_distances[closest]
                back_indices[frame] = indices[closest]
                chair_indices[frame] = extra_nearest[closest]
        return distances, back_indices, chair_indices

    def analyze(self, clip: dict) -> dict[str, np.ndarray]:
        """Return per-frame metres and closest sampled [back, chair] world points.

        Chair vertices and samples are scaled once in local space, including
        anisotropic scale. Query points then follow each frame's object pose.
        ``distances`` has shape (T,), ``points_world`` has shape (T, 2, 3).
        """
        if not {"object_vertices", "object_faces", "object_pos", "object_quat",
                "object_scale"} <= clip.keys():
            raise ValueError("Back proximity requires a chair mesh, poses and object_scale")
        clip = validate_motion(clip)
        vertices = np.asarray(clip["object_vertices"], dtype=np.float64) * clip["object_scale"]
        chair_points = np.concatenate((vertices, _sample_surface(
            vertices[clip["object_faces"]], self.chair_samples, seed=1)))
        tree = cKDTree(chair_points)

        addresses = []
        for name in clip["joint_names"]:
            joint = self.model.joint(str(name))
            if joint.type[0] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                raise ValueError(f"Expected a scalar robot joint: {name}")
            addresses.append(int(joint.qposadr[0]))
        n = len(clip["root_pos"])
        back_world = np.empty((n, len(self.back_points), 3), dtype=np.float64)
        query_points = np.empty_like(back_world)
        rotations = np.stack([_rotation(q) for q in clip["object_quat"]])
        self.data.qpos[:] = self.model.qpos0
        for frame in range(n):
            self.data.qpos[self.root_adr:self.root_adr + 3] = clip["root_pos"][frame]
            root_quat = np.asarray(clip["root_quat"][frame], dtype=np.float64)
            self.data.qpos[self.root_adr + 3:self.root_adr + 7] = root_quat / np.linalg.norm(root_quat)
            self.data.qpos[addresses] = clip["joint_pos"][frame]
            mujoco.mj_kinematics(self.model, self.data)
            back_world[frame] = (self.back_points @ self.data.xmat[self.body_id].reshape(3, 3).T
                                 + self.data.xpos[self.body_id])
            query_points[frame] = ((back_world[frame] - clip["object_pos"][frame])
                                   @ rotations[frame])
        distances, closest, nearest = self._nearest_pairs(tree, query_points)
        rows = np.arange(n)
        chair_local = chair_points[nearest]
        chair_world = np.einsum("nij,nj->ni", rotations, chair_local) + clip["object_pos"]
        return {
            "distances": distances,
            "points_world": np.stack((back_world[rows, closest], chair_world), axis=1),
        }


def summary(distances: np.ndarray, fps: float, threshold: float) -> dict:
    """Summarize frames at/below a chosen proximity threshold (in metres).

    ``longest_seconds`` is the longest consecutive near-frame count divided by
    fps. It is a sampled occupancy duration, not a physical contact duration.
    """
    distances = np.asarray(distances, dtype=np.float64)
    if (distances.ndim != 1 or not len(distances) or not np.isfinite(distances).all()
            or np.any(distances < 0)):
        raise ValueError("distances must be a nonempty finite nonnegative 1D array")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    near = distances <= threshold
    changes = np.diff(np.r_[False, near, False].astype(np.int8))
    lengths = np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)
    return {
        "fraction": float(near.mean()),
        "peak_frame": int(distances.argmin()),
        "longest_seconds": float(lengths.max(initial=0) / fps),
        "minimum_distance": float(distances.min()),
    }
