"""Create a kinematic G1 back-contact ledge reference and its static MJCF.

Run ``python -m tools.generate_ledge_reference`` from the project root.
The result uses the ordinary motion_data NPZ schema; it is not a trained policy
or a dynamically simulated rollout. All contact arrays are planned targets.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from motion_data import load_motion, save_motion
from supermimic.assets.g1.g1_constants import G1_JOINT_NAMES, make_g1_spec
from supermimic.tasks.open_door.geometry import _surface_candidates


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/motions/kimodo_side60_hug_elbow/side60_hug_elbow.npz"
OUTPUT = ROOT / "data/motions/ledge_back/ledge_back.npz"
WIDTH, HEIGHT, STEP = .08, .6, .1
# A small compliant overlap produces actual geometric back-contact pairs.
# It is inside the existing back task's 3 mm legal penetration tolerance;
# inverse kinematics still provides no measured contact-force labels.
BACK_GAP = -.0001
SCENE = '''<mujoco model="straight_back_contact_ledge">
  <compiler angle="radian" />
  <asset>
    <texture name="lower_floor_grid" type="2d" builtin="checker" rgb1="0.82 0.84 0.86" rgb2="0.72 0.75 0.78" width="512" height="512" />
    <material name="lower_floor_material" texture="lower_floor_grid" texrepeat="4 4" reflectance="0.05" />
  </asset>
  <worldbody>
    <!-- +X points away from the wall; +Y is the traversal direction. -->
    <geom name="lower_floor" type="plane" size="0 0 0.05" material="lower_floor_material" contype="1" conaffinity="0" friction="1 0.005 0.0001" />
    <body name="wall" pos="-0.06 0.6 1.1">
      <geom name="wall_panel" type="box" size="0.06 1.2 1.1" rgba="0.60 0.67 0.73 1" contype="2" conaffinity="0" priority="1" friction="0.7 0.005 0.0001" />
    </body>
    <body name="ledge" pos="0.04 0.6 0.54">
      <geom name="ledge_top" type="box" size="0.04 1.0 0.06" rgba="0.85 0.58 0.24 1" contype="1" conaffinity="0" friction="1 0.005 0.0001" />
    </body>
  </worldbody>
</mujoco>
'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smooth(t):
    return t ** 3 * (10 + t * (-15 + 6 * t))


class LedgeReference:
    def __init__(self, source):
        self.model = make_g1_spec(interaction_collisions=True).compile()
        self.data = mujoco.MjData(self.model)
        self.addresses = np.array([self.model.joint(n).qposadr[0] for n in G1_JOINT_NAMES])
        self.joint_ids = np.array([self.model.joint(n).id for n in G1_JOINT_NAMES])
        self.ranges = self.model.jnt_range[self.joint_ids].copy()
        clip = load_motion(source)
        order = [list(clip["joint_names"]).index(n) for n in G1_JOINT_NAMES]
        self.source_q = clip["joint_pos"][0, order].copy()
        self.feet = [self.model.body(s + "_ankle_roll_link").id for s in ("left", "right")]
        self.torso = self.model.body("torso_link").id
        self.foot_rotations = [Rotation.from_euler("z", a).as_matrix() for a in (np.pi / 2, -np.pi / 2)]
        self.surfaces = []
        for g in range(self.model.ngeom):
            if not self.model.geom_conaffinity[g] & 2:
                continue
            points = _surface_candidates(self.model, g)
            points = points[ConvexHull(points).vertices]
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, self.model.geom_quat[g])
            points = points @ matrix.reshape(3, 3).T + self.model.geom_pos[g]
            body = int(self.model.geom_bodyid[g])
            name = self.model.body(body).name
            self.surfaces.append((body, points, name))
        self.clearance = np.array([BACK_GAP if n in ("torso_link", "logo_link") else .002
                                   for _, _, n in self.surfaces])
        self.arms = self.make_hug()
        self.seed = np.r_[.065, 0., 1.36, 0., .05, 0., np.zeros(15)]
        self.seed[6 + 1], self.seed[6 + 7] = .4, -.4
        self.seed[6 + 2], self.seed[6 + 8] = np.pi / 2, -np.pi / 2
        self.seed[6 + 3] = self.seed[6 + 9] = .4
        self.seed[6 + 4] = self.seed[6 + 10] = -.15
        center = self.ranges[:15].mean(1)
        half = np.diff(self.ranges[:15], axis=1)[:, 0] * .45
        self.low = np.r_[.035, -.4, 1.08, -.45, -.4, -.7, center - half + 1e-5]
        self.high = np.r_[.25, .6, 1.42, .45, .4, .7, center + half - 1e-5]

    def make_hug(self):
        """Retain chest-front holding, with closer wrists for narrow support."""
        self.data.qpos[:] = 0
        self.data.qpos[3] = 1
        wrists = [self.model.body(s + "_wrist_yaw_link").id for s in ("left", "right")]
        targets = np.array([[.13, .155, .18], [.13, -.155, .18]])
        rotations = [Rotation.from_euler("z", a).as_matrix() for a in (-np.pi / 6, np.pi / 6)]
        initial = self.source_q[15:]
        ranges = self.ranges[15:]
        center = ranges.mean(1)
        half = (ranges[:, 1] - ranges[:, 0]) * .45

        def residual(q):
            self.data.qpos[self.addresses[15:]] = q
            mujoco.mj_forward(self.model, self.data)
            torso_R = self.data.xmat[self.torso].reshape(3, 3)
            residuals = []
            for b, position, rotation in zip(wrists, targets, rotations):
                relative = torso_R.T @ (self.data.xpos[b] - self.data.xpos[self.torso])
                residuals.extend(50 * (relative - position))
                error = rotation.T @ torso_R.T @ self.data.xmat[b].reshape(3, 3)
                residuals.extend(.05 * Rotation.from_matrix(error).as_rotvec())
            return np.r_[residuals, .05 * (q - initial)]

        result = least_squares(residual, np.clip(initial, center - half + 1e-6, center + half - 1e-6),
                               bounds=(center - half, center + half), max_nfev=300)
        return result.x

    def forward(self, v):
        self.data.qpos[:3] = v[:3]
        self.data.qpos[3:7] = Rotation.from_euler("xyz", v[3:6]).as_quat()[[3, 0, 1, 2]]
        self.data.qpos[self.addresses] = np.r_[v[6:], self.arms]
        mujoco.mj_forward(self.model, self.data)

    def solve(self, previous, foot_target, com_y):
        def residual(v):
            self.forward(v)
            residuals = []
            for b, position, rotation in zip(self.feet, foot_target, self.foot_rotations):
                residuals.extend(1000 * (self.data.xpos[b] - position))
                error = rotation.T @ self.data.xmat[b].reshape(3, 3)
                residuals.extend(100 * Rotation.from_matrix(error).as_rotvec())
            torso_R = self.data.xmat[self.torso].reshape(3, 3)
            residuals.extend(2 * Rotation.from_matrix(torso_R).as_rotvec())
            distances = np.array([(p @ self.data.xmat[b].reshape(3, 3).T + self.data.xpos[b])[:, 0].min()
                                  for b, p, _ in self.surfaces])
            back = min(x for x, (_, _, n) in zip(distances, self.surfaces)
                       if n in ("torso_link", "logo_link"))
            residuals.extend(1000 * np.minimum(distances - self.clearance, 0))
            residuals.append(300 * (back - BACK_GAP))
            com = self.data.subtree_com[1]
            residuals.extend([200 * max(com[0] - .073, 0), 500 * (com[1] - com_y)])
            residuals.extend(.5 * (v[6:] - self.seed[6:]))
            residuals.extend(.02 * v[3:6])
            return residuals

        result = least_squares(residual, np.clip(previous, self.low + 1e-7, self.high - 1e-7),
                               bounds=(self.low, self.high), max_nfev=80,
                               ftol=2e-8, xtol=2e-8, gtol=2e-7)
        self.forward(result.x)
        position_error = float(np.abs(self.data.xpos[self.feet] - foot_target).max())
        if position_error > .001:
            raise RuntimeError(f"IK foot error {position_error:.6f} m exceeds 1 mm")
        return result.x


def cycle_targets(fps):
    feet = np.array([[WIDTH - .031, .13, HEIGHT + .035], [WIDTH - .031, -.13, HEIGHT + .035]])
    com_y = 0.
    targets, centers, stance = [feet.copy()], [com_y], [[True, True]]
    # Weight shifts happen in double support; the other foot stays fixed during
    # each swing. Quintic timing makes endpoint velocity and acceleration zero.
    phases = [("shift", 1.5, -.16), ("left", 1.5, None),
              ("shift", 2., .13 + STEP + .035), ("right", 1.5, None),
              ("shift", 1.5, STEP)]
    for kind, seconds, goal in phases:
        initial, start_y = feet.copy(), com_y
        for frame in range(1, round(seconds * fps) + 1):
            t = frame / round(seconds * fps)
            weight = smooth(t)
            feet = initial.copy()
            support = [True, True]
            if kind == "shift":
                com_y = start_y + weight * (goal - start_y)
            else:
                side = 0 if kind == "left" else 1
                feet[side, 1] += STEP * weight
                feet[side, 2] += .035 * 64 * t ** 3 * (1 - t) ** 3
                support[side] = frame == round(seconds * fps)
            targets.append(feet.copy())
            centers.append(com_y)
            stance.append(support)
    return np.array(targets), np.array(centers), np.array(stance)


def generate(output, source, fps, cycles):
    solver = LedgeReference(source)
    targets, centers, support = cycle_targets(fps)
    pose = solver.solve(solver.seed, targets[0], centers[0])
    poses = [pose.copy()]
    for frame in range(1, len(targets)):
        pose = solver.solve(pose, targets[frame], centers[frame])
        poses.append(pose.copy())
        if frame % fps == 0:
            print(f"IK {frame}/{len(targets) - 1}", flush=True)
    poses = np.array(poses)
    closure = poses[-1] - poses[0]
    closure[1] -= STEP
    print(f"Cycle closure maximum: {np.abs(closure).max():.6g}", flush=True)
    if np.abs(closure).max() > 1e-3:
        raise RuntimeError("IK cycle does not close; refusing to blend a material pose difference")
    # A return to the same foot targets/COM must return to the same pose before
    # translating the cycle. Smooth a small optimizer null-space drift across
    # the final double-support transfer, then independently validate the clip.
    count = round(1.5 * fps)
    for i in range(count + 1):
        poses[-count - 1 + i] -= smooth(i / count) * closure
    sequence, target_sequence, support_sequence = [], [], []
    for cycle in range(cycles):
        p, t = poses[:-1].copy(), targets[:-1].copy()
        p[:, 1] += cycle * STEP
        t[:, :, 1] += cycle * STEP
        sequence.append(p)
        target_sequence.append(t)
        support_sequence.append(support[:-1])
    last = poses[-1].copy()
    last[1] += (cycles - 1) * STEP
    last_target = targets[-1].copy()
    last_target[:, 1] += (cycles - 1) * STEP
    sequence = np.concatenate([np.repeat(poses[:1], fps, 0), *sequence, np.repeat(last[None], fps + 1, 0)])
    target_sequence = np.concatenate([np.repeat(targets[:1], fps, 0), *target_sequence,
                                      np.repeat(last_target[None], fps + 1, 0)])
    support_sequence = np.concatenate([np.ones((fps, 2), bool), *support_sequence,
                                       np.ones((fps + 1, 2), bool)])
    output.parent.mkdir(parents=True, exist_ok=True)
    xml_path = output.with_suffix(".xml")
    xml_path.write_text(SCENE)
    mujoco.MjSpec.from_file(str(xml_path)).compile()
    metadata = {
        "method": "bounded kinematic IK with alternating foot swings and double-support COM transfers",
        "kinematic_only": True, "closed_loop_dynamics_validated": False,
        "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "source_sha256": digest(source), "generator_sha256": digest(__file__),
        "robot_xml_sha256": digest(ROOT / "data/g1_description/g1_29dof.xml"),
        "scene_xml_sha256": digest(xml_path), "fps": fps,
        "frames": len(sequence), "duration_s": (len(sequence) - 1) / fps,
        "travel_m": STEP * cycles, "cycles": cycles,
        "coordinate_frame": "+X away from wall, +Y along ledge, +Z up; wxyz quaternions",
        "joint_bounds": "all 29 joints constrained to the project's 90 percent soft ranges",
        "geometry": {"ledge_width_m": WIDTH, "ledge_length_m": 2., "ledge_top_m": HEIGHT,
                     "wall_front_x_m": 0., "wall_height_m": 2.2,
                     "foot_sphere_support_width_m": .07, "foot_mesh_width_m": .0756},
        "hug": {"source_wrist_target_m": [.23, .155, .18],
                "wrist_target_m": [.13, .155, .18], "right_y_sign": -1,
                "arm_joint_bounds": "90 percent soft joint range",
                "palm_orientation": "soft inward-facing target; adjusted to keep arms close"},
        "planned_contact": {"back_gap_m": BACK_GAP, "other_body_wall_clearance_m": .002,
                            "back_overlap_is_compliant_geometric_target": True,
                            "swing_height_m": .035, "force_labels": False},
        "cycle_ik_closure_before_blend": float(np.abs(closure).max()),
    }
    clip = dict(fps=np.float64(fps), root_pos=sequence[:, :3],
                root_quat=Rotation.from_euler("xyz", sequence[:, 3:6]).as_quat()[:, [3, 0, 1, 2]],
                joint_pos=np.c_[sequence[:, 6:], np.repeat(solver.arms[None], len(sequence), 0)],
                joint_names=np.array(G1_JOINT_NAMES), kinematic_only=np.array(True),
                planned_stance=support_sequence, planned_foot_positions=target_sequence,
                planned_back_contact=np.ones(len(sequence), bool),
                scene_xml_sha256=np.array(metadata["scene_xml_sha256"]),
                provenance=np.array(json.dumps(metadata, ensure_ascii=False)))
    save_motion(output, clip)
    metadata["reference_sha256"] = digest(output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {len(sequence)} frames to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cycles", type=int, default=10)
    args = parser.parse_args()
    if args.fps < 10 or args.cycles < 1 or args.cycles > 10:
        parser.error("fps must be >= 10 and cycles must be 1..10 for the fixed 2 m ledge")
    generate(args.output.resolve(), args.source.resolve(), args.fps, args.cycles)


if __name__ == "__main__":
    main()
