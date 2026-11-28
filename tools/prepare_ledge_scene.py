"""Fit recorded G1 references to a high ledge with two wide landings.

python -m tools.prepare_ledge_scene --source /path/to/video9_g1.npz
python tools/replay_motion.py data/motions/ledge_video9/video9_g1_ledge.npz \
    --scene-xml data/motions/ledge_video9/video9_g1_ledge.xml
python -m tools.prepare_ledge_scene --video11

The default video9 fit changes only world yaw/translation and root height.
The video11 fit preserves timing, waist and arms, while refining root/legs for
back contact and heel/forefoot support. Neither mode simulates a robot policy.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from motion_data import load_motion, save_motion
from supermimic.assets.g1.g1_constants import G1_JOINT_NAMES, make_g1_spec


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/motions/ledge_video9/source/video9_g1.npz"
OUTPUT = ROOT / "data/motions/ledge_video9/video9_g1_ledge.npz"
# Reviewed in the rotated video9 frame; both turns remain on wide landings.
START, ENTRY, EXIT, END = -.45, .65, 1.34, 2.10
REAR, FRONT = -.20, 1.10


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scene_xml(height, width):
    wall_top = height + 1.5
    return f'''<mujoco model="video9_high_ledge">
  <compiler angle="radian" />
  <asset>
    <texture name="lower_floor_grid" type="2d" builtin="checker" rgb1="0.23 0.27 0.31" rgb2="0.28 0.32 0.36" width="512" height="512" />
    <material name="lower_floor_material" texture="lower_floor_grid" texrepeat="3 3" reflectance="0" />
  </asset>
  <worldbody>
    <!-- +X: travel, +Y: away from wall, +Z: up. Central wall face Y=0. -->
    <!-- Ground category: sole spheres and body; wall category: full foot/body. -->
    <geom name="lower_floor" type="plane" size="0 0 0.05" material="lower_floor_material" contype="1" conaffinity="0" friction="1 0.005 0.0001" />
    <!-- Recess the landing walls by 20 cm to accommodate the recorded turns. -->
    <body name="wall">
      <geom name="entry_wall" type="box" pos="{(START + ENTRY) / 2} -0.275 {wall_top / 2}" size="{(ENTRY - START) / 2} 0.075 {wall_top / 2}" rgba="0.55 0.63 0.67 1" contype="2" conaffinity="0" priority="1" friction="0.7 0.005 0.0001" />
      <geom name="central_wall" type="box" pos="{(ENTRY + EXIT) / 2} -0.175 {wall_top / 2}" size="{(EXIT - ENTRY) / 2} 0.175 {wall_top / 2}" rgba="0.62 0.69 0.72 1" contype="2" conaffinity="0" priority="1" friction="0.7 0.005 0.0001" />
      <geom name="exit_wall" type="box" pos="{(EXIT + END) / 2} -0.275 {wall_top / 2}" size="{(END - EXIT) / 2} 0.075 {wall_top / 2}" rgba="0.55 0.63 0.67 1" contype="2" conaffinity="0" priority="1" friction="0.7 0.005 0.0001" />
    </body>
    <!-- Solid blocks: equal top heights and exact, gap-free junctions. -->
    <body name="platforms">
      <geom name="start_platform" type="box" pos="{(START + ENTRY) / 2} {(REAR + FRONT) / 2} {height / 2}" size="{(ENTRY - START) / 2} {(FRONT - REAR) / 2} {height / 2}" rgba="0.42 0.50 0.57 1" contype="1" conaffinity="0" friction="1 0.005 0.0001" />
      <geom name="narrow_ledge" type="box" pos="{(ENTRY + EXIT) / 2} {width / 2} {height / 2}" size="{(EXIT - ENTRY) / 2} {width / 2} {height / 2}" rgba="0.72 0.58 0.38 1" contype="1" conaffinity="0" friction="1 0.005 0.0001" />
      <geom name="goal_platform" type="box" pos="{(EXIT + END) / 2} {(REAR + FRONT) / 2} {height / 2}" size="{(END - EXIT) / 2} {(FRONT - REAR) / 2} {height / 2}" rgba="0.42 0.50 0.57 1" contype="1" conaffinity="0" friction="1 0.005 0.0001" />
    </body>
    <!-- Paint only: no guardrail or hidden support outside the narrow edge. -->
    <geom name="ledge_edge_mark" type="box" pos="{(ENTRY + EXIT) / 2} {width - .01} {height + .0005}" size="{(EXIT - ENTRY) / 2} 0.01 0.0005" rgba="0.98 0.66 0.16 1" contype="0" conaffinity="0" />
  </worldbody>
</mujoco>
'''


def prepare(source, output, height=1.2, width=.32, overwrite=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.suffix != ".npz":
        raise ValueError("Output must have the .npz suffix")
    if not np.isfinite([height, width]).all() or height <= 0 or not 0 < width < FRONT:
        raise ValueError("Height must be positive; width must be between 0 and 1.10 m")
    destinations = [output, output.with_suffix(".xml"), output.with_suffix(".json")]
    if source in destinations:
        raise ValueError("Output must not overwrite the source reference")
    if not overwrite and any(path.exists() for path in destinations):
        raise FileExistsError("Output already exists; use --overwrite to replace the derived asset")
    clip = load_motion(source)
    if set(clip["joint_names"]) != set(G1_JOINT_NAMES):
        raise ValueError("Expected all 29 named G1 joints")
    if len(clip["root_pos"]) != 767 or float(clip["fps"]) != 30:
        raise ValueError("This fit expects the 767-frame, 30 Hz video9 export")
    if "object_pos" in clip or "door_angle" in clip:
        raise ValueError("Expected a robot-only reference")

    # Use the recorded sideways segment to align travel with +X. No scaling,
    # time warping, joint editing, or per-frame horizontal projection is used.
    xy = clip["root_pos"][225:630, :2]
    _, _, axes = np.linalg.svd(xy - xy.mean(axis=0), full_matrices=False)
    tangent = axes[0]
    if tangent @ (xy[-1] - xy[0]) < 0:
        tangent = -tangent
    yaw = -np.arctan2(tangent[1], tangent[0])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    clip["root_pos"] = clip["root_pos"].copy()
    clip["root_pos"][:, :2] = clip["root_pos"][:, :2] @ rotation.T + [0., .15]
    yaw_quat = np.array([np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)])
    quaternions = np.empty_like(clip["root_quat"])
    for original, target in zip(clip["root_quat"], quaternions):
        mujoco.mju_mulQuat(target, yaw_quat, original)
    clip["root_quat"] = quaternions

    xml = scene_xml(height, width)
    scene = mujoco.MjSpec.from_string(xml)
    scene.compile()
    spec = make_g1_spec(interaction_collisions=True)
    spec.attach(scene, prefix="ledge/", frame=spec.worldbody.add_frame())
    model = spec.compile()
    data = mujoco.MjData(model)
    joints = [model.joint(str(name)).qposadr[0] for name in clip["joint_names"]]
    feet = np.array([g for g in range(model.ngeom)
                     if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
                     and model.body(model.geom_bodyid[g]).name in
                     ("left_ankle_roll_link", "right_ankle_roll_link")])
    if len(feet) != 8:
        raise ValueError("Expected the G1's eight sole support spheres")
    radius = model.geom_size[feet, 0]
    wall_ids = {model.geom("ledge/" + name).id for name in ("entry_wall", "central_wall", "exit_wall")}
    offsets = []
    wall_contacts = 0
    min_wall_distance = np.inf
    for frame in range(len(clip["root_pos"])):
        data.qpos[:3] = clip["root_pos"][frame]
        data.qpos[3:7] = clip["root_quat"][frame]
        data.qpos[joints] = clip["joint_pos"][frame]
        mujoco.mj_forward(model, data)
        dz = height - np.min(data.geom_xpos[feet, 2] - radius)
        offsets.append(float(dz))
        clip["root_pos"][frame, 2] += dz
        data.qpos[2] += dz
        mujoco.mj_forward(model, data)

        # Conservatively require each sole sphere's XY bounding square to lie
        # on the union of the three top surfaces, including during swings.
        centers = data.geom_xpos[feet, :2]
        for signs in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            points = centers + radius[:, None] * signs
            x, y = points.T
            on_start = (START <= x) & (x <= ENTRY) & (REAR <= y) & (y <= FRONT)
            on_ledge = (ENTRY <= x) & (x <= EXIT) & (0 <= y) & (y <= width)
            on_goal = (EXIT <= x) & (x <= END) & (REAR <= y) & (y <= FRONT)
            if not np.all(on_start | on_ledge | on_goal):
                raise ValueError(f"Frame {frame}: foot support extends outside the platforms")
        for contact in data.contact:
            if any(int(g) in wall_ids for g in contact.geom):
                wall_contacts += 1
                min_wall_distance = min(min_wall_distance, float(contact.dist))
                if contact.dist < -1e-6:
                    raise ValueError(f"Frame {frame}: wall penetration {contact.dist:g} m")

    metadata = {
        "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "source_sha256": digest(source),
        "generator_sha256": digest(__file__),
        "robot_xml_sha256": digest(ROOT / "data/g1_description/g1_29dof.xml"),
        "scene_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "frames": len(offsets), "fps": float(clip["fps"]),
        "duration_s": (len(offsets) - 1) / float(clip["fps"]),
        "coordinate_frame": "+X along traversal, +Y away from wall, +Z up; wxyz",
        "alignment": {"yaw_degrees": float(np.rad2deg(yaw)), "fit_frames_half_open": [225, 630],
                      "translation_xy_m": [0., .15], "root_z_offset_range_m": [min(offsets), max(offsets)],
                      "height_rule": "per-frame lowest sole sphere placed on platform top",
                      "joint_trajectories_preserved": True, "timing_preserved": True},
        "geometry": {"platform_top_m": height, "lower_floor_z_m": 0.,
                     "narrow_width_m": width, "narrow_length_m": EXIT - ENTRY,
                     "start_x_m": [START, ENTRY], "narrow_x_m": [ENTRY, EXIT],
                     "goal_x_m": [EXIT, END], "landing_y_m": [REAR, FRONT],
                     "wall_face_y_m": 0., "landing_wall_face_y_m": REAR,
                     "wall_top_m": height + 1.5},
        "geometry_checks": {"all_sole_sphere_projections_supported": True,
                            "wall_contact_pairs_over_clip": wall_contacts,
                            "min_wall_contact_distance_m": float(min_wall_distance) if wall_contacts else None},
        "kinematic_only": True, "closed_loop_dynamics_validated": False,
        "continuous_back_contact_validated": False,
    }
    clip.update(kinematic_only=np.array(True), scene_root_z_offset=np.array(offsets),
                scene_xml_sha256=np.array(metadata["scene_xml_sha256"]),
                provenance=np.array(json.dumps(metadata, ensure_ascii=False)))
    output.parent.mkdir(parents=True, exist_ok=True)
    save_motion(output, clip)
    output.with_suffix(".xml").write_text(xml)
    metadata["reference_sha256"] = digest(output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {output} and paired XML/JSON; yaw={np.rad2deg(yaw):.6f} deg, height={height:g} m, width={width:g} m")


def _make_ledge_solver(height, repeat_blocks=0):
    """Share the existing G1 IK for small reference repairs."""
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    from supermimic.tasks.open_door.geometry import _surface_candidates, _forbidden_back_points

    class Solver:
        def __init__(self):
            self.model = make_g1_spec(interaction_collisions=True).compile()
            self.data = mujoco.MjData(self.model)
            self.addresses = np.array([self.model.joint(name).qposadr[0] for name in G1_JOINT_NAMES])
            self.feet = [self.model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
            self.torso = self.model.body("torso_link").id
            self.spheres = np.array([[g for g in range(self.model.ngeom)
                                     if self.model.geom_bodyid[g] == body
                                     and self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
                                    for body in self.feet])
            if self.spheres.shape != (2, 4):
                raise ValueError("Expected four G1 sole spheres per foot")
            self.data.qpos[3] = 1
            mujoco.mj_forward(self.model, self.data)
            groups, legal, ids = [], [], []
            for geom in range(self.model.ngeom):
                if not self.model.geom_conaffinity[geom] & 2:
                    continue
                points = _surface_candidates(self.model, geom)
                name = self.model.body(self.model.geom_bodyid[geom]).name
                if name in ("torso_link", "logo_link"):
                    world = points @ self.data.geom_xmat[geom].reshape(3, 3).T + self.data.geom_xpos[geom]
                    local = (world - self.data.xpos[self.torso]) @ self.data.xmat[self.torso].reshape(3, 3)
                    mask = ~_forbidden_back_points(name, "torso_link", local)
                    for allowed in (False, True):
                        selected = points[mask == allowed]
                        if not len(selected):
                            continue
                        if len(selected) > 4:
                            selected = selected[ConvexHull(selected, qhull_options="QJ").vertices]
                        groups.append(selected)
                        legal.append(allowed)
                        ids.append(geom)
                else:
                    groups.append(points[ConvexHull(points).vertices])
                    legal.append(False)
                    ids.append(geom)
            self.local = np.zeros((len(groups), max(map(len, groups)), 3))
            for index, points in enumerate(groups):
                self.local[index] = points[0]
                self.local[index, :len(points)] = points
            self.geom_ids, self.legal = np.array(ids), np.array(legal)
            self.fixed = np.zeros(17)
            limits = self.model.jnt_range[[self.model.joint(name).id for name in G1_JOINT_NAMES[:12]]]
            center, half = limits.mean(1), np.diff(limits).ravel() * .45
            self.low = np.r_[-1.5, -.3, height + .65, -.35, -.35, -.8, center - half + 1e-7]
            self.high = np.r_[2.5 + repeat_blocks * .25, 1.8, height + .90, .35, .35, 2.5, center + half - 1e-7]
            self.low[[9, 15]] = np.maximum(self.low[[9, 15]], np.deg2rad(15.))

        def forward(self, pose):
            self.data.qpos[:3] = pose[:3]
            self.data.qpos[3:7] = Rotation.from_euler("xyz", pose[3:6]).as_quat()[[3, 0, 1, 2]]
            self.data.qpos[self.addresses] = np.r_[pose[6:], self.fixed]
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

        def gaps(self):
            # All three wall sections share Y=0. Convex support extrema suffice
            # for this plane constraint; final checks use actual finite geoms.
            values = (np.einsum("gj,gkj->gk", self.data.geom_xmat[self.geom_ids, 3:6], self.local)
                      + self.data.geom_xpos[self.geom_ids, 1, None])
            return values.min(1)

        def solve(self, reference, feet, yaws, back_weight, support_weights, previous, before_previous):
            rotations = Rotation.from_euler("z", yaws[:, None]).as_matrix()
            self.forward(reference)
            torso_rotation = self.data.xmat[self.torso].reshape(3, 3).copy()
            back_target = self.gaps()[self.legal].min()

            def residual(pose):
                self.forward(pose)
                gaps = self.gaps()
                position = (self.data.xpos[self.feet] - feet).ravel()
                orientation = np.concatenate([
                    Rotation.from_matrix(rotations[k].T @ self.data.xmat[body].reshape(3, 3)).as_rotvec()
                    for k, body in enumerate(self.feet)])
                torso = Rotation.from_matrix(torso_rotation.T @ self.data.xmat[self.torso].reshape(3, 3)).as_rotvec()
                values = [10000 * position, 800 * orientation,
                          8000 * np.minimum(gaps - np.where(self.legal, -.00015, .003), 0),
                          [20000 * (.02 + .98 * back_weight) * (gaps[self.legal].min() - back_target)],
                          35 * (pose[:3] - reference[:3]), np.array([60, 60, 60]) * torso,
                          1.5 * (pose[6:] - reference[6:])]
                if previous is not None:
                    values.extend([1.5 * (pose[6:] - previous[6:]), 1.5 * (pose[3:6] - previous[3:6])])
                if before_previous is not None:
                    expected = previous + .5 * (previous - before_previous)
                    values.extend([150 * (pose[:3] - expected[:3]),
                                   20 * (pose[3:6] - expected[3:6]), 3 * (pose[6:] - expected[6:])])
                # A soft posture preference, not a proof of static balance:
                # back contact permits some COM excursion beyond foot support.
                feet_xy = self.data.geom_xpos[self.spheres, :2]
                lower = np.array([feet_xy.reshape(-1, 2).min(0),
                                  feet_xy[0].min(0), feet_xy[1].min(0)])
                upper = np.array([feet_xy.reshape(-1, 2).max(0),
                                  feet_xy[0].max(0), feet_xy[1].max(0)])
                com = self.data.subtree_com[1, :2]
                allowance = np.array([.015 * back_weight, 0.])
                values.append(120 * back_weight * np.r_[np.maximum(support_weights @ lower + .006 - allowance - com, 0),
                                          np.maximum(com - support_weights @ upper + .006 - allowance, 0)])
                return np.concatenate(values)

            seed = reference if previous is None else previous
            result = least_squares(residual, np.clip(seed, self.low + 1e-7, self.high - 1e-7),
                                   bounds=(self.low, self.high), max_nfev=100,
                                   ftol=1e-8, xtol=1e-8, gtol=2e-6)
            self.forward(result.x)
            error = float(np.max(np.linalg.norm(self.data.xpos[self.feet] - feet, axis=1)))
            if not np.isfinite(result.x).all() or error > .001:
                raise ValueError(f"video11 IK failed: maximum ankle error {error:g} m")
            return result.x, error

    return Solver()


def prepare_video11(source, output, height=1.2, width=.22, overwrite=False, repeat_blocks=4):
    """Refine the recorded video11 timing with fixed stance feet and rear contact."""
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    from supermimic.tasks.open_door.geometry import _surface_candidates, _forbidden_back_points

    source, output = Path(source).resolve(), Path(output).resolve()
    destinations = [output, output.with_suffix(".xml"), output.with_suffix(".json")]
    inputs = [source, source.with_suffix(".xml"), source.with_suffix(".json"),
              Path(str(source) + ".json")]
    if output.suffix != ".npz" or {path.resolve() for path in inputs}.intersection(path.resolve() for path in destinations):
        raise ValueError("Use a separate .npz output; source files cannot be overwritten")
    if not np.isfinite([height, width]).all() or height <= 0 or not .22 <= width <= .28:
        raise ValueError("video11 requires positive height and a width between 0.22 and 0.28 m")
    if not isinstance(repeat_blocks, int) or not 0 <= repeat_blocks <= 10:
        raise ValueError("repeat_blocks must be an integer from 0 to 10")
    if not overwrite and any(path.exists() for path in destinations):
        raise FileExistsError("Output already exists; use --overwrite to replace the derived asset")
    clip = load_motion(source)
    if (clip["joint_names"].tolist() != G1_JOINT_NAMES or len(clip["root_pos"]) != 840
            or float(clip["fps"]) != 30 or "object_pos" in clip or "door_angle" in clip):
        raise ValueError("Expected the robot-only 840-frame, 30 Hz video11 G1 export")
    source_hash = digest(source)
    generator_hash = digest(__file__)
    count = len(clip["root_pos"])
    ankle_y = min(.12, width - .133)
    original_count = count

    def smooth(value):
        value = np.clip(value, 0., 1.)
        return value ** 3 * (10 + value * (-15 + 6 * value))

    solver = _make_ledge_solver(height, repeat_blocks)
    xy = clip["root_pos"][420:700, :2]
    _, _, axes = np.linalg.svd(xy - xy.mean(0), full_matrices=False)
    tangent = axes[0] if axes[0] @ (xy[-1] - xy[0]) > 0 else -axes[0]
    yaw = -np.arctan2(tangent[1], tangent[0])
    rotation = Rotation.from_euler("z", yaw)
    matrix = rotation.as_matrix()
    foot_positions = np.empty((count, 2, 3))
    foot_rotations = np.empty((count, 2, 3, 3))
    lowest = np.empty(count)
    for frame in range(count):
        solver.data.qpos[:3] = clip["root_pos"][frame]
        solver.data.qpos[3:7] = clip["root_quat"][frame]
        solver.data.qpos[solver.addresses] = clip["joint_pos"][frame]
        mujoco.mj_forward(solver.model, solver.data)
        foot_positions[frame] = solver.data.xpos[solver.feet] @ matrix.T
        foot_rotations[frame] = matrix @ solver.data.xmat[solver.feet].reshape(2, 3, 3)
        lowest[frame] = np.min(solver.data.geom_xpos[solver.spheres, 2]
                               - solver.model.geom_size[solver.spheres, 0])
    foot_yaw = np.unwrap(Rotation.from_matrix(foot_rotations.reshape(-1, 3, 3)).as_euler("xyz")
                         .reshape(count, 2, 3)[:, :, 2], axis=0)
    # Reviewed FK step intervals from this recording. They are planned phases,
    # not sensor-derived contact labels; every frame retains at least one stance.
    swings = [
        [(56, 72), (138, 156), (177, 195), (350, 366), (394, 411), (445, 463),
         (498, 515), (544, 562), (592, 608), (641, 657), (748, 767), (790, 807)],
        [(118, 137), (157, 174), (198, 213), (367, 384), (414, 430), (467, 483),
         (516, 532), (564, 579), (610, 626), (659, 669), (770, 787), (810, 824)],
    ]
    targets, angles = np.empty_like(foot_positions), np.empty((count, 2))
    stance = np.ones((count, 2), bool)
    for side, intervals in enumerate(swings):
        starts = [0] + [end for start, end in intervals]
        ends = [start for start, end in intervals] + [count - 1]
        anchors, directions = [], []
        for start, end in zip(starts, ends):
            anchor = np.median(foot_positions[start:end + 1, side], axis=0)
            anchor[2] = height + .035
            direction = float(np.median(foot_yaw[start:end + 1, side]))
            if start < 138:
                anchor[1] += .4
            elif start < 177:
                anchor[1] = .25 if side == 0 else .15
            elif start < 748:
                anchor[1] = ankle_y
                anchor[0] += .05 if side == 0 else -.05
            else:
                anchor[0] += .18 if side == 0 else .08
                anchor[1] = .14 if start == 767 else anchor[1] + .12
            anchors.append(anchor)
            directions.append(direction)
            targets[start:end + 1, side] = anchor
            angles[start:end + 1, side] = direction
        for index, (start, end) in enumerate(intervals):
            t = np.linspace(0., 1., end - start + 1)
            weight = smooth(t)
            targets[start:end + 1, side] = anchors[index] + weight[:, None] * (anchors[index + 1] - anchors[index])
            targets[start:end + 1, side, 2] += .025 * 64 * t ** 3 * (1 - t) ** 3
            delta = (directions[index + 1] - directions[index] + np.pi) % (2 * np.pi) - np.pi
            angles[start:end + 1, side] = directions[index] + weight * delta
            stance[start + 1:end, side] = False
    if not stance.any(1).all():
        raise ValueError("video11 plan has no supporting foot")

    # Blend the posture constraint before a foot lifts and after touchdown;
    # switching its bounds with a boolean contact mask creates root jumps.
    from scipy.ndimage import gaussian_filter1d
    support_modes = np.column_stack((stance.all(1), stance[:, 0] & ~stance[:, 1],
                                     stance[:, 1] & ~stance[:, 0])).astype(float)
    support_weights = gaussian_filter1d(support_modes, 3., axis=0, mode="nearest")

    frames = np.arange(count)
    root = clip["root_pos"] @ matrix.T
    root[:, 2] += height - lowest
    root[:, 2] -= .025
    orientation = (rotation * Rotation.from_quat(clip["root_quat"][:, [1, 2, 3, 0]])).as_euler("xyz")
    references = np.c_[root, orientation, clip["joint_pos"][:, :12]]
    back_weight = smooth((frames - 177) / 36) * (1 - smooth((frames - 762) / 35))
    back_required = (frames >= 213) & (frames <= 762)
    for frame in range(count):
        solver.fixed = clip["joint_pos"][frame, 12:]
        solver.forward(references[frame])
        back_y = references[frame, 1] - solver.gaps()[solver.legal].min() - .00015
        baseline = references[frame, 1] + .4
        blend, leave = smooth((frame - 138) / 75), smooth((frame - 762) / 50)
        references[frame, 1] = (1 - blend) * baseline + blend * back_y
        references[frame, 1] = ((1 - leave) * references[frame, 1]
                                + leave * (targets[frame, :, 1].mean() - .045))
        references[frame, 0] += .13 * leave

    # Repeat two recorded step pairs, closing their small endpoint differences.
    # Foot correction follows progress along the path, so planted feet stay fixed.
    source_frames = np.arange(count)
    blend_weight = np.zeros(count)
    extension = 0.
    extension_info = None
    if repeat_blocks:
        start, end = 435, 535
        period = end - start
        stride = float(np.mean(targets[end, :, 0] - targets[start, :, 0]))
        extension = repeat_blocks * stride
        indices = np.r_[np.arange(start), np.tile(np.arange(start, end), repeat_blocks),
                        np.arange(start, count)]
        source_joints = clip["joint_pos"].copy()
        corrected_targets, corrected_angles = targets.copy(), angles.copy()
        corrected_refs, corrected_joints = references.copy(), source_joints.copy()
        reference_delta = references[start] - references[end]
        reference_delta[0] += stride
        joint_delta = source_joints[start] - source_joints[end]
        weight = smooth(np.linspace(0., 1., period + 1))
        corrected_refs[start:end + 1] += weight[:, None] * reference_delta
        corrected_joints[start:end + 1] += weight[:, None] * joint_delta
        for side in range(2):
            foot_delta = targets[start, side] - targets[end, side]
            foot_delta[0] += stride
            yaw_delta = angles[start, side] - angles[end, side]
            progress = ((targets[start:end + 1, side, 0] - targets[start, side, 0])
                        / (targets[end, side, 0] - targets[start, side, 0]))
            corrected_targets[start:end + 1, side] += progress[:, None] * foot_delta
            corrected_angles[start:end + 1, side] += progress * yaw_delta
        # Insert only the closed copies; the original suffix then resumes at
        # the same source pose, translated by the added travel distance.
        def insert(values, corrected):
            return np.concatenate((values[:start], *[corrected[start:end] for _ in range(repeat_blocks)],
                                   values[start:]))
        offsets = np.r_[np.zeros(start), np.repeat(np.arange(repeat_blocks) * stride, period),
                        np.full(count - start, extension)]
        references = insert(references, corrected_refs)
        references[:, 0] += offsets
        targets = insert(targets, corrected_targets)
        targets[:, :, 0] += offsets[:, None]
        angles = insert(angles, corrected_angles)
        stance, back_weight, back_required = stance[indices], back_weight[indices], back_required[indices]
        source_frames = indices
        blend_weight = np.r_[np.zeros(start), np.tile(weight[:-1], repeat_blocks), np.zeros(count - start)]
        clip["joint_pos"] = insert(source_joints, corrected_joints)
        count = len(indices)
        support_modes = np.column_stack((stance.all(1), stance[:, 0] & ~stance[:, 1],
                                         stance[:, 1] & ~stance[:, 0])).astype(float)
        support_weights = gaussian_filter1d(support_modes, 3., axis=0, mode="nearest")
        extension_info = {"source_frames_half_open": [start, end], "added_copies": repeat_blocks,
                          "stride_m": stride, "added_travel_m": extension,
                          "added_frames": repeat_blocks * period, "insertion_before_source_frame": start,
                          "waist_arm_closure_max_rad": float(abs(joint_delta[12:]).max()),
                          "foot_correction_rule": "path progress, constant during stance"}
        # Joint/attitude seam blending changes the back's exact support point.
        for frame in np.flatnonzero(back_required):
            solver.fixed = clip["joint_pos"][frame, 12:]
            solver.forward(references[frame])
            references[frame, 1] -= solver.gaps()[solver.legal].min() + .00015

    scene = mujoco.MjSpec.from_string(scene_xml(height, width).replace("video9_high_ledge", "video11_back_contact_ledge"))
    exit_x, end_x = .72 + extension, 1.5 + extension
    starts, ends = [-.65, .30, exit_x], [.30, exit_x, end_x]
    for name, start, end in zip(("start_platform", "narrow_ledge", "goal_platform"), starts, ends):
        geom = scene.geom(name)
        geom.pos[0], geom.size[0] = (start + end) / 2, (end - start) / 2
    for name, start, end in zip(("entry_wall", "central_wall", "exit_wall"), starts, ends):
        geom = scene.geom(name)
        geom.pos[:] = [(start + end) / 2, -.175, (height + 1.5) / 2]
        geom.size[:] = [(end - start) / 2, .175, (height + 1.5) / 2]
    scene.geom("ledge_edge_mark").pos[0], scene.geom("ledge_edge_mark").size[0] = (.30 + exit_x) / 2, (exit_x - .30) / 2
    scene.compile()
    xml = scene.to_xml()
    poses, errors, previous, before_previous = [], [], None, None
    for frame in range(count):
        solver.fixed = clip["joint_pos"][frame, 12:]
        pose, error = solver.solve(references[frame], targets[frame], angles[frame],
                                   back_weight[frame], support_weights[frame], previous, before_previous)
        poses.append(pose)
        errors.append(error)
        before_previous, previous = previous, pose
        if frame % 60 == 0:
            print(f"video11 IK {frame}/{count}: ankle error {error:.6g} m", flush=True)
    poses = np.array(poses)
    refined = {**clip, "root_pos": poses[:, :3],
               "root_quat": Rotation.from_euler("xyz", poses[:, 3:6]).as_quat()[:, [3, 0, 1, 2]],
               "joint_pos": np.c_[poses[:, 6:], clip["joint_pos"][:, 12:]],
               "planned_stance": stance, "planned_foot_positions": targets,
               "planned_foot_quat": Rotation.from_euler("z", angles.reshape(-1, 1)).as_quat()[:, [3, 0, 1, 2]].reshape(count, 2, 4),
               "back_required": back_required, "back_alignment_weight": back_weight,
               "source_frame_index": source_frames, "cycle_blend_weight": blend_weight,
               "kinematic_only": np.array(True)}

    # Verify the actual finite colliders before publishing the pair. The mesh
    # tip may overhang; heel/forefoot support spheres must stay on the top.
    spec = make_g1_spec(interaction_collisions=True)
    spec.attach(scene, prefix="ledge/", frame=spec.worldbody.add_frame())
    model = spec.compile()
    data = mujoco.MjData(model)
    addresses = [model.joint(name).qposadr[0] for name in G1_JOINT_NAMES]
    torso = model.body("torso_link").id
    feet = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
    spheres = np.array([[g for g in range(model.ngeom) if model.geom_bodyid[g] == body
                        and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE] for body in feet])
    walls = {model.geom("ledge/" + name).id for name in ("entry_wall", "central_wall", "exit_wall")}
    platforms = {model.geom("ledge/" + name).id for name in ("start_platform", "narrow_ledge", "goal_platform")}
    back_frames, deepest, stance_gap = 0, 0., 0.
    for frame in range(count):
        data.qpos[:3], data.qpos[3:7] = refined["root_pos"][frame], refined["root_quat"][frame]
        data.qpos[addresses] = refined["joint_pos"][frame]
        mujoco.mj_forward(model, data)
        legal_contacts = 0
        for contact in data.contact:
            first, second = map(int, contact.geom)
            if first in walls or second in walls:
                robot_geom = second if first in walls else first
                body = model.body(model.geom_bodyid[robot_geom]).name
                local = (contact.pos - data.xpos[torso]) @ data.xmat[torso].reshape(3, 3)
                if _forbidden_back_points(body, "torso_link", local[None])[0] or contact.dist < -.0005:
                    raise ValueError(f"Frame {frame}: forbidden or excessive wall contact")
                legal_contacts += 1
            if first in platforms or second in platforms:
                robot_geom = second if first in platforms else first
                if model.geom_bodyid[robot_geom] not in feet or contact.dist < -.0005:
                    raise ValueError(f"Frame {frame}: invalid platform contact")
            deepest = min(deepest, float(contact.dist))
        if back_required[frame] and not legal_contacts:
            raise ValueError(f"Frame {frame}: missing required back contact")
        back_frames += bool(legal_contacts and back_required[frame])
        for side in np.flatnonzero(stance[frame]):
            ids = spheres[side]
            centers, radii = data.geom_xpos[ids], model.geom_size[ids, 0]
            gap = np.abs(centers[:, 2] - radii - height).max()
            stance_gap = max(stance_gap, float(gap))
            if gap > .001:
                raise ValueError(f"Frame {frame}: heel/forefoot is not on the platform")
            for signs in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                points = centers[:, :2] + radii[:, None] * signs
                x, y = points.T
                support = ((x >= -.65) & (x <= .30) & (y >= REAR) & (y <= FRONT)
                           | (x >= .30) & (x <= exit_x) & (y >= 0) & (y <= width)
                           | (x >= exit_x) & (x <= end_x) & (y >= REAR) & (y <= FRONT))
                if not support.all():
                    raise ValueError(f"Frame {frame}: heel/forefoot extends outside the platform")
    if digest(source) != source_hash or digest(__file__) != generator_hash:
        raise ValueError("Source or generator changed during optimization")
    metadata = {
        "method": "source-timed bounded root/leg IK with locked stance feet and rear-wall contact",
        "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "source_sha256": source_hash, "generator_sha256": generator_hash,
        "robot_xml_sha256": digest(ROOT / "data/g1_description/g1_29dof.xml"),
        "scene_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "frames": count, "fps": 30., "duration_s": (count - 1) / 30,
        "coordinate_frame": "+X along traversal, +Y away from wall, +Z up; wxyz",
        "alignment": {"yaw_degrees": float(np.rad2deg(yaw)), "fit_frames_half_open": [420, 700]},
        "preserved": {"source_frames": original_count, "step_timing": True,
                      "waist_and_arm_joints": "source-mapped, with documented cycle seam blending",
                      "source_file": True, "modified": "root/legs and cycle seam blending"},
        "extension": extension_info,
        "geometry": {"platform_top_m": height, "lower_floor_z_m": 0., "narrow_width_m": width,
                     "narrow_length_m": exit_x - .30, "start_x_m": [-.65, .30], "narrow_x_m": [.30, exit_x],
                     "goal_x_m": [exit_x, end_x], "landing_y_m": [REAR, FRONT],
                     "wall_face_y_m": 0., "landing_wall_face_y_m": 0., "wall_top_m": height + 1.5},
        "planned_contact": {"source_swings_frames_closed": swings,
                            "back_required_frames_closed": [int(np.flatnonzero(back_required)[0]), int(np.flatnonzero(back_required)[-1])],
                            "back_target_gap_m": -.00015, "other_wall_clearance_m": .003,
                            "swing_height_m": .025, "stance_ankle_y_m": ankle_y,
                            "heel_and_forefoot_support_required": True, "toe_mesh_overhang_allowed": True,
                            "support_posture_smoothing_frames": 3., "com_wall_allowance_m": .015,
                            "temporal_velocity_retention": .5, "back_target_minimum_weight": .02,
                            "core_ankle_x_adjustment_m": [.05, -.05],
                            "root_height_target_adjustment_m": -.025, "minimum_knee_angle_degrees": 15.,
                            "force_labels": False, "phases_are_inferred_from_source_fk": True},
        "ik_max_ankle_error_m": max(errors), "kinematic_only": True,
        "geometry_checks": {"required_back_frames": int(back_required.sum()),
                            "frames_with_required_back_contact": int(back_frames),
                            "max_stance_sphere_height_error_m": stance_gap,
                            "max_contact_penetration_m": -deepest},
        "closed_loop_dynamics_validated": False,
    }
    source_metadata = Path(str(source) + ".json")
    if source_metadata.is_file():
        metadata["source_metadata_sha256"] = digest(source_metadata)
    refined.update(scene_xml_sha256=np.array(metadata["scene_xml_sha256"]),
                   provenance=np.array(json.dumps(metadata, ensure_ascii=False)))
    output.parent.mkdir(parents=True, exist_ok=True)
    save_motion(output, refined)
    output.with_suffix(".xml").write_text(xml)
    metadata["reference_sha256"] = digest(output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {output} and paired XML/JSON; height={height:g} m, width={width:g} m")


def repair_video11_local(source, output, overwrite=False):
    """Filter noisy arm samples, shorten waits, and adjust one exit landing."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import savgol_filter
    from scipy.spatial.transform import Rotation, Slerp
    from supermimic.tasks.open_door.geometry import _forbidden_back_points

    source, output = Path(source).resolve(), Path(output).resolve()
    inputs = [source, source.with_suffix(".xml"), source.with_suffix(".json")]
    outputs = [output, output.with_suffix(".xml"), output.with_suffix(".json")]
    if output.suffix != ".npz" or set(inputs).intersection(outputs):
        raise ValueError("Use a separate .npz output; the original reference must be preserved")
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("Output exists; use --overwrite to replace it")
    signatures = {str(path): digest(path) for path in [*inputs, Path(__file__)]}
    base = load_motion(source)
    if (len(base["root_pos"]) != 1240 or float(base["fps"]) != 30
            or base["joint_names"].tolist() != G1_JOINT_NAMES):
        raise ValueError("Local repair expects the original 1240-frame long video11 reference")
    clip = {key: value.copy() for key, value in base.items()}

    def smooth(t):
        t = np.clip(t, 0., 1.)
        return t ** 3 * (10 + t * (-15 + 6 * t))

    # Preserve the original root, legs, turn and arm poses; remove only the
    # small frame-to-frame arm noise, fading the filter at the window edges.
    frames = np.arange(1240)
    arm_frames = np.arange(55, 145)
    arm_weight = smooth((arm_frames - 55) / 10) * smooth((144 - arm_frames) / 10)
    filtered = savgol_filter(base["joint_pos"][:, 15:], 11, 3, axis=0)
    clip["joint_pos"][55:145, 15:] += (filtered[55:145] - base["joint_pos"][55:145, 15:]) * arm_weight[:, None]
    # Only the first left exit landing changes. The following original left
    # swing gradually absorbs the offset and returns to its original target.
    offset = .14 * smooth((frames - 1148) / 19) * (1 - smooth((frames - 1190) / 17))
    clip["planned_foot_positions"][:, 0, 1] += offset
    solver = _make_ledge_solver(1.2, 4)
    references = np.c_[base["root_pos"], Rotation.from_quat(base["root_quat"][:, [1, 2, 3, 0]])
                       .as_euler("xyz"), base["joint_pos"][:, :12]]
    angles = Rotation.from_quat(base["planned_foot_quat"].reshape(-1, 4)[:, [1, 2, 3, 0]])
    angles = angles.as_euler("xyz")[:, 2].reshape(-1, 2)
    stance = base["planned_stance"]
    modes = np.column_stack((stance.all(1), stance[:, 0] & ~stance[:, 1],
                             stance[:, 1] & ~stance[:, 0])).astype(float)
    weights = gaussian_filter1d(modes, 3., axis=0, mode="nearest")
    previous, before = references[1147].copy(), references[1146].copy()
    for frame in range(1148, 1208):
        solver.fixed = base["joint_pos"][frame, 12:]
        pose, _ = solver.solve(references[frame], clip["planned_foot_positions"][frame], angles[frame],
                               float(base["back_alignment_weight"][frame]), weights[frame], previous, before)
        before, previous = previous, pose
        clip["root_pos"][frame] = pose[:3]
        clip["root_quat"][frame] = Rotation.from_euler("xyz", pose[3:6]).as_quat()[[3, 0, 1, 2]]
        clip["joint_pos"][frame, :12] = pose[6:]

    # Keep the start of each weight transfer at its original speed. Shorten
    # only the subsequent double-support pause; no replacement gait is used.
    source_time = np.r_[np.arange(238), np.linspace(238, 350, 29)[:-1],
                        np.arange(350, 1089), np.linspace(1089, 1148, 16)[:-1], np.arange(1148, 1240)]
    lower = np.floor(source_time).astype(int)
    upper = np.minimum(lower + 1, 1239)
    fraction = source_time - lower
    exact = fraction == 0
    refined = {}
    for key, value in clip.items():
        if value.shape and value.shape[0] == 1240:
            if key == "root_quat":
                refined[key] = Slerp(frames, Rotation.from_quat(value[:, [1, 2, 3, 0]]))(
                    source_time).as_quat()[:, [3, 0, 1, 2]]
            elif key == "planned_foot_quat":
                refined[key] = np.stack([Slerp(frames, Rotation.from_quat(value[:, side][:, [1, 2, 3, 0]]))(
                    source_time).as_quat()[:, [3, 0, 1, 2]] for side in range(2)], axis=1)
            elif value.dtype.kind == "b":
                refined[key] = value[lower] & value[upper]
            elif value.dtype.kind in "fiu":
                w = fraction.reshape((-1,) + (1,) * (value.ndim - 1))
                refined[key] = (1 - w) * value[lower] + w * value[upper]
            else:
                refined[key] = value[lower]
            refined[key][exact] = value[lower[exact]]
        else:
            refined[key] = value
    refined["base_frame_coordinate"] = source_time
    refined["base_frame_index"] = np.where(exact, lower, -1)
    # Reject any accidental whole-segment rewrite outside the approved edits.
    untouched = exact & ~((source_time >= 55) & (source_time < 145)) & ~((source_time >= 1148) & (source_time < 1208))
    for key in ("root_pos", "root_quat", "joint_pos"):
        if not np.array_equal(refined[key][untouched], base[key][lower[untouched]]):
            raise ValueError(f"Local repair changed an unrelated {key} sample")
    if not np.array_equal(clip["root_pos"][:1148], base["root_pos"][:1148]):
        raise ValueError("Local arm filtering must not change the original entry root trajectory")

    scene = mujoco.MjSpec.from_file(str(inputs[1]))
    for name in ("start_mark", "goal_mark"):
        marker = scene.geom(name)
        if marker is not None:
            scene.delete(marker)
    scene.compile()
    xml = scene.to_xml()
    spec = make_g1_spec(interaction_collisions=True)
    foot_names = [side + "_ankle_roll_link_object_collision" for side in ("left", "right")]
    spec.add_pair(name="local_repair_foot_pair", geomname1=foot_names[0], geomname2=foot_names[1])
    spec.attach(mujoco.MjSpec.from_string(xml), prefix="ledge/", frame=spec.worldbody.add_frame())
    model = spec.compile()
    data = mujoco.MjData(model)
    addresses = [model.joint(name).qposadr[0] for name in G1_JOINT_NAMES]
    torso = model.body("torso_link").id
    foot_ids = {model.geom(name).id for name in foot_names}
    foot_bodies = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
    spheres = [[g for g in range(model.ngeom) if model.geom_bodyid[g] == body
                and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE] for body in foot_bodies]
    walls = {model.geom("ledge/" + name).id for name in ("entry_wall", "central_wall", "exit_wall")}
    tops = {model.geom("ledge/" + name).id for name in ("start_platform", "narrow_ledge", "goal_platform")}
    ground = tops | {model.geom("ledge/lower_floor").id}
    mujoco.mj_forward(model, data)
    rectangles = [(data.geom_xpos[g, :2] - model.geom_size[g, :2],
                   data.geom_xpos[g, :2] + model.geom_size[g, :2]) for g in tops]
    for frame in range(len(source_time)):
        data.qpos[:3], data.qpos[3:7] = refined["root_pos"][frame], refined["root_quat"][frame]
        data.qpos[addresses] = refined["joint_pos"][frame]
        mujoco.mj_forward(model, data)
        back_contact = False
        for contact in data.contact:
            a, b = map(int, contact.geom)
            if {a, b} == foot_ids:
                raise ValueError(f"Frame {frame}: foot meshes overlap")
            if a in walls or b in walls:
                robot = b if a in walls else a
                point = (contact.pos - data.xpos[torso]) @ data.xmat[torso].reshape(3, 3)
                if (_forbidden_back_points(model.body(model.geom_bodyid[robot]).name, "torso_link", point[None])[0]
                        or contact.dist < -.0005):
                    raise ValueError(f"Frame {frame}: forbidden or excessive wall contact")
                back_contact = True
            if a in ground or b in ground:
                robot = b if a in ground else a
                if model.geom_bodyid[robot] not in foot_bodies or contact.dist < -.001:
                    raise ValueError(f"Frame {frame}: invalid ground contact")
        if refined["back_required"][frame] and not back_contact:
            raise ValueError(f"Frame {frame}: missing back contact")
        for side in np.flatnonzero(refined["planned_stance"][frame]):
            ids = spheres[side]
            centers, radius = data.geom_xpos[ids], model.geom_size[ids, 0]
            if abs(centers[:, 2] - radius - 1.2).max() > .001:
                raise ValueError(f"Frame {frame}: supporting foot leaves the top")
            for signs in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                points = centers[:, :2] + radius[:, None] * signs
                supported = np.zeros(len(ids), bool)
                for low, high in rectangles:
                    supported |= ((points >= low) & (points <= high)).all(1)
                if not supported.all():
                    raise ValueError(f"Frame {frame}: supporting heel/forefoot leaves the platform")
    if any(digest(path) != signature for path, signature in signatures.items()):
        raise ValueError("Local repair input or generator changed during processing")
    metadata = dict(method="bounded local filtering, pause retiming and one left exit landing adjustment",
                    source=str(source), source_sha256=digest(source), input_sha256=signatures,
                    generator_sha256=digest(__file__), scene_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
                    robot_xml_sha256=digest(ROOT / "data/g1_description/g1_29dof.xml"),
                    geometry=json.loads(inputs[2].read_text())["geometry"],
                    frames=len(source_time), fps=30., duration_s=(len(source_time) - 1) / 30,
                    arm_filter=dict(source_frames_half_open=[55, 145], window=11, order=3,
                                    max_joint_change_degrees=float(np.rad2deg(abs(clip["joint_pos"][:, 15:] - base["joint_pos"][:, 15:])).max())),
                    pause_retiming=[dict(source_frames_closed=[238, 350], output_intervals=28),
                                    dict(source_frames_closed=[1089, 1148], output_intervals=15)],
                    exit_local_repair=dict(source_frames_half_open=[1148, 1208], left_landing_y_offset_m=.14),
                    unchanged_samples=int(untouched.sum()), donor_segments_used=False,
                    kinematic_only=True, closed_loop_dynamics_validated=False)
    refined.update(provenance=np.array(json.dumps(metadata, ensure_ascii=False)),
                   scene_xml_sha256=np.array(metadata["scene_xml_sha256"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    save_motion(output, refined)
    output.with_suffix(".xml").write_text(xml)
    metadata["reference_sha256"] = digest(output)
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved local repair: {output} ({len(source_time)} frames)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video11", action="store_true", help="Refine video11 root/legs for continuous rear-wall contact")
    parser.add_argument("--local-repair", action="store_true", help="Locally smooth arm noise, shorten waits and adjust one exit landing")
    parser.add_argument("--height", type=float, default=1.2, help="Top surface height above lower floor, in metres")
    parser.add_argument("--width", type=float, help="Central wall-to-edge width, in metres (video9: 0.32, video11: 0.22)")
    parser.add_argument("--repeat-blocks", type=int, default=4, help="Additional two-step-pair blocks for video11 (default: 4)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.local_repair:
        source = args.source or ROOT / "data/motions/ledge_video11/archive/long_22cm/video11_g1_ledge.npz"
        output = args.output or ROOT / "data/motions/ledge_video11/video11_g1_ledge.npz"
        repair_video11_local(source, output, args.overwrite)
    elif args.video11:
        source = args.source or ROOT / "data/motions/ledge_video11/source/video11_g1.npz"
        output = args.output or ROOT / "data/motions/ledge_video11/video11_g1_ledge.npz"
        prepare_video11(source, output, args.height, .22 if args.width is None else args.width,
                        args.overwrite, args.repeat_blocks)
    else:
        prepare(args.source or SOURCE, args.output or OUTPUT, args.height,
                .32 if args.width is None else args.width, args.overwrite)


if __name__ == "__main__":
    main()
