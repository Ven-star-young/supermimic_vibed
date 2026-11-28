"""Prepare a fresh Kimodo G1 ledge motion, then import native candidates.

python -m tools.generate_kimodo_ledge prepare data/motions/kimodo_ledge_slow_raw
# Run the existing generate_kimodo_door.py generate command inside Kimodo.
python -m tools.generate_kimodo_ledge import RAW/ledge_00.csv OUTPUT.npz
python -m tools.generate_kimodo_ledge refine CANDIDATE.npz OUTPUT.npz --template RAW/template.npz
python -m tools.generate_kimodo_ledge repair-core ACCEPTED_CORE.npz OUTPUT.npz
python -m tools.generate_kimodo_ledge complete CORE.npz OUTPUT.npz
"""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from tools.generate_kimodo_door import fixed_arms, sha256


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "data/motions/ledge_video9/video9_g1_ledge.xml"


def smooth(t):
    t = np.clip(t, 0., 1.)
    return t * t * t * (10 + t * (-15 + 6 * t))


def prepare(folder, scene, overwrite=False):
    """Author five balanced side-step cycles for sparse Kimodo conditioning."""
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation, Slerp
    from motion_data import load_motion
    from supermimic.assets.g1.g1_constants import G1_JOINT_NAMES, G1_VELOCITY_LIMITS, make_g1_spec
    from supermimic.tasks.open_door.geometry import _surface_candidates

    folder, scene = Path(folder).resolve(), Path(scene).resolve()
    outputs = ("template.npz", "input.npz", "metadata.json", "scene.xml")
    if not overwrite and any((folder / name).exists() for name in outputs):
        raise ValueError("Prepared inputs already exist; use --overwrite")
    asset = mujoco.MjSpec.from_file(str(scene))
    if list(asset.joints) or any(body.mocap for body in asset.bodies):
        raise ValueError("Expected a static ledge scene")
    # Keep the existing materials and collision categories; resize this recipe's
    # three platforms and walls in place. Input scene files are never modified.
    geometry = {
        "start_platform": ([-.5, .45, .6], [.5, .65, .6]),
        "narrow_ledge": ([.6, .15, .6], [.6, .15, .6]),
        "goal_platform": ([1.7, .45, .6], [.5, .65, .6]),
        "entry_wall": ([-.5, -.275, 1.35], [.5, .075, 1.35]),
        "central_wall": ([.6, -.175, 1.35], [.6, .175, 1.35]),
        "exit_wall": ([1.7, -.275, 1.35], [.5, .075, 1.35]),
        "ledge_edge_mark": ([.6, .29, 1.2005], [.6, .01, .0005]),
    }
    for name, (position, size) in geometry.items():
        geom = asset.geom(name)
        if geom is None or geom.type != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError(f"Expected the existing ledge box: {name}")
        geom.pos[:], geom.size[:] = position, size
    # Older accepted references include markers in their scene signature.
    # Preserve those legacy inputs; newer ledge scenes do not contain them.
    for name, x in (("start_mark", -.5), ("goal_mark", 1.7)):
        marker = asset.geom(name)
        if marker is not None:
            marker.pos[:], marker.size[:] = [x, .75, 1.2005], [.25, .18, .0005]
    asset.compile()

    fps, cycles, step, height, back_gap = 30, 5, .15, 1.2, -.0001
    ankle_spacing, swing_height, root_height = .12, .015, 1.915
    arms_source = ROOT / "data/motions/kimodo_side60_hug_elbow/side60_hug_elbow.npz"
    source = load_motion(arms_source)
    names = np.array(G1_JOINT_NAMES)
    order = [source["joint_names"].tolist().index(name) for name in names]
    arms = source["joint_pos"][0, order][15:].copy()

    def timing(value):
        return value ** 3 * (10 + value * (-15 + 6 * value))

    # Transfer COM during double support, then hold it over the stance foot
    # during each 15 cm swing. The narrower stance and lower pelvis keep the
    # knees flexed instead of approaching full extension at the open stance.
    feet_target = np.array([[.2 - ankle_spacing / 2, .115, 1.235],
                            [.2 + ankle_spacing / 2, .115, 1.235]])
    com_x = .20
    targets, centers, stance = [feet_target.copy()], [com_x], [[True, True]]
    phase_keys, phases = [0], []
    for kind, count, goal in (("shift_left", 13, .2 - ankle_spacing / 2 + .02),
                             ("right", 20, None),
                             ("shift_right", 24, .2 + ankle_spacing / 2 + step - .02),
                             ("left", 20, None), ("settle", 13, .35)):
        initial, initial_com = feet_target.copy(), com_x
        start = len(targets) - 1
        for index in range(1, count + 1):
            fraction = index / count
            weight = timing(fraction)
            feet_target, contact = initial.copy(), [True, True]
            if kind in ("right", "left"):
                side = 1 if kind == "right" else 0
                feet_target[side, 0] += step * weight
                feet_target[side, 2] += swing_height * 64 * fraction ** 3 * (1 - fraction) ** 3
                contact[side] = index == count
            else:
                com_x = initial_com + weight * (goal - initial_com)
            targets.append(feet_target.copy())
            centers.append(com_x)
            stance.append(contact)
        phase_keys.append(len(targets) - 1)
        phases.append({"kind": kind, "frames": [start, len(targets) - 1]})
    targets, centers, stance = np.array(targets), np.array(centers), np.array(stance)

    model = make_g1_spec(interaction_collisions=True).compile()
    data = mujoco.MjData(model)
    addresses = np.array([int(model.joint(name).qposadr[0]) for name in names])
    bounds = model.jnt_range[[model.joint(name).id for name in names]]
    midpoint, half = bounds.mean(1), np.diff(bounds, axis=1)[:, 0] * .45
    low = np.r_[-.1, .035, root_height - .002, -.3, -.3, 1.35, midpoint[:15] - half[:15] + 1e-7]
    high = np.r_[.65, .18, root_height + .002, .3, .3, 1.80, midpoint[:15] + half[:15] - 1e-7]
    low[[9, 15]] = np.deg2rad(20.)
    if np.any(arms < midpoint[15:] - half[15:]) or np.any(arms > midpoint[15:] + half[15:]):
        raise ValueError("The requested arm keyframe exceeds the G1 soft joint range")
    feet = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
    torso = model.body("torso_link").id
    foot_rotation = Rotation.from_euler("z", np.pi / 2).as_matrix()
    surfaces = []
    for geom in range(model.ngeom):
        if model.geom_conaffinity[geom] & 2:
            points = _surface_candidates(model, geom)
            points = points[ConvexHull(points).vertices]
            surfaces.append((geom, points, model.body(int(model.geom_bodyid[geom])).name))
    back_mask = np.array([name in ("torso_link", "logo_link") for _, _, name in surfaces])
    clearance = np.where(back_mask, back_gap, .002)
    spheres = [np.array([geom for geom in range(model.ngeom)
                        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE
                        and model.geom_bodyid[geom] == foot]) for foot in feet]
    all_sole = np.concatenate(spheres)
    joints = np.zeros(15)
    joints[[0, 6]], joints[[3, 9]], joints[[4, 10]] = -.2, .32, -.12
    seed = np.r_[.2, .075, root_height, 0., 0., np.pi / 2, joints]

    def forward(pose):
        data.qpos[:3] = pose[:3]
        data.qpos[3:7] = Rotation.from_euler("xyz", pose[3:6]).as_quat()[[3, 0, 1, 2]]
        data.qpos[addresses] = np.r_[pose[6:], arms]
        mujoco.mj_forward(model, data)

    def wall_distances():
        return np.array([(points @ data.geom_xmat[geom].reshape(3, 3).T
                          + data.geom_xpos[geom])[:, 1].min() for geom, points, _ in surfaces])

    def solve(previous, target, target_com_x):
        def residual(pose):
            forward(pose)
            position = (data.xpos[feet] - target).ravel()
            orientation = np.concatenate([
                Rotation.from_matrix(foot_rotation.T @ data.xmat[body].reshape(3, 3)).as_rotvec()
                for body in feet])
            distances = wall_distances()
            com = data.subtree_com[1]
            tilt = Rotation.from_matrix(foot_rotation.T @ data.xmat[torso].reshape(3, 3)).as_rotvec()
            return np.r_[1000 * position, 120 * orientation,
                         1500 * np.minimum(distances - clearance, 0),
                         500 * (distances[back_mask].min() - back_gap),
                         1000 * (com[0] - target_com_x), 100 * (com[1] - .13),
                         100 * (pose[2] - root_height), 5 * tilt,
                         .8 * (pose[6:] - seed[6:]), .1 * pose[3:5]]

        result = least_squares(residual, np.clip(previous, low + 1e-8, high - 1e-8),
                               bounds=(low, high), max_nfev=120,
                               ftol=2e-9, xtol=2e-9, gtol=2e-8)
        forward(result.x)
        return result.x

    def check_poses(poses, foot_targets, planned_stance):
        qpos, com, margins, errors, back, other, lowest = [], [], [], [], [], [], []
        for frame, pose in enumerate(poses):
            forward(pose)
            qpos.append(np.r_[data.qpos[:7], data.qpos[addresses]])
            com.append(data.subtree_com[1].copy())
            errors.append(float(np.abs(data.xpos[feet] - foot_targets[frame]).max()))
            distances = wall_distances()
            back.append(distances[back_mask].min())
            other.append(distances[~back_mask].min())
            active = np.concatenate([spheres[side] for side in range(2) if planned_stance[frame, side]])
            hull = ConvexHull(data.geom_xpos[active, :2])
            margin = -(hull.equations[:, :2] @ data.subtree_com[1, :2] + hull.equations[:, 2])
            margins.append(float(margin.min()))
            lowest.append(float((data.geom_xpos[all_sole, 2]
                                 - model.geom_size[all_sole, 0] - height).min()))
        report = {"foot_origin_error_max_m": max(errors),
                  "back_plane_gap_range_m": [min(back), max(back)],
                  "other_body_wall_gap_min_m": min(other),
                  "com_support_margin_min_m": min(margins),
                  "com_outside_frames": int((np.array(margins) < 0).sum()),
                  "lowest_sole_gap_range_m": [min(lowest), max(lowest)],
                  "root_z_range_m": [float(poses[:, 2].min()), float(poses[:, 2].max())]}
        return report, np.array(qpos), np.array(com), np.array(margins)

    poses, pose = [], seed.copy()
    for frame, (target, target_com) in enumerate(zip(targets, centers)):
        pose = solve(pose, target, target_com)
        poses.append(pose.copy())
        if frame % fps == 0:
            print(f"Constraint IK {frame}/{len(targets) - 1}", flush=True)
    poses = np.array(poses)
    closure = poses[-1] - poses[0]
    closure[0] -= step
    cycle_report, _, _, _ = check_poses(poses, targets, stance)
    if (np.abs(closure).max() > .002 or cycle_report["foot_origin_error_max_m"] > .001
            or cycle_report["com_outside_frames"]):
        raise RuntimeError("Authored cycle failed closure, foot-target or support checks")
    # Only optimizer null-space noise is removed, over the final double support.
    count = phase_keys[-1] - phase_keys[-2]
    for index in range(count + 1):
        poses[-count - 1 + index] -= timing(index / count) * closure
    pose_parts, target_parts, stance_parts, center_parts, keys = [], [], [], [], []
    for cycle in range(cycles):
        part, target = poses[:-1].copy(), targets[:-1].copy()
        part[:, 0] += cycle * step
        target[:, :, 0] += cycle * step
        pose_parts.append(part)
        target_parts.append(target)
        stance_parts.append(stance[:-1])
        center_parts.append(centers[:-1] + cycle * step)
        keys.extend(cycle * (len(poses) - 1) + np.array(phase_keys[:-1]))
    endpoint = poses[-1].copy()
    endpoint[0] += (cycles - 1) * step
    full = np.concatenate(pose_parts)
    full_targets, full_stance = np.concatenate(target_parts), np.concatenate(stance_parts)
    full_centers = np.concatenate(center_parts)
    keys.append(len(full) - 1)
    period = phase_keys[-1]
    fullbody_keys = np.r_[np.arange(cycles) * period, len(full) - 1]
    right_keys = np.array([0, *[period * cycle + key for cycle in range(cycles) for key in (13, 23, 33)], len(full) - 1])
    left_keys = np.array([0, *[period * cycle + key for cycle in range(cycles) for key in (57, 67, 77)], len(full) - 1])
    hand_keys = np.array([0, *[period * cycle + key for cycle in range(cycles) for key in (33, 57)], len(full) - 1])
    report, qpos, com, margin = check_poses(full, full_targets, full_stance)
    if (report["foot_origin_error_max_m"] > .001 or report["com_outside_frames"]
            or report["other_body_wall_gap_min_m"] < .001):
        raise RuntimeError("Full authored template failed geometric support checks")

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "scene.xml").write_text(asset.to_xml())
    root_velocity = np.diff(full[:, :3], axis=0) * fps
    joint_velocity = np.diff(qpos[:, 7:], axis=0) * fps
    motor_limits = np.array([G1_VELOCITY_LIMITS[name] for name in names])
    report.update({
        "method": "Authored constrained kinematic cycle for Kimodo conditioning; not a model-generated trajectory or dynamics rollout.",
        "fps": fps, "frames": len(full), "sample_window_duration_s": len(full) / fps,
        "duration_s": (len(full) - 1) / fps, "cycles": cycles, "step_per_cycle_m": step,
        "right_foot_leads": True, "standing_ankle_separation_m": ankle_spacing,
        "swing_height_m": swing_height, "nominal_root_z_m": root_height,
        "root_z_bounds_m": [root_height - .002, root_height + .002],
        "knee_hard_min_deg": 20., "period_frames": period,
        "fullbody_keyframes": fullbody_keys.tolist(), "left_foot_keyframes": left_keys.tolist(),
        "right_foot_keyframes": right_keys.tolist(), "hand_keyframes": hand_keys.tolist(),
        "segments_frames": [2 * period, 2 * period, period], "phases": phases,
        "arms_source": str(arms_source.relative_to(ROOT)), "arms_source_sha256": sha256(arms_source),
        "arms_source_frame": 0, "arm_and_wrist_joints": names[15:].tolist(),
        "preparation_script_sha256": sha256(__file__), "scene_sha256": sha256(folder / "scene.xml"),
        "geometry": {"platform_top_m": height, "width_m": .30, "length_m": 1.2,
                     "x_range_m": [0., 1.2], "wall_face_y_m": 0.,
                     "start_x_m": [-1., 0.], "goal_x_m": [1.2, 2.2],
                     "landing_y_m": [-.2, 1.1], "landing_wall_face_y_m": -.2, "wall_top_m": 2.7},
        "closure_preblend_max": float(np.abs(closure).max()),
        "root_vertical_speed_max_m_s": float(np.abs(root_velocity[:, 2]).max()),
        "root_speed_max_m_s": float(np.linalg.norm(root_velocity, axis=1).max()),
        "root_acceleration_max_m_s2": float(np.linalg.norm(np.diff(root_velocity, axis=0) * fps, axis=1).max()),
        "joint_velocity_max_limit_ratio": float((np.abs(joint_velocity) / motor_limits).max()),
        "kinematic_only": True, "dynamics_validated": False, "contact_forces_measured": False,
    })
    indices = dict(keyframes=fullbody_keys, fullbody_keyframes=fullbody_keys,
                   phase_keyframes=np.array(keys), left_foot_keyframes=left_keys,
                   right_foot_keyframes=right_keys, hand_keyframes=hand_keys,
                   foot_keyframes=np.stack([left_keys, right_keys]), segment_lengths=np.array([2 * period, 2 * period, period]))
    np.savez(folder / "template.npz", fps=np.array(fps, dtype=float), qpos=qpos,
             root_pos=qpos[:, :3], root_quat=qpos[:, 3:7], joint_pos=qpos[:, 7:], joint_names=names,
             planned_feet=full_targets, planned_foot_positions=full_targets,
             planned_foot_quat=np.tile(np.array([2 ** -.5, 0., 0., 2 ** -.5]), (len(full), 2, 1)),
             planned_stance=full_stance, planned_com=np.c_[full_centers, np.full(len(full), .13), com[:, 2]],
             actual_com=com, support_margin=margin, endpoint_pose=endpoint,
             cycle_id=np.arange(len(full)) // period,
             kinematic_only=np.array(True), provenance=np.array(json.dumps(report)), **indices)
    # Generate one normal-cadence 288-frame clip, then import it uniformly at
    # the authored playback length. Sparse EE/root constraints all originate in this same template.
    origin = np.r_[qpos[0, :2], 0.]
    generation_frames = 288
    sample_frames = np.linspace(0, len(qpos) - 1, generation_frames)
    local_qpos = np.column_stack([
        np.interp(sample_frames, np.arange(len(qpos)), qpos[:, column]) for column in range(36)])
    local_qpos[:, 3:7] = Slerp(
        np.arange(len(qpos)), Rotation.from_quat(qpos[:, [4, 5, 6, 3]]))(
            sample_frames).as_quat()[:, [3, 0, 1, 2]]
    local_qpos[:, :3] -= origin + [0., 0., height]
    index_scale = (generation_frames - 1) / (len(qpos) - 1)
    np.savez(folder / "input.npz", qpos=local_qpos, joint_names=names,
             smooth_root_2d=local_qpos[:, [1, 0]],
             keyframes=np.rint(fullbody_keys * index_scale).astype(int),
             left_foot_keyframes=np.rint(left_keys * index_scale).astype(int),
             right_foot_keyframes=np.rint(right_keys * index_scale).astype(int),
             hand_keyframes=np.rint(hand_keys * index_scale).astype(int),
             root_pose_keyframes=np.rint(np.array(keys) * index_scale).astype(int))
    prompt = 'A person carefully takes five small sideways step-together cycles to their right, without crossing their feet. Each time they shift their weight onto the left leg, step the right foot a short distance to the right, transfer their weight onto the right leg, and bring the left foot alongside. Their back remains against a wall. They keep both hands close together in front of the chest as if carrying a small box. They face away from the wall without turning and finish standing steadily.'
    metadata = {
        "fps": fps, "joint_names": names.tolist(), "world_offset": origin.tolist(),
        "platform_height_m": height, "first_heading_angle": float(np.pi / 2),
        "output_stem": "ledge_slow", "num_frames": [generation_frames], "prompts": [prompt],
        "playback_frames": len(full), "root_path_source": "converted_smooth_root", "cfg_weight": [2., 3.],
        "provenance": {**report, "recipe": "kimodo_ledge_knee_repair_v3", "model": "Kimodo-G1-RP-v1",
                       "constraints": "Authored IK template with planned feet, stance and COM; no video9 or previous generated gait is used.",
                       "template_sha256": sha256(folder / "template.npz"),
                       "constraint_sampling": f"Single {generation_frames}-frame normal-cadence generation; resample poses uniformly to {len(full)} frames at {fps} Hz. Hips constrained at phase boundaries; feet at swing events.",
                       "generation_frame": "Input root starts at XY zero, ground Z=0; world XY and platform elevation are restored on import."},
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared {generation_frames} generation frames and {len(full)} template frames in {folder}; "
          f"COM support margin {report['com_support_margin_min_m'] * 1000:.3f} mm")


def import_candidate(source, output, overwrite=False):
    from motion_data import save_motion
    from supermimic.assets.g1.g1_constants import G1_JOINT_NAMES

    sources = [source, source.with_suffix(".json"), source.with_suffix(".npz"), source.parent / "scene.xml"]
    outputs = [output, output.with_suffix(".xml"), output.with_suffix(".json")]
    if output.suffix != ".npz" or set(sources).intersection(outputs):
        raise ValueError("Use a separate output .npz; inputs cannot be overwritten")
    if not overwrite and any(path.exists() for path in outputs):
        raise ValueError("Candidate output already exists; use --overwrite")
    metadata = json.loads(source.with_suffix(".json").read_text())
    if sha256(source.parent / "scene.xml") != metadata["provenance"]["scene_sha256"]:
        raise ValueError("Scene changed since constraint preparation")
    qpos = np.loadtxt(source, delimiter=",", ndmin=2)
    names = np.array(metadata["joint_names"])
    if qpos.shape != (sum(metadata["num_frames"]), 36) or set(names) != set(G1_JOINT_NAMES):
        raise ValueError("Unexpected native G1 export dimensions or joint names")
    with np.load(source.with_suffix(".npz"), allow_pickle=False) as native:
        contacts = native["foot_contacts"]
        if (native["root_positions"].shape != (len(qpos), 3)
                or not np.allclose(native["root_positions"][:, [2, 0, 1]], qpos[:, :3], atol=1e-5, rtol=0)):
            raise ValueError("Native motion and CSV must belong to the same candidate")
    if contacts.shape != (len(qpos), 4) or not np.isin(contacts, [0, 1]).all():
        raise ValueError("Expected native [left heel, left toe, right heel, right toe] contact labels")
    playback_frames = metadata.get("playback_frames", len(qpos))
    if not isinstance(playback_frames, int) or playback_frames < 2:
        raise ValueError("playback_frames must be an integer >= 2")
    if playback_frames != len(qpos):
        from scipy.spatial.transform import Rotation, Slerp
        original_frames = len(qpos)
        times = np.linspace(0, original_frames - 1, playback_frames)
        resampled = np.column_stack([np.interp(times, np.arange(original_frames), qpos[:, column])
                                     for column in range(36)])
        resampled[:, 3:7] = Slerp(
            np.arange(original_frames), Rotation.from_quat(qpos[:, [4, 5, 6, 3]]))(
                times).as_quat()[:, [3, 0, 1, 2]]
        contacts = contacts[np.rint(times).astype(int)]
        qpos = resampled
        metadata["retiming"] = {"source_frames": original_frames, "output_frames": playback_frames,
                                "source_fps": metadata["fps"], "output_fps": metadata["fps"],
                                "duration_scale": (playback_frames - 1) / (original_frames - 1),
                                "method": "uniform time scaling; linear xyz/joints, quaternion SLERP"}
    plan = {}
    if "template_sha256" in metadata["provenance"]:
        template = source.parent / "template.npz"
        if sha256(template) != metadata["provenance"]["template_sha256"]:
            raise ValueError("Conditioning template changed since generation")
        with np.load(template, allow_pickle=False) as data:
            for key in ("planned_stance", "planned_foot_positions", "planned_foot_quat", "planned_com"):
                if len(data[key]) != len(qpos):
                    raise ValueError("Conditioning template and candidate frame counts differ")
                plan[key] = data[key].copy()
        cycle_frames = len(qpos) // int(metadata["provenance"]["cycles"])
        plan["cycle_id"] = np.arange(len(qpos)) // cycle_frames
        sources.append(template)
    metadata["source_hashes"] = {path.name: sha256(path) for path in sources}
    metadata["import_script_sha256"] = sha256(__file__)
    offset = np.asarray(metadata["world_offset"]).copy()
    offset[2] += metadata["platform_height_m"]
    save_motion(output, dict(fps=np.float64(metadata["fps"]), root_pos=qpos[:, :3] + offset,
                            root_quat=qpos[:, 3:7], joint_pos=qpos[:, 7:], joint_names=names,
                            native_foot_contacts=contacts.astype(bool), kinematic_only=np.array(True),
                            scene_xml_sha256=np.array(sha256(source.parent / "scene.xml")),
                            provenance=np.array(json.dumps(metadata)), **plan))
    shutil.copy2(source.parent / "scene.xml", output.with_suffix(".xml"))
    metadata["reference_sha256"] = sha256(output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Imported {output}")


def refine(source, output, template_path, overwrite=False, *, _repair_style=False):
    """Project native Kimodo residuals onto the authored foot and COM plan."""
    import time
    import mujoco
    from scipy.ndimage import gaussian_filter1d
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation, Slerp
    from motion_data import load_motion, save_motion
    from supermimic.assets.g1.g1_constants import G1_JOINT_NAMES, make_g1_spec
    from supermimic.tasks.open_door.geometry import _surface_candidates, _forbidden_back_points

    candidate_path, template_path, output = [Path(path).resolve() for path in
                                           (source, template_path, output)]
    scene = candidate_path.with_suffix(".xml")
    arms_source = ROOT / "data/motions/kimodo_side60_hug_elbow/side60_hug_elbow.npz"
    inputs = {path.resolve() for path in (candidate_path, candidate_path.with_suffix(".json"),
              scene, template_path, template_path.parent / "scene.xml",
              template_path.parent / "metadata.json", arms_source, Path(__file__))}
    outputs = {path.resolve() for path in (output, output.with_suffix(".xml"), output.with_suffix(".json"))}
    if output.suffix != ".npz" or inputs.intersection(outputs):
        raise ValueError("Use a separate output NPZ/XML/JSON; inputs cannot be overwritten")
    if not overwrite and any(path.exists() for path in outputs):
        raise ValueError("Output exists; use --overwrite")
    candidate, template = load_motion(candidate_path), load_motion(template_path)
    if "postprocess" in template or (not _repair_style and "postprocess" in candidate):
        raise ValueError("Expected an unprocessed Kimodo candidate and an authored template")
    scene_hash, template_hash = sha256(scene), sha256(template_path)
    if str(candidate.get("scene_xml_sha256", "")) != scene_hash:
        raise ValueError("Candidate scene hash does not match its paired XML")
    candidate_report = json.loads(str(candidate["provenance"]))
    generation = candidate_report.get("provenance", {})
    template_report = json.loads(str(template["provenance"]))
    if _repair_style:
        if "postprocess" not in candidate:
            raise ValueError("repair-core requires the previously accepted core")
        old_template_path = Path(candidate_report["template"])
        if not old_template_path.is_absolute():
            old_template_path = ROOT / old_template_path
        if sha256(old_template_path) != candidate_report["template_sha256"]:
            raise ValueError("Accepted core's original template has changed")
        if old_template_path.resolve() in outputs:
            raise ValueError("Repair cannot overwrite the accepted core's template")
        old_template = load_motion(old_template_path)
        old_template_report = json.loads(str(old_template["provenance"]))
        if old_template_report.get("scene_sha256") != scene_hash:
            raise ValueError("Accepted core and its original template use different scenes")
    elif generation.get("template_sha256") != template_hash:
        raise ValueError("Candidate was not generated from this template")
    if ((not _repair_style and generation.get("scene_sha256") != scene_hash)
            or template_report.get("scene_sha256") != scene_hash):
        raise ValueError("Candidate and template must use the same scene")
    for clip in ((candidate, template, old_template) if _repair_style else (candidate, template)):
        if set(clip["joint_names"].tolist()) != set(G1_JOINT_NAMES):
            raise ValueError("Expected the 29 named G1 joints")
        order = [clip["joint_names"].tolist().index(name) for name in G1_JOINT_NAMES]
        clip["joint_pos"] = clip["joint_pos"][:, order]
        clip["joint_names"] = np.array(G1_JOINT_NAMES)
    count, fps = len(template["root_pos"]), float(template["fps"])
    if (count < 2 or fps != 30 or float(candidate["fps"]) != fps
            or (not _repair_style and len(candidate["root_pos"]) != count)
            or template_report.get("frames") != count):
        raise ValueError("Expected a matching 30 Hz template and source timing")
    root_z_bounds = np.asarray(template_report["root_z_bounds_m"], dtype=float)
    nominal_root_z = float(template_report["nominal_root_z_m"])
    if (root_z_bounds.shape != (2,) or not np.isfinite(root_z_bounds).all()
            or not root_z_bounds[0] <= nominal_root_z <= root_z_bounds[1]):
        raise ValueError("Invalid template root-height bounds")
    expected_geometry = {"platform_top_m": 1.2, "width_m": .3,
                         "length_m": 1.2, "x_range_m": [0, 1.2], "wall_face_y_m": 0}
    if any(template_report.get("geometry", {}).get(key) != value for key, value in expected_geometry.items()):
        raise ValueError("Unexpected template geometry")
    targets, planned_stance = template["planned_feet"], template["planned_stance"]
    com_targets = template["planned_com"]
    if (targets.shape != (count, 2, 3) or planned_stance.shape != (count, 2)
            or not np.isin(planned_stance, [0, 1]).all()
            or not planned_stance.any(1).all() or com_targets.shape != (count, 3)):
        raise ValueError("Invalid template foot/COM targets or unsupported template frames")
    stance = planned_stance.astype(bool)
    if not np.isfinite(targets).all() or not np.isfinite(com_targets).all():
        raise ValueError("Nonfinite template targets")
    foot_quat = template["planned_foot_quat"]
    if (foot_quat.shape != (count, 2, 4) or not np.isfinite(foot_quat).all()
            or not np.allclose(np.linalg.norm(foot_quat, axis=2), 1., atol=1e-8, rtol=0)):
        raise ValueError("Invalid planned foot orientations")
    foot_rotations = Rotation.from_quat(
        foot_quat.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(count, 2, 3, 3)
    native_contacts = candidate.get("native_foot_contacts")
    if not _repair_style and (native_contacts is None or native_contacts.shape != (count, 4)
            or not np.isin(native_contacts, [0, 1]).all()):
        raise ValueError("Expected the candidate's native heel/toe contact labels")

    # Check actual collision surfaces as well as provenance: the optimizer uses
    # their front Y planes and assumes these fixed, axis-aligned platform boxes.
    asset = mujoco.MjSpec.from_file(str(scene)).compile()
    asset_data = mujoco.MjData(asset)
    mujoco.mj_kinematics(asset, asset_data)
    if asset.nq or asset.nmocap:
        raise ValueError("Expected a static ledge scene")
    geometry = {
        "start_platform": ([-.5, .45, .6], [.5, .65, .6], 1),
        "narrow_ledge": ([.6, .15, .6], [.6, .15, .6], 1),
        "goal_platform": ([1.7, .45, .6], [.5, .65, .6], 1),
        "entry_wall": ([-.5, -.275, 1.35], [.5, .075, 1.35], 2),
        "central_wall": ([.6, -.175, 1.35], [.6, .175, 1.35], 2),
        "exit_wall": ([1.7, -.275, 1.35], [.5, .075, 1.35], 2),
    }
    for name, (position, size, category) in geometry.items():
        geom = asset.geom(name)
        if (geom.type[0] != mujoco.mjtGeom.mjGEOM_BOX
                or not np.allclose(asset_data.geom_xpos[geom.id], position, atol=1e-9, rtol=0)
                or not np.allclose(asset_data.geom_xmat[geom.id].reshape(3, 3), np.eye(3), atol=1e-9, rtol=0)
                or not np.allclose(geom.size, size, atol=1e-9, rtol=0)
                or geom.contype[0] != category or geom.conaffinity[0] != 0):
            raise ValueError(f"Unsupported ledge geometry: {name}")
    floor = asset.geom("lower_floor")
    collision_names = {asset.geom(index).name for index in range(asset.ngeom)
                       if asset.geom_contype[index] or asset.geom_conaffinity[index]}
    if (collision_names != set(geometry) | {"lower_floor"}
            or floor.type[0] != mujoco.mjtGeom.mjGEOM_PLANE
            or not np.allclose(asset_data.geom_xpos[floor.id], 0., atol=1e-9, rtol=0)
            or not np.allclose(asset_data.geom_xmat[floor.id].reshape(3, 3), np.eye(3), atol=1e-9, rtol=0)
            or floor.contype[0] != 1 or floor.conaffinity[0] != 0):
        raise ValueError("Unsupported floor or additional collision geometry")

    recorded_arms = Path(template_report["arms_source"])
    if not recorded_arms.is_absolute():
        recorded_arms = ROOT / recorded_arms
    if (recorded_arms.resolve() != arms_source.resolve()
            or template_report.get("arms_source_sha256") != sha256(arms_source)
            or template_report.get("arms_source_frame") != 0):
        raise ValueError("Template arm source or its recorded hash does not match")
    arm_clip = load_motion(arms_source)
    arm_order = [arm_clip["joint_names"].tolist().index(name) for name in G1_JOINT_NAMES]
    arms = template["joint_pos"][0, 15:].copy()
    if (not np.allclose(template["joint_pos"][:, 15:], arms, atol=1e-12, rtol=0)
            or not np.allclose(arms, arm_clip["joint_pos"][0, arm_order][15:], atol=1e-12, rtol=0)):
        raise ValueError("Template arms do not match the specified source frame")

    model = make_g1_spec(interaction_collisions=True).compile()
    data = mujoco.MjData(model)
    addresses = np.array([int(model.joint(name).qposadr[0]) for name in G1_JOINT_NAMES])
    bounds = model.jnt_range[[model.joint(name).id for name in G1_JOINT_NAMES]]
    midpoint, half = bounds.mean(1), np.diff(bounds, axis=1)[:, 0] * .45
    model_low = np.r_[np.full(6, -np.inf), midpoint[:15] - half[:15] + 1e-7]
    model_high = np.r_[np.full(6, np.inf), midpoint[:15] + half[:15] - 1e-7]
    knee_lower = float(template_report.get("knee_hard_min_deg", 0.))
    if not np.isfinite(knee_lower) or knee_lower < 0:
        raise ValueError("Invalid template knee minimum")
    model_low[[9, 15]] = np.maximum(model_low[[9, 15]], np.deg2rad(knee_lower))
    if np.any(arms < midpoint[15:] - half[15:]) or np.any(arms > midpoint[15:] + half[15:]):
        raise ValueError("Template arm pose exceeds the soft joint limits")
    feet = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
    torso = model.body("torso_link").id
    collision_surfaces = []
    for geom in range(model.ngeom):
        if model.geom_conaffinity[geom] & 2:
            points = _surface_candidates(model, geom)
            points = points[ConvexHull(points).vertices]
            collision_surfaces.append((geom, points, model.body(int(model.geom_bodyid[geom])).name))
    sole_spheres = [np.array([geom for geom in range(model.ngeom)
                            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_SPHERE
                            and model.geom_bodyid[geom] == foot]) for foot in feet]
    if any(len(group) != 4 for group in sole_spheres):
        raise ValueError("Expected four production sole spheres per foot")

    def forward(values):
        data.qpos[:3] = values[:3]
        data.qpos[3:7] = Rotation.from_euler("xyz", values[3:6]).as_quat()[[3, 0, 1, 2]]
        data.qpos[addresses] = np.r_[values[6:], arms]
        mujoco.mj_forward(model, data)

    def vector(clip):
        angles = Rotation.from_quat(clip["root_quat"][:, [1, 2, 3, 0]]).as_euler("xyz")
        return np.c_[clip["root_pos"], angles, clip["joint_pos"][:, :15]]


    def stats(values):
        values = np.asarray(values)
        return {"mean": float(values.mean()), "p95": float(np.quantile(values, .95)),
                "max": float(values.max())}


    def differences(first, second):
        position = first["root_pos"] - second["root_pos"]
        joints = first["joint_pos"] - second["joint_pos"]
        first_rotation = Rotation.from_quat(first["root_quat"][:, [1, 2, 3, 0]])
        second_rotation = Rotation.from_quat(second["root_quat"][:, [1, 2, 3, 0]])
        return {
            "root_position_distance_m": stats(np.linalg.norm(position, axis=1)),
            "root_position_max_abs_xyz_m": np.abs(position).max(0).tolist(),
            "root_rotation_distance_rad": stats((second_rotation.inv() * first_rotation).magnitude()),
            "joint_max_abs_per_joint_rad": np.abs(joints).max(0).tolist(),
            "joint_abs_rad": stats(np.abs(joints)),
            "leg_joint_rms_rad": float(np.sqrt(np.mean(joints[:, :12] ** 2))),
            "waist_joint_rms_rad": float(np.sqrt(np.mean(joints[:, 12:15] ** 2))),
            "arm_joint_rms_rad": float(np.sqrt(np.mean(joints[:, 15:] ** 2))),
        }

    if _repair_style:
        accepted = candidate
        phases = template_report["phases"]
        old_phases = old_template_report["phases"]
        if [phase["kind"] for phase in phases] != [phase["kind"] for phase in old_phases]:
            raise ValueError("Repair and accepted core must share the same gait phases")
        phase_knots = np.array([phases[0]["frames"][0], *[phase["frames"][1] for phase in phases]])
        old_knots = np.array([old_phases[0]["frames"][0], *[phase["frames"][1] for phase in old_phases]])
        period, old_period = int(phase_knots[-1]), int(old_knots[-1])
        cycles = int(template_report["cycles"])
        if (period <= 0 or old_period <= 0 or count != cycles * period
                or len(accepted["root_pos"]) != cycles * old_period
                or len(old_template["root_pos"]) != len(accepted["root_pos"])):
            raise ValueError("Accepted core and repair template have incompatible cycles")
        # Carry only the accepted root/waist style, aligned by support/swing
        # phase. The old near-straight leg trajectory is neither goal nor seed.
        source_residual = vector(accepted) - vector(old_template)
        source_residual[:, 3:6] = np.arctan2(np.sin(source_residual[:, 3:6]),
                                            np.cos(source_residual[:, 3:6]))
        source_residual[:, 6:18] = 0.
        frames = np.arange(count)
        style_samples = ((frames // period) * old_period
                         + np.interp(frames % period, phase_knots, old_knots))
        old_count = len(accepted["root_pos"])
        style_samples = np.minimum(style_samples, old_count - 1)
        mapped = np.column_stack([np.interp(style_samples, np.arange(old_count), source_residual[:, joint])
                                  for joint in range(21)])
        styled_values = vector(template) + mapped
        candidate = {key: value.copy() for key, value in template.items()}
        candidate["root_pos"] = styled_values[:, :3]
        candidate["root_quat"] = Rotation.from_euler("xyz", styled_values[:, 3:6]).as_quat()[:, [3, 0, 1, 2]]
        candidate["joint_pos"] = np.c_[styled_values[:, 6:], np.tile(arms, (count, 1))]
        candidate["provenance"] = np.array(json.dumps({
            "method": "Authored style target for kinematic repair; not a model export.",
            "source_core": str(candidate_path), "source_core_sha256": sha256(candidate_path),
            "source_core_provenance": candidate_report,
            "source_template": str(old_template_path),
            "style_fields": "root XYZ/RPY and three waist joints; leg residuals excluded",
            "phase_mapping_old": old_knots.tolist(), "phase_mapping_new": phase_knots.tolist(),
        }))
        retimed_source = {
            "root_pos": np.column_stack([np.interp(style_samples, np.arange(old_count), accepted["root_pos"][:, joint])
                                          for joint in range(3)]),
            "joint_pos": np.column_stack([np.interp(style_samples, np.arange(old_count), accepted["joint_pos"][:, joint])
                                           for joint in range(29)]),
            "root_quat": Slerp(np.arange(old_count), Rotation.from_quat(accepted["root_quat"][:, [1, 2, 3, 0]]))(
                style_samples).as_quat()[:, [3, 0, 1, 2]],
        }

    template_poses, candidate_poses = vector(template), vector(candidate)
    # Classify actual torso/logo surface points in the torso frame. Target the
    # legal logo surface specifically, so lower torso contact cannot substitute
    # for the accepted rear patch.
    forward(template_poses[0])
    surfaces, surface_legal, logo_groups = [], [], []
    for geom, old_points, name in collision_surfaces:
        points = _surface_candidates(model, geom)
        if name in ("torso_link", "logo_link"):
            world = points @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            local = (world - data.xpos[torso]) @ data.xmat[torso].reshape(3, 3)
            forbidden = _forbidden_back_points(name, "torso_link", local)
            for legal, mask in ((False, forbidden), (True, ~forbidden)):
                selected = points[mask]
                if not len(selected):
                    continue
                selected = selected[ConvexHull(selected).vertices]
                surfaces.append((geom, selected))
                surface_legal.append(legal)
                if legal and name == "logo_link":
                    logo_groups.append(len(surfaces) - 1)
        else:
            surfaces.append((geom, old_points))
            surface_legal.append(False)
    surface_legal = np.array(surface_legal)
    logo_groups = np.array(logo_groups)
    if not len(logo_groups):
        raise ValueError("Robot collision mesh has no legal logo contact patch")
    back_target = -.00015
    clearance = np.where(surface_legal, back_target, .0025)

    def wall_distances():
        return np.array([(points @ data.geom_xmat[geom].reshape(3, 3).T
                          + data.geom_xpos[geom])[:, 1].min() for geom, points in surfaces])

    residuals = candidate_poses - template_poses
    residuals[:, 3:6] = np.arctan2(np.sin(residuals[:, 3:6]), np.cos(residuals[:, 3:6]))
    # Smooth candidate residuals before optimization. The authored steps, foot
    # anchors and COM transfer remain the hard geometric plan throughout.
    residuals = gaussian_filter1d(residuals, sigma=5., axis=0, mode="nearest")
    limits = np.r_[.025, .015, .0006, .05, .05, .04, np.full(15, .15)]
    step_limits = np.r_[.00075, .00075, .00015, np.full(3, .002), np.full(15, .006)]
    desired = template_poses + np.clip(residuals, -limits, limits)
    output_poses = np.zeros_like(template_poses)
    foot_errors = np.zeros((count, 2))
    rotation_errors = np.zeros((count, 2))
    margins, back_gaps, other_gaps, actual_com = [], [], [], []
    previous_residual = np.zeros(21)
    started = time.monotonic()
    for frame in range(count):
        base, goal = template_poses[frame], desired[frame]
        lower, upper = model_low.copy(), model_high.copy()
        lower[:6], upper[:6] = base[:6] - limits[:6], base[:6] + limits[:6]
        lower[2], upper[2] = max(lower[2], root_z_bounds[0]), min(upper[2], root_z_bounds[1])
        # Bound changes of every residual relative to the already feasible
        # template; no per-frame jump to a different IK branch is allowed.
        if frame:
            lower = np.maximum(lower, base + previous_residual - step_limits)
            upper = np.minimum(upper, base + previous_residual + step_limits)
        goal = np.clip(goal, lower + 1e-8, upper - 1e-8)
        forward(goal)
        torso_goal = data.xmat[torso].reshape(3, 3).copy()

        def residual(values):
            forward(values)
            position_error = (data.xpos[feet] - targets[frame]).ravel()
            angle_error = np.concatenate([
                Rotation.from_matrix(foot_rotations[frame, side].T @
                                     data.xmat[body].reshape(3, 3)).as_rotvec()
                for side, body in enumerate(feet)])
            distance = wall_distances()
            com_error = data.subtree_com[1, :2] - com_targets[frame, :2]
            torso_error = Rotation.from_matrix(
                torso_goal.T @ data.xmat[torso].reshape(3, 3)).as_rotvec()
            delta = values - base
            return np.r_[
                2000 * position_error, 200 * angle_error,
                3000 * np.minimum(distance - clearance, 0),
                2000 * (distance[logo_groups].min() - back_target),
                np.array([1500., 500.]) * com_error,
                300 * (values[2] - goal[2]),
                8 * (values[:2] - goal[:2]), 12 * (values[3:6] - goal[3:6]),
                20 * torso_error, 3 * (values[6:] - goal[6:]),
                1500 * (delta[:3] - previous_residual[:3]),
                35 * (delta[3:6] - previous_residual[3:6]),
                8 * (delta[6:] - previous_residual[6:]),
            ]

        seed = np.clip(base + previous_residual, lower + 1e-8, upper - 1e-8)
        result = least_squares(residual, seed, bounds=(lower, upper), max_nfev=80,
                               ftol=2e-9, xtol=2e-9, gtol=2e-8)
        if not np.isfinite(result.x).all():
            raise ValueError(f"Frame {frame}: nonfinite projection")
        output_poses[frame] = result.x
        previous_residual = result.x - base
        forward(result.x)
        foot_errors[frame] = np.linalg.norm(data.xpos[feet] - targets[frame], axis=1)
        rotation_errors[frame] = [
            Rotation.from_matrix(foot_rotations[frame, side].T @
                                 data.xmat[body].reshape(3, 3)).magnitude()
            for side, body in enumerate(feet)]
        distance = wall_distances()
        back_gaps.append(float(distance[logo_groups].min()))
        other_gaps.append(float(distance[~surface_legal].min()))
        com = data.subtree_com[1].copy()
        actual_com.append(com)
        active = np.concatenate([sole_spheres[side] for side in range(2) if stance[frame, side]])
        hull = ConvexHull(data.geom_xpos[active, :2])
        margins.append(float(-(hull.equations[:, :2] @ com[:2] + hull.equations[:, 2]).max()))
        if foot_errors[frame].max() > .001 or margins[-1] < 0 or other_gaps[-1] < .002:
            raise ValueError(f"Frame {frame}: rejected geometry; foot={foot_errors[frame].max():.6f} m, "
                             f"COM margin={margins[-1]:.6f} m, other wall gap={other_gaps[-1]:.6f} m")
        if abs(back_gaps[-1] - back_target) > .0001:
            raise ValueError(f"Frame {frame}: lost planned back contact")
        if frame % 60 == 0:
            print(f"Projection {frame}/{count}: foot={foot_errors[:frame + 1].max():.6f} m, "
                  f"COM margin={min(margins):.6f} m, elapsed={time.monotonic() - started:.1f}s", flush=True)

    final = {key: value.copy() for key, value in candidate.items()}
    final["root_pos"] = output_poses[:, :3]
    final["root_quat"] = Rotation.from_euler("xyz", output_poses[:, 3:6]).as_quat()[:, [3, 0, 1, 2]]
    final["joint_pos"] = np.c_[output_poses[:, 6:], np.tile(arms, (count, 1))]
    from_candidate, from_template = differences(final, candidate), differences(final, template)
    root_residual = final["root_pos"] - template["root_pos"]
    near_template = bool(from_template["root_position_distance_m"]["max"] < .002
                         and from_template["root_rotation_distance_rad"]["max"] < np.deg2rad(.5)
                         and max(from_template["joint_max_abs_per_joint_rad"]) < .01)
    report = {
        "method": "Constrained IK projection of a Kimodo candidate onto authored foot/COM plans, with smoothed bounded candidate residuals and fixed source arms. Not a native Kimodo output.",
        "candidate": str(candidate_path), "candidate_sha256": sha256(candidate_path),
        "template": str(template_path), "template_sha256": sha256(template_path),
        "scene_sha256": sha256(scene), "script_sha256": sha256(__file__),
        "arms_source": str(arms_source.relative_to(ROOT)), "arms_source_sha256": sha256(arms_source),
        "robot_xml_sha256": sha256(ROOT / "data/g1_description/g1_29dof.xml"),
        "candidate_provenance": json.loads(str(candidate["provenance"])),
        "template_provenance": template_report,
        "frames": count, "fps": fps,
        "parameters": {"candidate_residual_gaussian_sigma_frames": 5.,
                       "maximum_template_root_delta_xyz_m": limits[:3].tolist(),
                       "maximum_template_root_rpy_delta_rad": limits[3:6].tolist(),
                       "root_z_residual_max_frame_change_m": .00015,
                       "root_xy_residual_max_frame_change_m": .00075,
                       "root_rpy_residual_max_frame_change_rad": .002,
                       "joint_residual_max_frame_change_rad": .006,
                       "legal_back_patch_torso_frame": "x<-.025, abs(y)<.13, .05<z<.38",
                       "contact_target": "legal logo surface", "back_target_gap_m": back_target,
                       "forbidden_surface_target_clearance_m": .0025,
                       "joint_soft_reference_residual_clip_rad": .15,
                       "arms_fixed_to_source_frame": True},
        "final_vs_candidate": from_candidate, "final_vs_template": from_template,
        "near_pure_template": bool(near_template),
        "near_pure_template_definition": "All frames: root distance <2mm, root rotation <0.5deg, and every joint difference <0.01rad relative to template.",
        "root_z_range_m": [float(final["root_pos"][:, 2].min()), float(final["root_pos"][:, 2].max())],
        "root_z_speed_max_m_s": float(np.abs(np.diff(final["root_pos"][:, 2]) * fps).max()),
        "root_z_residual_speed_max_m_s": float(np.abs(np.diff(root_residual[:, 2]) * fps).max()),
        "foot_target_error_max_m": float(foot_errors.max()),
        "foot_rotation_error_max_rad": float(rotation_errors.max()),
        "com_support_margin_min_m": min(margins), "other_body_wall_gap_min_m": min(other_gaps),
        "back_plane_gap_range_m": [min(back_gaps), max(back_gaps)],
        "kinematic_only": True, "dynamics_validated": False,
    }
    if _repair_style:
        report["method"] = ("Kinematic repair of the previously accepted core: new 15-second foot/COM plan, "
                            "fixed source arms and bounded accepted root/waist style residuals. No new Kimodo generation.")
        report["source_core"] = report.pop("candidate")
        report["source_core_sha256"] = report.pop("candidate_sha256")
        report["style_target_provenance"] = report.pop("candidate_provenance")
        report["final_vs_style_target"] = report.pop("final_vs_candidate")
        report["final_vs_retimed_source_core"] = differences(final, retimed_source)
        report["parameters"].update(knee_hard_min_deg=knee_lower,
                                     style_fields="root and waist only; old leg residuals excluded")

    final.update(
        postprocess=np.array(report["method"]), postprocess_stance=stance,
        postprocess_foot_targets=targets, postprocess_foot_target_quat=template["planned_foot_quat"],
        postprocess_com_targets=com_targets, postprocess_actual_com=np.array(actual_com),
        postprocess_ik_error=foot_errors, postprocess_foot_rotation_error=rotation_errors,
        postprocess_support_margin=np.array(margins),
        postprocess_candidate_root_delta=final["root_pos"] - candidate["root_pos"],
        postprocess_template_root_delta=root_residual,
        provenance=np.array(json.dumps(report)), kinematic_only=np.array(True))
    if _repair_style:
        final.pop("postprocess_candidate_root_delta")
        final.update(qpos=np.c_[final["root_pos"], final["root_quat"], final["joint_pos"]],
                     scene_xml_sha256=np.array(scene_hash), source_core_sha256=np.array(sha256(candidate_path)),
                     cycle_id=np.arange(count) // period, actual_com=np.array(actual_com),
                     support_margin=np.array(margins), style_source_sample_frames=style_samples,
                     postprocess_style_target_root_delta=final["root_pos"] - candidate["root_pos"],
                     postprocess_source_core_root_delta=final["root_pos"] - retimed_source["root_pos"])
    output.parent.mkdir(parents=True, exist_ok=True)
    save_motion(output, final)
    shutil.copy2(scene, output.with_suffix(".xml"))
    report["reference_sha256"] = sha256(output)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "near_pure_template": near_template,
                      "final_vs_candidate": from_candidate, "final_vs_template": from_template,
                      "foot_target_error_max_m": float(foot_errors.max())}, indent=2), flush=True)



def repair_core(source, output, overwrite=False):
    """Rebuild a bent-knee core while retaining the accepted root/waist style."""
    from motion_data import load_motion

    source, output = Path(source).resolve(), Path(output).resolve()
    inputs = {source, source.with_suffix(".xml"), source.with_suffix(".json")}
    outputs = {output, output.with_suffix(".xml"), output.with_suffix(".json")}
    if output.suffix != ".npz" or inputs.intersection(outputs):
        raise ValueError("repair-core needs a separate output NPZ/XML/JSON")
    if not overwrite and any(path.exists() for path in outputs):
        raise ValueError("Output exists; use --overwrite")
    accepted = load_motion(source)
    if "postprocess" not in accepted:
        raise ValueError("repair-core requires the previously accepted core")
    provenance = json.loads(str(accepted["provenance"]))
    old_template = Path(provenance["template"])
    if not old_template.is_absolute():
        old_template = ROOT / old_template
    if sha256(old_template) != provenance["template_sha256"]:
        raise ValueError("Accepted core's original template has changed")
    if old_template.resolve() in outputs:
        raise ValueError("Repair cannot overwrite its source template")
    if str(accepted.get("scene_xml_sha256", "")) != sha256(source.with_suffix(".xml")):
        raise ValueError("Accepted core's scene has changed")
    folder = output.parent / (output.stem + "_constraints")
    generated = {folder / name for name in ("template.npz", "input.npz", "scene.xml", "metadata.json")}
    if generated.intersection(inputs | outputs | {old_template.resolve()}):
        raise ValueError("Repair constraint outputs overlap an input")
    prepare(folder, source.with_suffix(".xml"), overwrite)
    refine(source, output, folder / "template.npz", overwrite, _repair_style=True)


def _full_ledge_plan(core_path):
    """Return the reviewed full-route foot/COM plan and its provenance.

    The central 450 samples are read through production FK. Entry and exit
    reuse the reviewed 90-frame cycles; this helper does not solve new poses.
    """
    import mujoco
    from scipy.spatial.transform import Rotation, Slerp
    from motion_data import load_motion
    from supermimic.assets.g1.g1_constants import G1_JOINT_NAMES, make_g1_spec

    core_path = Path(core_path).resolve()
    core = load_motion(core_path)
    if (len(core['root_pos']) != 450 or float(core['fps']) != 30
            or set(core['joint_names'].tolist()) != set(G1_JOINT_NAMES)):
        raise ValueError('Full-route planning requires a 450-frame, 30 Hz G1 core')
    stance_labels = core.get('planned_stance')
    if (stance_labels is None or stance_labels.shape != (450, 2)
            or not np.isin(stance_labels, [0, 1]).all() or not stance_labels.any(1).all()):
        raise ValueError('Core needs a valid supporting-foot plan on every frame')
    core['planned_stance'] = stance_labels.astype(bool)
    scene = core_path.with_suffix('.xml')
    if str(core.get('scene_xml_sha256', '')) != sha256(scene):
        raise ValueError('Core and paired ledge scene do not match')
    asset = mujoco.MjSpec.from_file(str(scene)).compile()
    asset_data = mujoco.MjData(asset)
    mujoco.mj_kinematics(asset, asset_data)
    if asset.nq or asset.nmocap:
        raise ValueError('Full-route planning requires a fixed ledge scene')
    expected_geometry = {
        'start_platform': ([-.5, .45, .6], [.5, .65, .6], 1),
        'narrow_ledge': ([.6, .15, .6], [.6, .15, .6], 1),
        'goal_platform': ([1.7, .45, .6], [.5, .65, .6], 1),
        'entry_wall': ([-.5, -.275, 1.35], [.5, .075, 1.35], 2),
        'central_wall': ([.6, -.175, 1.35], [.6, .175, 1.35], 2),
        'exit_wall': ([1.7, -.275, 1.35], [.5, .075, 1.35], 2),
    }
    for name, (position, size, category) in expected_geometry.items():
        geom = asset.geom(name)
        if (geom.type[0] != mujoco.mjtGeom.mjGEOM_BOX
                or not np.allclose(asset_data.geom_xpos[geom.id], position, atol=1e-9, rtol=0)
                or not np.allclose(asset_data.geom_xmat[geom.id].reshape(3, 3), np.eye(3), atol=1e-9, rtol=0)
                or not np.allclose(geom.size, size, atol=1e-9, rtol=0)
                or geom.contype[0] != category or geom.conaffinity[0] != 0):
            raise ValueError(f'Unsupported full-route geometry: {name}')
    arms_source = ROOT / 'data/motions/kimodo_side60_hug_elbow/side60_hug_elbow.npz'
    arms_clip = load_motion(arms_source)
    core_arm_ids = [core['joint_names'].tolist().index(name) for name in G1_JOINT_NAMES[15:]]
    source_arm_ids = [arms_clip['joint_names'].tolist().index(name) for name in G1_JOINT_NAMES[15:]]
    if not np.allclose(core['joint_pos'][:, core_arm_ids], arms_clip['joint_pos'][0, source_arm_ids],
                       atol=1e-10, rtol=0):
        raise ValueError('Core arms must retain the specified chest-hold source frame')

    def quintic(t):
        return t ** 3 * (10 + t * (-15 + 6 * t))

    model = make_g1_spec(interaction_collisions=True).compile()
    data = mujoco.MjData(model)
    addresses = [model.joint(name).qposadr[0] for name in core['joint_names']]
    feet_ids = [model.body(side + '_ankle_roll_link').id for side in ('left', 'right')]
    core_feet, core_rotations, core_com = [], [], []
    for frame in range(len(core['root_pos'])):
        data.qpos[:3] = core['root_pos'][frame]
        data.qpos[3:7] = core['root_quat'][frame]
        data.qpos[addresses] = core['joint_pos'][frame]
        mujoco.mj_forward(model, data)
        core_feet.append(data.xpos[feet_ids].copy())
        core_rotations.append(data.xmat[feet_ids].reshape(2, 3, 3).copy())
        core_com.append(data.subtree_com[1].copy())
    core_feet, core_rotations, core_com = map(np.asarray, (core_feet, core_rotations, core_com))
    if (not np.allclose(core_feet[0, :, :2], [[.14, .115], [.26, .115]], atol=.002, rtol=0)
            or not np.allclose(core_feet[-1, :, :2], [[.89, .115], [1.01, .115]], atol=.002, rtol=0)):
        raise ValueError('Core endpoints must match the reviewed five 15 cm step-together cycles')
    core_yaw = np.unwrap(Rotation.from_quat(core['root_quat'][:, [1, 2, 3, 0]]).as_euler('xyz')[:, 2])
    support_offsets = []
    for side in range(2):
        single = core['planned_stance'][:, side] & ~core['planned_stance'][:, 1 - side]
        stable = np.convolve(single.astype(int), np.ones(5, dtype=int), mode='same') == 5
        if not stable.any():
            raise ValueError('Core must contain stable single support for both feet')
        local = np.einsum('tji,tj->ti', core_rotations[stable, side],
                          core_com[stable] - core_feet[stable, side])
        support_offsets.append(np.median(local, axis=0))
    support_offsets = np.asarray(support_offsets)
    fps = 30
    states, phases = [], []

    def support_center(position, rotation, side):
        return (position + rotation @ support_offsets[side])[:2]

    def center(positions, rotations):
        return np.mean([support_center(positions[s], rotations[s], s) for s in range(2)], axis=0)

    def append(positions, rotations, com, yaw, stance, weight, label):
        states.append(dict(feet=positions.copy(), rotations=rotations.copy(), com=np.asarray(com).copy(),
                           yaw=float(yaw), stance=np.asarray(stance, dtype=bool), weight=float(weight), phase=label))

    def transition(label, frames, *, goal_com=None, goal_yaw=None, goal_weight=None):
        start = states[-1]
        goal_com = start['com'] if goal_com is None else np.asarray(goal_com)
        goal_yaw = start['yaw'] if goal_yaw is None else goal_yaw
        goal_weight = start['weight'] if goal_weight is None else goal_weight
        distance = np.linalg.norm(goal_com - start['com'])
        frames = max(frames, int(np.ceil(1.875 * distance / .55 * fps)))
        if frames == 0:
            return
        first = len(states) - 1
        for index in range(1, frames + 1):
            w = quintic(index / frames)
            append(start['feet'], start['rotations'], (1 - w) * start['com'] + w * goal_com,
                   (1 - w) * start['yaw'] + w * goal_yaw, [True, True],
                   (1 - w) * start['weight'] + w * goal_weight, label)
        phases.append(dict(name=label, start_frame=first, end_frame_inclusive=len(states) - 1))

    def step(label, side, goal, rotation=None, *, goal_yaw=None, settle_com=None, timing=(8, 15, 0)):
        start = states[-1]
        support = 1 - side
        shift_frames, swing_frames, settle_frames = timing
        transition(label + '_weight_shift', shift_frames,
                   goal_com=support_center(start['feet'][support], start['rotations'][support], support))
        start = states[-1]
        rotation = start['rotations'][side] if rotation is None else rotation
        goal_yaw = start['yaw'] if goal_yaw is None else goal_yaw
        interpolator = Slerp([0, 1], Rotation.from_matrix([start['rotations'][side], rotation]))
        first = len(states) - 1
        for index in range(1, swing_frames + 1):
            u = index / swing_frames
            w = quintic(u)
            positions, rotations = start['feet'].copy(), start['rotations'].copy()
            positions[side] = (1 - w) * start['feet'][side] + w * goal
            positions[side, 2] += .015 * 64 * u ** 3 * (1 - u) ** 3
            rotations[side] = interpolator(w).as_matrix()
            stance = [True, True]
            stance[side] = index == swing_frames
            append(positions, rotations, start['com'], (1 - w) * start['yaw'] + w * goal_yaw,
                   stance, start['weight'], label + '_swing')
        phases.append(dict(name=label + '_swing', side=side, start_frame=first,
                           end_frame_inclusive=len(states) - 1,
                           step_distance_m=float(np.linalg.norm(goal[:2] - start['feet'][side, :2])),
                           foot_rotation_deg=float(np.rad2deg((Rotation.from_matrix(start['rotations'][side]).inv()
                                                             * Rotation.from_matrix(rotation)).magnitude()))))
        # A normal landing keeps COM on the previous support. The next weight
        # transfer moves directly to the next supporting foot, with no fast
        # round trip through the feet midpoint.
        if settle_com is not None:
            transition(label + '_settle', max(settle_frames, 18), goal_com=settle_com)

    # Start standing on the wide platform facing +X. A 20 cm step pair brings
    # the pelvis near (-.4,.6); the feet then turn about this fixed wide-area pivot.
    reverse = Rotation.from_euler('z', -np.pi / 2).as_matrix()
    local_feet = (core_feet[0] - core['root_pos'][0]) @ reverse.T
    pivot = np.array([-.6, .6, core['root_pos'][0, 2]])
    positions = local_feet + pivot
    rotations = np.einsum('ij,sjk->sik', reverse, core_rotations[0])
    append(positions, rotations, center(positions, rotations), 0., [True, True], 0., 'start')
    transition('start_hold', 12)
    for side in (0, 1):
        goal = states[-1]['feet'][side].copy()
        goal[0] += .2
        step('approach_' + ('left' if side == 0 else 'right'), side, goal)
    pivot[0] += .2
    for turn in range(1, 4):
        heading = turn * np.pi / 6
        world_rotation = Rotation.from_euler('z', heading).as_matrix()
        goals = local_feet @ world_rotation.T + pivot
        final_rotations = np.einsum('ij,sjk->sik', world_rotation, rotations)
        # The outer foot moves first; each planted foot remains fixed.
        for side in (1, 0):
            intermediate_yaw = (turn - 1) * np.pi / 6 + (np.pi / 12 if side == 1 else np.pi / 6)
            step(f'turn_{turn}_' + ('left' if side == 0 else 'right'), side, goals[side],
                 final_rotations[side], goal_yaw=intermediate_yaw, timing=(6, 15, 3))
    staging = core_feet[0].copy()
    staging[:, 0] -= .6
    back_start = states[-1]['feet'].copy()
    for part in range(1, 4):
        goals = back_start + part / 3 * (staging - back_start)
        for side in (1, 0):
            step(f'backward_{part}_' + ('left' if side == 0 else 'right'), side, goals[side],
                 core_rotations[0, side])
    transition('align_before_entry', 18, goal_com=core_com[0, :2] - [.6, 0],
               goal_yaw=core_yaw[0], goal_weight=1.)
    def copied_cycle(label, source_start, shift_x):
        first = len(states)
        shift = np.array([shift_x, 0., 0.])
        for frame in range(source_start, source_start + 90):
            append(core_feet[frame] + shift, core_rotations[frame],
                   core_com[frame, :2] + shift[:2], core_yaw[frame],
                   core['planned_stance'][frame], 1., label)
        phases.append(dict(name=label, start_frame=first, end_frame_inclusive=len(states) - 1,
                           source_core_frames_half_open=[source_start, source_start + 90],
                           translation_x_m=shift_x))

    # The initial aligned anchor equals the first entry-cycle sample. Replace
    # that duplicate, then retain all ninety samples of each reviewed cycle.
    states.pop()
    for cycle in range(4):
        copied_cycle(f'entry_cycle_{cycle + 1}', 0, -.6 + .15 * cycle)

    # Replace the shared seam with the actual core sample. Preserve every core
    # FK target, heading and COM, and let the downstream solver copy core qpos.
    core_start = len(states)
    for frame in range(len(core_feet)):
        append(core_feet[frame], core_rotations[frame], core_com[frame, :2], core_yaw[frame],
               core['planned_stance'][frame], 1., 'core')
    core_end = len(states)
    phases.append(dict(name='core', start_frame=core_start, end_frame_inclusive=core_end - 1))
    for cycle in range(1, 4):
        copied_cycle(f'exit_cycle_{cycle}', len(core_feet) - 90, .15 * cycle)
    transition('release_after_exit', 18, goal_weight=0.)
    depart_start = states[-1]['feet'].copy()
    final_feet = core_feet[-1] - core['root_pos'][-1] + [1.7, .75, core['root_pos'][-1, 2]]
    for part in range(1, 5):
        goals = depart_start + part / 4 * (final_feet - depart_start)
        for side in (1, 0):
            step(f'depart_{part}_' + ('left' if side == 0 else 'right'), side,
                 goals[side], core_rotations[-1, side])
    transition('goal_settle', 24, goal_com=center(states[-1]['feet'], states[-1]['rotations']))
    transition('goal_hold', 12)

    positions = np.array([s['feet'] for s in states])
    rotations = np.array([s['rotations'] for s in states])
    com = np.array([s['com'] for s in states])
    body_yaw = np.array([s['yaw'] for s in states])
    weight = np.array([s['weight'] for s in states])
    stance = np.array([s['stance'] for s in states])
    back_required = (weight > .999) & (com[:, 0] > .12) & (com[:, 0] < 1.06)
    back_required[core_start:core_end] = True
    core_index = np.full(len(states), -1)
    core_index[core_start:core_end] = np.arange(len(core_feet))
    com_steps = np.linalg.norm(np.diff(com, axis=0), axis=1)
    foot_steps = np.linalg.norm(np.diff(positions, axis=0), axis=2)
    rotation_steps = (Rotation.from_matrix(rotations[:-1].reshape(-1, 3, 3)).inv()
                      * Rotation.from_matrix(rotations[1:].reshape(-1, 3, 3))).magnitude().reshape(-1, 2)
    anchored = stance[:-1] & stance[1:]
    external = np.ones(len(states) - 1, dtype=bool)
    external[core_start - 360:core_end + 270 - 1] = False
    checks = dict(max_com_step_m=float(com_steps.max()), max_foot_step_m=float(foot_steps.max()),
                  max_foot_rotation_step_rad=float(rotation_steps.max()),
                  max_planted_foot_step_m=float(foot_steps[anchored].max()),
                  max_planted_foot_rotation_step_rad=float(rotation_steps[anchored].max()),
                  max_planted_foot_rotation_outside_core_rad=float(rotation_steps[anchored & external[:, None]].max()),
                  core_seam_com_steps_m=com_steps[[core_start - 1, core_end - 1]].tolist())
    if (checks['max_com_step_m'] > .019 or checks['max_foot_step_m'] > .030
            or checks['max_foot_rotation_step_rad'] > .08
            or checks['max_planted_foot_step_m'] > .0001
            or checks['max_planted_foot_rotation_outside_core_rad'] > .001):
        raise ValueError(f'Plan continuity check failed: {checks}')
    report = dict(method='Discrete planted-foot plan with weight transfers; no human/robot IK generated here.',
                  core=str(core_path), core_sha256=sha256(core_path),
                  frames=len(states), fps=fps, duration_s=(len(states) - 1) / fps,
                  prefix_duration_s=core_start / fps, core_sample_window_s=len(core_feet) / fps,
                  suffix_duration_s=(len(states) - core_end) / fps,
                  core_range_frames=[core_start, core_end],
                  continuity_checks=checks, maximum_transfer_com_speed_m_s=.55,
                  free_step_lift_m=.015, free_step_com_offset_foot_local_m=support_offsets.tolist(),
                  entry_cycle_source_frames=[0, 90], exit_cycle_source_frames=[360, 450],
                  phase_bounds='start_frame and end_frame_inclusive include both shared endpoint anchors',
                  phases=phases, script_sha256=sha256(__file__),
                  scene=str(scene), scene_sha256=sha256(scene),
                  arms_source=str(arms_source), arms_source_sha256=sha256(arms_source))
    plan = dict(fps=np.array(float(fps)), planned_foot_positions=positions,
             planned_foot_quat=Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]].reshape(-1, 2, 4),
             planned_com=np.c_[com, np.full(len(states), core_com[:, 2].mean())],
             body_yaw=body_yaw, planned_stance=stance, back_required=back_required,
             back_alignment_weight=weight, core_range_frames=np.array([core_start, core_end]),
             core_frame_index=core_index, phase=np.array([s['phase'] for s in states]),
             phases=np.array(json.dumps(phases)), provenance=np.array(json.dumps(report)))
    return plan, report


def complete(core_path, output, overwrite=False):
    """Add planned approach/exit IK around the frozen core; no new Kimodo inference."""
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    from motion_data import load_motion, save_motion
    from supermimic.assets.g1.g1_constants import G1_JOINT_NAMES, make_g1_spec
    from supermimic.tasks.open_door.geometry import _surface_candidates, _forbidden_back_points

    core_path, output = Path(core_path).resolve(), Path(output).resolve()
    scene = core_path.with_suffix('.xml')
    arms_source = ROOT / 'data/motions/kimodo_side60_hug_elbow/side60_hug_elbow.npz'
    plan_path = output.with_name(output.stem + '_plan.npz')
    plan_json = plan_path.with_suffix('.json')
    input_paths = [core_path, scene, core_path.with_suffix('.json'), arms_source,
                   arms_source.with_suffix('.xml'), arms_source.with_suffix('.json')]
    output_paths = [output, output.with_suffix('.xml'), output.with_suffix('.json'), plan_path, plan_json]
    if output.suffix != '.npz' or {p.resolve() for p in input_paths}.intersection(p.resolve() for p in output_paths):
        raise ValueError('Complete needs separate output and plan files; inputs cannot be overwritten')
    if not overwrite and any(path.exists() for path in output_paths):
        raise ValueError('Complete output or plan already exists; use --overwrite')
    core = load_motion(core_path)
    if core['joint_names'].tolist() != G1_JOINT_NAMES:
        raise ValueError('Complete requires the standard G1 joint order to preserve the core arrays')
    plan, plan_metadata = _full_ledge_plan(core_path)

    class _FullLedgeSolver:

        def __init__(self, core, nominal_z=None):
            self.model = make_g1_spec(interaction_collisions=True).compile()
            self.data = mujoco.MjData(self.model)
            self.names = G1_JOINT_NAMES
            self.addresses = np.array([int(self.model.joint(name).qposadr[0]) for name in self.names])
            self.foot_bodies = [self.model.body(side + '_ankle_roll_link').id for side in ('left', 'right')]
            self.torso_body = self.model.body('torso_link').id
            self.pelvis_body = self.model.body('pelvis').id
            arms_clip = load_motion(ROOT / 'data/motions/kimodo_side60_hug_elbow/side60_hug_elbow.npz')
            self.arms = arms_clip['joint_pos'][0, [arms_clip['joint_names'].tolist().index(name) for name in self.names[15:]]]
            order = [core['joint_names'].tolist().index(name) for name in self.names]
            self.core_joints = core['joint_pos'][:, order]
            if not np.allclose(self.core_joints[:, 15:], self.arms, atol=1e-10):
                raise ValueError('Core arms differ from specified source frame')
            self.core_poses = np.c_[core['root_pos'],
                                    Rotation.from_quat(core['root_quat'][:, [1, 2, 3, 0]]).as_euler('xyz'),
                                    self.core_joints[:, :15]]
            self.nominal_z = float(np.median(core['root_pos'][:, 2])) if nominal_z is None else nominal_z
            ranges = self.model.jnt_range[[self.model.joint(name).id for name in self.names[:15]]]
            midpoint = ranges.mean(1)
            half = np.diff(ranges, axis=1)[:, 0] * 0.45
            self.lower_bounds = np.r_[-3.0, -0.2, self.nominal_z - 0.008, -0.24, -0.24, -4 * np.pi, midpoint - half + 1e-07]
            self.upper_bounds = np.r_[4.0, 2.0, self.nominal_z + 0.008, 0.24, 0.24, 4 * np.pi, midpoint + half - 1e-07]
            self.lower_bounds[[9, 15]] = np.maximum(self.lower_bounds[[9, 15]], np.deg2rad(20.0))
            self.forward(self.core_poses[0])
            self.surfaces = []
            legal_groups = []
            logo_groups = []
            for geom_id in range(self.model.ngeom):
                if not self.model.geom_conaffinity[geom_id] & 2:
                    continue
                points = _surface_candidates(self.model, geom_id)
                name = self.model.body(int(self.model.geom_bodyid[geom_id])).name
                if name in ('torso_link', 'logo_link'):
                    world = points @ self.data.geom_xmat[geom_id].reshape(3, 3).T + self.data.geom_xpos[geom_id]
                    local = (world - self.data.xpos[self.torso_body]) @ self.data.xmat[self.torso_body].reshape(3, 3)
                    forbidden = _forbidden_back_points(name, 'torso_link', local)
                    for legal, mask in ((False, forbidden), (True, ~forbidden)):
                        selected = points[mask]
                        if not len(selected):
                            continue
                        self.surfaces.append((geom_id, selected, name))
                        legal_groups.append(legal)
                        logo_groups.append(legal and name == 'logo_link')
                else:
                    points = points[ConvexHull(points).vertices]
                    self.surfaces.append((geom_id, points, name))
                    legal_groups.append(False)
                    logo_groups.append(False)
            self.legal_back = np.array(legal_groups)
            self.legal_logo = np.array(logo_groups)
            self.clearance = np.where(self.legal_back, -0.0001, 0.002)
            self.back_gain = 800.0
            self.wall_boxes = None
            self.sole_geoms = [np.array([
                geom_id for geom_id in range(self.model.ngeom)
                if self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
                and int(self.model.geom_bodyid[geom_id]) == body
            ]) for body in self.foot_bodies]

        def forward(self, pose):
            self.data.qpos[:3] = pose[:3]
            self.data.qpos[3:7] = Rotation.from_euler('xyz', pose[3:6]).as_quat()[[3, 0, 1, 2]]
            self.data.qpos[self.addresses] = np.r_[pose[6:], self.arms]
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

        def configure_scene(self, path):
            model = mujoco.MjSpec.from_file(str(path)).compile()
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            self.wall_boxes = []
            for geom_id in range(model.ngeom):
                if not model.geom_contype[geom_id] & 2:
                    continue
                rotation = data.geom_xmat[geom_id].reshape(3, 3)
                if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX or not np.allclose(np.abs(rotation).sum(0), 1.0, atol=1e-08):
                    raise ValueError('Finite-wall IK expects axis-aligned XML wall boxes')
                half = np.abs(rotation) @ model.geom_size[geom_id]
                self.wall_boxes.append((data.geom_xpos[geom_id] - half, data.geom_xpos[geom_id] + half))
            if not self.wall_boxes:
                raise ValueError('No wall-category boxes in paired XML')

        def plane_gaps(self):
            raw_y_values = []
            gaps = []
            for geom_id, points, _ in self.surfaces:
                world = points @ self.data.geom_xmat[geom_id].reshape(3, 3).T + self.data.geom_xpos[geom_id]
                world_low = world.min(0)
                world_high = world.max(0)
                raw_y_values.append(world_low[1])
                if self.wall_boxes is None:
                    face = 0.0
                else:
                    fronts = [box_high[1] for box_low, box_high in self.wall_boxes
                              if world_high[0] >= box_low[0] - 0.002
                              and world_low[0] <= box_high[0] + 0.002
                              and world_high[2] >= box_low[2] - 0.002
                              and world_low[2] <= box_high[2] + 0.002]
                    face = max(fronts) if fronts else -1000.0
                gaps.append(world_low[1] - face)
            self.surface_min_y = np.array(raw_y_values)
            return np.array(gaps)

        def qpos(self, pose):
            self.forward(pose)
            return np.r_[self.data.qpos[:7], self.data.qpos[self.addresses]]

        def solve(self, previous, feet, foot_rotations, com_xy, yaw, back_weight, reference,
                  previous_previous=None, max_nfev=70):
            yaw = previous[5] + np.arctan2(np.sin(yaw - previous[5]), np.cos(yaw - previous[5]))
            self.forward(reference)
            reference_torso_rotation = self.data.xmat[self.torso_body].reshape(3, 3).copy()
            desired_torso_rotation = Rotation.from_euler('z', yaw - reference[5]).as_matrix() @ reference_torso_rotation
            expected = previous.copy()
            if previous_previous is not None:
                delta = previous - previous_previous
                delta[:3] = np.clip(delta[:3], -0.01, 0.01)
                delta[3:] = np.clip(delta[3:], -0.04, 0.04)
                expected += delta

            def residual(pose):
                self.forward(pose)
                position_error = (self.data.xpos[self.foot_bodies] - feet).ravel()
                orientation_error = np.concatenate([
                    Rotation.from_matrix(foot_rotations[side].T @ self.data.xmat[body].reshape(3, 3)).as_rotvec()
                    for side, body in enumerate(self.foot_bodies)])
                gaps = self.plane_gaps()
                com = self.data.subtree_com[self.pelvis_body]
                torso_error = Rotation.from_matrix(desired_torso_rotation.T @ self.data.xmat[self.torso_body].reshape(3, 3)).as_rotvec()
                return np.r_[1200 * position_error, 120 * orientation_error, 900 * (com[:2] - com_xy),
                             1600 * np.minimum(gaps - self.clearance, 0.0),
                             self.back_gain * back_weight * (self.surface_min_y[self.legal_logo].min() + 0.0001),
                             100 * (pose[2] - reference[2]), np.array([8.0, 8.0, 30.0]) * torso_error,
                             np.array([0.6] * 12 + [1.2] * 3) * (pose[6:] - reference[6:]),
                             0.6 * (pose[6:] - expected[6:]), 2 * (pose[3:6] - expected[3:6]),
                             3 * (pose[:3] - expected[:3])]
            seed = np.clip(expected, self.lower_bounds + 1e-08, self.upper_bounds - 1e-08)
            result = least_squares(residual, seed, bounds=(self.lower_bounds, self.upper_bounds),
                                   max_nfev=max_nfev, ftol=2e-08, xtol=2e-08, gtol=2e-07)
            self.forward(result.x)
            return (result.x, dict(nfev=result.nfev, cost=float(result.cost), optimality=float(result.optimality), success=bool(result.success)))

        def diagnostics(self, pose, feet, foot_rotations, com_xy, stance):
            self.forward(pose)
            gaps = self.plane_gaps()
            active = np.concatenate([self.sole_geoms[side] for side in range(2) if stance[side]])
            support = ConvexHull(self.data.geom_xpos[active, :2])
            com = self.data.subtree_com[self.pelvis_body]
            margin = -(support.equations[:, :2] @ com[:2] + support.equations[:, 2]).max()
            return dict(
                foot_error_m=np.linalg.norm(self.data.xpos[self.foot_bodies] - feet, axis=1).tolist(),
                foot_rotation_error_rad=[
                    float(Rotation.from_matrix(foot_rotations[side].T @ self.data.xmat[body].reshape(3, 3)).magnitude())
                    for side, body in enumerate(self.foot_bodies)],
                com_error_m=float(np.linalg.norm(com[:2] - com_xy)), com=com.copy().tolist(),
                support_margin_m=float(margin), back_plane_gap_m=float(self.surface_min_y[self.legal_logo].min()),
                nonback_plane_gap_m=float(gaps[~self.legal_back].min()),
                knee_angles_deg=np.rad2deg(pose[[9, 15]]).tolist())

    solver = _FullLedgeSolver(core)
    solver.back_gain = 5000.
    solver.clearance[~solver.legal_back] = .002
    solver.configure_scene(scene)
    feet = plan['planned_foot_positions']
    rotations = Rotation.from_quat(plan['planned_foot_quat'].reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(-1, 2, 3, 3)
    com_targets = plan['planned_com'][:, :2]
    yaw, stance = plan['body_yaw'], plan['planned_stance']
    back_required, back_weight = plan['back_required'], plan['back_alignment_weight']
    core_start, core_end = map(int, plan['core_range_frames'])
    count = len(feet)
    poses = np.empty((count, 21))
    poses[core_start:core_end] = solver.core_poses
    diagnostics = [None] * count
    for label, indices, reference in (
        ('prefix', range(core_start - 1, -1, -1), solver.core_poses[0]),
        ('suffix', range(core_end, count), solver.core_poses[-1]),
    ):
        previous, previous_previous = reference.copy(), None
        for index, frame in enumerate(indices):
            current, details = solver.solve(previous, feet[frame], rotations[frame], com_targets[frame],
                                            float(yaw[frame]), float(back_weight[frame]), reference,
                                            previous_previous, 70)
            if not np.isfinite(current).all():
                raise ValueError(f'Nonfinite full-route IK at frame {frame}')
            poses[frame] = current
            diagnostics[frame] = solver.diagnostics(current, feet[frame], rotations[frame],
                                                     com_targets[frame], stance[frame])
            previous_previous, previous = previous, current
            if index % 60 == 0:
                error = max(diagnostics[frame]['foot_error_m'])
                print(f'{label} frame {frame}: foot error {error:.6f} m, {details["nfev"]} IK evaluations', flush=True)
    qpos = np.array([solver.qpos(pose) for pose in poses])
    qpos[core_start:core_end, :3] = core['root_pos']
    qpos[core_start:core_end, 3:7] = core['root_quat']
    qpos[core_start:core_end, 7:] = solver.core_joints
    for frame in range(core_start, core_end):
        diagnostics[frame] = solver.diagnostics(poses[frame], feet[frame], rotations[frame],
                                               com_targets[frame], stance[frame])
    actual_com = np.array([item['com'] for item in diagnostics])
    foot_errors = np.array([item['foot_error_m'] for item in diagnostics])
    rotation_errors = np.array([item['foot_rotation_error_rad'] for item in diagnostics])
    com_errors = np.array([item['com_error_m'] for item in diagnostics])
    support_margin = np.array([item['support_margin_m'] for item in diagnostics])
    back_gaps = np.array([item['back_plane_gap_m'] for item in diagnostics])
    nonback_gaps = np.array([item['nonback_plane_gap_m'] for item in diagnostics])
    if (not np.isfinite(qpos).all() or not np.isfinite(foot_errors).all()
            or foot_errors.max() > .002 or support_margin.min() < -1e-5):
        raise ValueError('Full-route IK failed finite-value, foot-target or COM-support checks')
    with np.load(core_path, allow_pickle=False) as archive:
        source_quaternions = archive['root_quat']
    quaternion_difference = np.abs(qpos[core_start:core_end, 3:7] - source_quaternions).max()
    angular_difference = (Rotation.from_quat(source_quaternions[:, [1, 2, 3, 0]]).inv()
                          * Rotation.from_quat(qpos[core_start:core_end, 3:7][:, [1, 2, 3, 0]])).magnitude().max()
    if angular_difference > 1e-12:
        raise ValueError('Frozen core rotations changed during full-route assembly')
    if (sha256(core_path) != plan_metadata['core_sha256'] or sha256(scene) != plan_metadata['scene_sha256']
            or sha256(arms_source) != plan_metadata['arms_source_sha256']):
        raise ValueError('A full-route input changed while solving')
    method = ('Planned prefix/suffix IK around a frozen repaired 15-second core, with fixed source chest-hold arms, '
              'finite-wall clearance and legal logo contact. No new Kimodo inference in this completion step.')
    phase_summaries = []
    for phase in dict.fromkeys(plan['phase'].tolist()):
        indices = np.flatnonzero(plan['phase'] == phase)
        phase_summaries.append(dict(
            name=phase, frame_range=[int(indices[0]), int(indices[-1]) + 1],
            frames=len(indices), sample_window_s=len(indices) / 30,
            max_foot_error_m=float(foot_errors[indices].max()),
            min_com_support_margin_m=float(support_margin[indices].min()),
            min_nonback_wall_gap_m=float(nonback_gaps[indices].min()),
            min_knee_angles_deg=np.rad2deg(poses[indices][:, [9, 15]]).min(0).tolist(),
        ))
    report = dict(
        method=method, core=str(core_path), core_sha256=sha256(core_path),
        core_metadata=str(core_path.with_suffix('.json')),
        core_metadata_sha256=sha256(core_path.with_suffix('.json')) if core_path.with_suffix('.json').is_file() else None,
        scene_sha256=sha256(scene), arms_source=str(arms_source), arms_source_sha256=sha256(arms_source),
        script_sha256=sha256(__file__), plan=str(plan_path), plan_metadata=str(plan_json),
        frames=count, fps=30, duration_s=(count - 1) / 30,
        core_range_frames=[core_start, core_end], core_sample_window_s=(core_end - core_start) / 30,
        prefix_duration_s=core_start / 30, suffix_duration_s=(count - core_end) / 30,
        core_pose_preservation=dict(
            root_pos_bitwise_equal=bool(np.array_equal(qpos[core_start:core_end, :3], core['root_pos'])),
            joint_pos_bitwise_equal=bool(np.array_equal(qpos[core_start:core_end, 7:], core['joint_pos'])),
            quaternion_max_abs_difference=float(quaternion_difference),
            quaternion_rotation_max_difference_rad=float(angular_difference),
            quaternion_rotation_tolerance_rad=1e-12,
            quaternion_note='The standard motion loader normalizes quaternions; core rotations remain equivalent.',
        ),
        parameters=dict(max_nfev=70, back_contact_weight=5000., forbidden_clearance_m=.002,
                        legal_back_patch='torso-local x<-.025, abs(y)<.13, .05<z<.38',
                        back_target_gap_m=-.0001, contact_target='legal logo surface',
                        root_z_bound_half_width_m=.008, minimum_knee_angle_deg=20.),
        max_foot_error_m=float(foot_errors.max()), max_foot_rotation_error_rad=float(rotation_errors.max()),
        min_support_margin_m=float(support_margin.min()), min_nonback_wall_gap_m=float(nonback_gaps.min()),
        min_knee_angles_deg=np.rad2deg(poses[:, [9, 15]]).min(0).tolist(),
        root_z_range_m=[float(poses[:, 2].min()), float(poses[:, 2].max())],
        max_joint_step_rad=float(np.abs(np.diff(qpos[:, 7:], axis=0)).max()),
        seams=[dict(index=frame, root_step_m=float(np.linalg.norm(qpos[frame, :3] - qpos[frame - 1, :3])),
                    max_joint_step_rad=float(np.abs(qpos[frame, 7:] - qpos[frame - 1, 7:]).max()))
               for frame in (core_start, core_end)],
        phases=phase_summaries, frame_ranges='start inclusive, end exclusive',
        kinematic_only=True, dynamics_validated=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(plan_path, **plan)
    plan_json.write_text(json.dumps(plan_metadata, indent=2) + '\n')
    report['plan_sha256'] = sha256(plan_path)
    report['plan_metadata_sha256'] = sha256(plan_json)
    out = dict(plan)
    out.update(
        fps=np.array(30.), qpos=qpos, root_pos=qpos[:, :3], root_quat=qpos[:, 3:7],
        joint_pos=qpos[:, 7:], joint_names=np.array(G1_JOINT_NAMES),
        planned_foot_positions=feet,
        planned_foot_quat=Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_quat()[:, [3, 0, 1, 2]].reshape(count, 2, 4),
        planned_stance=stance, postprocess_stance=stance, back_required=back_required,
        planned_com=np.c_[com_targets, actual_com[:, 2]], postprocess_actual_com=actual_com, actual_com=actual_com,
        postprocess_ik_error=foot_errors, postprocess_foot_rotation_error=rotation_errors,
        postprocess_com_error=com_errors, postprocess_support_margin=support_margin, support_margin=support_margin,
        postprocess_legal_back_gap=back_gaps, postprocess_nonback_wall_gap=nonback_gaps,
        core_range_frames=np.array([core_start, core_end]), kinematic_only=np.array(True),
        postprocess=np.array(method), provenance=np.array(json.dumps(report)), scene_xml_sha256=np.array(sha256(scene)),
    )
    save_motion(output, out)
    shutil.copy2(scene, output.with_suffix('.xml'))
    report['reference_sha256'] = sha256(output)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {output}: {count} frames, {(count - 1) / 30:.3f}s, maximum foot error {foot_errors.max():.6f} m')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("folder", type=Path)
    prep.add_argument("--scene", type=Path, default=SCENE)
    prep.add_argument("--overwrite", action="store_true")
    imp = commands.add_parser("import")
    imp.add_argument("source", type=Path)
    imp.add_argument("output", type=Path)
    imp.add_argument("--overwrite", action="store_true")
    ref = commands.add_parser("refine")
    ref.add_argument("source", type=Path)
    ref.add_argument("output", type=Path)
    ref.add_argument("--template", type=Path, required=True, help="Matching authored template.npz used for generation")
    ref.add_argument("--overwrite", action="store_true")
    repair = commands.add_parser("repair-core")
    repair.add_argument("source", type=Path, help="Previously accepted core, used only for root/waist style")
    repair.add_argument("output", type=Path)
    repair.add_argument("--overwrite", action="store_true")
    full = commands.add_parser("complete")
    full.add_argument("core", type=Path, help="Frozen repaired 450-frame ledge core")
    full.add_argument("output", type=Path)
    full.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.folder.resolve(), args.scene.resolve(), args.overwrite)
    elif args.command == "import":
        import_candidate(args.source.resolve(), args.output.resolve(), args.overwrite)
    elif args.command == "refine":
        refine(args.source.resolve(), args.output.resolve(), args.template.resolve(), args.overwrite)
    elif args.command == "repair-core":
        repair_core(args.source.resolve(), args.output.resolve(), args.overwrite)
    else:
        complete(args.core.resolve(), args.output.resolve(), args.overwrite)


if __name__ == "__main__":
    main()
