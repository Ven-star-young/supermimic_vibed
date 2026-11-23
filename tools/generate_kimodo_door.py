"""Prepare a back-door reference locally, then generate native G1 motion in Kimodo.

python -m tools.generate_kimodo_door prepare OUTPUT
python generate_kimodo_door.py generate OUTPUT --seed 42 --num-samples 4 --steps 150
python -m tools.generate_kimodo_door refine SOURCE.npz OUTPUT.npz --native-motion NATIVE.npz
python -m tools.generate_kimodo_door prepare-cycle OUTPUT
python -m tools.generate_kimodo_door assemble-cycle PREFIX.npz SUFFIX.npz OUTPUT.npz --prefix-native PREFIX_NATIVE.npz --suffix-native SUFFIX_NATIVE.npz

Copy this script and the prepared folder into the Kimodo deployment for generation.
Outputs are kinematic candidates; no robot dynamics are simulated.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "data/motions/open_door/video7_then_video1_then_video8_g1_door_cycle_asset.npz"
MODEL = "Kimodo-G1-RP-v1"
PROMPT = ("A person slowly takes small steps backwards while pushing a heavy door open "
          "with their upper back, keeping their elbows bent and hands in front of their chest.")
DEFAULT_BACK = ROOT / "data/motions/kimodo_back_door/kimodo_back_door.npz"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_revision(folder):
    try:
        result = subprocess.run(["git", "-C", str(folder), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def fixed_arms(names):
    """A single tucked-arm pose, fixed relative to the torso through the cycle."""
    mask = np.array([any(part in name for part in ("shoulder", "elbow", "wrist")) for name in names])
    if mask.sum() != 14:
        raise ValueError("Expected the fourteen named G1 arm joints")
    values = np.zeros(len(names))
    for index, name in enumerate(names):
        if "shoulder_pitch" in name:
            values[index] = -.12
        elif "shoulder_roll" in name:
            values[index] = .08 if name.startswith("left") else -.08
        elif "shoulder_yaw" in name:
            values[index] = -.25 if name.startswith("left") else .25
        elif "elbow" in name:
            values[index] = 1.15
    return mask, values[mask]


def prepare_cycle(folder: Path, back_motion: Path, *, overwrite=False):
    """Author approach/turn and exit/turn/walk constraints around the reviewed push."""
    from scipy.spatial.transform import Rotation
    from motion_data import load_motion
    from motion_data.door import validate_door_asset

    if not overwrite and any((folder / part / name).exists()
                             for part in ("prefix", "suffix") for name in ("input.npz", "metadata.json")):
        raise ValueError("Cycle input already exists; use --overwrite to replace it")
    back = load_motion(back_motion)
    validate_door_asset(back, back_motion.with_suffix(".xml"))
    if len(back["root_pos"]) != 270 or float(back["fps"]) != 30 or not np.isclose(back["door_angle"][235], 1.1):
        raise ValueError("Cycle preparation requires the reviewed 270-frame back-push reference")
    names = back["joint_names"]
    arms, pose = fixed_arms(names)
    back["joint_pos"][:, arms] = pose
    neutral = np.zeros(29)
    neutral[arms] = pose
    for index, name in enumerate(names):
        if "hip_pitch" in name:
            neutral[index] = -.2
        elif "knee" in name:
            neutral[index] = .32
        elif "ankle_pitch" in name:
            neutral[index] = -.12

    def smooth(value):
        value = np.clip(value, 0, 1)
        return value ** 2 * (3 - 2 * value)

    for part, count, seam in (("prefix", 255, 0), ("suffix", 330, 235)):
        times = np.arange(count) / 30
        start = back["root_pos"][seam]
        heading = Rotation.from_quat(back["root_quat"][seam], scalar_first=True).as_euler("xyz")[2]
        root = np.tile(start, (count, 1))
        joints = np.tile(neutral, (count, 1))
        if part == "prefix":
            root[:, 0] = 1.65 + (.4 - 1.65) * smooth(times / 3.6)
            root[:, 0] += (start[0] - .4) * smooth((times - 7) / (times[-1] - 7))
            root[:, 2] = .78540637
            yaw = np.pi + (np.pi + heading) * smooth((times - 4) / 3)
            keyframes, lengths, endpoint = [0, 108, 119, 210, 254], [120, 135], -1
            initial_heading, door_angle = np.pi, np.zeros(count)
            prompts = [
                "A person walks forward toward a door with both elbows bent and hands held still in front of the abdomen.",
                "A person stops, turns around in place, and takes a small step backwards with both hands held still in front of the abdomen.",
            ]
        else:
            # Accelerate along the free edge before the passive door is released.
            speed, acceleration_time, distance = .8, .3, .75
            cruise = distance / speed - acceleration_time
            travel = np.full(count, distance)
            for index, time in enumerate(times):
                if time < acceleration_time:
                    travel[index] = .5 * speed * (time - acceleration_time / np.pi * np.sin(np.pi * time / acceleration_time))
                elif time < acceleration_time + cruise:
                    travel[index] = .5 * speed * acceleration_time + speed * (time - acceleration_time)
                elif time < 2 * acceleration_time + cruise:
                    elapsed = time - acceleration_time - cruise
                    travel[index] = (.5 * speed * acceleration_time + speed * cruise
                                     + .5 * speed * (elapsed + acceleration_time / np.pi * np.sin(np.pi * elapsed / acceleration_time)))
            root[:, :2] += travel[:, None] * np.array([-np.sin(1.1), -np.cos(1.1)])
            root[:, 0] -= 1.3 * smooth((times - 6) / 4.5)
            yaw = heading + (-np.pi - heading) * smooth((times - 3) / 2.5)
            keyframes, lengths, endpoint = [0, 42, 89, 165, 179, 315, 329], [90, 90, 150], 0
            initial_heading, door_angle = heading, np.full(count, 1.1)
            prompts = [
                "A person briskly steps sideways and backwards around the edge of a door, then stops, holding both hands still in front of the abdomen.",
                "A person turns around in place and stands still with both hands held in front of the abdomen.",
                "A person walks forward and comes to a stop, keeping both hands still in front of the abdomen.",
            ]
        quaternions = np.column_stack([np.cos(yaw / 2), np.zeros(count), np.zeros(count), np.sin(yaw / 2)])
        root[endpoint], quaternions[endpoint], joints[endpoint] = (
            back["root_pos"][seam], back["root_quat"][seam], back["joint_pos"][seam])
        origin = np.r_[root[0, :2], 0.]
        destination = folder / part
        destination.mkdir(parents=True, exist_ok=True)
        np.savez(destination / "input.npz", qpos=np.c_[root - origin, quaternions, joints],
                 joint_names=names, keyframes=keyframes, smooth_root_2d=(root - origin)[:, [1, 0]])
        metadata = {
            "fps": 30, "joint_names": names.tolist(), "world_offset": origin.tolist(),
            "door_angle": door_angle.tolist(), "prompts": prompts, "num_frames": lengths,
            "first_heading_angle": float(initial_heading), "output_stem": part,
            "provenance": {"model": MODEL, "segment": part, "back_seam_frame": seam,
                           "source_back_sha256": sha256(back_motion),
                           "source_xml_sha256": sha256(back_motion.with_suffix(".xml")),
                           "script_sha256": sha256(__file__), "fixed_arm_pose": pose.tolist(),
                           "door_curve": "placeholder; actual rear-patch departure determines passive release during assembly"},
        }
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared approach and exit inputs in {folder}; central back-push frames 0..235")


def prepare(folder: Path, door_reference: Path, *, overwrite=False):
    if not overwrite and any((folder / name).exists() for name in ("input.npz", "metadata.json")):
        raise ValueError("Prepared input already exists; use --overwrite to replace it")
    # Preparation uses only this project's robot and door assets, never Kimodo.
    import mujoco
    from motion_data import load_motion
    from motion_data.converters._common import g1_joints
    from motion_data.door import door_metadata, validate_door_asset
    from supermimic.assets.g1 import make_g1_spec
    from supermimic.tasks.open_door.geometry import _surface_candidates

    source = load_motion(door_reference)
    door = door_metadata(source)
    validate_door_asset(source, door_reference.with_suffix(".xml"))
    expected = {"joint_axis": [0, 0, -1], "base_yaw": 0,
                "panel_half_size": [.02, .275, .9], "hinge_to_panel": [0, -.275, 0],
                "joint_range": [0, 1.8], "hinge_pos": [-.127574118618, .132627631264, .92]}
    for key, value in expected.items():
        if not np.allclose(door[key], value, atol=1e-6, rtol=0):
            raise ValueError(f"Unsupported door {key}: expected {value}")
    names = source["joint_names"]
    if set(names.tolist()) != set(g1_joints()[0].tolist()):
        raise ValueError("Door reference must contain all 29 G1 joint names")
    model = make_g1_spec(interaction_collisions=True).compile()
    data = mujoco.MjData(model)
    addresses = [model.joint(str(name)).qposadr[0] for name in names]
    joints = np.zeros(29)
    for index, name in enumerate(names):
        if "hip_pitch" in name:
            joints[index] = -.20
        elif "knee" in name:
            joints[index] = .32
        elif "ankle_pitch" in name:
            joints[index] = -.12
        elif "elbow" in name:
            joints[index] = 1.15
        elif "shoulder_pitch" in name:
            joints[index] = -.12
        elif "shoulder_roll" in name:
            joints[index] = .08 if name.startswith("left") else -.08
        elif "shoulder_yaw" in name:
            joints[index] = -.25 if name.startswith("left") else .25
    fps, count = 30, 270
    times = np.arange(count) / fps
    phase = np.clip((times - 2.5) / 5, 0, 1)
    angles = 1.10 * phase ** 2 * (3 - 2 * phase)
    approach = np.clip(times / 2.5, 0, 1)
    approach = .27 * (1 - approach ** 2 * (3 - 2 * approach))
    root, quat = np.zeros((count, 3)), np.zeros((count, 4))
    torso = model.body("torso_link").id
    back_local = np.array([-.06861, .00763, .213])
    feet = [model.geom(side + "_ankle_roll_link_object_collision").id for side in ("left", "right")]
    for frame, angle in enumerate(angles):
        yaw = -angle
        quat[frame] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]  # Zero root pitch; wxyz.
        data.qpos[:3], data.qpos[3:7], data.qpos[addresses] = [0, 0, 1], quat[frame], joints
        mujoco.mj_forward(model, data)
        floor = min((_surface_candidates(model, geom) @ data.geom_xmat[geom].reshape(3, 3).T
                     + data.geom_xpos[geom])[:, 2].min() for geom in feet)
        root[frame, 2] = 1 - floor + .002
        back = data.xpos[torso] + data.xmat[torso].reshape(3, 3) @ back_local
        c, s = np.cos(yaw), np.sin(yaw)
        panel_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        target = door["hinge_pos"] + panel_rotation @ np.array([.025, -.28, 0])
        root[frame, :2] = target[:2] - back[:2]
        root[frame, 0] += approach[frame]
    offset = np.r_[root[0, :2], 0]
    qpos = np.column_stack([root - offset, quat, np.tile(joints, (count, 1))])
    path = root[:, :2] - offset[:2]
    keyframes = [0, 75, 110, 145, 180, 225, 269]
    metadata = {
        "fps": fps, "joint_names": names.tolist(), "world_offset": offset.tolist(),
        "door_angle": angles.tolist(),
        "provenance": {
            "model": MODEL, "scope": "back push and hold; kinematic reference only",
            "door_reference": door_reference.name, "door_reference_sha256": sha256(door_reference),
            "door_xml_sha256": sha256(door_reference.with_suffix(".xml")),
            "preparation_script_sha256": sha256(__file__), "project_revision": git_revision(ROOT),
            "opening_start_s": 2.5, "opening_end_s": 7.5, "target_angle_rad": 1.1,
            "root_pitch_rad": 0., "hip_pitch_rad": -.2, "knee_rad": .32,
            "ankle_pitch_rad": -.12, "back_along_m": .28, "keyframes": keyframes,
        },
    }
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / "input.npz", qpos=qpos, joint_names=names, keyframes=keyframes,
             smooth_root_2d=path[:, [1, 0]], heading=np.column_stack([np.cos(-angles), np.sin(-angles)]))
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared {count} frames at {fps} Hz in {folder}; world offset {offset.tolist()}")


def generate(folder: Path, *, seed=42, num_samples=4, steps=150, overwrite=False):
    if num_samples < 1 or steps < 1:
        raise ValueError("num-samples and steps must be positive")
    if not overwrite and ((folder / "constraints.json").exists() or any(folder.glob("back_door_*"))):
        raise ValueError("Generated output already exists; use --overwrite to replace it")
    # Generation runs inside Kimodo and needs none of the local project imports.
    import xml.etree.ElementTree as ET
    import torch
    import kimodo
    from kimodo import load_model
    from kimodo.constraints import FullBodyConstraintSet, Root2DConstraintSet, save_constraints_lst
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.exports.motion_io import save_kimodo_npz
    from kimodo.skeleton import G1Skeleton34
    from kimodo.tools import seed_everything

    with np.load(folder / "input.npz", allow_pickle=False) as archive:
        inputs = dict(archive)
    metadata = json.loads((folder / "metadata.json").read_text())
    skeleton = G1Skeleton34()
    converter = MujocoQposConverter(skeleton)
    names = [joint.get("name") for joint in ET.parse(converter.xml_path).findall("./worldbody//joint")]
    source_names = inputs["joint_names"].tolist()
    if len(names) != 29 or len(source_names) != 29 or set(names) != set(source_names):
        raise ValueError("Prepared input and native Kimodo G1 joint names must match")
    qpos = inputs["qpos"].copy()
    prompts = metadata.get("prompts", [PROMPT])
    lengths = metadata.get("num_frames", [270])
    initial_heading = float(metadata.get("first_heading_angle", 0.))
    output_stem = metadata.get("output_stem", "back_door")
    if (not isinstance(prompts, list) or not all(isinstance(text, str) and text for text in prompts)
            or len(prompts) != len(lengths) or not lengths
            or any(not isinstance(count, int) or not 2 <= count <= 300 for count in lengths)
            or qpos.shape != (sum(lengths), 36) or not np.isfinite(qpos).all()
            or metadata["fps"] != 30 or not np.isfinite(initial_heading)
            or not isinstance(output_stem, str) or not output_stem
            or Path(output_stem).name != output_stem):
        raise ValueError("Expected finite 30 Hz G1 inputs with one 2–300-frame duration per prompt")
    if not overwrite and any(folder.glob(f"{output_stem}_*")):
        raise ValueError("Generated candidates already exist; use --overwrite to replace them")
    qpos[:, 7:] = qpos[:, 7:][:, [source_names.index(name) for name in names]]
    motion = converter.qpos_to_motion_dict(qpos.astype(np.float32), source_fps=metadata["fps"])
    positions, rotations, heading = motion["posed_joints"], motion["global_rot_mats"], motion["global_root_heading"]
    positions = positions[0] if positions.ndim == 4 else positions
    rotations = rotations[0] if rotations.ndim == 5 else rotations
    heading = heading[0] if heading.ndim == 3 else heading
    frames = torch.as_tensor(inputs["keyframes"], dtype=torch.long)
    path = torch.as_tensor(inputs["smooth_root_2d"], dtype=torch.float32)
    # Both constraint types share the same root path at pose keyframes.
    constraints = [Root2DConstraintSet(skeleton, torch.arange(len(path)), path, global_root_heading=heading),
                   FullBodyConstraintSet(skeleton, frames, positions[frames], rotations[frames],
                                         smooth_root_2d=path[frames])]
    save_constraints_lst(str(folder / "constraints.json"), constraints)
    model = load_model(MODEL, device="cuda")
    if model.fps != metadata["fps"]:
        raise ValueError("Kimodo model FPS differs from the prepared input")
    seed_everything(seed)
    output = model(prompts, lengths, constraint_lst=[constraint.to("cuda") for constraint in constraints],
                   num_denoising_steps=steps, num_samples=num_samples, multi_prompt=True,
                   first_heading_angle=initial_heading, cfg_type="separated", cfg_weight=[2., 3.],
                   num_transition_frames=5,
                   post_processing=False, return_numpy=True)
    exported = np.asarray(MujocoQposConverter(model.skeleton).dict_to_qpos(output, "cuda"))
    if exported.shape != (num_samples, len(path), 36) or not np.isfinite(exported).all():
        raise ValueError("Kimodo returned an unexpected or nonfinite native G1 export")
    metadata["joint_names"] = names
    metadata["provenance"].update(
        prompts=prompts, num_frames=lengths, seed=seed, num_samples=num_samples, diffusion_steps=steps,
        cfg_type="separated", cfg_weight=[2., 3.], first_heading_angle=initial_heading,
        num_transition_frames=5, post_processing=False,
        input_sha256=sha256(folder / "input.npz"), constraints_sha256=sha256(folder / "constraints.json"),
        input_metadata_sha256=sha256(folder / "metadata.json"), generation_script_sha256=sha256(__file__),
        kimodo_revision=git_revision(Path(kimodo.__file__).resolve().parent),
        keyframes=inputs["keyframes"].tolist())
    for index in range(num_samples):
        stem = folder / f"{output_stem}_{index:02d}"
        np.savetxt(stem.with_suffix(".csv"), exported[index], delimiter=",")
        one = {key: (value[index] if hasattr(value, "shape") and len(value.shape) > 0
                     and value.shape[0] == num_samples else value) for key, value in output.items()}
        save_kimodo_npz(str(stem.with_suffix(".npz")), one)
        metadata["provenance"]["sample_index"] = index
        stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {num_samples} kinematic candidates in {folder}")


def refine(source: Path, output: Path, native_motion: Path, *, overwrite=False):
    """Fit this recipe's back/feet geometry, then reduce stance sliding with leg IK."""
    import shutil
    import mujoco
    from scipy.ndimage import gaussian_filter1d, maximum_filter1d
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from motion_data import load_motion, save_motion
    from motion_data.converters._common import g1_joints
    from motion_data.door import validate_door_asset
    from supermimic.assets.g1 import make_g1_spec
    from supermimic.tasks.open_door.geometry import _surface_candidates

    source_xml, output_xml = source.with_suffix(".xml"), output.with_suffix(".xml")
    inputs = {path.resolve() for path in (source, source_xml, native_motion)}
    if output.suffix != ".npz" or output.resolve() in inputs or output_xml.resolve() in inputs:
        raise ValueError("Refine needs a separate output NPZ/XML pair; input files cannot be overwritten")
    if not overwrite and (output.exists() or output_xml.exists()):
        raise ValueError("Refined output already exists; use --overwrite to replace it")
    clip = load_motion(source)
    if "postprocess" in clip:
        raise ValueError("Refine requires the original imported native G1 motion, not an already refined clip")
    validate_door_asset(clip, source_xml)
    expected = {"door_joint_axis": [0, 0, -1], "door_base_yaw": 0,
                "door_panel_half_size": [.02, .275, .9], "door_hinge_to_panel": [0, -.275, 0],
                "door_joint_range": [0, 1.8], "door_hinge_pos": [-.127574118618, .132627631264, .92]}
    for key, value in expected.items():
        if not np.allclose(clip[key], value, atol=1e-6, rtol=0):
            raise ValueError(f"Unsupported refinement {key}: expected {value}")
    count, fps = len(clip["root_pos"]), float(clip["fps"])
    if count != 270 or fps != 30 or set(clip["joint_names"].tolist()) != set(g1_joints()[0].tolist()):
        raise ValueError("Refine supports only this recipe's 270-frame, 30 Hz, 29-joint G1 export")
    if str(clip.get("door_coordinate_frame", "")) != "asset" or str(clip.get("door_contact_region", "")) != "back":
        raise ValueError("Refine requires a back-contact motion in the door asset frame")
    phase = np.clip((np.arange(count) / fps - 2.5) / 5, 0, 1)
    if not np.allclose(clip["door_angle"], 1.1 * phase ** 2 * (3 - 2 * phase), atol=1e-9, rtol=0):
        raise ValueError("Refine requires the authored 2.5–7.5 second, 1.1 rad door trajectory")
    if native_motion.stem != Path(str(clip.get("source_name", ""))).stem:
        raise ValueError("Native motion name must match the imported source CSV name")
    with np.load(native_motion, allow_pickle=False) as native:
        if "foot_contacts" not in native:
            raise ValueError("Native Kimodo motion is missing foot_contacts")
        labels = np.asarray(native["foot_contacts"])
        if (labels.shape != (count, 4) or labels.dtype.kind not in "bifu"
                or not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all()):
            raise ValueError("Native foot_contacts must contain finite binary values with shape (270, 4)")
        if "root_positions" not in native or native["root_positions"].shape != (count, 3):
            raise ValueError("Native motion frame count does not match the imported clip")
    native_csv = native_motion.with_suffix(".csv")
    if native_csv.exists() and sha256(native_csv) != str(clip.get("source_sha256", "")):
        raise ValueError("Native motion's paired CSV does not match the imported source hash")
    labels = labels.astype(bool)
    model = make_g1_spec(interaction_collisions=True).compile()
    data = mujoco.MjData(model)
    names = clip["joint_names"]
    addresses = np.array([model.joint(str(name)).qposadr[0] for name in names])
    bodies = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
    leg_indices, bounds, shapes = [], [], []
    for side, label in enumerate(("left", "right")):
        ids = np.array([index for index, name in enumerate(names) if str(name).startswith(label)
                        and any(part in str(name) for part in ("hip_", "knee_", "ankle_"))])
        if len(ids) != 6:
            raise ValueError(f"Expected six {label} leg joints")
        leg_indices.append(ids)
        bounds.append(model.jnt_range[[model.joint(str(names[index])).id for index in ids]])
        geom = model.geom(label + "_ankle_roll_link_object_collision").id
        data.qpos[:] = model.qpos0
        mujoco.mj_forward(model, data)
        vertices = _surface_candidates(model, geom) @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        shapes.append((vertices - data.xpos[bodies[side]]) @ data.xmat[bodies[side]].reshape(3, 3))

    def set_frame(frame):
        data.qpos[:3], data.qpos[3:7], data.qpos[addresses] = (
            clip["root_pos"][frame], clip["root_quat"][frame], clip["joint_pos"][frame])
        mujoco.mj_kinematics(model, data)

    def foot_poses():
        positions, rotations = np.empty((count, 2, 3)), np.empty((count, 2, 3, 3))
        for frame in range(count):
            set_frame(frame)
            for side, body in enumerate(bodies):
                positions[frame, side] = data.xpos[body]
                rotations[frame, side] = data.xmat[body].reshape(3, 3)
        return positions, rotations

    def solve_feet(targets, rotations, max_nfev):
        errors = np.empty((count, 2))
        for frame in range(count):
            set_frame(frame)
            for side, ids in enumerate(leg_indices):
                original = clip["joint_pos"][frame, ids].copy()
                lower, upper = bounds[side].T

                def residual(joints):
                    data.qpos[addresses[ids]] = joints
                    mujoco.mj_kinematics(model, data)
                    rotation = data.xmat[bodies[side]].reshape(3, 3)
                    return np.r_[30 * (data.xpos[bodies[side]] - targets[frame, side]),
                                 3 * Rotation.from_matrix(rotations[frame, side].T @ rotation).as_rotvec(),
                                 .02 * (joints - original)]

                result = least_squares(residual, np.clip(original, lower + 1e-7, upper - 1e-7),
                                       bounds=(lower + 1e-8, upper - 1e-8), max_nfev=max_nfev,
                                       ftol=1e-8, xtol=1e-8, gtol=1e-8)
                errors[frame, side] = np.linalg.norm(residual(result.x)[:3]) / 30
                if not np.isfinite(result.x).all() or not np.isfinite(errors[frame, side]) or errors[frame, side] > .03:
                    raise ValueError(f"Leg IK failed at frame {frame}, side {side}: {errors[frame, side]:.6f} m")
                clip["joint_pos"][frame, ids] = result.x
        return errors

    normals = np.column_stack([np.cos(clip["door_angle"]), -np.sin(clip["door_angle"]), np.zeros(count)])
    phase = np.clip(np.arange(count) / fps / 2.5, 0, 1)
    goal_gap = .005 + .27 * (1 - phase ** 2 * (3 - 2 * phase))
    shifts = np.empty(count)
    torso = model.body("torso_link").id
    for frame in range(count):
        set_frame(frame)
        back = data.xpos[torso] + data.xmat[torso].reshape(3, 3) @ np.array([-.06861, .00763, .213])
        gap = (back - clip["door_hinge_pos"]) @ normals[frame] - .02
        shifts[frame] = goal_gap[frame] - gap
        clip["root_pos"][frame] += normals[frame] * shifts[frame]
    feet, rotations = foot_poses()
    targets = feet.copy()
    for side in range(2):
        offsets = np.einsum("tij,vj->tvi", rotations[:, side], shapes[side])
        signed = np.einsum("tvi,ti->tv", offsets + feet[:, side, None] - clip["door_hinge_pos"], normals) - .02
        shift = np.maximum(.004 - signed.min(1), 0)
        shift = np.maximum(shift, gaussian_filter1d(maximum_filter1d(shift, size=11), 2))
        targets[:, side] += shift[:, None] * normals
        targets[:, side, 2] += np.maximum(.002 - (offsets[:, :, 2] + feet[:, side, None, 2]).min(1), 0)
    clearance_errors = solve_feet(targets, rotations, max_nfev=35)

    # Freeze root/upper body after clearance fitting; move only each leg's six joints.
    feet, rotations = foot_poses()
    targets, target_rotations = feet.copy(), rotations.copy()
    stance = np.zeros((count, 2), dtype=bool)
    for side in range(2):
        active = np.flatnonzero(labels[:, 2 * side:2 * side + 2].all(axis=1))
        groups = [group for group in np.split(active, np.where(np.diff(active) > 1)[0] + 1) if len(group) >= 5]
        for group in groups:
            anchor = np.median(feet[group, side], axis=0)
            middle = rotations[group[len(group) // 2], side]
            rotation = Rotation.from_euler("z", np.arctan2(middle[1, 0], middle[0, 0])).as_matrix()
            targets[group, side], target_rotations[group, side], stance[group, side] = anchor, rotation, True
        off = np.flatnonzero(~stance[:, side])
        swings = [group for group in np.split(off, np.where(np.diff(off) > 1)[0] + 1) if len(group)]
        for group in swings:
            lo, hi = max(int(group[0]) - 1, 0), min(int(group[-1]) + 1, count - 1)
            if hi - lo < 20:
                middle = (lo + hi) // 2
                lo, hi = max(middle - 10, 0), min(middle + 10, count - 1)
            indices = np.arange(lo + 1, hi)
            stance[indices, side] = False
            for frame in indices:
                fraction = (frame - lo) / max(hi - lo, 1)
                weight = fraction ** 2 * (3 - 2 * fraction)
                targets[frame, side] = (1 - weight) * targets[lo, side] + weight * targets[hi, side]
                rotation = target_rotations[lo, side]
                delta = Rotation.from_matrix(rotation.T @ target_rotations[hi, side]).as_rotvec()
                target_rotations[frame, side] = rotation @ Rotation.from_rotvec(weight * delta).as_matrix()
                targets[frame, side, 2] += .03 * np.sin(np.pi * fraction) ** 2
    heel_relax = np.empty((count, 2))
    for side in range(2):
        offsets = np.einsum("tij,vj->tvi", target_rotations[:, side], shapes[side])
        targets[:, side, 2] = np.maximum(targets[:, side, 2], .002 - offsets[:, :, 2].min(1))
        signed = np.einsum("tvi,ti->tv", offsets + targets[:, side, None] - clip["door_hinge_pos"], normals) - .02
        heel_relax[:, side] = np.maximum(.004 - signed.min(1), 0)
        targets[:, side] += heel_relax[:, side, None] * normals
    stance_errors = solve_feet(targets, target_rotations, max_nfev=50)
    source_hashes = {"source_npz": sha256(source), "source_xml": sha256(source_xml),
                     "native_motion": sha256(native_motion), "script": sha256(__file__)}
    parameters = {"recipe": "back_door_v2_gentle", "frames": count, "fps": fps,
                  "contact_gap_m": .005, "approach_distance_m": .27, "approach_seconds": 2.5,
                  "heel_clearance_m": .004, "floor_clearance_m": .002, "stance_min_frames": 5,
                  "swing_min_frames": 20, "swing_lift_m": .03, "ik_max_residual_m": .03,
                  "clearance_ik_evaluations": 35, "stance_ik_evaluations": 50,
                  "ik_weights": [30, 3, .02], "heel_max_filter_frames": 11, "heel_gaussian_sigma": 2,
                  "root_changes": "first-stage translation along door normal only",
                  "joint_changes": "six joints per leg; upper body and root orientation preserved"}
    clip.update(
        kinematic_only=np.asarray(True),
        postprocess=np.asarray("Back normal fitting, heel/floor clearance, stance anchors and swing leg IK; no dynamics"),
        postprocess_parameters=np.asarray(json.dumps(parameters, sort_keys=True)),
        postprocess_source_name=np.asarray(source.name),
        postprocess_source_hashes=np.asarray(json.dumps(source_hashes, sort_keys=True)),
        postprocess_chain_sha256=np.asarray(hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()),
        postprocess_labels_source=np.asarray(native_motion.name + ":foot_contacts; generated labels, not measured contacts"),
        postprocess_foot_contacts=labels, postprocess_stance=stance,
        postprocess_root_normal_shift=shifts, postprocess_root_shift_max=np.asarray(np.abs(shifts).max()),
        postprocess_clearance_ik_error_m=clearance_errors, postprocess_stance_ik_error_m=stance_errors,
        postprocess_ik_error_max_m=np.asarray(max(clearance_errors.max(), stance_errors.max())),
        postprocess_heel_relax_m=heel_relax,
        postprocess_leg_velocity_max_rad_s=np.asarray(np.abs(np.diff(clip["joint_pos"][:, np.concatenate(leg_indices)], axis=0)).max() * fps))
    if any(not np.isfinite(clip[key]).all() for key in ("root_pos", "root_quat", "joint_pos")):
        raise ValueError("Refinement produced nonfinite poses")
    validate_door_asset(clip, source_xml)
    save_motion(output, clip)
    shutil.copy2(source_xml, output_xml)
    print(json.dumps({"output": str(output), "kinematic_only": True,
                      "clearance_ik_error_max_m": float(clearance_errors.max()),
                      "stance_ik_error_max_m": float(stance_errors.max()),
                      "root_shift_max_m": float(np.abs(shifts).max()),
                      "leg_velocity_max_rad_s": float(clip["postprocess_leg_velocity_max_rad_s"])}, indent=2))


def _cycle_exit_targets(targets, rotations, shapes):
    """Continuous alternating steps between the exact exit poses at frames 0 and 45."""
    from scipy.spatial.transform import Rotation, Slerp

    schedules = [[(.15, .55), (.75, 1.30)], [(0., .45), (.60, 1.10)]]
    endpoints, endpoint_rotations = targets[[0, 45]].copy(), rotations[[0, 45]].copy()
    for side in range(2):
        slerp = Slerp([0, 1], Rotation.from_matrix(endpoint_rotations[:, side]))
        for index in range(1, 45):
            time, progress, height = index / 30, 0., 0.
            for start, end in schedules[side]:
                fraction = np.clip((time - start) / (end - start), 0, 1)
                progress += .5 * fraction * fraction * (3 - 2 * fraction)
                if start < time < end:
                    height = .03 * np.sin(np.pi * fraction) ** 2
            targets[index, side] = (1 - progress) * endpoints[0, side] + progress * endpoints[1, side]
            targets[index, side, 2] += height
            rotations[index, side] = slerp(progress).as_matrix()
            vertices = shapes[side] @ rotations[index, side].T + targets[index, side]
            targets[index, side, 2] += max(.002 - vertices[:, 2].min(), 0)
    return schedules


def assemble_cycle(prefix: Path, suffix: Path, output: Path, prefix_native: Path,
                   suffix_native: Path, back_motion: Path, *, overwrite=False):
    """Assemble the fixed approach/push/exit recipe; output remains kinematic only."""
    import shutil
    import mujoco
    from scipy.optimize import least_squares
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation
    from motion_data import load_motion, save_motion
    from motion_data.converters._common import g1_joints
    from motion_data.door import validate_door_asset
    from supermimic.assets.g1 import make_g1_spec
    from supermimic.tasks.open_door.geometry import _surface_candidates

    sources = [prefix, suffix, back_motion, prefix_native, suffix_native]
    inputs = sources + [path.with_suffix('.xml') for path in (prefix, suffix, back_motion)]
    inputs += [native.with_suffix(ext) for native in (prefix_native, suffix_native) for ext in ('.csv', '.json')]
    inputs += [native.parent / name for native in (prefix_native, suffix_native) for name in ('input.npz', 'metadata.json')]
    outputs = [output, output.with_suffix('.xml'), output.with_suffix('.json')]
    if output.suffix != '.npz' or {p.resolve() for p in outputs}.intersection(p.resolve() for p in inputs):
        raise ValueError('Assembly needs a separate output NPZ/XML/JSON; input files cannot be overwritten')
    if not overwrite and any(path.exists() for path in outputs):
        raise ValueError('Cycle output already exists; use --overwrite to replace it')
    back = load_motion(back_motion)
    xml = back_motion.with_suffix('.xml')
    validate_door_asset(back, xml)
    names = back['joint_names']
    if (len(back['root_pos']) != 270 or float(back['fps']) != 30
            or set(names.tolist()) != set(g1_joints()[0].tolist())
            or not np.isclose(back['door_angle'][235], 1.1)):
        raise ValueError('Cycle assembly requires the reviewed 270-frame, 30 Hz G1 back-push reference')
    static_keys = ['door_hinge_pos', 'door_base_yaw', 'door_panel_half_size', 'door_hinge_to_panel',
                   'door_joint_axis', 'door_joint_range', 'door_coordinate_frame', 'door_contact_region', 'root_quat_order']
    expected = {'door_joint_axis': [0, 0, -1], 'door_base_yaw': 0,
                'door_panel_half_size': [.02, .275, .9], 'door_hinge_to_panel': [0, -.275, 0]}
    if any(not np.allclose(back[key], value, atol=1e-6, rtol=0) for key, value in expected.items()):
        raise ValueError('Cycle assembly requires the fixed 0.55 m, -Z-hinge door asset')
    arms, arm_pose = fixed_arms(names)
    back['joint_pos'][:, arms] = arm_pose
    segments, labels, designs = [], [], []
    for part, path, native, count in zip(('prefix', 'suffix'), (prefix, suffix),
                                        (prefix_native, suffix_native), (255, 330)):
        clip = load_motion(path)
        validate_door_asset(clip, path.with_suffix('.xml'))
        if (len(clip['root_pos']) != count or float(clip['fps']) != 30
                or not np.array_equal(clip['joint_names'], names)
                or any(not np.array_equal(clip[key], back[key]) for key in static_keys)
                or sha256(path.with_suffix('.xml')) != sha256(xml)):
            raise ValueError(f'{part}: expected {count} frames at 30 Hz with matching G1 joints and paired door asset')
        if 'postprocess' in clip or str(clip.get('door_contact_region', '')) != 'back':
            raise ValueError(f'{part}: assembly requires an original imported back-contact candidate')
        native_json, native_csv = native.with_suffix('.json'), native.with_suffix('.csv')
        if (native.stem != Path(str(clip.get('source_name', ''))).stem
                or sha256(native_csv) != str(clip.get('source_sha256', ''))
                or sha256(native_json) != str(clip.get('source_metadata_sha256', ''))):
            raise ValueError(f'{part}: native NPZ/CSV/metadata do not match the imported candidate')
        meta = json.loads((native.parent / 'metadata.json').read_text())
        generated_meta = json.loads(native_json.read_text())
        with np.load(native, allow_pickle=False) as archive:
            if not {'foot_contacts', 'root_positions'}.issubset(archive.files):
                raise ValueError(f'{part}: native motion needs root_positions and foot_contacts')
            contact = np.asarray(archive['foot_contacts'])
            if (archive['root_positions'].shape != (count, 3) or contact.shape != (count, 4)
                    or contact.dtype.kind not in 'bifu' or not np.isfinite(contact).all()
                    or not np.isin(contact, [0, 1]).all()):
                raise ValueError(f'{part}: native pose/contact arrays must match the imported frame count')
            native_root = np.asarray(archive['root_positions'])[:, [2, 0, 1]]
        csv = np.loadtxt(native_csv, delimiter=',', ndmin=2)
        if (csv.shape != (count, 36) or not np.isfinite(csv).all()
                or not np.allclose(native_root, csv[:, :3], atol=1e-6, rtol=0)
                or generated_meta['joint_names'] != names.tolist()):
            raise ValueError(f'{part}: native NPZ and CSV must describe the same G1 trajectory')
        with np.load(native.parent / 'input.npz', allow_pickle=False) as archive:
            qpos = np.asarray(archive['qpos']).copy()
            if not np.array_equal(archive['joint_names'], names):
                raise ValueError(f'{part}: prepared input joint order differs from the imported candidate')
        input_hash = generated_meta.get('provenance', {}).get('input_sha256')
        if input_hash != sha256(native.parent / 'input.npz'):
            raise ValueError(f'{part}: generated native metadata does not match the prepared input')
        offset = np.asarray(meta['world_offset'], dtype=float)
        if (meta['fps'] != 30 or generated_meta['fps'] != 30 or qpos.shape != (count, 36)
                or sum(meta['num_frames']) != count or offset.shape != (3,)
                or not np.isfinite(qpos).all() or not np.isfinite(offset).all()
                or not np.array_equal(offset, np.asarray(generated_meta['world_offset']))):
            raise ValueError(f'{part}: expected a finite matching 30 Hz design path')
        csv_quaternions = csv[:, 3:7] / np.linalg.norm(csv[:, 3:7], axis=1, keepdims=True)
        if (not np.allclose(clip['root_pos'], csv[:, :3] + offset, atol=1e-6, rtol=0)
                or not np.allclose(abs(np.sum(clip['root_quat'] * csv_quaternions, axis=1)), 1, atol=1e-6, rtol=0)
                or not np.allclose(clip['joint_pos'], csv[:, 7:], atol=1e-6, rtol=0)
                or not np.array_equal(clip['door_angle'], np.asarray(generated_meta['door_angle']))):
            raise ValueError(f'{part}: imported poses must still match the native CSV and metadata')
        source_back_hash = meta.get('provenance', {}).get('source_back_sha256')
        if source_back_hash is not None and source_back_hash != sha256(back_motion):
            raise ValueError(f'{part}: cycle design was prepared for a different back-push reference')
        qpos[:, :3] += offset
        segments.append(clip)
        labels.append(contact.astype(bool))
        designs.append(qpos)

    robot = make_g1_spec(interaction_collisions=True).compile()
    data = mujoco.MjData(robot)
    addresses = np.array([robot.joint(str(name)).qposadr[0] for name in names])
    foot_bodies = [robot.body(side + '_ankle_roll_link').id for side in ('left', 'right')]
    torso, logo = robot.body('torso_link').id, robot.body('logo_link').id
    leg_ids, bounds, shapes = [], [], []
    data.qpos[:] = robot.qpos0
    mujoco.mj_forward(robot, data)
    for side, body in zip(('left', 'right'), foot_bodies):
        ids = np.array([j for j, name in enumerate(names) if str(name).startswith(side)
                        and any(part in str(name) for part in ('hip_', 'knee_', 'ankle_'))])
        if len(ids) != 6:
            raise ValueError(f'Expected six {side} leg joints')
        leg_ids.append(ids)
        bounds.append(robot.jnt_range[[robot.joint(str(names[j])).id for j in ids]])
        geom = robot.geom(side + '_ankle_roll_link_object_collision').id
        vertices = _surface_candidates(robot, geom) @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        shapes.append((vertices - data.xpos[body]) @ data.xmat[body].reshape(3, 3))
    back_patch = []
    for geom in range(robot.ngeom):
        if robot.geom_bodyid[geom] not in (torso, logo) or not robot.geom_conaffinity[geom] & 2:
            continue
        vertices = _surface_candidates(robot, geom) @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        local = (vertices - data.xpos[torso]) @ data.xmat[torso].reshape(3, 3)
        x, y, z = local.T
        back_patch.extend(local[(x < -.025) & (abs(y) < .13) & (z > .05) & (z < .38)])
    back_patch = np.asarray(back_patch)
    if not len(back_patch):
        raise ValueError('G1 interaction asset has no legal rear torso patch')
    door_model = mujoco.MjModel.from_xml_path(str(xml))
    door_data = mujoco.MjData(door_model)
    box_ids = [g for g in range(door_model.ngeom) if door_model.geom(g).name == 'door_panel'
               or door_model.geom(g).name.startswith('door_wall')]
    if len(box_ids) != 3 or any(door_model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX for g in box_ids):
        raise ValueError('Cycle assembly requires one finite door panel and two wall boxes')

    def smooth(value):
        value = np.clip(value, 0, 1)
        return value * value * (3 - 2 * value)

    def frame(clip, index):
        data.qpos[:3], data.qpos[3:7], data.qpos[addresses] = (
            clip['root_pos'][index], clip['root_quat'][index], clip['joint_pos'][index])
        mujoco.mj_kinematics(robot, data)

    def foot_poses(clip):
        positions = np.empty((len(clip['root_pos']), 2, 3))
        rotations = np.empty((len(positions), 2, 3, 3))
        for index in range(len(positions)):
            frame(clip, index)
            for side, body in enumerate(foot_bodies):
                positions[index, side], rotations[index, side] = data.xpos[body], data.xmat[body].reshape(3, 3)
        return positions, rotations

    def solve_feet(clip, targets, rotations, indices, *, previous=False):
        errors = []
        for index in indices:
            frame(clip, index)
            for side, ids in enumerate(leg_ids):
                reference = clip['joint_pos'][index, ids].copy()
                initial = clip['joint_pos'][index - 1, ids] if previous and index else reference
                lower, upper = bounds[side].T

                def residual(joints):
                    data.qpos[addresses[ids]] = joints
                    mujoco.mj_kinematics(robot, data)
                    rotation = data.xmat[foot_bodies[side]].reshape(3, 3)
                    return np.r_[30 * (data.xpos[foot_bodies[side]] - targets[index, side]),
                                 3 * Rotation.from_matrix(rotations[index, side].T @ rotation).as_rotvec(),
                                 .02 * (joints - reference)]

                result = least_squares(residual, np.clip(initial, lower + 1e-7, upper - 1e-7),
                                       bounds=(lower + 1e-8, upper - 1e-8), max_nfev=100,
                                       ftol=1e-9, xtol=1e-9, gtol=1e-9)
                error = np.linalg.norm(residual(result.x)[:3]) / 30
                if not result.success or not np.isfinite(result.x).all() or not np.isfinite(error) or error > .02:
                    raise ValueError(f'Cycle foot IK failed at frame {index}, side {side}: {error:.6f} m')
                clip['joint_pos'][index, ids] = result.x
                errors.append(float(error))
        return max(errors, default=0.)

    def clear_feet(clip, *, preserve=()):
        targets, rotations = foot_poses(clip)
        changed = set()
        for index in range(len(targets)):
            if index in preserve:
                continue
            door_data.joint('door_hinge').qpos[0] = clip['door_angle'][index]
            mujoco.mj_kinematics(door_model, door_data)
            for side in range(2):
                offsets = shapes[side] @ rotations[index, side].T
                target = targets[index, side]
                original = target.copy()
                target[2] += max(.002 - (offsets[:, 2] + target[2]).min(), 0)
                for _ in range(20):
                    moved = False
                    for geom in box_ids:
                        rotation = door_data.geom_xmat[geom].reshape(3, 3)
                        if not np.allclose(rotation[:, 2], [0, 0, 1], atol=1e-6):
                            raise ValueError('Only vertical finite door/wall boxes are supported')
                        local = (offsets + target - door_data.geom_xpos[geom]) @ rotation
                        lo, hi = local.min(0), local.max(0)
                        half = door_model.geom_size[geom] + .004
                        if np.all(hi > -half + 1e-10) and np.all(lo < half - 1e-10):
                            moves = np.array([half[0] - lo[0], -half[0] - hi[0], half[1] - lo[1], -half[1] - hi[1]])
                            selected = int(np.argmin(abs(moves)))
                            target += rotation[:, selected // 2] * moves[selected]
                            moved = True
                    if not moved:
                        break
                else:
                    raise ValueError(f'No consistent finite foot clearance at frame {index}, side {side}')
                if np.linalg.norm(target - original) > 1e-8:
                    changed.add(index)
        return solve_feet(clip, targets, rotations, sorted(changed))

    def lock_stance(clip, contact, part):
        feet, rotations = foot_poses(clip)
        targets, target_rotations = feet.copy(), rotations.copy()
        stance = np.zeros((len(feet), 2), dtype=bool)
        start = 35 if part == 'suffix' else 0

        def groups(mask):
            indices = np.flatnonzero(mask)
            return [group for group in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if len(group)]

        for side in range(2):
            active = contact[:, 2 * side:2 * side + 2].all(axis=1)
            active[:start] = False
            for group in groups(active):
                if len(group) < 5:
                    continue
                if part == 'prefix' and group[-1] == len(feet) - 1:
                    anchor, rotation = feet[-1, side], rotations[-1, side]
                else:
                    anchor = np.median(feet[group, side], axis=0)
                    middle = rotations[group[len(group) // 2], side]
                    rotation = Rotation.from_euler('z', np.arctan2(middle[1, 0], middle[0, 0])).as_matrix()
                targets[group, side], target_rotations[group, side], stance[group, side] = anchor, rotation, True
            for group in groups(~stance[:, side]):
                lo, hi = max(int(group[0]) - 1, 0), min(int(group[-1]) + 1, len(feet) - 1)
                if part == 'suffix' and side == 0 and lo == 222:
                    lo -= 10  # Leave the long generated stance before the leg becomes unreachable.
                if part == 'suffix' and lo == 0:
                    lo, hi = start - 1, max(start + 19, hi)
                elif hi - lo < 20:
                    middle = (lo + hi) // 2
                    lo, hi = max(middle - 10, start), min(middle + 10, len(feet) - 1)
                if hi <= lo:
                    continue
                rotation = target_rotations[lo, side]
                delta = Rotation.from_matrix(rotation.T @ target_rotations[hi, side]).as_rotvec()
                for index in range(max(lo + 1, start), hi):
                    fraction = (index - lo) / (hi - lo)
                    weight = smooth(fraction)
                    targets[index, side] = (1 - weight) * targets[lo, side] + weight * targets[hi, side]
                    targets[index, side, 2] += .03 * np.sin(np.pi * fraction) ** 2
                    target_rotations[index, side] = rotation @ Rotation.from_rotvec(weight * delta).as_matrix()
            offsets = np.einsum('tij,vj->tvi', target_rotations[:, side], shapes[side])
            targets[start:, side, 2] = np.maximum(targets[start:, side, 2], .002 - offsets[start:, :, 2].min(1))
        preserve = [len(feet) - 1] if part == 'prefix' else list(range(start))
        error = solve_feet(clip, targets, target_rotations, [i for i in range(start, len(feet)) if i not in preserve])
        return max(error, clear_feet(clip, preserve=preserve))

    def release_curve(clip):
        along = []
        tangent = np.array([-np.sin(1.1), -np.cos(1.1), 0])
        for index in range(len(clip['root_pos'])):
            frame(clip, index)
            patch_world = back_patch @ data.xmat[torso].reshape(3, 3).T + data.xpos[torso]
            along.append(((patch_world - clip['door_hinge_pos']) @ tangent).min())
        along = np.asarray(along)
        departures = np.flatnonzero(along > .55)
        if not len(departures) or departures[0] == 0:
            raise ValueError('Exit does not contain a rear-patch departure from the finite door edge')
        index = int(departures[0])
        release = (index - 1 + (.55 - along[index - 1]) / (along[index] - along[index - 1])) / 30
        mujoco.mj_resetData(door_model, door_data)
        door_data.joint('door_hinge').qpos[0] = 1.1
        dt, passive = door_model.opt.timestep, [1.1]
        for _ in range(int(np.ceil(len(along) / 30 / dt)) + 2):
            mujoco.mj_step(door_model, door_data)
            passive.append(max(float(door_data.joint('door_hinge').qpos[0]), 0.))
        if not np.isfinite(passive).all():
            raise ValueError('Passive door integration produced nonfinite angles')
        times = np.arange(len(along)) / 30
        clip['door_angle'] = np.interp(np.maximum(times - release, 0), np.arange(len(passive)) * dt, passive)
        closed = np.flatnonzero(np.asarray(passive) <= 1e-5)
        return float(release), float(release + closed[0] * dt) if len(closed) else None

    errors, release, closing = {}, None, None
    for part, clip, contact, design, seam in zip(('prefix', 'suffix'), segments, labels, designs, (0, 235)):
        count = len(clip['root_pos'])
        clip['root_pos'][:, :2] = design[:, :2]
        rpy = Rotation.from_quat(clip['root_quat'], scalar_first=True).as_euler('xyz')
        rpy[:, 2] = Rotation.from_quat(design[:, 3:7], scalar_first=True).as_euler('xyz')[:, 2]
        clip['root_quat'] = Rotation.from_euler('xyz', rpy).as_quat(scalar_first=True)
        clip['joint_pos'][:, arms] = arm_pose
        indices = np.arange(count - 24, count) if part == 'prefix' else np.arange(24)
        weights = smooth(np.linspace(0, 1, 24) if part == 'prefix' else np.linspace(1, 0, 24))
        rotation = Rotation.from_quat(back['root_quat'][seam], scalar_first=True)
        for index, weight in zip(indices, weights):
            clip['root_pos'][index, 2] = (1 - weight) * clip['root_pos'][index, 2] + weight * back['root_pos'][seam, 2]
            current = Rotation.from_quat(clip['root_quat'][index], scalar_first=True)
            clip['root_quat'][index] = (current * Rotation.from_rotvec(weight * (current.inv() * rotation).as_rotvec())).as_quat(scalar_first=True)
            clip['joint_pos'][index] = (1 - weight) * clip['joint_pos'][index] + weight * back['joint_pos'][seam]
        endpoint = -1 if part == 'prefix' else 0
        for key in ('root_pos', 'root_quat', 'joint_pos'):
            clip[key][endpoint] = back[key][seam]
        if part == 'suffix':
            normal = np.array([np.cos(1.1), -np.sin(1.1), 0])
            for index in range(count):
                weight = 1 - smooth((index / 30 - .5) / .5)
                if weight:
                    frame(clip, index)
                    back_point = data.xpos[torso] + data.xmat[torso].reshape(3, 3) @ np.array([-.06861, .00763, .213])
                    gap = (back_point - clip['door_hinge_pos']) @ normal - .02
                    clip['root_pos'][index] += weight * (.005 - gap) * normal
            release, closing = release_curve(clip)
        errors[part + '_clearance_m'] = clear_feet(clip, preserve=[count - 1] if part == 'prefix' else [0])
        errors[part + '_stance_m'] = lock_stance(clip, contact, part)
        if part == 'suffix':
            if release >= .5:
                raise ValueError('This exit recipe requires rear-patch departure before the 0.5 s clearance step')
            clip['root_pos'][:, :2] += smooth((np.arange(count) / 30 - .5) / .3)[:, None] * np.array([-.028, -.04])
            release, closing = release_curve(clip)
            targets, rotations = foot_poses(clip)
            early_schedules = _cycle_exit_targets(targets, rotations, shapes)
            errors['suffix_early_steps_m'] = solve_feet(clip, targets, rotations, range(1, 45), previous=True)
        for key in ('root_pos', 'root_quat', 'joint_pos'):
            if not np.allclose(clip[key][endpoint], back[key][seam], atol=1e-12, rtol=0):
                raise ValueError(f'{part}: endpoint no longer matches the preserved back-push seam')

    # Native contact labels may include a hovering stance. Translate the body
    # down smoothly, without pushing either foot through the ground or moving
    # any joint. The exit's initial pose and the preserved push remain intact.
    feet, rotations = foot_poses(segments[1])
    lowest = np.stack([(np.einsum('tij,vj->tvi', rotations[:, side], shapes[side])[:, :, 2]
                       + feet[:, side, None, 2]).min(1) for side in range(2)], axis=1).min(1)
    ground_shift = np.maximum(lowest - .003, 0)
    ground_shift = np.minimum(ground_shift, gaussian_filter1d(ground_shift, 2))
    ground_shift *= smooth(np.arange(len(ground_shift)) / 30 / .3)
    segments[1]['root_pos'][:, 2] -= ground_shift

    result = {key: back[key] for key in static_keys}
    for key in ('root_pos', 'root_quat', 'joint_pos', 'door_angle'):
        result[key] = np.concatenate([segments[0][key][:-1], back[key][:235], segments[1][key]])
    result.update(fps=np.asarray(30.), joint_names=names, kinematic_only=np.asarray(True),
                  door_angle_source=np.asarray('authored push; passive nominal spring release after rear-patch departure'))
    report = {'recipe': 'kimodo_back_door_cycle', 'kinematic_only': True, 'frames': len(result['root_pos']),
              'fps': 30, 'stages': {'approach_turn': [0, 254], 'back_push': [254, 489], 'exit_turn_walk': [489, 819]},
              'frame_ranges': 'start inclusive, end exclusive',
              'fixed_arm_joint_names': names[arms].tolist(), 'fixed_arm_joint_positions': arm_pose.tolist(),
              'endpoint_blend_frames': 24, 'stance_min_frames': 5, 'swing_min_frames': 20,
              'swing_lift_m': .03, 'suffix_left_liftoff_advance_frames': 10,
              'floor_clearance_m': .002, 'finite_box_clearance_m': .004,
              'ik_position_tolerance_m': .02, 'ik_weights': [30, 3, .02],
              'early_exit_end_frame': 45, 'early_exit_swing_intervals_s': early_schedules,
              'early_exit_displacement_per_step': .5, 'exit_retreat_xy_m': [-.028, -.04],
              'exit_retreat_start_s': .5, 'exit_retreat_duration_s': .3,
              'suffix_grounding_floor_m': .003, 'suffix_grounding_gaussian_sigma': 2,
              'suffix_grounding_fade_s': .3, 'suffix_grounding_max_m': float(ground_shift.max()),
              'release_suffix_s': release, 'closing_suffix_s': closing,
              'release_cycle_s': 489 / 30 + release,
              'closing_cycle_s': 489 / 30 + closing if closing is not None else None,
              'ik_error_max_m': errors,
              'source_sha256': {str(path): sha256(path) for path in inputs}, 'script_sha256': sha256(__file__)}
    result.update(postprocess=np.asarray('Fixed arm pose, authored root path, stance/continuous swing foot IK; no robot dynamics'),
                  postprocess_root_grounding_m=np.r_[np.zeros(489), ground_shift],
                  provenance=np.asarray(json.dumps(report, sort_keys=True)))
    validate_door_asset(result, xml)
    save_motion(output, result)
    shutil.copy2(xml, output.with_suffix('.xml'))
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'frames': report['frames'], 'kinematic_only': True,
                      'ik_error_max_m': errors, 'release_cycle_s': report['release_cycle_s']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Prepare reference constraints using the local G1 asset")
    prep.add_argument("folder", type=Path)
    prep.add_argument("--door-reference", type=Path, default=DEFAULT_REFERENCE)
    prep.add_argument("--overwrite", action="store_true")
    cycle = commands.add_parser("prepare-cycle", help="Prepare approach/turn and exit/turn/walk extensions")
    cycle.add_argument("folder", type=Path)
    cycle.add_argument("--back-motion", type=Path, default=DEFAULT_BACK)
    cycle.add_argument("--overwrite", action="store_true")
    gen = commands.add_parser("generate", help="Generate candidates inside the Kimodo environment")
    gen.add_argument("folder", type=Path)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--num-samples", type=int, default=4)
    gen.add_argument("--steps", type=int, default=150)
    gen.add_argument("--overwrite", action="store_true")
    ref = commands.add_parser("refine", help="Fit back/foot geometry for the fixed 270-frame back-door recipe")
    ref.add_argument("source", type=Path)
    ref.add_argument("output", type=Path)
    ref.add_argument("--native-motion", type=Path, required=True, help="Matching native Kimodo NPZ with foot_contacts")
    ref.add_argument("--overwrite", action="store_true")
    assembly = commands.add_parser('assemble-cycle', help='Fit and assemble the fixed approach/back-push/exit cycle')
    assembly.add_argument('prefix', type=Path)
    assembly.add_argument('suffix', type=Path)
    assembly.add_argument('output', type=Path)
    assembly.add_argument('--prefix-native', type=Path, required=True)
    assembly.add_argument('--suffix-native', type=Path, required=True)
    assembly.add_argument('--back-motion', type=Path, default=DEFAULT_BACK)
    assembly.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare(args.folder, args.door_reference, overwrite=args.overwrite)
        elif args.command == "prepare-cycle":
            prepare_cycle(args.folder, args.back_motion, overwrite=args.overwrite)
        elif args.command == "generate":
            generate(args.folder, seed=args.seed, num_samples=args.num_samples,
                     steps=args.steps, overwrite=args.overwrite)
        elif args.command == 'assemble-cycle':
            assemble_cycle(args.prefix, args.suffix, args.output, args.prefix_native,
                           args.suffix_native, args.back_motion, overwrite=args.overwrite)
        else:
            refine(args.source, args.output, args.native_motion, overwrite=args.overwrite)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
