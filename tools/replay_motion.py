"""Replay a standard motion NPZ using only forward kinematics (no mj_step).

uv run python tools/replay_motion.py data/motions/sitting
uv run python tools/replay_motion.py clip.npz --camera-view
uv run python tools/replay_motion.py clip.npz --sensor-view
uv run python tools/replay_motion.py clip.npz --video replay.mp4
"""

import argparse
import os
from pathlib import Path
import subprocess
import threading
import time

import numpy as np

from motion_data import load_motion

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "data/g1_description/g1_29dof.xml"
HDMI_MIN_DEPTH = 0.1
HDMI_MAX_DEPTH = 4.0


def make_scene(model_path, clip, door_xml=None):
    import mujoco

    if Path(model_path).resolve() == DEFAULT_MODEL.resolve():
        from supermimic.assets.g1 import HDMI_CAMERA_NAME, make_g1_spec
        spec = make_g1_spec()
        camera_name = HDMI_CAMERA_NAME
    else:
        spec = mujoco.MjSpec.from_file(str(Path(model_path).resolve()))
        camera_name = "d435_camera"
    # Only render visual meshes; neither drives nor collisions are needed.
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    if Path(model_path).resolve() == DEFAULT_MODEL.resolve():
        for geom in spec.geoms:
            if geom.contype or geom.conaffinity:
                geom.group = 3  # This G1 has separate visual meshes in group 1.
    spec.worldbody.add_geom(name="replay_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, .05], rgba=[.86, .88, .9, 1])
    spec.worldbody.add_light(pos=[0, 0, 5], dir=[0, 0, -1])
    if "object_vertices" in clip:
        spec.add_mesh(name="replay_object_mesh", uservert=clip["object_vertices"].ravel(),
                      userface=clip["object_faces"].ravel(), scale=clip["object_scale"])
        body = spec.worldbody.add_body(name="replay_object", mocap=True)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="replay_object_mesh",
                      rgba=[.56, .36, .18, 1], mass=0)
    elif "object_pos" in clip:
        raise ValueError("Clip has object poses but no mesh; convert it with --object-usd-dir")
    if "door_angle" in clip:
        from motion_data.door import validate_door_asset
        if door_xml is None:
            raise ValueError("Clip has door_angle; provide its matching door_xml for replay")
        if not Path(door_xml).is_file():
            raise FileNotFoundError(f"Missing paired door XML: {door_xml}")
        validate_door_asset(clip, door_xml)
        spec.attach(mujoco.MjSpec.from_file(str(Path(door_xml).resolve())),
                    prefix="replay_door/", frame=spec.worldbody.add_frame())
    for geom in spec.geoms:
        geom.contype = geom.conaffinity = 0

    model = spec.compile()
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
    model.vis.headlight.ambient[:] = .4
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError("Replay expects one floating robot root")
    root_adr = int(model.jnt_qposadr[free[0]])
    addresses = []
    for name in clip["joint_names"]:
        joint = model.joint(str(name))
        if joint.type[0] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError(f"Expected a scalar joint: {name}")
        addresses.append(int(joint.qposadr[0]))
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    return model, mujoco.MjData(model), root_adr, addresses, camera_id


def set_frame(model, data, clip, frame, root_adr, joint_adrs):
    import mujoco

    data.qpos[root_adr:root_adr + 3] = clip["root_pos"][frame]
    data.qpos[root_adr + 3:root_adr + 7] = clip["root_quat"][frame]
    data.qpos[joint_adrs] = clip["joint_pos"][frame]
    if "object_pos" in clip:
        mocap = int(model.body("replay_object").mocapid[0])
        data.mocap_pos[mocap] = clip["object_pos"][frame]
        data.mocap_quat[mocap] = clip["object_quat"][frame]
    if "door_angle" in clip:
        data.joint("replay_door/door_hinge").qpos[0] = clip["door_angle"][frame]
    data.time = frame / float(clip["fps"])
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)


def capture_hdmi_sensor_frame(renderer, data, camera_id, scene_option):
    """Return one HDMI-shaped RGB/depth observation from the asset camera."""
    import torch
    from supermimic.camera import preprocess_depth

    renderer.update_scene(data, camera=camera_id, scene_option=scene_option)
    rgb = renderer.render().copy()  # [H, W, 3], uint8

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_id, scene_option=scene_option)
    raw_depth = renderer.render().copy()  # [H, W], float32, metres
    renderer.disable_depth_rendering()

    depth = preprocess_depth(torch.from_numpy(raw_depth)[None, ..., None])[0].numpy()
    return rgb, depth, raw_depth[..., None]


def camera_replay_clip(clip):
    """Interpolate reference poses on the same 50 Hz clock as the student."""
    import torch
    from supermimic.camera import HDMI_CAMERA_METADATA
    from supermimic.rotations import quat_slerp

    dt = HDMI_CAMERA_METADATA["update_period"]
    count = len(clip["root_pos"])
    if count < 2:
        raise ValueError("Camera replay requires at least two reference frames")
    fps = float(clip["fps"])
    n = int(np.floor((count - 1) / fps / dt + 1e-9)) + 1
    time_index = np.arange(n) * dt * fps
    index = np.minimum(time_index.astype(int), count - 2)
    alpha = np.clip(time_index - index, 0, 1)
    result = dict(clip, fps=1. / dt)
    for name in ("root_pos", "joint_pos", "object_pos", "door_angle"):
        if name in clip:
            values = clip[name]
            weight = alpha.reshape((-1,) + (1,) * (values.ndim - 1))
            result[name] = values[index] * (1 - weight) + values[index + 1] * weight
    for name in ("root_quat", "object_quat"):
        if name in clip:
            values = torch.as_tensor(clip[name])
            result[name] = quat_slerp(values[index], values[index + 1], torch.as_tensor(alpha)).numpy()
    return result


def run_sensor_replay(
    model,
    data,
    clip,
    n,
    fps,
    speed,
    camera_id,
    option,
    apply,
    *,
    sensor_view=False,
    sensor_output=None,
    once=False,
):
    """Replay HDMI-shaped camera observations, optionally with live plots."""
    import mujoco

    width, height = (int(v) for v in model.cam_resolution[camera_id])
    rgb_frames = [] if sensor_output is not None else None
    depth_frames = [] if sensor_output is not None else None
    plane_frames = [] if sensor_output is not None else None

    plot = None
    if sensor_view:
        import matplotlib.pyplot as plt

        plt.ion()
        plot = plt.subplots(1, 2, figsize=(10, 4.5))
        fig, (rgb_axis, depth_axis) = plot
        rgb_image = rgb_axis.imshow(np.zeros((height, width, 3), dtype=np.uint8))
        depth_image = depth_axis.imshow(
            np.full((height, width), HDMI_MAX_DEPTH, dtype=np.float32),
            cmap="turbo", vmin=HDMI_MIN_DEPTH, vmax=HDMI_MAX_DEPTH,
        )
        rgb_axis.set_title("HDMI RGB")
        depth_axis.set_title("HDMI depth (m)")
        fig.colorbar(depth_image, ax=depth_axis, label="m")
        for axis in (rgb_axis, depth_axis):
            axis.set_axis_off()
        fig.tight_layout()

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        while True:
            for frame in range(n):
                start = time.monotonic()
                apply(frame)
                rgb, depth, plane = capture_hdmi_sensor_frame(renderer, data, camera_id, option)
                if rgb_frames is not None:
                    rgb_frames.append(rgb)
                    depth_frames.append(depth)
                    plane_frames.append(plane)
                if plot is not None:
                    import matplotlib.pyplot as plt

                    if not plt.fignum_exists(plot[0].number):
                        plot = None
                        break
                    plot[1][0].images[0].set_data(rgb)
                    plot[1][1].images[0].set_data(depth[0])
                    plot[0].canvas.draw_idle()
                    plot[0].canvas.flush_events()
                    plt.pause(max(0.001, 1 / (fps * speed) - (time.monotonic() - start)))
            if once or sensor_output is not None or plot is None:
                break

    if sensor_output is not None:
        sensor_output = Path(sensor_output)
        sensor_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            sensor_output,
            rgb=np.stack(rgb_frames, axis=0),
            depth=np.stack(depth_frames, axis=0),
            distance_to_image_plane=np.stack(plane_frames, axis=0),
            fps=np.float32(fps),
            capture_time=np.arange(len(rgb_frames), dtype=np.float64) / fps,
            frame_id=np.arange(len(rgb_frames), dtype=np.int64),
        )
        print(
            f"Saved HDMI sensor output {sensor_output} "
            f"(rgb={np.asarray(rgb_frames).shape}, depth={np.asarray(depth_frames).shape})"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path, help="NPZ file or directory of clips")
    parser.add_argument("--index", type=int, default=0, help="Clip index in a sorted directory (default: 0)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--once", action="store_true", help="Exit after one pass; default loops")
    parser.add_argument("--headless", action="store_true", help="Check all frames without a display")
    parser.add_argument("--camera-view", action="store_true",
                        help="View the replay from the mounted d435 camera")
    parser.add_argument("--sensor-view", action="store_true",
                        help="Show live HDMI RGB and depth sensor outputs")
    parser.add_argument("--sensor-output", type=Path,
                        help="Save HDMI-shaped RGB/depth outputs to a compressed NPZ")
    parser.add_argument("--video", type=Path, help="Render one pass to MP4 (requires ffmpeg)")
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    files = sorted(args.motion.glob("*.npz")) if args.motion.is_dir() else [args.motion]
    if not files or not 0 <= args.index < len(files):
        parser.error(f"No clip at index {args.index}; found {len(files)} clips")
    if args.video or args.sensor_output:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    path = files[args.index]
    clip = load_motion(path)
    if args.sensor_view or args.sensor_output:
        clip = camera_replay_clip(clip)
    n, fps = len(clip["root_pos"]), float(clip["fps"])
    print(f"[{args.index}/{len(files) - 1}] {path.name}: {n} frames, {fps:g} Hz, {n / fps:g}s")
    model, data, root_adr, joint_adrs, camera_id = make_scene(
        args.model, clip, door_xml=path.with_suffix(".xml") if "door_angle" in clip else None)

    def apply(frame):
        set_frame(model, data, clip, frame, root_adr, joint_adrs)

    camera = mujoco.MjvCamera()
    points = clip["root_pos"]
    if "object_pos" in clip:
        points = np.concatenate((points, clip["object_pos"]))
    if "door_angle" in clip:
        points = np.concatenate((points, clip["door_hinge_pos"][None]))
    camera.lookat[:] = (points.min(axis=0) + points.max(axis=0)) / 2
    camera.lookat[2] = .7
    camera.distance = max(3.0, float(np.ptp(points[:, :2], axis=0).max()) + 1.8)
    camera.azimuth, camera.elevation = 135, -20
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0  # G1 collision meshes

    if args.camera_view:
        if camera_id < 0:
            raise ValueError(
                "Camera view requires a 'd435_camera' camera in the selected model; "
                "use the shared G1 asset or omit --camera-view."
            )
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = camera_id
    if args.sensor_view or args.sensor_output:
        if camera_id < 0:
            raise ValueError("Sensor output requires the shared G1 d435_camera asset")
        run_sensor_replay(
            model, data, clip, n, fps, args.speed, camera_id, option, apply,
            sensor_view=args.sensor_view, sensor_output=args.sensor_output, once=args.once,
        )
    elif args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        with mujoco.Renderer(model, height=720, width=960) as renderer:
            command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", "960x720", "-r", str(fps * args.speed), "-i", "-", "-an", "-c:v",
                       "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(args.video)]
            with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
                try:
                    for frame in range(n):
                        apply(frame)
                        renderer.update_scene(data, camera=camera, scene_option=option)
                        process.stdin.write(renderer.render().tobytes())
                finally:
                    process.stdin.close()
                if process.wait():
                    raise RuntimeError("ffmpeg failed")
        print(f"Saved {args.video}")
    elif args.headless:
        for frame in range(n):
            apply(frame)
            if not np.isfinite(data.xpos).all():
                raise ValueError(f"Invalid kinematics at frame {frame}")
        print(f"Checked {n} frames; no physics steps")
    else:
        import mujoco.viewer

        state = {"frame": 0, "paused": False}

        def keypress(key):
            if key == 32:  # Space
                state["paused"] = not state["paused"]
            elif key in (262, 263):  # right/left arrow
                state["paused"] = True
                state["frame"] = (state["frame"] + (1 if key == 262 else -1)) % n
            elif key in (82, 114):  # R
                state["frame"] = 0

        print("Space: pause/resume | Left/Right: step frame | R: restart | close window: quit")
        apply(0)
        existing_threads = set(threading.enumerate())
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=keypress)
        viewer_threads = set(threading.enumerate()) - existing_threads
        try:
            with viewer.lock():
                viewer.cam.type = camera.type
                viewer.cam.fixedcamid = camera.fixedcamid
                viewer.cam.lookat[:] = camera.lookat
                viewer.cam.distance = camera.distance
                viewer.cam.azimuth, viewer.cam.elevation = camera.azimuth, camera.elevation
                viewer.opt.geomgroup[:] = option.geomgroup
            while viewer.is_running():
                start = time.monotonic()
                apply(state["frame"])
                viewer.sync()
                time.sleep(max(0, 1 / (fps * args.speed) - (time.monotonic() - start)))
                if not state["paused"]:
                    if args.once and state["frame"] == n - 1:
                        break
                    state["frame"] = (state["frame"] + 1) % n
        finally:
            viewer.close()
            # MuJoCo 3.11 closes asynchronously; finish rendering before GLFW's
            # atexit handler destroys the display connection.
            for thread in viewer_threads:
                thread.join()


if __name__ == "__main__":
    main()
