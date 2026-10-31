"""Replay a standard motion NPZ using only forward kinematics (no mj_step).

uv run python tools/replay_motion.py data/motions/sitting
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


def make_scene(model_path, clip, door_xml=None):
    import mujoco

    spec = mujoco.MjSpec.from_file(str(Path(model_path).resolve()))
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
    return model, mujoco.MjData(model), root_adr, addresses


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path, help="NPZ file or directory of clips")
    parser.add_argument("--index", type=int, default=0, help="Clip index in a sorted directory (default: 0)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--once", action="store_true", help="Exit after one pass; default loops")
    parser.add_argument("--headless", action="store_true", help="Check all frames without a display")
    parser.add_argument("--video", type=Path, help="Render one pass to MP4 (requires ffmpeg)")
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    files = sorted(args.motion.glob("*.npz")) if args.motion.is_dir() else [args.motion]
    if not files or not 0 <= args.index < len(files):
        parser.error(f"No clip at index {args.index}; found {len(files)} clips")
    if args.video:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    path = files[args.index]
    clip = load_motion(path)
    n, fps = len(clip["root_pos"]), float(clip["fps"])
    print(f"[{args.index}/{len(files) - 1}] {path.name}: {n} frames, {fps:g} Hz, {n / fps:g}s")
    model, data, root_adr, joint_adrs = make_scene(
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
    if args.video:
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
