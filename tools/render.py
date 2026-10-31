"""Render recorded evaluation trajectories (python -m supermimic.scripts.evaluate --record) with MuJoCo (EGL).

The simulated robot is drawn solid; the reference motion is a translucent ghost.
Usage: MUJOCO_GL=egl python tools/render.py traj.npz out.mp4 --clips dance1_subject1 jumps1_subject1 --seconds 12
For OpenDoor, add --door-xml paired.xml --motion-file paired.npz. The door
follows recorded policy angles; the blue robot follows the reference in world coordinates.
"""

import argparse
import os
from pathlib import Path
import re
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
# use the NVIDIA EGL vendor (GPU) rather than Mesa's CPU rasterizer when the image does not register it
_NV_EGL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docker", "10_nvidia.json")
if os.path.exists(_NV_EGL):
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", os.path.abspath(_NV_EGL))
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MJCF = os.path.join(ROOT, "data", "g1_description", "g1_29dof.xml")


def two_robot_xml() -> str:
    xml = open(MJCF).read()
    xml = re.sub(r'meshdir="[^"]*"', 'meshdir="' + os.path.join(ROOT, 'data', 'g1_description') + '/"', xml)
    body = xml[xml.index('<body name="pelvis"') : xml.index("</worldbody>")]
    ghost = re.sub(r' name="([^"]+)"', r' name="ref_\1"', body)
    ghost = re.sub(r'rgba="[^"]*"', 'rgba="0.35 0.65 1.0 0.25"', ghost)
    ghost = re.sub(r' (contype|conaffinity|group)="[^"]*"', "", ghost)
    ghost = ghost.replace("<geom ", '<geom contype="0" conaffinity="0" group="1" ')
    scene = """
    <light pos="0 0 6" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="0" conaffinity="0"/>
    """
    xml = xml.replace("</worldbody>", ghost + scene + "</worldbody>")
    assets = """
    <texture name="grid" type="2d" builtin="checker" rgb1="0.92 0.93 0.94" rgb2="0.80 0.82 0.85" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.95 0.96 0.98" rgb2="0.75 0.8 0.88" width="256" height="256"/>
    """
    xml = xml.replace("<asset>", "<asset>" + assets, 1)
    xml = xml.replace("<mujoco", '<mujoco', 1)
    if "<visual>" not in xml:
        xml = xml.replace("<asset>", '<visual><global offwidth="1280" offheight="720"/><quality shadowsize="4096"/></visual>\n  <asset>', 1)
    return xml


def make_model(trajectory, door_xml=None, motion_file=None):
    """Add the recorded policy's door without altering either G1's qpos layout."""
    spec = mujoco.MjSpec.from_string(two_robot_xml())
    if "theta" in trajectory:
        from motion_data import load_motion
        from motion_data.door import validate_door_asset

        if door_xml is None or motion_file is None:
            raise ValueError("Door policy recordings require --door-xml and --motion-file")
        validate_door_asset(load_motion(motion_file), door_xml)
        spec.attach(mujoco.MjSpec.from_file(str(Path(door_xml).resolve())),
                    prefix="policy_door/", frame=spec.worldbody.add_frame())
    elif door_xml is not None or motion_file is not None:
        raise ValueError("Door geometry was supplied but the recording has no theta")
    model = spec.compile()
    if (model.joint("floating_base_joint").qposadr[0] != 0
            or model.joint("ref_floating_base_joint").qposadr[0] != 36):
        raise ValueError("Recorded trajectories require the standard two-G1 joint layout")
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.55
    return model


def set_frame(model, data, trajectory, clip, t, global_ghost=False):
    """Replay measured state by forward kinematics, including the actual hinge."""
    d, c = trajectory, clip
    f = int(d["f"][t, c])
    data.qpos[0:3] = d["root_p"][t, c]
    data.qpos[3:7] = d["root_quat"][t, c]
    data.qpos[7:36] = d["q"][t, c]
    data.qpos[36:39] = d["ref_root_p"][f]
    # Door interaction uses absolute scene coordinates: anchoring the ghost to
    # the policy's xy would conceal displacement relative to the hinge.
    if not global_ghost and "theta" not in d:
        data.qpos[36:38] = d["root_p"][t, c][:2]
    data.qpos[39:43] = _mat_to_quat(d["ref_root_R"][f])
    data.qpos[43:72] = d["ref_q"][f]
    if "theta" in d:
        data.joint("policy_door/door_hinge").qpos[0] = d["theta"][t, c]
    if not np.isfinite(data.qpos).all():
        raise ValueError(f"Nonfinite recorded state at clip {c}, frame {t}")
    data.time = t / 50
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    return f


def frame_indices(trajectory, clip, start, seconds, fps):
    """Include the terminal state, but never show later steps of a finished world."""
    alive = trajectory["alive"][:, clip]
    stopped = np.flatnonzero(~alive)
    last = int(stopped[0] - 1) if len(stopped) else len(alive) - 1
    first = int(round(start * 50))
    end = min(last, first + int(round(seconds * 50)) - 1)
    if first < 0 or end < first:
        raise ValueError(f"No recorded episode frames at start={start:g}s for clip {clip}")
    frames = np.unique(np.rint(np.arange(first, end + 1, 50 / fps)).astype(int))
    frames = frames[frames <= end]
    # Downsampling must not hide the terminal frame.
    if frames[-1] != end:
        frames = np.append(frames, end)
    return frames, last


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("traj")
    ap.add_argument("out")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--title", type=str, default="")
    ap.add_argument("--door-xml", type=Path, help="Paired passive door MJCF for a policy recording")
    ap.add_argument("--motion-file", type=Path, help="Original motion NPZ used to validate the door geometry")
    ap.add_argument("--global_ghost", action="store_true", help="draw the reference at its global xy instead of anchored to the robot")
    a = ap.parse_args(argv)
    if not 1 <= a.fps <= 50 or not np.isfinite(a.seconds) or a.seconds <= 0:
        ap.error("--fps must be 1..50 and --seconds must be finite and positive")
    if not np.isfinite(a.start) or a.start < 0:
        ap.error("--start must be finite and nonnegative")
    if a.width <= 0 or a.height <= 0 or a.width % 2 or a.height % 2:
        ap.error("--width and --height must be positive even numbers")

    with np.load(a.traj, allow_pickle=False) as archive:
        d = dict(archive)
    names = list(d["names"])
    model = make_model(d, a.door_xml, a.motion_file)
    model.vis.global_.offwidth, model.vis.global_.offheight = a.width, a.height
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, a.height, a.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.6, 135.0, -12.0
    font = ImageFont.load_default(size=22) if hasattr(ImageFont, "load_default") else None
    clips = a.clips or names
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{a.width}x{a.height}", "-r", str(a.fps),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    try:
        for clip in clips:
            c = names.index(clip)
            frames, last = frame_indices(d, c, a.start, a.seconds, a.fps)
            if "theta" in d:
                set_frame(model, data, d, c, int(frames[0]))
                hinge = data.xanchor[model.joint("policy_door/door_hinge").id]
                points = np.concatenate((d["root_p"][frames, c],
                                         d["ref_root_p"][d["f"][frames, c]], hinge[None]))
                cam.lookat[:] = (points.min(0) + points.max(0)) / 2
                cam.lookat[2] = .85
                cam.distance = max(3.6, float(np.ptp(points[:, :2], axis=0).max()) + 2)
                cam.elevation = -18
            for t in frames:
                f = set_frame(model, data, d, c, t, a.global_ghost)
                if "theta" not in d:
                    cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.75]
                renderer.update_scene(data, camera=cam)
                img = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(img)
                status = f"t={(f - int(d['clip_start'][c])) * 0.02:5.1f}s"
                if t == last:
                    at_end = f >= int(d["clip_start"][c] + d["clip_len"][c] - 1)
                    status += "  [reference end]" if at_end else "  [episode ended]"
                labels = [((16, 12), f"{a.title}  {clip}"), ((16, 44), status),
                          ((16, a.height - 38),
                          "solid: simulated G1 policy   blue ghost: reference")]
                if "theta" in d:
                    labels.append(((16, 76), f"door actual {np.rad2deg(d['theta'][t, c]):.1f} deg"
                                   f"   reference {np.rad2deg(d['ref_theta'][f]):.1f} deg"))
                for xy, txt in labels:
                    bb = draw.textbbox(xy, txt, font=font)
                    draw.rectangle((bb[0] - 6, bb[1] - 4, bb[2] + 6, bb[3] + 4), fill=(255, 255, 255))
                    draw.text(xy, txt, fill=(20, 20, 20), font=font)
                ff.stdin.write(np.asarray(img).tobytes())
    finally:
        ff.stdin.close()
        renderer.close()
        returncode = ff.wait()
    if returncode:
        raise RuntimeError(f"ffmpeg failed with exit status {returncode}")
    print("wrote", a.out)


def _mat_to_quat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1).astype(np.float64))
    return q


if __name__ == "__main__":
    main()
