"""Clean single-camera rendering of previously saved native MuJoCo states.

Only visual camera/material settings are changed; the trajectory is not rerun.
"""
import argparse
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(Path("docker/10_nvidia.json").resolve()))
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FPS = 25
WIDTH, HEIGHT = 1920, 1080


def render(kind, preview=False, azimuth=None):
    source = ROOT / kind
    archive = np.load(source / "physics_states.npz")
    qpos, times = archive["qpos"], archive["time"]
    model = mujoco.MjModel.from_binary_path(str(source / "scene.mjb"))
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = HEIGHT
    model.vis.global_.fovy = 36
    model.vis.headlight.ambient[:] = .48
    model.vis.headlight.diffuse[:] = .65
    model.vis.headlight.specular[:] = .12
    model.vis.quality.shadowsize = 4096
    model.vis.rgba.fog[:] = (.88, .91, .95, 1)
    model.vis.rgba.haze[:] = (.88, .91, .95, 1)
    model.geom_rgba[model.geom("door/door_panel").id] = (.34, .52, .69, 1)
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            model.geom_rgba[g] = (.88, .91, .95, 1)
            model.geom_matid[g] = -1
    data = mujoco.MjData(model)
    data.mocap_pos[:] = archive["mocap_pos"]
    data.mocap_quat[:] = archive["mocap_quat"]
    option = mujoco.MjvOption()
    option.geomgroup[3] = False
    camera = mujoco.MjvCamera()
    camera.distance = 3.55
    camera.azimuth = (80 if kind == "back" else 70) if azimuth is None else azimuth
    camera.elevation = -23
    camera.lookat[:] = (-.10, -.06, .89)
    title = "背部开门" if kind == "back" else "侧面开门"
    font = ImageFont.truetype(FONT, 38)
    duration = float(times[-1])
    count = int(np.ceil(duration * FPS)) + 1
    queries = np.minimum(np.arange(count) / FPS, duration)
    indices = np.searchsorted(times, queries)
    indices = np.minimum(indices, len(times) - 1)
    indices[-1] = len(times) - 1
    if preview:
        indices = [0, 1166 if kind == "back" else 2037, len(times) - 1]
    output = ROOT / f"{kind}_clean.mp4"
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    process = None if preview else subprocess.Popen(command, stdin=subprocess.PIPE)
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        try:
            for i, step in enumerate(indices):
                data.qpos[:] = qpos[step]
                mujoco.mj_kinematics(model, data)
                mujoco.mj_camlight(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                image = Image.fromarray(renderer.render().copy())
                draw = ImageDraw.Draw(image)
                draw.text((58, 35), title, font=font, fill=(38, 55, 73))
                if preview:
                    image.save(source / f"clean_preview_{i}_az{int(camera.azimuth)}.png")
                else:
                    process.stdin.write(np.asarray(image).tobytes())
                    if i in (0, len(indices)//2, len(indices)-1):
                        image.save(source / f"clean_video_{i}.png")
                    if i % 75 == 0:
                        print(f"{kind}: rendered {i+1}/{len(indices)}", flush=True)
            if process is not None:
                for _ in range(FPS):
                    process.stdin.write(np.asarray(image).tobytes())
        finally:
            if process is not None:
                process.stdin.close()
                if process.wait():
                    raise RuntimeError("ffmpeg failed")
    print(output if not preview else source, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("back", "side"))
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--azimuth", type=float)
    args = parser.parse_args()
    render(args.kind, args.preview, args.azimuth)
