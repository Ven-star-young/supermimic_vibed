"""Record physical Sitting deployment and render a synchronized nine-chair grid.

Run from repository root with .venv/bin/python.  Each recording is isolated in
its own process; no reference poses replace the measured policy poses.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
CHECKPOINT = ROOT / "logs/cuhk_sitting_deployment/model_8600.pt"
INDICES = [0, 8, 10, 13, 14, 19, 28, 34, 45]
LABELS = ["Adirondack", "Bauhaus armchair", "Berber chair", "Metal mesh chair",
          "Brass / velvet", "Hot dog chair", "Chinese mahjong chair",
          "Curved concrete", "Gold bamboo chair"]


def record(index: int):
    import numpy as np
    import mujoco
    import torch
    from motion_data.chair_split import load_split, file_digest
    from supermimic.scripts.play_policy import (
        build_parser, validate_arguments, resolve_references, make_deployment_cfg,
        DeploymentEnv, load_actor, policy_digest,
    )
    from mjlab.envs import ManagerBasedRlEnv
    from supermimic.rl.runner import SuperMimicRunner

    args = build_parser().parse_args([
        "--checkpoint", str(CHECKPOINT), "--partition", "all", "--index", str(index),
        "--init", "clean", "--seed", "0", "--episodes", "1", "--viewer", "none",
    ])
    validate_arguments(args)
    manifest, data_root = load_split(CHECKPOINT.parent / "split.json", args.data_root)
    references = resolve_references(args, manifest, data_root)
    reference = references[index]
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    native = ManagerBasedRlEnv(make_deployment_cfg(checkpoint, reference, args), device=args.device)
    env = None
    try:
        env = DeploymentEnv(native, 1, None, seeds=[0])
        train_cfg = checkpoint["train_cfg"] | dict(amp="none", no_compile=True)
        runner = SuperMimicRunner(env, train_cfg, device=args.device)
        load_actor(runner, checkpoint, manifest)
        before = policy_digest(runner.pi)
        policy = runner.get_inference_policy()
        command = native.command_manager.get_term("motion")
        qpos, qvel, phases = [], [], []

        def capture():
            qpos.append(native.sim.data.qpos[0].cpu().numpy().copy())
            qvel.append(native.sim.data.qvel[0].cpu().numpy().copy())
            phases.append(int(command.time_steps[0]))

        capture()
        with torch.inference_mode():
            while not env.finished:
                env.step(policy(env.get_observations()))
                capture()
        if policy_digest(runner.pi) != before:
            raise RuntimeError("Deployment changed frozen policy")
        model_path = OUT / f"chair_{index:03d}.mjb"
        mujoco.mj_saveModel(native.sim.mj_model, str(model_path), None)
        np.savez_compressed(OUT / f"chair_{index:03d}.npz", qpos=np.asarray(qpos),
                            qvel=np.asarray(qvel), phases=np.asarray(phases),
                            dt=np.asarray(native.step_dt), reference=str(reference))
        metadata = dict(index=index, label=LABELS[INDICES.index(index)],
                        reference=str(reference), checkpoint=str(CHECKPOINT),
                        checkpoint_sha256=file_digest(CHECKPOINT),
                        split_fingerprint=manifest["fingerprint"], condition="clean", seed=0,
                        frames=len(qpos), duration_s=(len(qpos)-1)*native.step_dt,
                        control_dt=native.step_dt, policy_unchanged=True,
                        episode=env.records[0])
        (OUT / f"chair_{index:03d}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("RECORDED " + json.dumps({k: metadata[k] for k in ("index", "label", "frames", "duration_s")}), flush=True)
    finally:
        if env is not None:
            env.close_metrics()
        native.close()


def record_all():
    for index in INDICES:
        if all((OUT / f"chair_{index:03d}.{suffix}").is_file() for suffix in ("json", "mjb", "npz")):
            print(f"Already recorded {index}", flush=True)
            continue
        with (OUT / f"chair_{index:03d}.log").open("w") as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "record", "--index", str(index)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Finished recording {index}", flush=True)


def render():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(ROOT / "docker/10_nvidia.json"))
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    fps, width, height = 25, 1920, 1080
    tw, th = width // 3, height // 3
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 23)
    small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)

    def fit_camera(model, data, trajectory, renderer, camera, option):
        # Use the full physical trajectory and the visible mesh bounds.  This
        # makes the camera fit the robot AND chair between the two text bars.
        meshes = []
        from itertools import product
        for geom in range(model.ngeom):
            if model.geom_group[geom] != 1 or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = int(model.geom_dataid[geom])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            vertices = model.mesh_vert[start:start+count]
            lo, hi = vertices.min(0), vertices.max(0)
            corners = np.asarray(list(product(*zip(lo, hi))))
            meshes.append((geom, corners))
        points = []
        frames = sorted(set(range(0, len(trajectory["qpos"]), 5)) | {len(trajectory["qpos"])-1})
        for frame in frames:
            data.qpos[:] = trajectory["qpos"][frame]
            mujoco.mj_kinematics(model, data)
            for geom, corners in meshes:
                points.append(corners @ data.geom_xmat[geom].reshape(3,3).T + data.geom_xpos[geom])
        points = np.concatenate(points)
        tangent = np.tan(np.deg2rad(model.vis.global_.fovy / 2))
        for _ in range(20):
            renderer.update_scene(data, camera=camera, scene_option=option)
            gl = renderer.scene.camera
            position = (gl[0].pos + gl[1].pos) / 2
            forward = np.asarray(gl[0].forward)
            up = np.asarray(gl[0].up)
            right = np.cross(forward, up)
            relative = points - position
            depth = relative @ forward
            x = (relative @ right) / depth / tangent / (tw/th)
            y = (relative @ up) / depth / tangent
            bounds = np.array([x.min(), x.max(), y.min(), y.max()])
            cx, cy = (bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2
            camera.lookat[:] += (right * cx * camera.distance * tangent * (tw/th)
                                 + up * (cy + .025) * camera.distance * tangent)
            factor = max((bounds[1]-bounds[0])/1.80, (bounds[3]-bounds[2])/1.43)
            if factor > 1:
                camera.distance *= factor * 1.02
            if abs(cx) < .003 and abs(cy + .025) < .003 and factor <= 1:
                break
        return dict(lookat=camera.lookat.tolist(), distance=camera.distance,
                    azimuth=camera.azimuth, elevation=camera.elevation)

    scenes = []
    for index in INDICES:
        model = mujoco.MjModel.from_binary_path(str(OUT / f"chair_{index:03d}.mjb"))
        model.vis.global_.offwidth = tw
        model.vis.global_.offheight = th
        model.vis.headlight.ambient[:] = .42
        model.vis.headlight.diffuse[:] = .65
        model.vis.headlight.specular[:] = .12
        data = mujoco.MjData(model)
        trajectory = dict(np.load(OUT / f"chair_{index:03d}.npz"))
        metadata = json.loads((OUT / f"chair_{index:03d}.json").read_text())
        data.qpos[:] = trajectory["qpos"][-1]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        robot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis")
        chair_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chair/chair")
        if robot_id < 0 or chair_id < 0:
            raise RuntimeError("Expected named robot and chair bodies in recorded native model")
        points = []
        for qpos in trajectory["qpos"][::20]:
            data.qpos[:] = qpos
            mujoco.mj_kinematics(model, data)
            points.extend((data.xpos[robot_id].copy(), data.xpos[chair_id].copy()))
        points = np.asarray(points)
        center = (points.min(0) + points.max(0)) / 2
        center[2] = .65
        # Frame the full approach, and choose the oblique view from the final
        # seated heading so chair backs cannot conceal the result.
        data.qpos[:] = trajectory["qpos"][-1]
        mujoco.mj_kinematics(model, data)
        R = data.xmat[robot_id].reshape(3, 3)
        heading = np.rad2deg(np.arctan2(R[1, 0], R[0, 0]))
        camera = mujoco.MjvCamera()
        camera.lookat[:] = center
        camera.distance, camera.azimuth, camera.elevation = 2.5, heading + 235, -24
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0
        for geom in range(model.ngeom):
            name = model.geom(geom).name
            if name == "floor" or model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_PLANE:
                model.geom_matid[geom] = -1
                model.geom_rgba[geom] = [.60, .64, .68, 1]
        renderer = mujoco.Renderer(model, height=th, width=tw)
        metadata["render_camera"] = fit_camera(model, data, trajectory, renderer, camera, option)
        print(f"CAMERA {index} {metadata['render_camera']}", flush=True)
        scenes.append(dict(model=model, data=data, trajectory=trajectory, metadata=metadata,
                           renderer=renderer,
                           camera=camera, option=option))
    duration = max(s["metadata"]["duration_s"] for s in scenes) + 1.0
    frame_count = int(np.ceil(duration * fps)) + 1
    video = OUT / "sitting_grid.mp4"
    process = subprocess.Popen([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", "-movflags", "+faststart", str(video)
    ], stdin=subprocess.PIPE)
    try:
        for frame in range(frame_count):
            seconds = frame / fps
            canvas = Image.new("RGB", (width, height), (242, 245, 248))
            for position, s in enumerate(scenes):
                traj, meta = s["trajectory"], s["metadata"]
                last = len(traj["qpos"]) - 1
                step = min(int(round(seconds / float(traj["dt"]))), last)
                s["data"].qpos[:] = traj["qpos"][step]
                s["data"].qvel[:] = traj["qvel"][step]
                s["data"].time = min(seconds, meta["duration_s"])
                mujoco.mj_kinematics(s["model"], s["data"])
                mujoco.mj_camlight(s["model"], s["data"])
                s["renderer"].update_scene(s["data"], camera=s["camera"], scene_option=s["option"])
                tile = Image.fromarray(s["renderer"].render())
                draw = ImageDraw.Draw(tile)
                draw.rectangle((0, 0, tw, 41), fill=(244, 247, 250))
                draw.text((14, 9), f"{meta['index']:02d}  {meta['label']}", font=title_font, fill=(28, 44, 62))
                finished = seconds >= meta["duration_s"]
                success = meta["episode"]["full_reference_success"]
                status = ("Complete | final pose held" if success else "Episode ended | final pose held") if finished else f"t = {seconds:.2f} s"
                if not success and finished:
                    status += " | " + meta["episode"]["outcome"]
                draw.rectangle((0, th - 32, tw, th), fill=(244, 247, 250))
                draw.text((14, th - 26), status, font=small_font, fill=(28, 92, 65) if finished and success else (47, 58, 74))
                canvas.paste(tile, ((position % 3)*tw, (position // 3)*th))
            draw = ImageDraw.Draw(canvas)
            for x in (tw, 2*tw): draw.line((x, 0, x, height), fill=(204, 213, 223), width=3)
            for y in (th, 2*th): draw.line((0, y, width, y), fill=(204, 213, 223), width=3)
            process.stdin.write(np.asarray(canvas).tobytes())
            if frame in {0, int(frame_count*.5), frame_count-1}:
                canvas.save(OUT / f"preview_{frame:04d}.png")
            if frame % fps == 0:
                print(f"RENDER {frame}/{frame_count}", flush=True)
    finally:
        process.stdin.close()
        code = process.wait()
        for s in scenes: s["renderer"].close()
    if code: raise RuntimeError(f"ffmpeg failed: {code}")
    (OUT / "manifest.json").write_text(json.dumps(dict(
        video=str(video), fps=fps, resolution=[width,height], duration_s=frame_count/fps,
        time_alignment="All episodes start at t=0 at recorded real-time speed. Final recorded poses held after episode completion.",
        checkpoint=str(CHECKPOINT), recordings=[s["metadata"] for s in scenes]), indent=2) + "\n")
    print(video, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("record", "record-all", "render"))
    parser.add_argument("--index", type=int, choices=INDICES)
    args = parser.parse_args()
    if args.mode == "record":
        if args.index is None: parser.error("record needs --index")
        record(args.index)
    elif args.mode == "record-all": record_all()
    else: render()
