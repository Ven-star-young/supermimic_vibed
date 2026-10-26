"""Inspect robot/chair contacts at recorded poses, without physics integration.

uv run python tools/check_collisions.py data/motions/sitting --index 0 --render
First prepare convex parts with tools/prepare_collisions.py.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from motion_data import load_motion


def make_scene(clip, parts, *, interaction=True):
    import mujoco
    from motion_data.collision import add_object_geoms
    from supermimic.assets.g1.g1_constants import make_g1_spec

    spec = make_g1_spec(interaction_collisions=interaction)
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.group = 3
            geom.rgba = [.15, .55, .8, 1]
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0, 0, .05], contype=1, conaffinity=0,
                            rgba=[.85, .87, .9, 1])
    spec.worldbody.add_light(pos=[0, 0, 5], dir=[0, 0, -1])
    body = spec.worldbody.add_body(name="chair", mocap=True)
    add_object_geoms(spec, body, clip, parts)
    if not interaction:
        for geom in body.geoms:
            if geom.contype:
                geom.contype = 1  # Original G1's environment bit, for geometry comparison.
    model = spec.compile()
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
    model.vis.headlight.ambient[:] = .4
    return model, mujoco.MjData(model)


def set_frame(model, data, clip, frame):
    import mujoco

    root = int(model.joint("floating_base_joint").qposadr[0])
    data.qpos[root:root + 3] = clip["root_pos"][frame]
    data.qpos[root + 3:root + 7] = clip["root_quat"][frame]
    for name, value in zip(clip["joint_names"], clip["joint_pos"][frame]):
        data.joint(str(name)).qpos[0] = value
    mocap = int(model.body("chair").mocapid[0])
    data.mocap_pos[mocap], data.mocap_quat[mocap] = clip["object_pos"][frame], clip["object_quat"][frame]
    data.time = frame / float(clip["fps"])
    mujoco.mj_fwdPosition(model, data)  # Includes collision detection; never advances time.


def inspect(model, data, clip):
    chair = model.body("chair").id
    records = []
    for frame in range(len(clip["root_pos"])):
        set_frame(model, data, clip, frame)
        contacts = []
        for contact in data.contact[:data.ncon]:
            g1, g2 = map(int, contact.geom)
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            if chair not in (b1, b2):
                continue
            robot = b2 if b1 == chair else b1
            contacts.append(dict(body=model.body(robot).name, distance_m=float(contact.dist),
                                 position=contact.pos.tolist()))
        records.append(dict(frame=frame, time_s=frame / float(clip["fps"]), contacts=contacts))
    depths = [-c["distance_m"] for record in records for c in record["contacts"]]
    return dict(frames=len(records), frames_with_chair_contact=sum(bool(r["contacts"]) for r in records),
                max_penetration_m=max([0., *depths]),
                frames_over_2cm=sum(any(c["distance_m"] < -.02 for c in r["contacts"]) for r in records),
                records=records)


def snapshot(model, data, clip, frame, *, collision):
    import mujoco
    from PIL import Image

    set_frame(model, data, clip, frame)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (clip["root_pos"][frame] + clip["object_pos"][frame]) / 2
    camera.lookat[2] = .65
    camera.distance, camera.azimuth, camera.elevation = 2.7, 45, -18
    option = mujoco.MjvOption()
    option.geomgroup[1], option.geomgroup[3] = (0, 1) if collision else (1, 0)
    option.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = collision
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = collision
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=option)
        return Image.fromarray(renderer.render().copy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/collisions/chairs"))
    parser.add_argument("--output", type=Path, default=Path("results/analysis/collision"))
    parser.add_argument("--render", action="store_true", help="Save visual / single hull / decomposed comparison")
    args = parser.parse_args()
    if args.render:
        os.environ.setdefault("MUJOCO_GL", "egl")
    from motion_data.collision import load_collision_parts
    from scipy.spatial import ConvexHull

    files = sorted(args.motion.glob("*.npz")) if args.motion.is_dir() else [args.motion]
    if not 0 <= args.index < len(files):
        parser.error("Motion index is out of range")
    path = files[args.index]
    clip = load_motion(path)
    parts = load_collision_parts(clip, cache_dir=args.cache_dir)
    vertices = clip["object_vertices"]
    hull = ConvexHull(vertices)
    faces = hull.simplices.copy()
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    flip = np.einsum("ij,ij->i", normals, hull.equations[:, :3]) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    remap = np.zeros(len(vertices), dtype=int)
    remap[hull.vertices] = np.arange(len(hull.vertices))
    source_parts = [(vertices[hull.vertices], remap[faces])]
    output = args.output / path.stem
    output.mkdir(parents=True, exist_ok=True)
    results, images = {}, []
    labels = dict(single_hull="Original G1 + single hull", decomposed="Interaction G1 + decomposed")
    frame = int(np.argmin(clip["root_pos"][:, 2]))
    for label, shapes, interaction in (("single_hull", source_parts, False),
                                        ("decomposed", parts, True)):
        model, data = make_scene(clip, shapes, interaction=interaction)
        results[label] = inspect(model, data, clip)
        if args.render:
            if not images:
                images.append(("Reference geometry", snapshot(model, data, clip, frame, collision=False)))
            images.append((labels[label], snapshot(model, data, clip, frame, collision=True)))
    report = dict(clip=str(path), convex_parts=len(parts), render_frame=frame, comparisons=results,
                  robot_profiles=dict(single_hull="original", decomposed="interaction"),
                  note="Combined before/after comparison changes both robot and chair colliders. Recorded poses are imposed; overlaps do not establish dynamic stability or isolate chair approximation error.")
    (output / "contacts.json").write_text(json.dumps(report, indent=2) + "\n")
    if images:
        from PIL import Image, ImageDraw

        canvas = Image.new("RGB", (960 * len(images), 760), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (label, picture) in enumerate(images):
            canvas.paste(picture, (960 * i, 40))
            draw.text((960 * i + 20, 10), label, fill="black", font_size=22)
        canvas.save(output / "comparison.png")
    print(json.dumps({labels[k]: {f: v for f, v in result.items() if f != "records"}
                      for k, result in results.items()}, indent=2))
    print(f"Convex parts: {len(parts)}; report: {output}")


if __name__ == "__main__":
    main()
