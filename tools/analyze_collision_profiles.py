"""Compare original chair surfaces with an explicitly prepared collision profile.

No collision assets are generated or changed. Surface samples, distances,
provenance, and review figures are written beneath --output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

from motion_data.collision import (COLLISION_PROFILES, DEFAULT_CACHE_DIR,
                                   collision_key, collision_parameters, load_collision_parts)


def surface_sample(triangles, count, rng):
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    indices = rng.choice(len(triangles), count, p=lengths / lengths.sum())
    uv = rng.random((count, 2))
    root = np.sqrt(uv[:, :1])
    weights = np.column_stack((1 - root, root * (1 - uv[:, 1:]), root * uv[:, 1:]))
    points = (triangles[indices] * weights[:, :, None]).sum(axis=1)
    normals = cross[indices] / lengths[indices, None]
    return points, normals


def inside_union(points, parts):
    inside = np.zeros(len(points), dtype=bool)
    for vertices, faces in parts:
        triangle = vertices[faces]
        normal = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        plane = np.einsum("ij,ij->i", normal, triangle[:, 0])
        # Chunk the half-space checks to keep memory bounded for larger probes.
        for start in range(0, len(points), 1024):
            inside[start:start + 1024] |= np.all(points[start:start + 1024] @ normal.T <= plane + 1e-10, axis=1)
    return inside


def triangle_distance(points, triangles):
    """Exact nearest triangle distances using conservative spatial candidates.

    The nearest mesh vertex supplies a valid distance upper bound. Any closer
    triangle's centroid must lie within upper_bound + its bounding radius.
    Use the maximum radius for cKDTree discovery, then each actual radius to
    filter candidates before exact point/triangle evaluation. No rtree needed.
    """
    centers = triangles.mean(axis=1)
    radii = np.linalg.norm(triangles - centers[:, None], axis=2).max(axis=1)
    tree = cKDTree(centers)
    vertex_tree = cKDTree(triangles.reshape(-1, 3))
    upper, _ = vertex_tree.query(points)
    distances = np.empty(len(points))
    max_radius = radii.max()
    for index, (point, bound) in enumerate(zip(points, upper)):
        candidates = np.asarray(tree.query_ball_point(point, bound + max_radius + 1e-10))
        near = np.linalg.norm(centers[candidates] - point, axis=1) <= bound + radii[candidates] + 1e-10
        candidates = candidates[near]
        projected = trimesh.triangles.closest_point(triangles[candidates],
                                                   np.broadcast_to(point, (len(candidates), 3)))
        distances[index] = np.linalg.norm(projected - point, axis=1).min()
    return distances


def describe(distances):
    return dict(samples=len(distances), mean_mm=float(np.mean(distances) * 1000),
                median_mm=float(np.median(distances) * 1000),
                p95_mm=float(np.quantile(distances, .95) * 1000),
                sampled_max_mm=float(np.max(distances) * 1000),
                fraction_over_30mm=float(np.mean(distances > .03)))


def draw(source, compact, output, title):
    all_points = np.concatenate((source.reshape(-1, 3), compact.reshape(-1, 3)))
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    center, radius = (high + low) / 2, (high - low).max() * .54
    fig = plt.figure(figsize=(14, 9), layout="constrained", facecolor="white")
    for row, (elev, azim, view) in enumerate(((18, -60, "Oblique"), (90, -90, "Top"))):
        for column, label in enumerate(("Original source surface", "Compact collision hulls", "Overlay")):
            ax = fig.add_subplot(2, 3, row * 3 + column + 1, projection="3d")
            if column != 1:
                ax.add_collection3d(Poly3DCollection(source, facecolors="#96999e", edgecolor="none",
                                                     alpha=1 if column == 0 else .40, shade=True))
            if column != 0:
                ax.add_collection3d(Poly3DCollection(compact, facecolors="#0072B2" if column == 1 else "#D55E00",
                                                     edgecolor="#303030", linewidth=.16,
                                                     alpha=1 if column == 1 else .42, shade=True))
            ax.set(xlim=(center[0] - radius, center[0] + radius),
                   ylim=(center[1] - radius, center[1] + radius),
                   zlim=(center[2] - radius, center[2] + radius),
                   xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
            ax.set_box_aspect((1, 1, 1))
            ax.set_proj_type("ortho")
            ax.view_init(elev=elev, azim=azim)
            if row == 1:
                ax.set_zticks([])
                ax.set_zlabel("")
            ax.set_title(f"{view}: {label}", fontsize=11)
            ax.tick_params(labelsize=8)
    fig.suptitle(title + "\nReference-frame orientation; equal metric scale across panels", fontsize=14)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def analyze(path, cache_dir, profile, count, seed, output):
    with np.load(path, allow_pickle=False) as archive:
        clip = dict(archive)
    parts = load_collision_parts(clip, cache_dir, profile=profile)
    scale = clip["object_scale"]
    parts = [(v * scale, f) for v, f in parts]
    source = clip["object_vertices"][clip["object_faces"]] * scale
    compact = np.concatenate([v[f] for v, f in parts])
    rng = np.random.default_rng(seed)
    source_points, _ = surface_sample(source, count, rng)
    raw_points, raw_normals = surface_sample(compact, 2 * count, rng)
    # Exclude overlapping/internal hull surfaces. A tiny outward offset must
    # lie outside every hull. Retained samples are area weighted on the union.
    external = ~inside_union(raw_points + 1e-7 * raw_normals, parts)
    compact_points = raw_points[external]
    source_to_hulls = triangle_distance(source_points, compact)
    compact_to_source = triangle_distance(compact_points, source)
    outside = ~inside_union(source_points, parts)
    uncovered_distance = np.where(outside, source_to_hulls, 0.)
    if not all(np.isfinite(value).all() for value in (source_to_hulls, compact_to_source, uncovered_distance)):
        raise ValueError(f"Non-finite surface comparison for {path}")
    prefix = output / path.stem.removeprefix("sitting__")
    np.savez_compressed(str(prefix) + "_samples.npz", source_points=source_points,
                        collision_union_surface_points=compact_points,
                        source_to_hull_surface_m=source_to_hulls,
                        collision_union_surface_to_source_m=compact_to_source,
                        source_outside_collision_union=outside,
                        source_to_collision_solid_m=uncovered_distance)
    # Rotate only for display; metrics use the scaled local object frame.
    quaternion = clip["object_quat"][0]
    rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    visual_source, visual_compact = source @ rotation.T, compact @ rotation.T
    draw(visual_source, visual_compact, str(prefix) + ".png",
         path.stem.removeprefix("sitting__").replace("_", " "))
    default = load_collision_parts(clip, cache_dir)
    result = dict(reference=str(path.resolve()),
                  reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  profile=profile, collision_key=collision_key(clip, profile),
                  parameters=collision_parameters(profile), object_scale=scale.tolist(),
                  seed=seed, source_triangles=len(source), collision_triangles=len(compact),
                  original_hulls=len(default), compact_hulls=len(parts),
                  max_compact_vertices=max(len(v) for v, f in parts),
                  surface_to_hull_surface=describe(source_to_hulls),
                  collision_union_surface_to_source=describe(compact_to_source),
                  source_to_collision_solid=describe(uncovered_distance),
                  source_fraction_outside_union=float(outside.mean()),
                  union_surface_samples_accepted=len(compact_points),
                  union_surface_samples_generated=len(raw_points),
                  image=str(prefix) + ".png")
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motions", type=Path, nargs="+")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--profile", choices=COLLISION_PROFILES, default="compact")
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/analysis/mixed_chair_collisions"))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    results = [analyze(path, args.cache_dir, args.profile, args.samples, args.seed, args.output)
               for path in args.motions]
    (args.output / "metrics.json").write_text(json.dumps(dict(
        units="metres for saved sample arrays; millimetres for distance summaries",
        methodology="Area-weighted deterministic surface sampling; exact nearest-triangle distance. "
                    "Internal compact hull surfaces removed using outward-offset half-space checks. "
                    "Sampled maxima are not Hausdorff bounds. No volume or seat/contact guarantee.",
        results=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
