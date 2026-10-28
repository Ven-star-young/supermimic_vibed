"""Audit object-frame conventions without changing motions, meshes or training.

uv run python tools/check_object_origins.py

Exact mesh bounds/area centroids are separate from a reference-conditioned
support-surface proxy. The proxy is NOT an annotated semantic seat center.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from motion_data import load_motion
from motion_data.chair_split import file_digest, load_split, verify_files


def rotation(q):
    return Rotation.from_quat(q, scalar_first=True).as_matrix()


def support_proxy(triangles, pelvis, floor):
    """81 vertical rays in a 16-cm square below a low-reference pelvis point."""
    offsets = np.linspace(-.08, .08, 9)
    xy = pelvis[:2] + np.array(np.meshgrid(offsets, offsets)).reshape(2, -1).T
    a = triangles[:, 0]
    b, c = triangles[:, 1] - a, triangles[:, 2] - a
    det = b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]
    valid = np.abs(det) > 1e-12
    safe = np.where(valid, det, 1.)
    hits = []
    for point in xy:
        delta = point - a[:, :2]
        u = (delta[:, 0] * c[:, 1] - delta[:, 1] * c[:, 0]) / safe
        v = (b[:, 0] * delta[:, 1] - b[:, 1] * delta[:, 0]) / safe
        z = a[:, 2] + u * b[:, 2] + v * c[:, 2]
        inside = (valid & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
                  & (z > floor + .06) & (z < pelvis[2] - .02))
        if inside.any():
            hits.append([*point, z[inside].max()])
    flags = ["semantic_seat_unverified"]
    if not hits:
        return dict(coverage=0., point=None, flags=flags + ["no_hits"])
    hits = np.asarray(hits)
    center = np.median(hits, axis=0)
    q10, q90 = np.percentile(hits[:, 2], [10, 90])
    gap = float(pelvis[2] - center[2])
    coverage = len(hits) / len(xy)
    if coverage < .5:
        flags.append("low_ray_coverage")
    if q90 - q10 > .06:
        flags.append("inclined_or_mixed_surface")
    if not .04 < gap < .25:
        flags.append("atypical_pelvis_gap")
    return dict(coverage=coverage, point=center.tolist(), z_q10_q90=[float(q10), float(q90)],
                pelvis_gap_m=gap, flags=flags)


def analyze(path, partition):
    from supermimic.tasks.interaction.assets import mass_properties
    clip = load_motion(path)
    vertices = clip["object_vertices"].astype(np.float64) * clip["object_scale"]
    triangles = vertices[clip["object_faces"]]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    lower, upper = vertices.min(0), vertices.max(0)
    bbox = (lower + upper) / 2
    surface = np.average(triangles.mean(1), weights=areas, axis=0)
    with warnings.catch_warnings(record=True) as notes:
        com, _ = mass_properties(clip)
    rotations = rotation(clip["object_quat"])
    first = rotations[0]
    up = first.T @ [0., 0., 1.]
    # Choose one horizontal local axis; do not infer semantic chair-forward.
    axis = np.eye(3)[0 if abs(up[0]) < .9 else 2]
    horizontal = axis - np.dot(axis, up) * up
    horizontal /= np.linalg.norm(horizontal)
    basis = np.stack((horizontal, np.cross(up, horizontal), up), axis=1)
    upright = vertices @ basis
    # All reference poses mapped into the same object frame; moving-chair pose
    # changes must not be mistaken for a change in its static mesh origin.
    pelvis_local = np.einsum("tji,tj->ti", rotations, clip["root_pos"] - clip["object_pos"])
    pelvis = pelvis_local @ basis
    low = np.median(pelvis[pelvis[:, 2] <= np.percentile(pelvis[:, 2], 20)], axis=0)
    proxy = support_proxy(upright[clip["object_faces"]], low, upright[:, 2].min())
    if proxy["point"] is not None:
        proxy["local_point"] = (basis @ proxy["point"]).tolist()
        proxy["origin_vertical_offset_m"] = proxy["point"][2]
    row = dict(reference=path.name, partition=partition,
        sha256=file_digest(path), scale=clip["object_scale"].tolist(),
        bounds=[lower.tolist(), upper.tolist()], bbox_center=bbox.tolist(),
        bbox_offset_m=float(np.linalg.norm(bbox)), surface_center=surface.tolist(),
        surface_offset_m=float(np.linalg.norm(surface)), simulation_com=np.asarray(com).tolist(),
        simulation_com_offset_m=float(np.linalg.norm(com)), mass_property_notes=[str(n.message) for n in notes],
        origin_inside_bbox=bool(np.all(lower <= 0) and np.all(upper >= 0)),
        height_m=float(np.ptp(upright[:, 2])), origin_above_bottom_m=float(-upright[:, 2].min()),
        origin_height_ratio=float(-upright[:, 2].min() / np.ptp(upright[:, 2])),
        local_up=up.tolist(), local_y_up_error_deg=float(np.degrees(np.arccos(np.clip(up[1], -1, 1)))),
        display_basis=basis.tolist(), low_reference_pelvis=low.tolist(), support_proxy=proxy)
    return row


def statistics(rows):
    result = dict(count=len(rows), origins_inside_bbox=sum(r["origin_inside_bbox"] for r in rows),
                  bbox_within_5mm=sum(r["bbox_offset_m"] <= .005 for r in rows),
                  bbox_within_10mm=sum(r["bbox_offset_m"] <= .01 for r in rows))
    for key in ("bbox_offset_m", "surface_offset_m", "simulation_com_offset_m", "origin_height_ratio", "local_y_up_error_deg"):
        result[key] = dict(zip(("min", "median", "p95", "max"),
                              np.quantile([r[key] for r in rows], [0, .5, .95, 1]).tolist()))
    result["support_proxy_available"] = sum(r["support_proxy"]["point"] is not None for r in rows)
    result["support_proxy_flagged"] = sum(len(r["support_proxy"]["flags"]) > 1 for r in rows)
    return result


def plots(rows, root, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = {"train": "#0072B2", "test": "#D55E00"}
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
        for part, marker in (("train", "o"), ("test", "^")):
            selected = [r for r in rows if r["partition"] == part]
            x = np.sort([1000 * r["bbox_offset_m"] for r in selected])
            axes[0].step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=colors[part], label=f"{part} (n={len(x)})")
            axes[0].scatter(x, np.arange(1, len(x) + 1) / len(x), s=12, marker=marker, color=colors[part])
            axes[1].scatter([r["height_m"] for r in selected], [r["origin_above_bottom_m"] for r in selected],
                            label=part, color=colors[part], marker=marker, s=28, alpha=.7)
        axes[0].axvline(10, color="0.4", linestyle="--", linewidth=1, label="10 mm")
        axes[0].set(xlabel="Origin to asset-local bounding-box center (mm)", ylabel="Fraction of chairs at or below this offset",
                    xlim=(0, max(12., max(r["bbox_offset_m"] for r in rows) * 1100)), ylim=(0, 1.04), title="Distance from asset origin to box center")
        axes[0].legend(loc="lower right")
        median = np.median([r["bbox_offset_m"] for r in rows]) * 1000
        small = sum(r["bbox_offset_m"] <= .01 for r in rows)
        axes[0].text(.42, .6, f"All {len(rows)}: median {median:.2f} mm\n{small} / {len(rows)} within 10 mm", transform=axes[0].transAxes)
        extent = max(r["height_m"] for r in rows) * 1.06
        axes[1].plot([0, extent], [0, extent / 2], "--", color="0.4", label="Half of chair height")
        axes[1].set(xlabel="Mesh height along world-up at reference start (m)", ylabel="Origin height above mesh bottom (m)",
                    xlim=(0, extent), ylim=(0, extent * .55), title="Origin is near mid-height, not ground level")
        axes[1].legend()
        counts = {p: sum(r["partition"] == p for r in rows) for p in ("train", "test")}
        fig.suptitle(f"Selected chair references: {counts['train']} train + {counts['test']} test | exact scaled-mesh bounds", fontsize=14)
        fig.savefig(output / "origin_summary.png", dpi=180)
        fig.savefig(output / "origin_summary.svg")
        plt.close(fig)

        examples = ["cane_back_peacock", "bentwood_thonet", "marcel_breuer_cesca",
                    "adirondack_chair", "platform_rocking", "bistro_wicker"]
        selected = [next((r for r in rows if name in r["reference"]), None) for name in examples]
        fallback = sorted(rows, key=lambda r: r["bbox_offset_m"], reverse=True)
        titles = ["Peacock lounge chair", "Bentwood rocker", "Cesca cantilever chair",
                  "Adirondack chair", "Platform rocker", "Bistro wicker chair"]
        fig, axes = plt.subplots(2, 3, figsize=(13, 9), layout="constrained")
        for ax, row, title in zip(axes.flat, selected, titles):
            if row is None:
                row = fallback.pop(0) if fallback else rows[0]
                title = row["reference"].removeprefix("sitting__").split("__")[0][:36]
            clip = load_motion(root / row["reference"])
            basis = np.asarray(row["display_basis"])
            vertices = (clip["object_vertices"] * clip["object_scale"]) @ basis
            triangles = vertices[clip["object_faces"]]
            areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
            rng = np.random.default_rng(0)
            tri = triangles[rng.choice(len(triangles), 8000, p=areas / areas.sum())]
            uv = rng.random((len(tri), 2)); sq = np.sqrt(uv[:, :1])
            weights = np.concatenate((1-sq, sq*(1-uv[:, 1:]), sq*uv[:, 1:]), axis=1)
            cloud = (tri * weights[..., None]).sum(1)
            # Asset-local X supplies depth for these Y-up assets; basis accounts
            # for non-Y-up inputs without pretending this is semantic forward.
            ax.scatter(cloud[:, 0], cloud[:, 2], color="0.38", s=.8, alpha=.25, rasterized=True)
            bbox = np.asarray(row["bbox_center"]) @ basis
            ax.scatter(bbox[0], bbox[2], marker="o", s=140, facecolors="none", edgecolors="#0072B2", linewidths=1.8, zorder=5)
            ax.scatter(0, 0, marker="x", s=85, color="#D55E00", linewidths=2, zorder=6)
            proxy = row["support_proxy"]
            if proxy["point"] is not None:
                point = proxy["point"]
                ax.scatter(point[0], point[2], marker="*", s=150, color="#CC9900", edgecolors="black", linewidths=.4, zorder=7)
                sub = f"Support proxy: {point[2]*100:+.1f} cm vertically from origin"
                if len(proxy["flags"]) > 1:
                    sub += "\nHeuristic flagged: inspect surface / reference"
            else:
                sub = "Support proxy unavailable"
            ax.set_title(f"{title}\nOrigin–box center: {row['bbox_offset_m']*1000:.1f} mm", fontsize=11)
            ax.set(xlabel="Horizontal asset axis (m)", ylabel="Height relative to asset origin (m)")
            ax.set_aspect("equal", adjustable="datalim")
            ax.axhline(0, color="#D55E00", linewidth=.6, alpha=.5)
            ax.text(.02, .02, sub, transform=ax.transAxes, fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
        handles = [Line2D([], [], marker="x", color="#D55E00", linestyle="", label="Asset origin"),
                   Line2D([], [], marker="o", markerfacecolor="none", color="#0072B2", linestyle="", label="Asset-local box center"),
                   Line2D([], [], marker="*", color="#CC9900", linestyle="", label="Support proxy under reference pelvis (unverified)")]
        fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=10)
        fig.suptitle("Vertical projections | origin and box center nearly coincide; support location varies\nGold stars are local support estimates, NOT annotated seat centers", fontsize=13)
        fig.savefig(output / "origin_examples.png", dpi=180)
        fig.savefig(output / "origin_examples.svg")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=Path("data/splits/sitting_back_references_seed0.json"))
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/analysis/object_origins_20261028"))
    args = parser.parse_args()
    manifest, root = load_split(args.split, args.data_root)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for part in ("train", "test"):
        verify_files(manifest, root, part)
        for group in manifest[part]:
            if len(group["clips"]) != 1:
                raise ValueError("Use the one-reference-per-chair split for this audit")
            rows.append(analyze(root / group["clips"][0]["path"], part))
    summary = {part: statistics([r for r in rows if part == "all" or r["partition"] == part])
               for part in ("all", "train", "test")}
    summary.update(split=str(args.split.resolve()), fingerprint=manifest["fingerprint"],
                   units="metres unless suffixed mm/deg; scale applied once",
                   note="Support proxies are reference-conditioned mesh intersections, not semantic seat annotations or physical contacts")
    (args.output / "chairs.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    scalar_fields = [k for k, v in rows[0].items() if isinstance(v, (str, int, float, bool))]
    with (args.output / "chairs.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in scalar_fields} for r in rows)
    plots(rows, root, args.output)
    print(json.dumps(summary["all"], indent=2))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
