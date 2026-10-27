# Optional compact chair collisions

The `compact` profile reduces collision complexity without modifying reference
NPZs, the visual mesh, or the original `default` collision caches. It is an
explicit physical approximation, not merely a memory allocation change.

| Recipe | Default | Compact |
| --- | ---: | ---: |
| CoACD version | 1.0.14 | 1.0.14 |
| Concavity threshold (`real_metric=True`) | 0.01 | 0.03 |
| Maximum convex parts | Unbounded | 32 |
| Maximum vertices per part | 64 | 32 |

The threshold is measured in the unscaled source mesh coordinate system. Object
scale is applied later, exactly once. A hull budget can force CoACD to exceed its
requested concavity threshold. Consequently the compact recipe does **not**
guarantee a 3 cm collision error, preservation of every opening, or equivalent
robot contact behavior. Thin bars, woven seats, armrests, and leg gaps merit
particular attention when comparing the two approximations.

Prepare the selected references offline:

```bash
OMP_NUM_THREADS=2 uv run --extra motion python tools/prepare_collisions.py \
  --split-manifest data/splits/sitting_back_references_seed0.json \
  --partition all --profile compact --workers 2
```

For a small initial inspection, use `--partition train --limit 2`. The limit is
for preprocessing only: training still validates every selected reference and
never drops a chair because its collision cache is missing. Use the same profile
for preparation and training/evaluation. Preparing a compact file does not
automatically opt any existing training run into it.

Both profiles can coexist in `data/collisions/chairs`. Cache names hash the full
recipe and source mesh. Existing default keys and version-1 NPZ arrays remain
compatible. New caches also record the profile. The compact loader checks the
exact recipe, profile, convex closed-manifold parts, offsets, and both bounds.
Missing or corrupt compact assets fail explicitly; the loader never substitutes
the default mesh, a whole-chair hull, or a truncated list of parts. Runtime
loading does not import CoACD or recompute collisions.

The neural geometry bank must use the selected collision recipe in its geometry
identity and sample that actual collision surface. A reference split still
groups the original source shapes; choosing another approximation must not
change which chairs are held out. Changing collision profile also changes the
physical training problem and should use a new experiment directory.

## Local preprocessing check

On 2026-10-27, two existing training references were decomposed with two OpenMP
threads. Both compact caches passed the runtime convexity and bound validator.

| Chair reference | Default parts | Compact parts | Largest compact hull | Time |
| --- | ---: | ---: | ---: | ---: |
| `cocoon_woven_rattan_hanging_chair_stand_free__002` | 785 | 32 | 32 vertices | 34.7 s |
| `brutalist_1972_raw_concrete_sculpted_armchair_detailed__008` | 167 | 25 | 32 vertices | 11.7 s |

CoACD reported that the woven chair exceeded the concavity threshold under the
32-part budget (reported maximum 0.1025 versus threshold 0.0246 in its normalized
internal coordinates). The bounded asset is valid but should not be described
as geometrically equivalent to the original. These checks measure preprocessing
cost and structure; they do not establish GPU memory use or policy quality.

Fewer parts reduce the number of physical collision geoms and associated work.
Simulator contact/constraint capacities, model layout, and neural training
buffers still determine actual GPU memory. Measure complete training updates
with the intended environment count rather than inferring memory from hull
count alone.

## Geometry fidelity measurements

`tools/analyze_collision_profiles.py` compares original source surfaces with
the actual cached collision hulls, after physical scale is applied. It writes
metric JSON, sampled points/distances, and orthographic comparison figures under
`results/analysis/mixed_chair_collisions`. Example:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 uv run python tools/analyze_collision_profiles.py \
  data/motions/sitting/sitting__cocoon_woven_rattan_hanging_chair_stand_free__002.npz \
  data/motions/sitting/sitting__brutalist_1972_raw_concrete_sculpted_armchair_detailed__008.npz
```

Sampling uses seed 0 and 8,192 area-weighted original surface points. Another
16,384 points are sampled from compact hull surfaces; internally overlapping
faces are rejected by testing whether a 0.1 micrometre outward displacement is
outside the union of convex parts. This retains 15,116 and 16,104 union-boundary
samples respectively. Nearest-triangle distances use conservative spatial
candidate bounds and exact point/triangle projection. A check against exhaustive
search on 32 independent probes of the 40,000-triangle concrete mesh agreed
exactly. No distances are inferred from rendered pixels or point-cloud spacing.

| Metric, physical millimetres | Woven hanging chair | Concrete armchair |
| --- | ---: | ---: |
| Original surface → nearest compact hull surface, median / p95 | 6.8 / 30.1 | 3.6 / 8.0 |
| Original surface → compact **solid**, p95 | 8.5 | 2.2 |
| Compact union boundary → original surface, median / p95 | 10.1 / 80.7 | 14.9 / 100.1 |
| Compact union boundary → original surface, sampled maximum | 168.9 | 153.8 |

Distance to the solid is zero for points inside any convex part, so it measures
uncovered source surface rather than extra collision material. The reverse
surface metric detects generated surfaces far from the original mesh, including
cut faces or narrow inter-hull cracks inside the original object. It must not be
interpreted as purely outward inflation. The first metric includes internal
part faces and does not measure distance to the union boundary.

These are sampled distributions, not Hausdorff bounds, volume errors, or contact
accuracy guarantees. In particular a small original-to-solid distance says
nothing about filling an important opening or preserving a seat's support.
Both profiles still need representative contact/trajectory validation. The
rendered source is gray, compact hulls are blue in separate panels, and the
overlay adds translucent orange hulls. Views share metric bounds and use the
first reference orientation, with translation removed. Transparency and
lighting aid inspection; neither is used to calculate metrics.

Only the woven chair hits the 32-part cap; the concrete chair already uses 25.
Increasing the cap to 64 alone would not fix the latter's differences. For now
the compact recipe remains opt-in instead of adding an unvalidated second
recipe. Hull budget, concavity threshold, and per-hull vertex budget should be
evaluated together if higher physical fidelity is needed.
