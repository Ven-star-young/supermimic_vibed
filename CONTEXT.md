# Task-specific learning contracts

This branch trains a separate Actor, World Model and Critic per task family.
Tracking and Sitting share optimization machinery, not model parameters or a cross-task
state schema. Sitting contains exactly G1 + one free rigid chair per environment.
Different worlds may use different native chair variants and matching references.

## Ownership

- `motion_data/`: standard NumPy NPZ, per-source converters, collision caches, selected
  references and immutable chair-disjoint train/test manifests.
- `supermimic/assets/g1/`: 29-joint G1, actuator settings and differentiable FK.
- `supermimic/mdp/`: common G1 math and native manager adapters.
- `supermimic/tasks/<task>/`: state, model factories, observations, goals, reward,
  termination, environment and checkpoint semantics.
- `supermimic/rl/`: replay collection, WM regression, actor imagination, critic fitting,
  running statistics, geometry components and explicit distributed gradient reduction.
- `scripts/train_sitting.py`: one mixed-chair training workflow.
- `scripts/play_policy.py`: one Sitting deployment workflow with frozen Actor only.
- `scripts/evaluate.py`: full-library scoring/trajectory export for a registered task.

No simulator or data files are loaded at module import. Native mjlab train/play remain
available. Model construction belongs to the task; runner has no task-type dispatch.
The former SharedTracking registration and cross-task WM/replay adapters are removed.

## Numerical contracts

Control dt = 0.02s, four 0.005s physics steps. Data resampling preserves duration:
125 frames at25Hz become249 at50Hz (4.96s first-to-last). Standard quaternions are wxyz;
all physical units are SI. Scene origins are removed from stored world positions.

| Contract | Tracking | Sitting |
|---|---:|---:|
| Replay state | 71 | 84 |
| Context storage | 33 | 46 (45 physics + geometry ID) |
| Dynamic actor frame | 93 | 102 |
| Actor history (10) | 930 | 1020 |
| Robot reference (10) → latent | 730→32 | 730→32 |
| Static geometry | none | 64, concatenated once |
| Actor decoder | 962→29 | 1116→29 |
| WM input → increment | 133→70 | 224→82 |
| Critic | 914→1 | 1101→1 |

Sitting state is `(robot_p,R,v,w,q,qd,chair_p,R,v,w)`. Replay packs rotations as
quaternions. Root velocities refer to body origins, not COM. WM invariant G1 features
are height1, yaw-free Rot6D6, body-frame linear/angular velocity6, q29 and qd29.
Chair features are position3 and Rot6D6 relative to actual robot heading, plus world
linear/angular velocities6 expressed in that heading. They are not relative velocities.
WM receives71+15+29+45+64=224; it predicts robot70 and chair12 increments.

Chair position/velocity increments use the current actual robot heading; its rotation
increment uses the chair body frame. State integration, regression targets and physical
state reads must agree. WM does not receive reference, reward or task ID.

Actor sees robot93 + chair relative pose9 per frame, no privileged chair velocity or
physics. Robot future goals are73/frame in the current reference heading, including
0.1-scaled joint velocity. Critic receives robot184 + chair15 + current chair error6
+ chair physics12 + geometry64 + robot future730 + chair future90 =1101.
Chair future poses use the current actual robot heading for every horizon frame.
See [the complete model contract](docs/task-specific-sitting-design.md).

## Geometry and optimization

GeometryBank version3 uses only real assets, starting at ID0. Each asset is deterministic
1024×6 local metric xyz+normal samples, scaled once, with metric extents. PointNet
6→64→128, max-pool, concatenate extents3, linear131→64. IDs select assets only.

Actor and WM reference the same encoder instance; Critic consumes its detached output.
Only WM optimizer owns encoder parameters. Actor/Critic recompute current embeddings
and never put learned geometry in history or Replay. Running normalizers exclude learned
geometry. During Actor BPTT, WM and Critic parameters freeze while action/state gradients
through both networks remain intact.

WM fits actual replay transitions with episode masks. Actor starts from current task
Replay, imagines differentiable states/rewards/soft terminations, and uses Critic terminal
value. Critic learns lambda-returns. Real and imagined rewards share one tensor function;
Sitting adds only chair pose reward to existing tracking terms and retains termination.

## Simulation and distributed execution

Real transition state is copied before resetting terminal environments. Reward compares
integrated state to the next reference frame. Reset installs reference positions AND
velocities before history backfill. Clean deployment turns off noise; perturbations occur
before observations, not after them.

Each mixed world has one fixed chair variant and a matching reference sampler. Remainder
environments are assigned with counts differing by at most one; each GPU must have at
least as many environments as training chairs. No periodic scene-rebuild scheduler.

Mass5–15kg and robot-chair friction0.3–1.2 randomize each reset (eval10kg/1.0). Inertia
scales from each world's own default chair; contact meshes separate robot and floor roles.
Native variant padding must preserve these roles. Collision sizing uses25% headroom and
128-contact/512-constraint alignment; sticky overflow is rejected before Replay insertion.
Missing collision caches fail explicitly. EPA scratch is enlarged before graph capture.

One simulator and Replay per rank; gradients average before clipping; initial parameters
broadcast; normalizer statistics use global samples. Buffers/batches/environment counts
are per rank, logged sim_steps are global. Only rank0 writes models/logs. Rank-local
physics errors are synchronized before entering the next learning collective.

## Checkpoints and retained tests/tools

Sitting schema4 identifies model/coordinate/reward semantics, independently of scene ID.
Full checkpoints contain pi/wm/critic, optimizer states, geometry_bank, policy_config,
iteration counters and task metadata. Experiment infos contain split fingerprint,
collision profile and training exposure. Full resume validates the asset bank and experiment;
Actor deployment validates task/policy schema but keeps the destination scene's geometry.
Shared encoder tensors in pi and wm must agree. No old entity checkpoint conversion.

Tools retain source conversion, kinematic replay, collision preparation/checks, reference
selection, split generation, plotting and trajectory rendering. Test helpers live under
tests/. One-off diagnostics stay on the historical branch. Tests protect coordinate and
clock conventions, gradients, physical transitions, reset/variants/collisions, train/heldout
separation, checkpoint roundtrips and multi-process synchronization.
