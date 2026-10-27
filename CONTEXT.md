# SuperMimic task and learning contracts

Physical rollouts use the stock mjlab 1.6.0 `ManagerBasedRlEnv` and MuJoCo-Warp.
The package registers its tasks through the `mjlab.tasks` entry-point group. Standard
mjlab `train` and `play` own argument parsing, GPU selection, environment creation,
log-directory allocation and viewer lifecycle. `SuperMimicRunner` supplies learning,
checkpoint save/load and policy inference; it has no command-line launcher.

## Ownership

| Layer | Owns |
|---|---|
| `motion_data/` (repository root) | NumPy-only standard motion NPZ schema, validation and file IO |
| `motion_data/converters/` | One source-specific raw-data conversion script per dataset |
| `assets/` | Native robot/object models, joint names, actuators and differentiable FK |
| `motion.py` | Standard NPZ loading, G1 joint-name mapping, reference resampling and FK |
| `tasks/` | mjlab task registration, native manager configuration, task dimensions and checkpoint schema |
| `mdp/` | Native commands/actions/events, shared tensor observation/reward/termination heads and state/dynamics |
| `rl/` | Runner, replay adapter, models, storage, model fitting, imagined BPTT and critic updates |
| `tasks/interaction/` | Optional rigid-object references, task heads, collision assets and physical randomization |
| `rl/shared_world_model.py` | Task-independent state/integration, geometry encoder and masked normalization |
| `rl/shared_replay.py` | Portable real transitions, episode masks and geometry ID remapping |
| `tasks/entities/` | Variable-entity scenes, task references, raw observations, randomization and registration |
| `rl/entity_state.py`, `entity_world_model.py` | Entity physics schema, masks, graph interactions and shared dynamics weights |
| `rl/entity_policy.py`, `entity_replay.py` | Independent Actor/Critic scene encoders and portable entity replay |
| `scripts/evaluate.py` | Full motion-library scoring and trajectory recording, separate from interactive `play` |

Task IDs are `Mjlab-SuperMimic-Tracking-Unitree-G1` (baseline),
`Mjlab-SuperMimic-SharedTracking-Unitree-G1` (shared physics, no object), and
`Mjlab-SuperMimic-Sitting-Unitree-G1` (shared physics, one dynamic chair), plus
`Mjlab-SuperMimic-EntityTracking-Unitree-G1` and `Mjlab-SuperMimic-EntitySitting-Unitree-G1`
(new entity interaction physics). The latter implementation branches from
`f714fc134d0f8e79ea908da5dc809a194932aaea`; original task interfaces remain available.
Configuration factories are independent and can be imported without loading motion
files or initializing CUDA.
The registered `MotionCommand` creates the task runtime when the environment loads
its managers. The pure `build_task()` API supports tensor/model use without a simulator.

## Motion input

Raw-source handling belongs in `motion_data/converters/`: `grail.py` handles the GRAIL
robot/object exports and `lafan1.py` handles G1 CSVs. Invoke each with
`python -m motion_data.converters.<source>`. New sources add a converter that writes
the same NPZ format; downstream code does not branch on dataset identity.

`motion_data/__init__.py` supplies `validate_motion`, `load_motion` and `save_motion`
without importing Torch or mjlab. Clips use metres, radians, Z-up world coordinates,
wxyz quaternions and named scalar joints. Required fields are `fps`, `root_pos`,
`root_quat`, `joint_pos` and `joint_names`; optional object poses, scale and mesh fields
remain available to the shared kinematic replay tool. Arrays never require pickle.
The existing NPZ format is unchanged.

`MotionLib` accepts only these NPZs and maps `joint_names` to `G1_JOINT_NAMES`, rejecting
missing or extra joints. It reads each clip's own `fps` and resamples the robot poses
on the 50 Hz control clock: linear interpolation for positions/joint angles and SLERP
for root orientation, followed by velocities at 20 ms intervals and tracked-body FK.
It never changes playback speed by relabeling a file's frame rate. A GRAIL clip with
125 frames at 25 Hz becomes 249 frames at 50 Hz, with the same 4.96 s first-to-last
sample span. Samples do not extrapolate beyond the source's final timestamp. Training
requires at least two resampled frames; the portable format and kinematic replay allow
a single frame.

The tracking tasks default to `data/motions/lafan1`; Sitting defaults to
`data/motions/sitting`. `InteractionMotionLib` resamples object poses on the robot's
same frame grid and derives world velocities. Sitting selects clips with the same
mesh, collision recipe and scale as one anchor clip. One MuJoCo batch has one physical
object geometry; different shapes require separate runs. The original Tracking task
continues to read only robot references, including when given sitting NPZs.

## Baseline state and action

Collision inspection and object asset preparation use `motion_data/collision.py` and
`tools/prepare_collisions.py` / `tools/check_collisions.py`. Convex object parts retain
the original object frame; visual and collision geometry each apply `object_scale`
once. Cache keys include geometry, the complete decomposition recipe and engine version.
Geometry preparation does not add objects to the training task or change its state layout.
G1 omits both rubber-hand visual and collision geometries. Its 29 joints, including the
wrists, and fixed hand-body masses/inertias remain unchanged; hand contact is not modeled.
`make_g1_spec(interaction_collisions=True)` and the matching entity-config option add
full-foot object colliders. Ground uses collision bit 1, objects bit 2; the original
foot spheres still contact only the ground. Robot self-collisions remain disabled.
The tracking task uses the base collision profile unless explicitly configured otherwise.

The G1 state is `S = (p, R, v, w, q, qd)`: world pelvis-link position, rotation matrix,
world linear/angular link-origin velocity, and 29 named joint positions/velocities.
Replay stores orientation as a wxyz quaternion, giving 71 state values. State reads
use integrated generalized coordinates; reward and termination therefore agree with
that transition rather than lagging one physics substep behind derived MuJoCo fields.

An action has 29 values and sets joint targets around the default pose, with physical
calibration offsets and actuator scales. The 33 context values are friction, 29 joint
offsets and a three-dimensional torso CoM offset. The single-frame policy observation
has 93 values: 29 relative joint positions, 29 joint velocities, 3 body-frame angular
velocities, 3 projected-gravity components and 29 previous actions. This follows
Appendix A, Table A.1 of [SuperMimic v1](https://arxiv.org/pdf/2610.28480v1#page=13).
The actor observation is independent of the reference frame; its former six-dimensional
reference-relative root rotation, noise and configuration flag have been removed.

Ten observation frames give 930 values. Goals retain 73 values per future reference
frame (730 for ten frames), encoded into a 32-value latent; the policy decoder receives
962 values. This goal encoding is an independent implementation choice whose exact
agreement with the paper authors' implementation is unverified. The world model keeps
133 input values and 70 predicted increments. The critic keeps 184 current features
including privileged reference-relative rotation, plus the 730-value goal (914 total).
These are G1 task properties, not universal dimensions for future tasks.

The world model predicts frame-local increments. Its features, integration, regression
targets and loss must describe the same state.

## Legacy single-object contract

`SharedDynamics` fixes a common G1, 20 ms control interval, flat ground and at most
one free rigid body. SharedTracking uses the same schema with object presence zero.
Reference trajectories, rewards and task identifiers never enter the world model.

| Interface | SharedTracking | Sitting |
|---|---:|---:|
| Replay state | 84 | 84 |
| Replay context, including metadata | 47 | 47 |
| Actor observation / ten-frame history | 93 / 930 | 99 / 990 |
| Actor future robot goal / encoded goal | 730 / 32 | 730 / 32 |
| Actor decoder input / action | 962 / 29 | 1022 / 29 |
| World-model input / predicted increment | 228 / 82 | 228 / 82 |
| Critic current features / future goal / total | 285 / 790 / 1075 | 285 / 790 / 1075 |

State is `(p,R,v,w,q,qd, object_p,object_R,object_v,object_w)` in pure tensor code.
Replay packs each rotation as wxyz, yielding robot 71 + object 13 = 84. Positions
exclude parallel-environment origins. Linear and angular velocities are world-frame
body-origin velocities. Robot features/increments retain the baseline convention.
Object features are heading-relative position 3, heading-relative Rot6D orientation
6, and world linear/angular velocity expressed in current robot heading 3+3.
Object position/velocity increments use that same current heading; rotation increments
are local to the object. Twelve object increments extend the robot's 70 to 82.

Context stores robot physical values 33, object mass 1, object COM 3, symmetric
inertia 6, robot-object friction 1, object-ground friction 1, presence 1 and geometry
ID 1. The inertia is about the COM, expressed in the object body frame, ordered
`xx,yy,zz,xy,xz,yz`. Presence and ID are metadata preserved by context normalization.
The geometry ID is used only for lookup and is never a numeric network feature.

The dynamics input is robot 71 + object 15 + action 29 + physical context 45 +
geometry 67 + presence 1 = 228. Geometry consists of a jointly trained PointNet's
64 features and three metric extents. Its 1024 collision-surface points retain the
object origin and are divided isotropically by maximum extent for encoding. GeometryBank
caches points and device copies; it does not cache learned embeddings across updates.

Missing-object samples do not update object normalization statistics. Object input
blocks are masked after normalization; presence stays binary. Object prediction loss
and integrated increments are masked. Placeholder object quaternion is identity.
The robot receives the object state and geometry through the same joint dynamics
network, permitting robot-object effects in both directions.

Sitting appends relative position 3 and rotation-vector orientation 3 to Actor
observations. It uses pelvis origin and robot yaw only; body tilt does not rotate
these six coordinates. The Actor does not see object velocities, geometry, physical
parameters or future object goals. The Critic adds current object state/error,
physical parameters and detached shared geometry features; it also receives ten
future object reference poses, 6 per frame, in the current actual robot heading
frame. Its future goal has 730 robot + 60 object values. SharedTracking masks all
object features and emits zero object goals. Rotation vectors have a branch at pi.

This defines a reusable dynamics interface, not universal task generalization. New
contact regimes or shapes may require additional real transitions. A moving-object
objective that the Actor must distinguish also requires Actor goal conditioning.
See [shared-world-model.md](docs/shared-world-model.md) for implementation and CLI use.

## Entity interaction contract

`EntityDynamics` keeps G1 and an object axis: `71+13N` state values, `70+12N` output
increments, and context `33+13N+3(N+2)^2+4`. At default capacity 8 these are 175, 166
and 441. Per-object context contains mass, COM, inertia, valid/dynamic masks and a
geometry ID; pair friction and a plane normal/offset follow. IDs are lookup metadata.
All objects share encoder/head weights. Robot133, object90 and ground4 inputs become
256-wide tokens with four interaction layers; edge inputs18/13 become32-wide messages.
Geometry bank format2 stores 1024 metric xyz/unit-normal points and extents, not learned codes.

Raw Actor observations are `93+13N` (197 default), retaining per-object pose9 and
valid/dynamic/target/geometry metadata. Shared geometry and the Actor's own set encoder produce
128 scene features, hence221 per frame,2210 per ten-frame history and2242 after the
32-value robot goal latent. Actor, Critic and WM share the same geometry encoder module.
Only the WM optimizer owns these parameters. Actor/Critic geometry lookup runs without
autograd; their own scene encoders still learn and state/action gradients are preserved.
History stores raw observations so geometry codes are recomputed after WM updates.

Critic raw current features are `184+38N+3(N+2)^2+4` (792 default). It receives the
730-value robot future goal and `N*K*9` object future poses (720 default). Per-object
future tracks are encoded to32 and included in scene interaction; scene128 + robot184
+ robot goal730 form the1042-input value MLP. Current-object features and all future
poses use actual G1-relative frames. Target flags/references never enter the WM.

EntitySitting currently assigns the existing NPZ's one object trajectory to target
slot0. Additional configured dynamic bodies/static obstacles are real simulation
entities, but have no target trajectory; there is no multi-target NPZ parser yet.
Their physical effects still reach all three networks. Scene JSON or Python ObjectCfg
records add bodies; reference clocks, robot rewards and termination remain shared.
Only the target-object pose reward is added. Dynamic mass/friction use5–15kg/.3–1.2;
evaluation uses10kg/1.0 for the chair and configured values for extra objects.
Object-object effective friction is the componentwise max; ground copies are isolated.

Entity full checkpoints use schema5 and preserve geometry mappings and scene identity.
Schema4 full checkpoints had independent encoders and incompatible task optimizer groups;
they are rejected for full/inference task restore. Their standalone entity WM weights
remain compatible and initialize the shared encoder. New full checkpoints require equal
geometry-encoder weights in all three network paths; inference loads the actor's copy.
Standalone entity WM weights support changing capacity; archived replay requires an
exact schema including capacity. Old single-object model weights are not implicitly
converted. New defaults are128 training worlds,256 replay steps and WM/Actor batches128.
See [entity-world-model.md](docs/entity-world-model.md) for layouts, CLI, JSON and limits.

## Physical and imagined rollouts

Both paths use the same tensor MDP configuration for observations, rewards and
termination. Set reward weights through `commands["motion"].reward_cfg`; the replay adapter
rejects additional simulator-only reward terms, weights or a second dt scaling.
Physical stepping uses hard failure tests; imagination uses smooth
survival probabilities and differentiable rewards. A real contact sensor does not
supply a gradient through predicted object motion.

Removing reference-dependent actor observation does not remove reference-dependent
tracking objectives. Root orientation reward remains as explicitly defined in the
paper's Table A.2 (weight 0.5, scale 0.4), together with body orientation reward and
tilt-based termination. The baseline maximum per-step reward remains 0.1. The critic's privileged
features are separate from the actor observation contract; retaining them does not
establish that its exact encoding matches the paper's implementation.

Four 5 ms physics steps produce one 20 ms control step. At reference frame `f`,
the action advances physics; reward and failure compare the integrated state to
`f + 1`. The command then advances its reference index before observations are built.
History observations reset only selected worlds and refill with the new initial frame.
The inference callable consumes the observation dictionary passed by mjlab `play`.

Training disables native auto-reset. `ReplayEnvAdapter` records `s`, `s1`, `a`,
`a_prev`, `f`, `xi`, `r`, `term`, `done` and `ep_step`, then resets ended worlds.
`s1` is copied before reset. Native `play` uses auto-reset; quantitative evaluation
tracks a separate per-clip alive mask so reset rollouts cannot add to finished scores.
Stock mjlab applies step/interval events after reward and termination; replay's next
state includes such events, while reward belongs to the preceding physics result.

Sitting resets robot and chair from the same reference frame, then lets the free
chair evolve under gravity and contact. It adds only
`dt * weight * exp(-position_error_squared / position_std**2 - rotation_error_squared / orientation_std**2)`
to the original robot reward. Defaults are weight 1, position_std 0.1 m,
orientation_std 0.4 rad, giving a maximum step reward of 0.12 at dt=0.02.
Robot termination and reset-curriculum definitions are inherited.

Object mass 5–15 kg and G1-object sliding friction 0.3–1.2 are sampled each reset;
evaluation fixes them to 10 kg and 1.0. Inertia scales with mass. A robot-only
contact copy and a ground-only contact copy isolate sampled pair friction from
the unchanged G1-ground friction and fixed chair-ground friction 1.0. Collision
parts come from validated offline caches; missing caches never silently fall back
to one whole-chair hull. The handless G1 uses full-foot object colliders.
Unmodeled extra scene entities remain rejected.

## Checkpoints and validation

### Chair-disjoint experiments

`motion_data/chair_split.py` defines an immutable geometry-level split, grouping
identical unscaled mesh arrays before choosing train/test so scale variants cannot
leak. Seed 0 and fraction 0.8 yield 151/38 chair shapes (1391/357 clips). File hashes,
geometry identities and membership form a relocatable manifest fingerprint.
`scripts/chair_experiment.py` is an explicit multi-scene experiment entry point;
ordinary native train/play tasks keep their single-scene behavior.

`ChairExperimentRunner` keeps one native scene batch resident while preserving one
learner, all optimizer state, shared geometry encoder and bounded recent Replay.
Its training-only reference library assigns stable global frame IDs and its geometry
bank contains only training assets. Physical scene transitions retain local reference
clocks; the Replay adapter offsets recorded frames into the global reference library.
Scene switches mark a replay boundary and reset local observation history. Sampling
cycles uniformly through scenes in a deterministic shuffled order. Missing collision
assets never silently remove a chair from the experiment.

Full experiment checkpoints record the split, scheduling contract and visited training
shapes. Held-out actor transfer is a separate, explicit evaluation API: it validates
the physical/policy contract and training geometry provenance, but uses the new test
scene's own geometry IDs. It creates no world model, critic, optimizer or replay and
does not update network weights or normalization buffers. Metrics aggregate by chair
as well as clip. Reference-conditioned zero-shot geometry evaluation is distinct from
unseen-motion evaluation or a dedicated stable-sitting success test.
See [chair-generalization.md](docs/chair-generalization.md) for commands and limitations.

Full checkpoints require task metadata: baseline schema version 2, legacy interaction
schema version 3, and entity schema version 5. Incompatible layouts and missing metadata are rejected before loading.
Earlier 99-observation policies cannot load into either current task; their observation
semantics differ from Sitting's object-relative values. No implicit weight slicing or conversion is
performed. See [ADR 0003](docs/adr/0003-paper-observation-contract.md).
Fresh training and resumed training both fill a new replay
buffer before model fitting because replay is not checkpointed. Interactive play and
quantitative evaluation load policy weights without allocating replay or optimizers.

Shared physics checkpoints under `world_models/model_N.pt` contain the dynamics
schema, network/geometry encoder weights and all world-model normalization state.
`agent.world_model_file` loads these independently of task policy/critic schemas.
Full task resume uses `model_N.pt`; auxiliary directories do not affect native resume
selection. Inference does not require a world model.

Optional exports under `replays/model_N.pt` include real `s,s1,a,xi,done` and the
geometry bank. `agent.world_model_replays` loads archives, remaps geometry IDs and
mixes them into world-model fitting at `world_model_replay_fraction` (default 0.5).
Actor imagination starts remain sampled from the current task's replay. Window
masks include the terminal transition and exclude subsequent reset-episode steps;
windows never cross archive or environment boundaries. Shared model normalization
updates include valid archived transitions and exclude absent object slots.

Tests cover native task registration and CLI discovery, import safety, CPU FK against
MuJoCo, manager timing, partial/terminal resets, histories, randomization, contact
sensors and gradients through the learned model. Runtime smoke checks are separate
from a completed training benchmark. Earlier Isaac Lab results remain historical.
