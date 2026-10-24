# SuperMimic task and learning contracts

Physical rollouts use the stock mjlab 1.6.0 `ManagerBasedRlEnv` and MuJoCo-Warp.
The package registers its task through the `mjlab.tasks` entry-point group. Standard
mjlab `train` and `play` own argument parsing, GPU selection, environment creation,
log-directory allocation and viewer lifecycle. `SuperMimicRunner` supplies learning,
checkpoint save/load and policy inference; it has no command-line launcher.

## Ownership

| Layer | Owns |
|---|---|
| `assets/` | Native robot/object models, joint names, actuators and differentiable FK |
| `tasks/` | mjlab task registration, native manager configuration, task dimensions and checkpoint schema |
| `mdp/` | Native commands/actions/events, shared tensor observation/reward/termination heads and state/dynamics |
| `rl/` | Runner, replay adapter, models, storage, model fitting, imagined BPTT and critic updates |
| `scripts/evaluate.py` | Full motion-library scoring and trajectory recording, separate from interactive `play` |

The task ID is `Mjlab-SuperMimic-Tracking-Unitree-G1`. Configuration factories are
independent and can be imported without loading motion files or initializing CUDA.
The registered `MotionCommand` creates the task runtime when the environment loads
its managers. The pure `build_task()` API supports tensor/model use without a simulator.

## State and action

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
targets and loss must describe the same state. Object interaction must extend all of
these together; changing replay width or scene entities alone is insufficient.

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
tilt-based termination. The maximum per-step reward remains 0.1. The critic's privileged
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

Object entities/contact sensors can be inspected in an ordinary native environment.
The replay adapter rejects scene entities absent from the task's modeled state.
A future interaction task must supply object references, state/dynamics, context,
goals, reset behavior and differentiable contact objectives before training.

## Checkpoints and validation

Checkpoints require task metadata with schema version 2. Version 1, incompatible layouts
and missing metadata are rejected before loading. Earlier 99-observation policies cannot
load into the current 93-observation task; no implicit weight slicing or conversion is
performed. See [ADR 0003](docs/adr/0003-paper-observation-contract.md).
Fresh training and resumed training both fill a new replay
buffer before model fitting because replay is not checkpointed. Interactive play and
quantitative evaluation load policy weights without allocating replay or optimizers.

Tests cover native task registration and CLI discovery, import safety, CPU FK against
MuJoCo, manager timing, partial/terminal resets, histories, randomization, contact
sensors and gradients through the learned model. Runtime smoke checks are separate
from a completed training benchmark. Earlier Isaac Lab results remain historical.
