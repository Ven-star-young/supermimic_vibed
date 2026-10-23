# SuperMimic task and learning contracts

This repository trains G1 motion tracking with SuperMimic's learned dynamics and imagined
policy gradients. Physical rollouts use mjlab 1.6.0 and MuJoCo-Warp. HDMI supplies a useful
organization pattern for assets, task configuration and MDP terms; its PPO update is not
the learning algorithm used here.

## Ownership

| Layer | Owns | Does not own |
|---|---|---|
| `assets/` | Native robot/object models, joint names, actuators | Rewards or optimizer updates |
| `tasks/` | A task's composition, dimensions and checkpoint contract | Generic rollout optimization |
| `envs/` | Scene/simulation lifecycle, physical steps, terminal snapshots and reset | Imagined-state integration |
| `mdp/` | Commands, actions, events, pure observation/reward/termination functions, state/dynamics representations | Training loop scheduling |
| `rl/` | Models, replay, model fitting, imagined BPTT updates, critic and evaluation | Task-specific scene contents |

`tasks.g1_tracking.env_cfgs.make_env_cfg()` creates a fresh configuration. Mutating one
configuration must not affect another. `tasks.load_task("g1_tracking", ...)` constructs
the task runtime from motion data and the device. The runtime supplies dimensions and
environment construction, allowing replay and model input/output allocation to use the same
task contract. CLI modules delegate to runtime functions only when called explicitly.

## Current state and action

The G1 state is `S = (p, R, v, w, q, qd)`: world pelvis-link position, rotation matrix,
world linear/angular link-origin velocity, and 29 named joint positions/velocities. Replay
stores the rotation as a wxyz quaternion, giving 71 state values. These conventions match
the motion library and differentiable FK. Joint order is defined by the asset, not by
incidental MuJoCo index order.

An action has 29 values and sets joint targets around the default pose, with physical
calibration offsets and actuator scales. The context currently has 33 values: friction,
29 joint offsets and a three-dimensional torso center-of-mass offset. The observation has
99 values by default, including the existing six-dimensional reference-relative rotation;
the observation configuration can disable that addition. Goals contain 73 values per
reference frame. These are properties of the G1 tracking task, not universal dimensions
for future tasks.

The world model predicts frame-local increments. Its integration, regression targets and
error representation must describe the same state. A future object task must extend all
three together; increasing replay width alone does not make a model object-aware.

## Physical and imagined rollouts

Both paths use the same task MDP instance and configuration for observations, rewards and
termination. Rewards must remain differentiable with respect to imagined states. Hard
termination is used in physics; the corresponding smooth survival probability is used in
imagination. Calling a MuJoCo contact sensor inside an imagined reward cannot supply a
gradient through predicted object motion.

Physical `step(action)` returns `s`, `s1`, `a`, `a_prev`, `f`, `xi`, `r`, `term`, `done`,
`ep_step` and diagnostic reward terms. `s1` is copied **before** automatic reset. A failed
world resets without changing other worlds; replay windows mask transitions across episode
boundaries. Physics advances four 5 ms steps per 20 ms control step.

The scene can contain additional native entities and sensors for simulation experiments.
The current task's training factory rejects entities outside its robot-only learned state.
Adding object training requires object motion data, state/dynamics, reset, goals, observation,
reward and checkpoint contracts. This refactor establishes boundaries for that work; it does
not implement an object interaction policy.

## Compatibility and validation

Historical top-level imports remain adapters so existing smoke scripts and integrations keep
working. New checkpoints carry task metadata; incompatible task layouts must be rejected
before model loading. Historical checkpoints remain tied to their original task/backend and
must be evaluated before reuse.

The regression suite covers CPU FK against MuJoCo Jacobians, native CUDA reset and stepping,
terminal snapshots, physical randomization, configuration isolation, import safety and
gradients through the world model. Contact histories remain unchanged in worlds excluded
from a reset, and step/interval state changes are included in the saved next state. Small
tests validate software behavior; no completed
mjlab training benchmark is claimed. Earlier Isaac Lab results remain historical evidence.
