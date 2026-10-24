# SuperMimic motion tracking with mjlab

**Unofficial** reproduction of *SuperMimic: Scaling Humanoid Whole-Body Control with One
Differentiable World Model* ([arXiv:2610.28480](https://arxiv.org/abs/2610.28480)). The target
experiment is the controlled LAFAN1 comparison (Fig. 4a): 40 G1-retargeted motion clips
(2.45 h), one GPU. This repository independently implements the method from the paper.

The simulator now uses **mjlab 1.6.0 with MuJoCo-Warp**, following the native simulation
organization used in `HDMI-mjlab-rewrite`. Training and evaluation load the G1 MJCF directly;
Isaac Lab, Isaac Sim, Newton, USD conversion, and their launchers are no longer required.
The learned world model, differentiable imagined rollouts, policy and critic remain the
SuperMimic algorithm; this migration does not replace them with HDMI's PPO trainer.

**The mjlab port has no completed training benchmark yet.** The GIF, curve, training times,
and checkpoints discussed in [MILESTONES.md](MILESTONES.md) belong to the earlier Isaac Lab
implementation. They do not validate performance or checkpoint transfer under mjlab.

Original implementation: **Chong Zhang** (ETH Zurich). Code license: [MIT](LICENSE).

## Layout

| Path | Purpose |
|---|---|
| `pyproject.toml`, `uv.lock` | Native Python package and locked runtime dependencies |
| `supermimic/assets/g1/` | G1 MJCF, mjlab entity, actuator settings, joint ordering and differentiable forward kinematics |
| `supermimic/tasks/g1_tracking/` | Current task's scene, simulation, MDP and learning configuration |
| `supermimic/tasks/` | Native mjlab task registration, dimensions and checkpoint contract |
| `supermimic/mdp/` | Commands, actions, events, observations, rewards, termination and state/dynamics functions |
| `supermimic/motion.py` | Motion loading and resampling |
| `supermimic/rl/` | Runner, replay adapter, world model, imagined BPTT updates and critic |
| `supermimic/scripts/evaluate.py` | Full motion-library scoring and trajectory recording |
| `tools/` | Setup, data download, URDF-to-MJCF conversion, checks, plotting and rendering |
| `docker/` | Optional plain Python/uv container; no simulator SDK image |
| `data/` | Downloaded motions and G1 assets; not tracked |
| `logs/<run>/` | Training/evaluation logs and checkpoints; not tracked |
| `results/` | Historical Isaac Lab result figures and GIF |

## Native setup

Requirements: Linux, `uv`, and an NVIDIA GPU with a driver compatible with the installed
PyTorch and Warp CUDA runtimes. `.python-version` selects Python 3.13. Install `ffmpeg`
separately if rendering videos. Docker is optional.

```bash
tools/setup.sh                 # uv sync: creates .venv and installs the package
tools/download_data.sh         # downloads motions/assets and creates the MJCF
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

A CUDA GPU is required for batched mjlab simulation. Default training uses 4096 worlds
and a large replay buffer. On a smaller GPU, reduce `--env.scene.num-envs`,
`--agent.buffer-steps`, `--agent.wm-batch` and `--agent.pi-batch`.

The package registers **`Mjlab-SuperMimic-Tracking-Unitree-G1`** with mjlab through
`[project.entry-points."mjlab.tasks"]`. As in HDMI, the stock `train` and `play`
commands create an ordinary `ManagerBasedRlEnv`; no project launcher or custom
simulation loop is involved. `SuperMimicRunner` implements the framework's runner
interface while retaining SuperMimic's learned dynamics and BPTT algorithm.

```text
mjlab train TASK
  → registered env/agent configuration
  → ManagerBasedRlEnv → motion/action/observation/reward/event managers
  → SuperMimicRunner.learn()
      → physical collection → replay → world-model fitting
      → imagined rollouts → BPTT policy update → critic fitting
      → checkpoint

mjlab play TASK → checkpoint policy → native viewer
scripts.evaluate → checkpoint policy → complete-clip scores / recorded trajectories
```

Physics uses 50 Hz control and four 5 ms MuJoCo-Warp steps per action. Training's
replay adapter copies terminal transitions before resetting ended worlds; `play`
uses native auto-reset. Real and imagined trajectories share the same tensor
observation, reward and termination heads. See [CONTEXT.md](CONTEXT.md) and
[ADR 0002](docs/adr/0002-native-mjlab-entrypoints.md) for the contracts and timing.

### Extending toward object interaction

Add assets to a new task's `cfg.scene.entities` and native `ContactSensorCfg`
instances to `cfg.scene.sensors`. Ordinary mjlab environments support these scenes;
the robot-only replay adapter rejects training when unmodeled entities are present.
Object-conditioned training additionally requires:

1. Object reference data, state layout, dynamics features/integration/losses and schema.
2. Goals, observations, context, reset behavior and critic inputs for robot and object.
3. Shared rewards and hard/soft termination. Contact-dependent imagined rewards need
   a differentiable predictor or approximation; real sensors alone cannot supply it.
4. Checks of terminal transitions, partial resets, model fitting and reward gradients.

`BuiltinPositionActuatorCfg` preserves torque caps but has no equivalent to Isaac
Lab's `velocity_limit_sim`. Motor speed limits remain metadata rather than a
solver-enforced clamp, so the backends are not physically identical.

### Data and robot assets

| Data | Source | License | Destination |
|---|---|---|---|
| G1-retargeted LAFAN1 clips: root position, root quaternion xyzw and 29 joint angles at 30 Hz | HF [`lvhaidong/LAFAN1_Retargeting_Dataset`](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset), `g1/` | CC BY-NC-ND 4.0 (Ubisoft La Forge) | `data/lafan1_g1/*.csv` |
| G1 URDF and meshes used by these retargets | Same dataset, `robot_description/g1/` | BSD-3-Clause (Unitree) | `data/g1_description/` |

The downloader resolves a single immutable dataset revision before downloading all files,
and records its SHA and file list in `data/source.json`. To select a known revision:

```bash
LAFAN1_REVISION=<dataset-commit-sha> tools/download_data.sh
```

`tools/urdf_to_mjcf.py` compiles the source URDF into a free-floating G1 MJCF. This same
model supplies simulation and differentiable FK. There is no USD intermediate:

```text
data/
├── source.json
├── lafan1_g1/                 # expected full dataset: 40 CSVs
└── g1_description/
    ├── g1_29dof_rev_1_0.urdf
    ├── g1_29dof.xml           # generated native MuJoCo asset
    └── meshes/*.STL
```

Run the runtime checks after downloading assets:

```bash
uv run python tools/test_spawn.py --device cuda:0
uv run python tools/test_env.py --device cuda:0
uv run python tools/test_reset.py --device cuda:0
```

These checks exercise the native simulator, joint/body mapping, FK consistency and reset
behavior. Passing them is separate from demonstrating learned tracking performance.

The regression suite also compares FK against CPU MuJoCo Jacobians and checks that replay
receives the terminal state before a partial reset:

```bash
uv run pytest
# If a system-wide ROS/pytest plugin interferes with this virtual environment:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

The regression suite covers native task registration, import safety, integrated-state
reward timing, terminal transitions, partial resets and policy histories, randomization,
object/contact scene composition and gradients through the world model. Small runtime
checks validate software behavior; they are not trained-policy performance evidence.

The native-entrypoint migration passed 30 CPU/CUDA tests on an RTX 4070 Laptop GPU.
The unmodified mjlab `train` command also completed a three-iteration real walking-clip
check and resumed from iteration 3 to 4 with replay refilling. The saved policy loaded
in stock `play` (Viser), and the quantitative evaluator wrote scores and a trajectory.
These small runs used reduced batches and disabled compilation.

## Training, play and quantitative evaluation

```bash
# The stock mjlab entry point owns GPU selection and the timestamped log directory.
uv run train Mjlab-SuperMimic-Tracking-Unitree-G1 \
  --env.commands.motion.motion-dir data/lafan1_g1 \
  --gpu-ids '[0]' --log-root logs --agent.experiment-name lafan1_mjlab

# Convenience alias; extra arguments are passed unchanged to mjlab train.
./run_train.sh lafan1_mjlab --gpu-ids '[0]'
CPUS=0-23 ./run_train.sh lafan1_mjlab --gpu-ids '[0]'

# Use the run directory printed by train.
RUN_DIR=logs/lafan1_mjlab/YOUR_RUN_DIRECTORY

# Interactively view a saved policy.
# For other datasets, prefix play with SUPERMIMIC_MOTION_DIR=/path/to/csvs.
uv run play Mjlab-SuperMimic-Tracking-Unitree-G1 \
  --checkpoint-file "$RUN_DIR/model_14000.pt"

# Score every clip deterministically after training.
uv run python -m supermimic.scripts.evaluate \
  --task Mjlab-SuperMimic-Tracking-Unitree-G1 \
  --run-dir "$RUN_DIR" \
  --checkpoint-file "$RUN_DIR/model_14000.pt" --device cuda:0

# Record and render the physical evaluation trajectory.
uv run python -m supermimic.scripts.evaluate \
  --run-dir "$RUN_DIR" \
  --checkpoint-file "$RUN_DIR/model_14000.pt" --record traj.npz
uv run python tools/render.py traj.npz out.mp4 --clips dance2_subject1 --seconds 10
```

Native mjlab stores runs under `log_root/experiment_name/<timestamp>[_run_name]`.
Checkpoints are `model_<iteration>.pt`; each contains policy, world model, critic,
optimizer state and task metadata. `params/env.yaml` and `params/agent.yaml` record
resolved configuration. Resuming uses standard `--agent.resume True`, `--agent.load-run`
and `--agent.load-checkpoint`; replay is refilled before any resumed model updates.

Use `uv run train Mjlab-SuperMimic-Tracking-Unitree-G1 --help` for all native options.
The first iterations include compilation; `--agent.no-compile True` is useful for short
checks. Default batches remain 4096 environments, 1000 replay steps per world,
8-step model fitting, 32-step imagination, 10 model batches of 2048 windows,
4096 imagined rollouts and 500 world-model-only iterations. `--agent.amp none`
disables BF16; TF32 remains enabled for FP32 matmuls. Training is single-GPU.

Quantitative evaluation starts each clip at its first frame, disables randomization
and observation/action noise, and scores until the first failure or clip completion.
It evaluates every clip in the selected directory; no train/test split is inferred.
It loads the policy only, without training replay or optimizers. `--watch` scans for
new checkpoints; use a separate GPU or evaluate after training for timing comparisons.

### Optional Docker

The Docker image installs the same locked Python dependencies. It does not download motion
data or include an Isaac Lab checkout. Build from the repository root:

```bash
docker build -f docker/Dockerfile -t supermimic:mjlab .
docker/run_container.sh 0 128g
# The repository is mounted; the container's environment lives separately at /opt/venv.
docker exec supermimic tools/download_data.sh
docker exec -it supermimic ./run_train.sh lafan1_mjlab --gpu-ids 0
```

This path requires Docker and NVIDIA Container Toolkit. GPU index `0` inside the container
refers to the GPU selected by `docker/run_container.sh`.

## Algorithm and retained choices

* G1 29-DoF with BeyondMimic actuator parameters: 10 Hz natural frequency, damping ratio 2,
  motor armatures and action scale `0.25 * torque_limit / stiffness`.
* Motion library: 30 to 50 Hz resampling, finite-difference velocities, FK of 14 tracked
  bodies, and 1 s sampling bins.
* Paper Table A.1's 93-dimensional proprioceptive observation: 29 joint positions,
  29 joint velocities, 3 base angular velocities, 3 projected-gravity components and
  29 previous actions. Ten frames give a 930-dimensional history. Reference-relative
  root orientation is not part of this actor observation.
* Ten future reference frames, analytic tracking rewards,
  no self-collision reward, joint-limit weight −2; hard termination at root height error
  above 0.25 m or projected-gravity z difference above 0.8.
* Return-based reset sampling: 50% uniform, 40% softmax of negative bin return, and 10%
  historical softmax.
* World model: MLP `(2048, 2048, 2048)` on heading-invariant state, action and physical
  context, predicting local state increments. Training uses 8-step autoregressive windows.
* Motion encoder and decoder: MLP `(2048, 1024, 512, 256)` with a 32-dimensional tanh latent;
  critic: MLP `(256, 128)`. Policy gradients pass through 32 imagined steps, differentiable
  rewards, soft survival probabilities and the final critic value.
* Critic targets use GAE with γ=0.99 and λ=0.95, followed by 5 epochs × 4 minibatches.
* Evaluation starts each clip at its first frame with friction 1.0, no domain randomization,
  no observation/action noise and deterministic actions. A failure uses the training
  termination conditions; normalized return also reflects tracking accuracy.

The observation definition now follows Appendix A, Table A.1 of the
[paper v1](https://arxiv.org/pdf/2610.28480v1#page=13). The earlier implementation's extra
six-dimensional reference-relative root rotation and its noise/configuration option
have been removed. This changes the policy decoder input to `930 + 32 = 962` values.
Root orientation tracking reward remains: it is explicitly part of Table A.2, with
weight 0.5 and scale 0.4. Body orientation tracking and tilt-based failure tests also
remain. The critic retains its privileged reference-relative orientation feature;
Table A.1 defines the actor observation, not the critic's feature encoding.

Other implementation choices remain independently specified. The goal is still
73 values per future frame (730 for ten frames); the paper does not fully specify
this encoding, so its exact correspondence to the authors' implementation is unverified.
BF16 differs from the paper's TF32; physical self-collisions are disabled; analytic
termination uses sigmoid smoothing; and 500 world-model-only warm-up iterations
precede policy updates. Friction and torso CoM are sampled once per environment,
while joint offsets are sampled on reset. Training episodes end after 10 s or at clip end;
evaluation uses the complete clip. These choices need controlled ablations before claiming
strict paper reproduction. See [ADR 0003](docs/adr/0003-paper-observation-contract.md)
for the observation change and checkpoint contract.

## Historical result: earlier Isaac Lab backend

The previous implementation reported normalized return **0.824 at 1.24 h** and **0.829 with
0/40 failed clips at 1.47 h** (iteration 14000) on one RTX PRO 6000 Blackwell. The paper reports
0.82 within 1.2 h on an H100. Hardware, precision and observation differences prevent a
controlled speed comparison. No corresponding mjlab training result is claimed here.

![Historical Isaac Lab tracking demonstration](results/supermimic_lafan1_highlight.gif)

*Historical simulation: Isaac Lab with Newton/MuJoCo-Warp. Solid: policy; blue ghost: reference.*

![Historical Isaac Lab success curve](results/success_curve.png)

See [MILESTONES.md](MILESTONES.md) for the original run log. Those policies used the former
99-dimensional observation and cannot load into the current task. Checkpoints now require
task schema version 2; version 1 and checkpoints without task metadata are rejected.
There is no automatic weight slicing or legacy conversion. Train a new policy for the
93-dimensional observation contract; the historical scores do not validate this change.

## Attribution and licenses

* Method: Q. Liao, Y. Shao, J. Long, K. Zakka, R. Horowitz, P. Abbeel, K. Sreenath,
  *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*,
  CoRL 2026, arXiv:2610.28480. This is an independent implementation, unaffiliated with the
  paper's authors.
* The current simulation backend uses mjlab, MuJoCo and MuJoCo-Warp; their upstream licenses
  apply. Native simulation organization follows `HDMI-mjlab-rewrite`.
* The historical implementation used Isaac Lab (BSD-3-Clause), Newton and MuJoCo-Warp
  (Apache-2.0). It was originally described by its author as an unreviewed, one-hour
  AI-assisted implementation.
* LAFAN1 motions (Ubisoft La Forge): CC BY-NC-ND 4.0; Unitree G1 description: BSD-3-Clause.
  Motion data is not redistributed here. The historical demonstration GIF includes reference
  motions and is not covered by the repository's MIT license.

Code copyright (c) 2026 ETH Zurich, Chong Zhang. Released under the [MIT License](LICENSE).
