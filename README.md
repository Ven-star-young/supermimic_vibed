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
| `supermimic/robot.py` | G1 MJCF, mjlab entity, actuator settings and joint ordering |
| `supermimic/env.py` | mjlab scene/simulation wrapper, state reads, resets and domain randomization |
| `supermimic/motion.py`, `kinematics.py` | Motion loading, resampling and differentiable forward kinematics |
| `supermimic/mdp.py` | Observations, goals, reward and termination functions |
| `supermimic/networks.py`, `buffer.py`, `train.py` | World model, replay, imagined BPTT policy updates and critic training |
| `supermimic/evaluate.py` | Deterministic per-clip checkpoint evaluation |
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

A CUDA GPU is required for batched mjlab simulation. The default training configuration uses
4096 environments and a large replay buffer; reduce `--num_envs`, `--buffer_steps`,
`--wm_batch`, and `--pi_batch` when checking the runtime on a smaller GPU.

Physics uses 50 Hz control and four 5 ms MuJoCo-Warp simulation steps per action. The native
scene owns the ground and a G1 entity; mjlab position actuators apply the same joint PD
parameters and action scaling as the previous implementation. `TrackingSim` owns simulation
creation and cleanup, with no global application launcher:

```python
from supermimic.env import TrackingSim

with TrackingSim(num_envs, motion, mdp, device="cuda:0") as env:
    transition = env.step(action)
```

mjlab's `BuiltinPositionActuatorCfg` preserves the joint torque caps, but has no equivalent
to Isaac Lab's `velocity_limit_sim`. Motor speed limits remain metadata rather than a
solver-enforced clamp, so the two backends are not physically identical.

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

The migration was checked on an RTX 4070 Laptop GPU: ten CPU/CUDA regression cases passed,
and a three-iteration run on one real LAFAN1 walking clip exercised simulation collection,
world-model fitting, BPTT policy/critic updates, checkpoint reload and evaluation recording.
This short check used reduced batches and `--no_compile`; it is not a trained policy benchmark.

## Training and evaluation

```bash
# Foreground training, with stdout saved under logs/lafan1_mjlab/.
./run_train.sh lafan1_mjlab --device cuda:0

# Equivalent direct entry point.
uv run python -m supermimic.train --log_dir logs/lafan1_mjlab --device cuda:0

# Optional CPU affinity; start in a tmux session yourself if desired.
CPUS=0-23 ./run_train.sh lafan1_mjlab --device cuda:0

# Score a checkpoint after training (replace the path with an existing checkpoint).
uv run python -m supermimic.evaluate --run_dir logs/lafan1_mjlab \
  --ckpt logs/lafan1_mjlab/ckpt_014000.pt --device cuda:0

# Plot, record trajectories, and render a clip.
uv run python tools/plot.py logs/lafan1_mjlab --out results/mjlab_curve.png
uv run python -m supermimic.evaluate --run_dir logs/lafan1_mjlab \
  --ckpt logs/lafan1_mjlab/ckpt_014000.pt --record traj.npz --device cuda:0
uv run python tools/render.py traj.npz out.mp4 --clips dance2_subject1 --seconds 10
```

The first iterations include `torch.compile` kernel compilation. Checkpoints default to every
2000 iterations. `--watch` is available for continuous evaluation, but evaluate after training
or on a separate GPU when measuring training time. The launcher now runs locally in the
foreground and preserves individual command-line arguments; it no longer starts Docker/tmux.
For a short runtime check, `--no_compile` disables compilation overhead; it is not the
configuration to use when comparing training throughput.

The default optimizer and network configuration still follows the original implementation:
4096 environments, 1000 replay steps/environment, 8-step model fitting, 32-step imagination,
10 model batches of 2048 windows, 4096 imagined rollouts, 500 model-only warm-up iterations,
and BF16 MLP autocast. Use `--amp none` to disable BF16; TF32 remains enabled for FP32 matmuls.

### Optional Docker

The Docker image installs the same locked Python dependencies. It does not download motion
data or include an Isaac Lab checkout. Build from the repository root:

```bash
docker build -f docker/Dockerfile -t supermimic:mjlab .
docker/run_container.sh 0 128g
# The repository is mounted; the container's environment lives separately at /opt/venv.
docker exec supermimic tools/download_data.sh
docker exec -it supermimic ./run_train.sh lafan1_mjlab --device cuda:0
```

This path requires Docker and NVIDIA Container Toolkit. GPU index `0` inside the container
refers to the GPU selected by `docker/run_container.sh`.

## Algorithm and retained choices

* G1 29-DoF with BeyondMimic actuator parameters: 10 Hz natural frequency, damping ratio 2,
  motor armatures and action scale `0.25 * torque_limit / stiffness`.
* Motion library: 30 to 50 Hz resampling, finite-difference velocities, FK of 14 tracked
  bodies, and 1 s sampling bins.
* Ten-frame observation history, ten future reference frames, analytic tracking rewards,
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

This simulator migration preserves existing method choices rather than resolving all
paper differences. In particular, actor observations include an extra six-dimensional
relative reference-root rotation; BF16 differs from the paper's TF32; physical self-collisions
are disabled; analytic termination uses sigmoid smoothing; and 500 world-model-only warm-up
iterations precede policy updates. Friction and torso CoM are sampled once per environment,
while joint offsets are sampled on reset. Training episodes end after 10 s or at clip end;
evaluation uses the complete clip. These choices need controlled ablations before claiming
strict paper reproduction.

## Historical result: earlier Isaac Lab backend

The previous implementation reported normalized return **0.824 at 1.24 h** and **0.829 with
0/40 failed clips at 1.47 h** (iteration 14000) on one RTX PRO 6000 Blackwell. The paper reports
0.82 within 1.2 h on an H100. Hardware, precision and observation differences prevent a
controlled speed comparison. No corresponding mjlab training result is claimed here.

![Historical Isaac Lab tracking demonstration](results/supermimic_lafan1_highlight.gif)

*Historical simulation: Isaac Lab with Newton/MuJoCo-Warp. Solid: policy; blue ghost: reference.*

![Historical Isaac Lab success curve](results/success_curve.png)

See [MILESTONES.md](MILESTONES.md) for the original run log. Old checkpoints may have compatible
network shapes, but policy behavior must be reevaluated in the new simulator before reuse.

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
