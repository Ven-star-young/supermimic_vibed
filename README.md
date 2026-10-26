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
| `supermimic/tasks/interaction/` | Shared tracking/sitting task heads, object references, dynamic chair and randomization |
| `supermimic/tasks/` | Native mjlab task registration, dimensions and checkpoint contract |
| `supermimic/mdp/` | Commands, actions, events, observations, rewards, termination and state/dynamics functions |
| `motion_data/__init__.py` | NumPy-only shared NPZ schema, validation, loading and saving |
| `motion_data/converters/` | One raw-data converter per source: `grail.py`, `lafan1.py` |
| `supermimic/motion.py` | Standard NPZ loading, named G1 joint mapping and 50 Hz reference resampling |
| `supermimic/rl/` | Runner, replay adapter, world model, imagined BPTT updates and critic |
| `supermimic/rl/shared_world_model.py`, `shared_replay.py` | Shared robot/object physics, geometry encoder and portable replay across tasks |
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
tools/download_data.sh         # downloads raw motions/assets, creates MJCF and standard NPZs
uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

A CUDA GPU is required for batched mjlab simulation. Default training uses 4096 worlds
and a large replay buffer. On a smaller GPU, reduce `--env.scene.num-envs`,
`--agent.buffer-steps`, `--agent.wm-batch` and `--agent.pi-batch`.

The package registers three tasks with mjlab through
`[project.entry-points."mjlab.tasks"]`:

| Task suffix after `Mjlab-SuperMimic-` | Actor observation | Replay state | WM input → increment | Critic input |
|---|---:|---:|---:|---:|
| `Tracking-Unitree-G1` — original baseline | 93 | 71 | 133 → 70 | 914 |
| `SharedTracking-Unitree-G1` — no object, shared dynamics | 93 | 84 | 228 → 82 | 1075 |
| `Sitting-Unitree-G1` — one dynamic chair | 99 | 84 | 228 → 82 | 1075 |

Dimensions assume the default ten future reference frames. SharedTracking and Sitting
reuse the same world-model checkpoint and geometry encoder. Their policies remain
task-specific. As in HDMI, the stock `train` and `play`
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

### Shared world model and sitting

`Sitting` adds a free-moving chair, its relative pose in the actor observation,
object state and future reference in the critic, and one object-pose tracking reward.
The existing robot rewards, termination conditions, 29 actions and 730-value robot
goal remain unchanged. Chair mass is randomized to 5–15 kg and G1–chair sliding
friction to 0.3–1.2 per episode; evaluation uses 10 kg and 1.0.

`SharedTracking` and `Sitting` share 84-value states, 82 predicted increments and
the same 228-value dynamics input. The 47 stored context values contain 45 physical
values, presence and a geometry ID; the ID only selects geometry and never enters
the neural network. A PointNet jointly trained with dynamics produces 64 shape
features, accompanied by three metric extents. Reference trajectories and rewards
stay outside the world model.

One native simulation run uses one chair geometry and matching motion clips. Different
chair runs can export real transitions and mix their replays when training the shared
model. This does not establish generalization to unseen geometry or new contact modes.
See [the interface and usage guide](docs/shared-world-model.md) for dimensions,
checkpoint transfer, replay mixing and current validation limits.

`BuiltinPositionActuatorCfg` preserves torque caps but has no equivalent to Isaac
Lab's `velocity_limit_sim`. Motor speed limits remain metadata rather than a
solver-enforced clamp, so the backends are not physically identical.

### Data and robot assets

| Data | Source | License | Destination |
|---|---|---|---|
| G1-retargeted LAFAN1 clips: root position, root quaternion xyzw and 29 joint angles at 30 Hz | HF [`lvhaidong/LAFAN1_Retargeting_Dataset`](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset), `g1/` | CC BY-NC-ND 4.0 (Ubisoft La Forge) | Raw: `data/lafan1_g1/*.csv`; standard: `data/motions/lafan1/*.npz` |
| G1 URDF and meshes used by these retargets | Same dataset, `robot_description/g1/` | BSD-3-Clause (Unitree) | `data/g1_description/` |

The downloader resolves a single immutable dataset revision before downloading all files,
records its SHA and file list in `data/source.json`, builds the G1 MJCF, and converts the
CSVs with `motion_data.converters.lafan1`. To select a known revision:

```bash
LAFAN1_REVISION=<dataset-commit-sha> tools/download_data.sh
```

`tools/urdf_to_mjcf.py` compiles the source URDF into a free-floating G1 MJCF. This same
model supplies simulation and differentiable FK. There is no USD intermediate:

```text
data/
├── source.json
├── lafan1_g1/                 # original LAFAN1 source: 40 CSVs
├── grail/sitting/             # optional GRAIL source: robot/, objects/, meta/
├── motions/
│   ├── lafan1/*.npz           # default input for training, play and evaluation
│   └── sitting/*.npz          # converted GRAIL clips, including chair meshes/poses
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

### 统一动作数据与纯运动学回放

原始数据转换集中在 `motion_data/converters/`，每个来源一个脚本。转换结果统一为
`NPZ`，训练、评估和纯运动学回放共同使用此格式。`motion_data/__init__.py` 提供
`validate_motion`、`load_motion`、`save_motion`，只依赖 NumPy；已有 NPZ 无需重新转换。
今后接入新来源，在该目录新增转换脚本即可，格式约定和下游代码保持一致。

`tools/replay_motion.py` 仅设置根节点、关节和物体姿态并执行正运动学，没有控制器、
重力积分或物理步进；不需要训练模型或 CUDA。

```bash
# 首次转换 GRAIL sitting（默认路径如下；已存在的文件会跳过）
uv run --extra motion python -m motion_data.converters.grail data/grail/sitting data/motions/sitting

# 回放目录中的第 0 段，也可以直接传某个 .npz 文件
uv run python tools/replay_motion.py data/motions/sitting --index 0
# Space 暂停/继续；左右箭头逐帧；R 重播；默认循环，--once 只播放一次
uv run python tools/replay_motion.py data/motions/sitting --index 10 --speed 0.5

# 无窗口检查，或导出一遍 MP4（视频需要系统 ffmpeg）
uv run python tools/replay_motion.py data/motions/sitting --headless
uv run python tools/replay_motion.py data/motions/sitting --video results/sitting_kinematic.mp4

# 原有 LAFAN1 CSV 也转成同一种格式；--fps 仅指定 CSV 原始帧率
uv run python -m motion_data.converters.lafan1 data/lafan1_g1 data/motions/lafan1 --fps 30
```

GRAIL 的 1748 段动作均为 125 帧、25 Hz，转换保留原帧率。椅子模型会从
`source/object_usd` 或 `COPY_MANIFEST.json` 记录的原始目录读取，也可显式指定
`--object-usd-dir /path/to/object_usd`。转换后椅子三角网格保存在 NPZ 内，回放不再依赖
外置盘、USD 或 pickle；椅子使用统一材质，不复制纹理。重新转换用 `--overwrite`。

统一格式：米、弧度、Z 向上、世界坐标，所有四元数为 **wxyz**。

| 字段 | 形状 | 含义 |
|---|---|---|
| `fps` | 标量 | 文件内各帧的实际采样率；现有转换器保留原始帧率 |
| `root_pos`, `root_quat` | `(T,3)`, `(T,4)` | 机器人根节点世界姿态 |
| `joint_pos`, `joint_names` | `(T,J)`, `(J,)` | 关节角和对应名称；训练与回放均按名称匹配 |
| `object_pos`, `object_quat` | `(T,3)`, `(T,4)` | 可选单个物体的世界姿态 |
| `object_vertices`, `object_faces` | `(V,3)`, `(F,3)` | 可选物体局部坐标顶点、三角面索引 |
| `object_scale` | `(3,)` | 只作用于物体网格的三轴缩放 |

GRAIL **机器人原四元数是 xyzw，椅子原四元数已经是 wxyz**，转换分别处理；
关节顺序和轴与当前 G1 MJCF 核对，世界位置不平移、不缩放。新来源需负责单位、坐标系、
四元数和目标机器人关节的转换；已经重定向到同一个 G1 的数据可共用现有训练流程。
回放其他机器人可用 `--model /path/to/robot.xml`（一个浮动根节点及具名标量关节）；
当前策略仍要求完整的 G1 29 个关节。详见 [转换器接入约定](motion_data/converters/README.md)。

`MotionLib` 只读取标准 NPZ，将每段 `joint_names` 重排到 G1 顺序，并读取该段自己的
`fps` 重采样到 **50 Hz**：根位置与关节角线性插值，根四元数用 SLERP，随后按
**20 ms** 间隔计算速度和 FK。不同原始帧率可放在同一目录；不会仅修改 `fps` 来加速动作。
例如 sitting 的 **125 帧 @25 Hz → 249 帧 @50 Hz**，首末采样时间跨度均为
`(125 - 1) / 25 = 4.96 s`。采样只取 20 ms 时间网格内的点，不向源动作末尾外推。
训练要求重采样后至少两帧，纯运动学回放仍可读取单帧。

原 `Tracking` 任务保持单帧策略观测 **93 维**、每个未来参考帧 goal **73 维**，
仍只读取机器人参考。新 `Sitting` 任务读取同一 NPZ 中的机器人和物体数据，
同步重采样到 50 Hz，并建立动态椅子场景；其 Actor 新增物体相对位姿 **6 维**。
选择数据目录不会自动切换任务，需要使用对应任务 ID。

可先在 CPU 上检查一段 sitting 的训练输入，不启动物理仿真：

```bash
uv run python - <<'PY'
from pathlib import Path
from supermimic.assets.g1.kinematics import G1Kinematics
from supermimic.motion import MotionLib

path = sorted(Path("data/motions/sitting").glob("*.npz"))[0]
motion = MotionLib([str(path)], G1Kinematics("cpu"), "cpu")
print("frames:", motion.clip_len.tolist(), "dt:", motion.control_dt)
print("10-frame goal:", tuple(motion.goal(motion.clip_start, 10).shape))
# frames: [249], dt: 0.02; goal: (1, 730)
PY
```

### 碰撞几何检查

G1 模型已移除左右 `rubber_hand` 的视觉和碰撞几何，当前不建模手部接触。
29 个关节（包括腕部）以及固定手部 body 的质量、惯量保持原值，数据格式和策略维度不变。
G1 提供可选的交互碰撞配置：`make_g1_spec(interaction_collisions=True)` 或
`make_g1_robot_cfg(interaction_collisions=True)`。它增加仅碰物体的整脚 mesh；原脚底
四球继续负责地面支撑。地面使用 `contype=1`，椅子使用
`contype=2`，两者 `conaffinity=0`；机器人自碰撞保持关闭。现有 tracking 任务默认
仍使用基础碰撞配置，整脚物体碰撞供交互场景显式选用。

椅子不能直接用完整视觉 mesh 做单凸包碰撞。下面的工具读取标准 NPZ，离线生成
多个凸块并按几何内容缓存；同一椅子的不同动作共用缓存，原动作文件无需改写。

```bash
# 将路径换成待检查的动作；批量处理也可以传目录
uv run --extra motion python tools/prepare_collisions.py /path/to/motion.npz
uv run python tools/check_collisions.py /path/to/motion.npz --render
```

缓存默认在 `data/collisions/chairs/`；检查结果及视觉/单凸包/分解对比图在
`results/analysis/collision/<clip>/`。检查仅按参考姿态计算碰撞，不推进动力学。
通用 `tools/replay_motion.py` 仍用于纯运动学回放。

`motion_data.collision.add_object_geoms(spec, body, clip, parts)` 把视觉几何和凸碰撞
几何加入指定 body，统一施加一次 `object_scale`；动态椅子的惯量由交互任务资产层设置。
缓存缺失或损坏会明确报错，不会退回整个椅子的凸包。分解阈值是近似算法参数，不能
视为已保证的表面误差；复杂椅子还需检查凸块数量、坐人空腔、座面支撑和参考动作穿透。
详见 [碰撞审计](docs/collision-audit.md)。

### 策略训练与评估

```bash
# The stock mjlab entry point owns GPU selection and the timestamped log directory.
uv run train Mjlab-SuperMimic-Tracking-Unitree-G1 \
  --env.commands.motion.motion-dir data/motions/lafan1 \
  --gpu-ids '[0]' --log-root logs --agent.experiment-name lafan1_mjlab

# Convenience alias; extra arguments are passed unchanged to mjlab train.
./run_train.sh lafan1_mjlab --gpu-ids '[0]'
CPUS=0-23 ./run_train.sh lafan1_mjlab --gpu-ids '[0]'

# Use the run directory printed by train.
RUN_DIR=logs/lafan1_mjlab/YOUR_RUN_DIRECTORY

# Interactively view a saved policy.
# For other datasets, prefix play with SUPERMIMIC_MOTION_DIR=/path/to/standard_npzs.
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

The two tracking tasks default to `data/motions/lafan1`; Sitting defaults to
`data/motions/sitting`. Select another converted
directory with train's `--env.commands.motion.motion-dir`, play's `SUPERMIMIC_MOTION_DIR`,
or evaluation's `--motion-dir`. Only `Mjlab-SuperMimic-Sitting-Unitree-G1` adds the dynamic
chair. Its motion loader selects clips matching one collision geometry and scale;
the full library is not treated as interchangeable chair shapes in one simulation batch.

Native mjlab stores runs under `log_root/experiment_name/<timestamp>[_run_name]`.
Checkpoints are `model_<iteration>.pt`; each contains policy, world model, critic,
optimizer state and task metadata. `params/env.yaml` and `params/agent.yaml` record
resolved configuration. Resuming uses standard `--agent.resume True`, `--agent.load-run`
and `--agent.load-checkpoint`; replay is refilled before any resumed model updates.
Shared-dynamics runs also save independent physics weights under `world_models/`.
With `--agent.export-world-model-replay True`, transitions and their geometry bank
are exported under `replays/`. These auxiliary files stay outside the resume search.

Use `uv run train Mjlab-SuperMimic-Tracking-Unitree-G1 --help` for all native options.
The first iterations include compilation; `--agent.no-compile True` is useful for short
checks. Default batches remain 4096 environments, 1000 replay steps per world,
8-step model fitting, 32-step imagination, 10 model batches of 2048 windows,
4096 imagined rollouts and 500 world-model-only iterations. `--agent.amp none`
disables BF16; TF32 remains enabled for FP32 matmuls. Training is single-GPU.

Quantitative evaluation starts each clip at its first frame, disables randomization
and observation/action noise, and scores until the first failure or clip completion.
It evaluates the clips selected by the task: every clip for tracking, or the matching
single-geometry subset for Sitting. No train/test split is inferred.
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

## Baseline algorithm and retained choices

* G1 29-DoF with BeyondMimic actuator parameters: 10 Hz natural frequency, damping ratio 2,
  motor armatures and action scale `0.25 * torque_limit / stiffness`.
* Motion library: standard NPZ input, named G1 joint mapping, each clip's own frame rate
  resampled to 50 Hz, finite-difference velocities, FK of 14 tracked bodies, and 1 s sampling bins.
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
baseline task schema version 2; version 1 and checkpoints without task metadata are rejected.
There is no automatic weight slicing or legacy conversion. Train a new policy for the
current task contract; the historical scores do not validate this change. The new
Sitting Actor also has 99 values, but its six additional values describe the chair,
so it is not compatible with the historical 99-value observation. Interaction tasks
use full-checkpoint schema version 3 and a separate shared-dynamics schema.

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
