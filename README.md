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
| `supermimic/tasks/entities/` | Multi-object scenes, raw entity observations and object reference trajectories for the critic |
| `supermimic/tasks/` | Native mjlab task registration, dimensions and checkpoint contract |
| `supermimic/mdp/` | Commands, actions, events, observations, rewards, termination and state/dynamics functions |
| `motion_data/__init__.py` | NumPy-only shared NPZ schema, validation, loading and saving |
| `motion_data/converters/` | One raw-data converter per source: `grail.py`, `lafan1.py` |
| `supermimic/motion.py` | Standard NPZ loading, named G1 joint mapping and 50 Hz reference resampling |
| `supermimic/rl/` | Runner, replay adapter, world model, imagined BPTT updates and critic |
| `supermimic/rl/shared_world_model.py`, `shared_replay.py` | Shared robot/object physics, geometry encoder and portable replay across tasks |
| `supermimic/rl/entity_*.py` | Shared entity interaction model, Actor/Critic scene encoders, entity state and portable replay |
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

A CUDA GPU is required for batched mjlab simulation. Original tasks default to 4096 worlds;
the new entity tasks default to 128 worlds and smaller batches. Reduce `--env.scene.num-envs`,
`--agent.buffer-steps`, `--agent.wm-batch` and `--agent.pi-batch`.

The package registers five tasks with mjlab through
`[project.entry-points."mjlab.tasks"]`:

| Task suffix after `Mjlab-SuperMimic-` | Actor observation | Replay state | WM input → increment | Critic input |
|---|---:|---:|---:|---:|
| `Tracking-Unitree-G1` — original baseline | 93 | 71 | 133 → 70 | 914 |
| `SharedTracking-Unitree-G1` — no object, shared dynamics | 93 | 84 | 228 → 82 | 1075 |
| `Sitting-Unitree-G1` — one dynamic chair | 99 | 84 | 228 → 82 | 1075 |
| `EntityTracking-Unitree-G1` — entity model, no default objects | 197 raw | 175 | entity tokens → 166 | 2242 raw → 1042 encoded |
| `EntitySitting-Unitree-G1` — entity model, target chair and optional extra objects | 197 raw | 175 | entity tokens → 166 | 2242 raw → 1042 encoded |

Dimensions assume ten future reference frames; entity rows also assume capacity 8.
SharedTracking and Sitting
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

### Multi-entity shared world model

The `codex/shared-entity-world-model` implementation starts from
`f714fc134d0f8e79ea908da5dc809a194932aaea`. The three existing tasks retain their interfaces.
`EntityTracking` and `EntitySitting` add a shared graph of robot, object and ground tokens:
256-wide tokens, four interaction layers, and shared object encoders/output heads.
The same weights support varying numbers of objects. Each task keeps its own Actor,
Critic, references and reward. Only object pose tracking is added to the original reward.

At capacity 8, replay stores state **175** and context **441**. Actor observations contain
**197 raw values** per frame; its scene encoder produces a **221-value encoded frame**.
Critic receives **792 current values + 730 robot references + 720 object references**,
then encodes the scene before its 1042-input value MLP. Actor, Critic and world model
share one geometry encoder, updated only by the world-model dynamics loss. Actor and
Critic train their own downstream scene encoders; geometry codes are recomputed from
the current shared weights. Default training uses 128 worlds, 256 replay steps and
WM/Actor batches of 128.

```bash
uv run train Mjlab-SuperMimic-EntitySitting-Unitree-G1 \
  --env.commands.motion.motion-dir data/motions/sitting/sitting__acadian_rustic_wooden_dining_chair_distressed_white__000.npz \
  --agent.experiment-name entity_sitting --gpu-ids '[0]' --log-root logs
```

Prepare the chair collision cache first. Add physical boxes, fixed obstacles or other
mesh assets through `--env.commands.motion.extra-objects-file scene_objects.json`.
Current NPZs provide one target object trajectory; extra objects participate in physics
but have no target trajectory. The new model does not implicitly load f714's single-object
weights. See [the entity interface and usage guide](docs/entity-world-model.md) for exact
dimensions, JSON configuration, playback, checkpoint transfer and replay compatibility.
Software/learning smoke checks do not establish trained sitting performance or generalization.

### 80/20 chair generalization experiment

For one policy trained across many chair geometries, use the dedicated experiment
entry point rather than the single-scene task launcher. The fixed seed-0 split of
the current assets contains **151 training chairs / 1391 clips** and **38 held-out
chairs / 357 clips**. All clips and scale variants of an identical source mesh stay
in one partition. The learner retains its models, optimizers and Replay while
rotating physical scenes; held-out evaluation freezes policy/encoder/normalizers.

```bash
uv run python tools/split_chairs.py  # Run once; never overwrites a saved split.
uv run --extra motion python tools/prepare_collisions.py \
  --split-manifest data/splits/sitting_80_20_seed0.json --partition all --workers 4
uv run python -m supermimic.scripts.chair_experiment train \
  --split data/splits/sitting_80_20_seed0.json --run-dir logs/sitting_80_20 \
  --num-envs 128 --iterations-per-chair 50 --iterations 50000
uv run python -m supermimic.scripts.chair_experiment evaluate \
  --split data/splits/sitting_80_20_seed0.json \
  --checkpoint-file logs/sitting_80_20/model_50000.pt \
  --output results/analysis/sitting_zero_shot.json
```

The local split has already been generated. Collision preparation is incomplete
(133 training and 32 held-out shapes missing at split creation); the experiment
fails explicitly on missing assets instead of changing the split. No full-scale
training result is claimed. See [experiment protocol and configuration](docs/chair-generalization.md).

### Legacy single-object shared world model and sitting

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

### Sitting 背部接触筛选

浏览器筛选器可旋转查看机器人与椅子，慢放、逐帧查看、跳到背部最贴近椅子的帧，
并将动作标记为保留、排除或待定。只使用 CPU 正运动学，不需要训练权重或 GPU。

```bash
uv run python tools/select_motions.py
# 浏览器打开 http://127.0.0.1:8086
# 自定义数据、记录文件和端口：
uv run python tools/select_motions.py data/motions/sitting \
  --selection data/selections/sitting_back.json --port 8086
```

先点击“扫描全部并排序”，再按“背部贴近占比优先”或“持续贴近优先”浏览。
扫描在后台运行，期间可以播放和标记；可停止、继续，已分析结果会缓存。
可按椅子文件名搜索、筛选标记状态、设置最少贴近占比；大于零时只显示已经分析的动作。
“距离阈值”默认 3 cm，可根据视觉检查调整；“最贴近帧”暂停并定位候选接触瞬间。
青色点仅覆盖 `torso_link` 的后侧表面，排除了臀部、手臂和手。
距离来自背侧与椅子三角表面的确定性采样，可能高估间隙；它是筛选线索，
不能判断穿透、接触力或是否真正形成背部支撑。应从侧面及斜后方确认。

每次标记、备注和接触标注自动原子保存到 `data/selections/sitting_back.json`，
重启使用同一记录文件即可继续。浏览器标签页共享同一筛选会话。
“标记后自动下一段”默认开启，关闭后可以反复检查当前动作。
“导出保留子集”会在 `data/motions/sitting_back_selected/selection-0001/`
这样的新目录中生成保留的 NPZ、`manifest.json` 和 `kept.txt`。
默认软链接节省空间，复制模式可生成独立副本；源动作不会移动或删除。
再次导出使用新的编号目录，避免混入之前保留、后来排除的动作。
导出目录可直接作为训练的 `motion-dir`；椅子分组和 train/test 划分仍应在选好子集后单独生成。
其他数据集请同时指定独立的 `--selection`，记录文件会校验数据目录以避免混用标签。

初筛完成后，用椅子分组回放器从每种椅子的已保留动作中确定一个 reference：

```bash
uv run python -m tools.review_chairs
# 浏览器打开 http://127.0.0.1:8088
```

界面默认只显示尚未确定 reference 的多候选椅子，候选多的优先。
同一椅子的已保留动作默认依次循环回放；关闭“同椅子的候选依次回放”可循环当前动作。
暂停并手动切换候选时默认保持时间位置与相机视角，便于对比同一阶段。
点击“选当前 motion 为这把椅子的 reference”后自动保存并转到下一待选椅子，
也可关闭自动转到下一椅子、继续对比并替换选择。
单候选椅子直接采用唯一候选；“椅子范围”可切换查看全部、已确定或单候选椅子。
reference 单独保存到 `data/selections/sitting_back_references.json`，初筛的保留记录不变。
初筛列表有更新时点击“重新读取最新保留列表”。全部椅子确定后可导出每椅一个 reference，
生成 `data/motions/sitting_back_references/references-0001/` 等独立目录。
导出会重新核对最新初筛记录，禁止尚未选择的多候选组或已排除的旧候选混入结果。

### 使用最终 reference 训练

`run_train_references.sh` 将最终 reference 接入已有的多椅子训练器。
它读取 `sitting_back_references.json` 中的人工选择及初筛记录中的唯一候选，
为每把椅子固定一段 reference，并沿用原始划分的训练／测试归属。
当前最终选择为 **105 把椅子 / 105 段 reference：83 训练、22 测试**。
生成的清单为 `data/splits/sitting_back_references_seed0.json`，
运行时每个场景使用清单中的单个 `motion_files`，不会再加载同椅子的其他候选。

```bash
# 生成或复用固定清单，并检查碰撞缓存；不启动训练。
./run_train_references.sh --check

# 训练前补齐所选椅子的碰撞资产；已有缓存会复用。
uv run --extra motion python tools/prepare_collisions.py \
  --split-manifest data/splits/sitting_back_references_seed0.json --partition all --workers 4

# 使用已有 Actor / WM / Critic、多椅子场景轮换及训练配置。
./run_train_references.sh --num-envs 128 --iterations-per-chair 50 \
  --iterations 50000 --device cuda:0

# 用同一清单评估这次保留的 22 把测试椅子。
uv run python -m supermimic.scripts.chair_experiment evaluate \
  --split data/splits/sitting_back_references_seed0.json \
  --checkpoint-file logs/sitting_back_references/model_50000.pt \
  --output results/analysis/sitting_back_reference_zero_shot.json
```

启动脚本可透传原实验入口的 `--agent-json`、`--resume`、`--run-dir` 等参数。
新清单包含 reference 文件哈希、选择记录来源和原划分 fingerprint；
修改选择后应通过 `tools/prepare_reference_training.py --output <新清单>` 建立新实验，
避免将不同 reference 当作相同实验恢复。`--resume` 应使用这份清单产生的 checkpoint。
在其他机器上使用启动脚本时，保留两个选择 JSON、原始 split 和所选 NPZ 的相对目录布局；
脚本会覆盖 JSON 中旧机器的绝对数据路径，无需改写原始选择记录。
若只有导出目录，也可以向准备工具传 `--references <导出目录>/manifest.json`。

### 本地策略部署：统一 reference 与初始化入口

`tools/deploy_policy.py` 在真实MuJoCo物理中执行冻结Actor。默认使用本地下载的
`logs/cuhk_reference_deployment/model_2000.pt` 及旁边的 `split.json`，一个环境、50 Hz。
同一文件处理训练reference、扰动初始化、测试reference和以后人工挑选的NPZ/清单。

```bash
# 查看训练reference编号；选中的一段会单独建物理场景，不混入同椅子的其它动作。
uv run python tools/deploy_policy.py --list

# 参考第一帧的准确位姿和速度；重复播放，关闭窗口结束。
uv run python tools/deploy_policy.py --index 0 --init clean --episodes 0

# 同一reference，仅给机器人初始状态加扰动，运行5个回合。
uv run python tools/deploy_policy.py --index 0 --init perturbed --noise-scale 1 --episodes 5

# 以后选定的reference，不需要修改脚本。
uv run python tools/deploy_policy.py --reference /path/to/selected.npz --init clean

# 无窗口验证并保存每回合初态、长度、回报和结束原因。
uv run python tools/deploy_policy.py --viewer none --init perturbed --episodes 10 \
  --output results/analysis/deployment_perturbed.json
```

浏览器预览用 `--viewer viser --port 8087`，访问 `http://127.0.0.1:8087`；
`--partition test` 切换到测试清单，`--references ...json --selection ...json` 接收人工选择记录。
替换模型用 `--checkpoint /path/to/model_N.pt --split /path/to/its/split.json`。
初始化语义、扰动范围和批量验证见 [policy-deployment.md](docs/policy-deployment.md)。

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

All three tracking tasks default to `data/motions/lafan1`; Sitting and EntitySitting default to
`data/motions/sitting`. Select another converted
directory with train's `--env.commands.motion.motion-dir`, play's `SUPERMIMIC_MOTION_DIR`,
or evaluation's `--motion-dir`. Both sitting task variants add a dynamic target
chair. Their motion loaders select clips matching one collision geometry and scale;
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
checks. Original-task default batches remain 4096 environments, 1000 replay steps per world,
8-step model fitting, 32-step imagination, 10 model batches of 2048 windows,
4096 imagined rollouts and 500 world-model-only iterations. `--agent.amp none`
disables BF16; TF32 remains enabled for FP32 matmuls.

Synchronous multi-GPU training uses the stock mjlab launcher:

```bash
uv run train Mjlab-SuperMimic-EntitySitting-Unitree-G1 \
  --gpu-ids '[0,1]' --env.scene.num-envs 128 \
  --agent.experiment-name entity_sitting_2gpu --log-root logs
```

Each GPU owns a simulator batch and Replay; all ranks average Actor, WM and Critic
gradients before clipping. Shared geometry still updates only through the WM optimizer.
Normalizer moments use global valid samples, excluding masked objects. Environment
counts, `wm_batch`, `pi_batch` and Replay capacity are **per GPU**: two cards with
128 environments each give 256 environments overall. Learning rate is not scaled
automatically. Model and optimizer memory is replicated, not pooled across cards.
Only rank 0 writes metrics and model checkpoints; `sim_steps` counts all ranks.
Optional Replay exports are named `replays/model_N.rank0000.pt`, `.rank0001.pt`, etc.;
list the desired archives in `agent.world_model_replays` when reusing them.
For distributed training across different chair scenes, use the explicit `torchrun`
command in [chair-generalization.md](docs/chair-generalization.md#多卡训练).
Evaluation remains single-process and can load a distributed training checkpoint.

Two-process CPU/Gloo tests cover synchronization, BPTT, checkpoints and resume.
Native NCCL training, BF16, compiled updates and checkpoint resume have also been
checked on CUHK's two A100 GPUs; see [the validation report](docs/distributed-validation-cuhk.md).
These short correctness checks do not establish long-run convergence or speedup.

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
