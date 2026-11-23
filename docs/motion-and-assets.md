# 动作数据与碰撞资产

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
当前策略仍要求完整的 G1 29 个关节。详见 [转换器接入约定](../motion_data/converters/README.md)。

`MotionLib` 只读取标准 NPZ，将每段 `joint_names` 重排到 G1 顺序，并读取该段自己的
`fps` 重采样到 **50 Hz**：根位置与关节角线性插值，根四元数用 SLERP，随后按
**20 ms** 间隔计算速度和 FK。不同原始帧率可放在同一目录；不会仅修改 `fps` 来加速动作。
例如 sitting 的 **125 帧 @25 Hz → 249 帧 @50 Hz**，首末采样时间跨度均为
`(125 - 1) / 25 = 4.96 s`。采样只取 20 ms 时间网格内的点，不向源动作末尾外推。
训练要求重采样后至少两帧，纯运动学回放仍可读取单帧。

原 `Tracking` 任务保持单帧策略观测 **93 维**、每个未来参考帧 goal **73 维**，
仍只读取机器人参考。新 `Sitting` 任务读取同一 NPZ 中的机器人和物体数据，
同步重采样到 50 Hz，并建立动态椅子场景；其 Actor 新增物体相对位姿 **9 维**，并使用共享点云编码。
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

### PA-HOI 抱瑜伽球与大气球

`data/pahoi/hold/` 保存从 Lexar `data_reviewed_torso_v1` 迁入的完整物体子集：
yoga ball 15 个尺寸版本、big balloon 18 个尺寸版本，共 5535 帧；每个源片段包含
原尺寸、0.85、0.74 三档。原始轨迹、接触任务、几何触觉、物体/机器人资产和来源记录
均保留，本地读取不依赖外置盘。`episodes.csv` 是索引，`manifest.json` 记录文件哈希。

标准动作在 `data/motions/hold/<episode_id>.npz`，包含 G1、物体轨迹、内嵌网格和
三列 `contact_*` 字段。转换保留 30 Hz 原帧率、源时间戳和全部帧，不重复贴地或缩放。
此前演示使用 yoga ball `1_273__s0p85` 和 balloon `1_241__s0p85`，对应视频在
`data/pahoi/hold/previews/`。

```bash
uv run python -m motion_data.converters.pahoi
uv run python tools/replay_motion.py data/motions/hold/1_273__s0p85.npz
uv run python tools/replay_motion.py data/motions/hold/1_241__s0p85.npz
uv run python data/pahoi/hold/reader.py data/pahoi/hold --verify
```

原始 episode 中的接触信息有两类：

| 文件 | 逐帧字段 | 含义 |
|---|---|---|
| `contact_task.npz` | `active`、`weight`：`(T,3)`；`target_object`、`normal_object`：`(T,3,3)` | 左手、右手、前胸腹的优化任务，顺序由 `patch_ids` 标识；不是接触成功标签 |
| `tactile.npz` | `geometric_contact`、`surface_distance_m`：`(T,464)` | 原内臂布局的逐点几何接触及表面距离 |
| `tactile.npz` | `channel_geometric_contact`、`channel_surface_distance_m`：`(T,112)` | 逐通道任一成员接触、成员最小距离 |

标准 NPZ 保留 `contact_task.npz` 的全部字段并加 `contact_` 前缀；完整触觉时序保留在
原始 episode，标准文件的 `source_episode` 给出相对仓库根目录的位置。inactive 目标
可能为 NaN，读取时应用 `contact_active` 掩码。几何接触使用 4 mm 阈值并记录内部点；
内部穿插不能等同于有效接触。这些字段均不是力、压力或硬件 counts。

原始 `gmr_reference.npz` 是优化前的动作；其中的贴地量只用于显示原始 GMR，不能叠加到
优化后的 `motion.npz`。人工 keep、优化后人工验收标志和数值 QC 分别保存在 metadata，
迁移未改变这些状态。reference 的动力学可行性和训练后的抱持表现需独立评估。
数据沿用仓库 `/data` 忽略规则，不进入 Git；字段详情见 `data/pahoi/hold/FORMAT.md`。

### YogaBall 任务

任务 ID 为 `Mjlab-SuperMimic-YogaBall-Unitree-G1`，默认单条 reference 为
`1_273__s0p85.npz`。注册不加载数据，创建环境时再读取轨迹和配套资产；
运行依赖仓库内 `data/motions/hold/` 与 `data/pahoi/hold/`，不依赖 Lexar 原路径。
可将 `--env.commands.motion.motion-dir` 指向另一条瑜伽球 NPZ；首版每个运行只用一种尺寸、
一段 reference，目录或 Balloon 文件会明确报错。

```bash
# JSON 加载和最终参数检查，不创建仿真或训练
./run_yogaball.sh --check
# 使用 configs/yogaball.json；也可用 --config 指定另一份 JSON
./run_yogaball.sh
# 临时覆盖 JSON；GPU、环境变量和 CLI 约定与 run_open_door.sh 相同
CUDA_VISIBLE_DEVICES=0 ./run_yogaball.sh --env.scene.num-envs 256 --agent.pi-lr 0.0001
# 原生环境查看（零动作，不代表训练后的策略）
uv run play Mjlab-SuperMimic-YogaBall-Unitree-G1 --agent zero --num-envs 1 --viewer native
# 纯运动学 reference 回放仍使用原有工具
uv run python tools/replay_motion.py data/motions/hold/1_273__s0p85.npz
```

`configs/yogaball.json` 沿用 OpenDoor 的 `training` / `task` 结构，由
`python -m supermimic.scripts.train_yogaball` 读取；`run_yogaball.sh` 复用已有启动脚本。
默认每卡128环境、WM/Actor batch各128、10000轮，先进行500轮仅 WM 更新。
`training.lr` 为 WM/Critic 学习率，`training.pi_lr=null` 时 Actor 也使用它。
`task.reward_cfg` 为机器人奖励，`task.object_reward_cfg` 为球位姿奖励；
`task.reset_cfg`、`observation_cfg`、`physical_cfg` 分别控制机器人初始化扰动、观测噪声和物性随机化。
`task.object_physical_cfg.mass` / `robot_friction` 是训练时每次 reset 的采样范围，
两端相等表示固定值；`evaluation_mass` / `evaluation_robot_friction` 是名义建模与 play 值。
例如要统一使用2kg球，应将 `mass` 设为 `[2.0, 2.0]`，并将 `evaluation_mass` 设为 `2.0`。
原生 `train` / `play` 仍使用注册默认值，不自动读取此 JSON。

场景为 `robot`＋`ball`，球有六自由度、重力及机器人／地面碰撞，reset 同步写入机器人和球的
reference 位姿及速度，之后由物理推进。球使用迁入的原网格与 3 个原凸部件，缩放已烘焙；
不伪装成椅子的 CoACD 缓存。球质量默认固定 1 kg，机器人／球和球／地面摩擦均为 1，
惯量按原网格均匀密度近似；这些是可配置的未标定仿真值，非源数据的实测物性。
JSON 中的物性参数在 `task.object_physical_cfg`，物体跟踪权重与误差尺度在
`task.object_reward_cfg`；CLI 对应 `--env.commands.motion.object-physical-cfg.*`
和 `--env.commands.motion.object-reward-cfg.*`。

模型和优化接口复用 Sitting：state84、context46、每帧 observation102、action29；
机器人和球在同一 50 Hz 时钟上重采样（本段 186 帧 @30 Hz → 309 帧 @50 Hz）。
Actor 使用机器人本体、当前球相对位姿、机器人未来 goal 和物体几何；Critic 额外使用球的
未来 reference。奖励为原机器人跟踪项＋球位置／朝向跟踪，真实环境和想象 rollout 共用
同一函数。终止条件沿用机器人跟踪失败、reference 完成与超时。

任务名 `g1_yogaball`、实验目录 `logs/rsl_rl/supermimic_yogaball/` 独立，拒绝 Sitting
checkpoint。现有全库评估器可以记录球的实际与参考轨迹，不能使用 Sitting 的坐下成功判据。
本阶段不消费接触标签、不增加接触奖励或触觉策略输入，也不代表已经学会稳定抱持。
共享 G1 仍未建模 rubber hand 掌部碰撞；原双手任务锚点的物理承托需后续单独适配。

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
详见 [碰撞审计](collision-audit.md)。
