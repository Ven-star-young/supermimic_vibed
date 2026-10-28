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
