# 椅子级 80/20 泛化实验

本实验训练同一个 Actor、WM、Critic，然后冻结策略，在未参与训练的椅子几何上评估。
共享几何编码器只随 WM 的动力学损失更新。它不是 151 个单椅子策略，也不是按动作文件
随机划分的实验。

## 固定划分

```bash
uv run python tools/split_chairs.py data/motions/sitting \
  --output data/splits/sitting_80_20_seed0.json --train-fraction 0.8 --seed 0
```

本地已生成上述文件，重复运行不会覆盖。当前数据共 189 种网格、1748 段动作：

| 分区 | 椅子几何 | 动作片段 |
|---|---:|---:|
| train | 151 | 1391 |
| test | 38 | 357 |

训练椅子数量取 `floor(189 * 0.8)`。同一未缩放网格的全部动作和尺度变体在同一侧；
物理仿真按几何和尺度分别建场景。分组依据网格数组内容，不依赖椅子名称。
这避免了同一资产的重复动作泄漏，但不声称识别所有语义相似椅型或重拓扑近似副本。
划分保存种子、每个动作 SHA-256、几何身份和整体 fingerprint；加载时验证内容。
manifest 的数据路径可通过 `--data-root` 覆盖，方便搬到服务器后保持相同划分。

测试含义是**未见物体几何上的、参考动作条件化的 zero-shot**：测试时仍提供原设计要求的
机器人参考动作和当前物体几何。它不是没有参考动作或没有几何信息的盲测。不同椅子上的
动作可能来自相同动作模板，这个划分没有同时实现“未见动作模板”的实验。

## 准备碰撞资产

```bash
uv run python -m supermimic.scripts.chair_experiment check \
  --split data/splits/sitting_80_20_seed0.json

uv run --extra motion python tools/prepare_collisions.py \
  --split-manifest data/splits/sitting_80_20_seed0.json --partition all --workers 4
```

本次划分时，本地只有 24 种已生成的碰撞缓存；train 缺 133 种，test 缺 32 种。
完整 151/38 实验尚未启动。准备工具复用已有缓存，按唯一网格生成缺失缓存，失败会报错。
训练或评估启动前检查对应分区的全部缓存，**不自动跳过缺资产或分解失败的椅子**。
复杂椅子的凸块数量、惯量有效性及接触质量仍可能影响仿真速度或使资产校验失败；需要
修复资产或明确建立另一个实验划分，不能悄悄缩小测试集。
测试几何的离线碰撞预处理不训练任何网络。

2026-10-27 检查时，CUHK 上该划分的 train/test 碰撞缓存已全部存在；本地此前的缺失
统计不能代表服务器现状。实体任务在场景组装时测量初始接触/约束数量，按需扩大
`nconmax` / `njmax`，避免复杂椅子超过旧的每物体512接触、2000约束下限。当前保留25%
余量，并分别按128接触/512约束对齐，保留手动设置的更大容量。早期的“乘2再取2的幂”
会过度放大CUDA graph工作区，512环境下已出现OOM，参见[复测记录](cuda-oom-visit27.md)。
检查增加一次CPU场景编译。这是启动容量估计，不能保证所有未来姿态都不溢出；Replay
会在记录转移、重置环境之前检查Warp累积溢出标记，发现截断物理数据就明确报错。

## 多椅子训练

```bash
uv run python -m supermimic.scripts.chair_experiment train \
  --split data/splits/sitting_80_20_seed0.json \
  --num-envs 128 --max-objects 8 --iterations-per-chair 50 \
  --iterations 50000 --seed 0 --device cuda:0 \
  --run-dir logs/sitting_80_20
```

这是多场景实验入口。原有 `uv run train ... EntitySitting` 入口继续用于单种椅子场景。
实验入口复用相同的 native mjlab 环境、SuperMimic 更新逻辑、奖励和域随机化；多椅子
调度器负责管理物理场景和统一数据索引；也支持下述同步多卡训练。

- 一次只驻留一种椅子的原生物理环境批次，避免同时装载 151 套 MuJoCo 场景。
- 每 50 次学习迭代切换场景；每轮确定性打乱训练场景顺序，每种场景访问一次。
  当前数据每轮是 `151 * 50 = 7550` 次迭代。一次访问包含多个并行环境和匹配的动作片段。
- Actor、WM、Critic、共享几何编码器、优化器和 Replay 持续保留。不是每换椅子从头训练。
- Replay 使用整个 train 分区的稳定参考帧索引和几何 ID，想象奖励会匹配对应椅子的轨迹。
- 切换场景时将旧批次的最后一行标为边界，保留该转移，但不允许时间窗口或历史跨场景。
- 当前 Replay 是有限的近期缓冲区，并非保留全部训练历史的永久档案。跨椅子知识通过
  重复混合训练积累；这个实现不保证消除遗忘，需要比较不同轮换周期与 Replay 容量。
- 几何采样和所有训练数据读取只使用 train 成员；默认拒绝无法核实来源的预训练模型和
  外部 Replay 参数，避免测试椅子通过预训练数据泄漏。

默认超参数沿用 entity 任务。其它训练设置可通过 JSON 覆盖 `SuperMimicRunnerCfg` 字段：

```json
{
  "lr": 0.0002,
  "wm_batch": 128,
  "pi_batch": 128,
  "buffer_steps": 256,
  "horizon": 32,
  "wm_window": 8,
  "wm_only_iters": 500,
  "no_compile": true
}
```

把文件传给 `--agent-json /path/to/agent.json`。`no_compile: true` 适合小规模检查或避免
首次编译等待；长训练可按实际性能决定是否开启编译。实验配置、可迁移划分副本、访问过
的训练椅子身份和调度进度都会保存。复用同一划分、容量、环境数和轮换参数续训：

```bash
uv run python -m supermimic.scripts.chair_experiment train \
  --split data/splits/sitting_80_20_seed0.json \
  --num-envs 128 --max-objects 8 --iterations-per-chair 50 --seed 0 \
  --run-dir logs/sitting_80_20 --resume logs/sitting_80_20/model_50000.pt \
  --iterations 10000
```

完整续训恢复三类网络和优化器；在线 Replay 重新填充。场景序列从保存的迭代位置继续，
但不保证恢复仿真 RNG 和 episode 的逐位一致运行。

## 多卡训练

在具备完整训练碰撞缓存的机器上运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run python -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  -m supermimic.scripts.chair_experiment train \
  --split data/splits/sitting_80_20_seed0.json \
  --num-envs 128 --max-objects 8 --iterations-per-chair 50 \
  --iterations 50000 --seed 0 --run-dir logs/sitting_80_20_2gpu
```

四卡将 `CUDA_VISIBLE_DEVICES` 改为 `0,1,2,3`，`--nproc_per_node` 改为 `4`。
`LOCAL_RANK` 自动选择各进程的 CUDA 设备；无需为每个进程手工指定 `--device`。
原生单场景任务直接用 `uv run train ... --gpu-ids '[0,1]'`，不要再套 torchrun。

- 每卡独立驻留一个物理场景批次和本地 Replay，保存相同的完整网络与优化器。
  每次更新先平均各卡梯度再裁剪；WM、Actor、Critic 参数同步，几何编码器仍只归 WM 更新。
- 各卡从同一打乱的训练场景序列按 `visit * world_size + rank` 取场景，统一每 50 次迭代
  换场景。151 个场景在两卡下前 76 次访问（3800 次迭代）可全部覆盖；跨序列边界的一批
  可能同时包含下一轮场景，不保证每次各卡都互不重复。测试椅子不进入训练。
- `--num-envs`、`wm_batch`、`pi_batch`、`buffer_steps` 都是**每卡**配置。例如两卡各
  128 个环境，总计 256 个环境；两卡 WM batch 各 128，每次更新合计 256 个窗口。
  学习率保持配置值。多卡不合并显存，每张卡仍需放下完整模型和自己的 Replay。
- 归一化使用所有卡的有效样本统计，排除被 mask 的物体；空物体批次也参与通信。
  `sim_steps` 统计所有卡的累计采样步数，日志聚合各卡指标，迭代编号不乘卡数。
- rank 0 保存模型、实验元数据和日志；checkpoint 的训练覆盖率是所有卡访问集合的并集。
  如开启 Replay 导出，各卡写入 `replays/model_N.rank0000.pt`、`.rank0001.pt` 等独立文件。
- 完整多椅子续训要求卡数、划分、容量、每卡环境数和轮换设置一致，在上述命令追加
  `--resume logs/sitting_80_20_2gpu/model_50000.pt`，并设置需要**追加**的 `--iterations`。
  各卡从自己的场景调度位置继续，重新填充本地 Replay。
- zero-shot 评估仍使用下方单进程命令，可以在一张卡上读取多卡 checkpoint，全部权重和
  归一化统计冻结。

## 未见椅子评估

```bash
uv run python -m supermimic.scripts.chair_experiment evaluate \
  --split data/splits/sitting_80_20_seed0.json \
  --checkpoint-file logs/sitting_80_20/model_50000.pt \
  --output results/analysis/sitting_zero_shot.json --device cuda:0
```

- 逐个 test 椅子构建真实物理环境，评估该几何对应的所有动作，从参考第一帧开始。
- 加载训练后的 Actor（包含共享几何编码器和归一化统计），保持 `eval` 和冻结状态。
  不创建 WM、Critic、Replay 或优化器，不调用归一化更新。
- 新椅子使用自己的几何 bank 和局部 ID；训练 bank 的 ID 数值没有语义，不能直接套到新场景。
- 每种测试几何评估前后对策略参数及 buffer 做摘要校验，确保没有测试时自适应。
- 固定椅子质量 10 kg、G1–椅子摩擦系数 1.0。按当前机器人 tracking 终止规则结束。
- 默认要求 checkpoint 已实际访问全部 train 椅子；`--allow-incomplete-training` 仅用于
  明确标记的开发检查，报告会记录训练覆盖率。

输出包含每动作归一化回报、每椅子均值、按椅子等权的 macro 回报和轨迹完成率，以及
按动作加权的回报、checkpoint SHA-256 和 split fingerprint。轨迹完成率是“未触发原
机器人终止并走到参考结束”，**不是新增的坐稳成功判据**；目前没有额外稳定坐姿检测器。
不要用 test 反复选择超参数或 checkpoint 后仍将其称作最终独立测试；需调参时应在 train
内部另划 validation。

## 已验证与未完成

新增测试覆盖椅子级划分、尺度变体不跨分区、文件变化拒绝、缺缓存不改变划分、均衡轮换、
全局参考帧/几何 ID、场景边界、原生 GPU 跨场景学习和续训、冻结未见几何评估。
本修复对应的原有测试与新增回归共 **179 项通过**。另在真实现有椅子上用完整默认网络完成两种训练几何的
两次迭代，随后在第三种未见几何上冻结评估；流程、参数不变性和报告生成通过，未进行长训练。
该诊断策略的测试轨迹未完成，归一化回报约 0.143，不能当作泛化成功结果。产物保存在
`logs/chair_generalization_smoke/`，与正式 151/38 划分分开。
新增双进程 CPU/Gloo 测试覆盖真实网络的 BPTT 更新、全局归一化、梯度平均、参数一致性、
检查点与每卡 Replay 保存、续训以及跨卡训练椅子覆盖汇总。其转移数据是解析测试夹具，
不代表双 GPU 原生仿真验证。另已在CUHK两张A100上完成原生NCCL训练、BF16、编译更新、
逐卡参数/统计/优化器一致性和续训检查，详见[检查报告](distributed-validation-cuhk.md)。
完整151把训练椅子、38把测试椅子的长期训练和泛化效果仍需实际测量。
