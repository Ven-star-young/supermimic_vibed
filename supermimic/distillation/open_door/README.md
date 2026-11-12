# OpenDoor 本体＋触觉蒸馏

学生使用 10 帧本体观测、10 帧 112 通道触觉和原 motion goal，输出 29 维动作。
教师沿用已有 OpenDoor checkpoint 的完整状态观测，参数和归一化统计冻结。
学生不读取当前门的 9 维真值、物性、接触力或 Critic 输入；goal 仍是预先给定的参考轨迹。

## 运行

在仓库根目录编辑 [配置](../../../configs/open_door_dagger_observable.json)，随后运行：

```bash
.venv/bin/python -m supermimic.scripts.distill_open_door --check
.venv/bin/python -m supermimic.scripts.distill_open_door \
  --config configs/open_door_dagger_observable.json
```

默认读取背部开门教师 `logs/wm_transfer/back_model_10000.pt`，并行环境数由 JSON 配置决定，
输出至 `logs/open_door_dagger_tactile/`。训练使用原任务的随机初始化和物性分布，
开始前从教师 checkpoint 恢复并核对任务配置。侧面任务可替换 `expert_checkpoint` 和
`motion_dir` 为匹配的侧面模型及 NPZ；入口按 checkpoint 的任务类型选择环境。
输出目录必须为空，避免覆盖已有实验。

也可使用由 `configs/open_door_tactile.json` 训练的背部触觉expert：将
`expert_checkpoint` 指向该模型，并设 `expert_observation_dim=326`。入口恢复教师的触觉
布局和counts参数；教师历史为原102维状态＋112维counts＋112维validity。
学生仍只提取前93维本体，`use_tactile` 独立决定学生是否使用触觉，教师保持冻结。

训练顺序是教师执行采集 → BC 初始化 → 逐轮混合执行／DAgger 更新。
`bc_rollout_steps`、`bc_train_epochs`、`bc_batches` 控制 BC；
`rounds`、`rollout_steps`、`train_epochs`、`updates_per_round` 控制 DAgger。
每轮 beta 为 `beta_start + min(round_index / beta_decay_rounds, 1) * (beta_end - beta_start)`，
`round_index` 从 0 开始；beta 表示每个环境选择教师动作的概率，不混合关节动作数值。

无触觉对照使用 `configs/open_door_dagger_proprio.json`：仅将 `use_tactile` 设为
`false`，输出目录改为 `logs/open_door_dagger_proprio/`，其余训练参数与触觉配置一致。
学生不创建触觉编码器，Actor 输入为本体历史930维＋goal latent32维，共962维；
触觉读数和有效性掩码均不参与动作计算。仿真中的传感器与现有日志仍保留作诊断，
因此日志中有触觉读数不表示学生使用了触觉。

当前两份正式配置将 `initialize_goal_from_teacher` 设为 `true`。训练开始时，学生复制教师的 `enc` 和
`goal_norm` 参数及统计量；初始化完成后，`enc` 会继续参与 BC/DAgger 的梯度更新。
首次 BC 更新前，`goal_norm` 统计量再和 BC 初始数据合并一次，之后由现有 normalizer
生命周期保持不变。将该选项设为
`false` 可进行随机目标编码器的消融实验。

在 CUHK 项目根目录提交无触觉训练（共用现有 Slurm 脚本）：

```bash
mkdir -p logs/slurm
sbatch --job-name=door-proprio-distill tools/distill_open_door.sbatch configs/open_door_dagger_proprio.json
```

Slurm 脚本的首个参数可选；省略时继续使用原触觉配置。每次运行在配置的输出目录下
新建 `slurm-JOB_ID/`，保存该次配置和模型。

## 输入与编码

```text
教师：policy_history [B,10,102] = 本体93 + 当前门状态9
      goal           [B,750]
      触觉expert可选 [B,10,326] = 上述102 + counts112 + validity112

学生：proprio_history [B,10,93]
      tactile_history [B,10,112] uint8，原始0–255 counts
      tactile_valid   [B,10,112] bool
      goal            [B,750]
```

两个历史都按最旧到最新排列，最新帧对应同一控制状态。50 Hz 下，10 帧覆盖当前及过去
180 ms；不是10个未来动作。reset 后用首次观测填满该环境历史，不能混入上一回合。

触觉编码借鉴 Uni-VLaT 的空间／时间融合，但采用本项目六区和小网络：

1. 屏蔽无效 counts 后除以 255，按胸48、背40、左右臂8各、左右肩4各拆分。
2. 每区把读数与有效性 mask 拼接，经独立 `2×通道数 → 64 → 64` MLP。
3. 六个64维 learned queries 经4-head cross-attention和LayerNorm聚合区域特征。
4. 每个token的10帧加入时间嵌入，经一层64维、4-head Transformer（FFN128，dropout0）。
   只读取当前及过去窗口，取最后位置输出，与当前空间token残差相加。
5. 六个token展平，线性投影得到64维触觉latent。

全无效输入仍有明确mask且输出有限，不使用会把全部key屏蔽的注意力掩码。
固定通道顺序来自资产；demo背部图的显示旋转不会作用于网络输入。
仿真counts仍是未标定合成响应，参数和来源见[触觉文档](../../../docs/tactile.md)。

Actor保留原本体／goal归一化、goal编码器及decoder隐层，输入为：

```text
本体10×93 + goal latent32 + tactile latent64 = 1026 → Actor → 29
```

`STUDENT_INPUT_DIM=930` 仍表示本体历史长度，完整decoder输入是1026。
触觉latent只拼接一次；默认教师保持原网络，可选触觉expert需先单独在RL阶段训练。
蒸馏不更新教师，也不训练世界模型或Critic。
新增学生从头训练，动作仍由现有action manager转换：

```text
q_target = q_default + calibration_offset + joint_scale * action
```

## 采集接口

`DaggerCoordinator` 保持仿真无关，现有环境调用者可直接使用：

```python
from supermimic.distillation.open_door import TactileHistory

history = TactileHistory(env.num_envs, device=env.device)
sensor = env.scene["tactile"]

# 每个控制步、执行动作前，读取与policy_history同一状态的触觉。
sensor.update(0.0)
reading = sensor.data
counts, valid = history.update(reading.channel_counts, reading.channel_valid,
                               env.common_step_counter)
step = coordinator.label_and_select(
    observations["policy_history"], observations["goal"], beta,
    tactile_history=counts, tactile_valid=valid,
)
observations, reward, terminated, truncated, extras = env.step(step.executed_action)
done = terminated | truncated
coordinator.record(step, done)
ids = done.nonzero().flatten()
if len(ids):
    observations, extras = env.reset(env_ids=ids)
    history.reset(ids)
```

上例使用原生 mjlab 环境且 `auto_reset=False`；可运行入口还检查参考末帧与仿真有效性。
`label_and_select` 在步进前复制输入，避免仿真缓存重用或reset改变旧标签。
教师和学生在同一真实状态取动作，下一帧的previous action是实际执行的动作。
同一 `step_id` 重复读取不会推进触觉历史；局部reset只重新填充指定环境。

Replay保存完整本体历史、goal、教师动作、done、uint8触觉历史和bool mask。
CPU环形容量受 `capacity` 限制，采样时搬到训练设备；归一化初始化仅分块读取本体和goal，
不把整库触觉迁到GPU。BC和DAgger使用同一个数据集，优化器只更新学生，目标为动作MSE。

## 保存与使用

输出包含 `config.json`、`train_log.csv`、BC后的 `model_bc.pt`，以及按 `save_every`
保存的 `model_N.pt`（最后一轮总会保存）。checkpoint记录 `kind=open_door_tactile_student`、
`schema_version=1`、输入契约、教师来源与任务信息、触觉布局／通道摘要及counts映射参数。
`student` 包含编码器、Actor与归一化统计；这是学生推理权重，不是完整训练续训快照。

```python
import torch
from supermimic.distillation.open_door import OpenDoorStudentPolicy

checkpoint = torch.load("logs/open_door_dagger_tactile/model_10.pt", weights_only=False)
student = OpenDoorStudentPolicy(use_tactile=checkpoint["config"].get("use_tactile", True)).eval()
student.load_state_dict(checkpoint["student"], strict=True)
with torch.no_grad():
    action = student(proprio_history, goal, tactile_history, tactile_valid)
```

部署时必须使用同一布局、counts参数及10帧历史规则。现有触觉回放工具支持学生 checkpoint，
会恢复保存的触觉映射，并在重置环境时清空触觉历史：

```bash
.venv/bin/python tools/play_tactile.py --task open_door \
  --checkpoint logs/open_door_dagger_tactile/model_10.pt \
  --reference data/motions/open_door_prepared/video1_g1_door_video45_centered_asset.npz
```

学生 checkpoint 仍使用独立格式；通用 SuperMimic Runner 不直接加载它。checkpoint 的
`goal_encoder` 字段记录是否使用了教师初始化、`enc` 是否继续训练，以及归一化统计来源。
无触觉模型保存为 `kind=open_door_proprio_student`，触觉输入及latent维度均为0；
同一回放工具可加载它，画面中的触觉面板仅用于诊断。
