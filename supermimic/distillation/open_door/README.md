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

两份正式配置默认读取触觉教师
`logs/rsl_rl/supermimic_open_door/2026-11-12_11-21-19_back_tracking_dr_rsi_tactile/model_7800.pt`，
使用3072环境、batch1024、16轮DAgger，触觉学生输出至
`logs/open_door_dagger_tactile_teacher7800/`。训练使用原任务的随机初始化和物性分布，
开始前从教师 checkpoint 恢复并核对任务配置。侧面任务可替换 `expert_checkpoint` 和
`motion_dir` 为匹配的侧面模型及 NPZ；入口按 checkpoint 的任务类型选择环境。
输出目录必须为空，避免覆盖已有实验。

两份配置均设 `expert_observation_dim=326`。入口恢复教师的触觉
布局和counts参数；教师历史为原102维状态＋112维counts＋112维validity。
学生仍只提取前93维本体，`use_tactile` 独立决定学生是否使用触觉，教师保持冻结。

训练顺序是教师执行采集 → BC 初始化 → 逐轮混合执行／DAgger 更新。
`bc_rollout_steps`、`bc_train_epochs`、`bc_batches` 控制 BC；
`rounds`、`rollout_steps`、`train_epochs`、`updates_per_round` 控制 DAgger。
每轮 beta 为 `beta_start + min(round_index / beta_decay_rounds, 1) * (beta_end - beta_start)`，
`round_index` 从 0 开始；beta 表示每个环境选择教师动作的概率，不混合关节动作数值。

无触觉对照使用 `configs/open_door_dagger_proprio.json`：仅将 `use_tactile` 设为
`false`，输出目录改为 `logs/open_door_dagger_proprio_teacher7800/`，其余训练参数与触觉配置一致。
学生不创建触觉编码器，Actor 输入为本体历史930维＋goal latent32维，共962维；
触觉读数和有效性掩码均不参与动作计算。仿真中的传感器与现有日志仍保留作诊断，
因此日志中有触觉读数不表示学生使用了触觉。

当前两份正式配置将 `initialize_goal_from_teacher` 设为 `true`。训练开始时，学生复制教师的 `enc` 和
`goal_norm` 参数及统计量；初始化完成后，`enc` 会继续参与 BC/DAgger 的梯度更新。
首次 BC 更新前，`goal_norm` 统计量再和 BC 初始数据合并一次，之后由现有 normalizer
生命周期保持不变。将该选项设为
`false` 可进行随机目标编码器的消融实验。

CUHK两份同名JSON需将 `motion_dir` 设为
`/data/wangfeng/supermimic_open_door_tracking/data/motions/open_door_prepared/video1_g1_door_video45_centered_asset.npz`，
其余内容与本地一致。以下命令每组申请1张A100，直接读取对应JSON：

```bash
cd /home/wangfeng/supermimic_vibed
mkdir -p logs/slurm
sbatch --account=interns --partition=gpu --nodes=1 --ntasks=1 --gres=gpu:a100:1 \
  --cpus-per-task=8 --mem=32G --time=24:00:00 \
  --job-name=door-distill-t7800-touch --output=logs/slurm/%x-%j.out \
  --wrap='OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 srun --ntasks=1 .venv/bin/python -u -m supermimic.scripts.distill_open_door --config configs/open_door_dagger_observable.json'
sbatch --account=interns --partition=gpu --nodes=1 --ntasks=1 --gres=gpu:a100:1 \
  --cpus-per-task=8 --mem=32G --time=24:00:00 \
  --job-name=door-distill-t7800-proprio --output=logs/slurm/%x-%j.out \
  --wrap='OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 srun --ntasks=1 .venv/bin/python -u -m supermimic.scripts.distill_open_door --config configs/open_door_dagger_proprio.json'
```

现有 `tools/distill_open_door.sbatch` 会覆盖JSON中的教师路径为旧无触觉10000，
不适用于本次7800实验；以上命令绕开该覆盖，CUHK无需修改代码。
模型和生效配置直接保存到JSON指定的两个新输出目录，CUHK的 `logs` 指向
`/data/wangfeng/supermimic_vibed/logs`。重新提交同一组前应指定新的空输出目录。

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
触觉latent只拼接一次；当前两组共用已训练的触觉expert7800。
蒸馏不更新教师，也不训练世界模型或Critic。
学生动作解码器及触觉编码器从头训练，goal分支按初始化配置处理；动作仍由现有action manager转换：

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

checkpoint = torch.load("logs/open_door_dagger_tactile_teacher7800/model_16.pt", weights_only=False)
student = OpenDoorStudentPolicy(use_tactile=checkpoint["config"].get("use_tactile", True)).eval()
student.load_state_dict(checkpoint["student"], strict=True)
with torch.no_grad():
    action = student(proprio_history, goal, tactile_history, tactile_valid)
```

部署时必须使用同一布局、counts参数及10帧历史规则。现有触觉回放工具支持学生 checkpoint，
会恢复保存的触觉映射，并在重置环境时清空触觉历史：

```bash
.venv/bin/python tools/play_tactile.py --task open_door \
  --checkpoint logs/open_door_dagger_tactile_teacher7800/model_16.pt \
  --reference data/motions/open_door_prepared/video1_g1_door_video45_centered_asset.npz
```

学生 checkpoint 仍使用独立格式；不能作为完整RL续训快照加载。checkpoint 的
`goal_encoder` 字段记录是否使用了教师初始化、`enc` 是否继续训练，以及归一化统计来源。
无触觉模型保存为 `kind=open_door_proprio_student`，触觉输入及latent维度均为0；
同一回放工具可加载它，画面中的触觉面板仅用于诊断。

## 蒸馏后继续 MBRL

使用现有 `train_open_door` 入口继续优化学生：

```bash
.venv/bin/python -m supermimic.scripts.train_open_door --config configs/open_door_student_mbrl_tactile.json
# 无触觉对照
.venv/bin/python -m supermimic.scripts.train_open_door --config configs/open_door_student_mbrl_proprio.json
```

`training.student_checkpoint` 分别指向本地归档的
`logs/open_door_dagger_tactile_teacher7800/model_16.pt` 和
`logs/open_door_dagger_proprio_teacher7800/model_16.pt`；
`training.teacher_checkpoint` 均指向
`logs/rsl_rl/supermimic_open_door/2026-11-12_11-21-19_back_tracking_dr_rsi_tactile/model_7800.pt`。
这两个来源必须同时指定，不能与 `wm_checkpoint` 或 `finetune_checkpoint` 混用。

Actor完整继承学生的 `enc`、`dec`、可选触觉编码器及归一化统计，保留已经学到的goal编码。
WM和Critic从教师初始化，包括各自的归一化统计，随后都继续更新；优化器和训练计数重新建立。
训练改为想象轨迹上的奖励优化：真实仿真转移训练WM，想象轨迹的λ-return用于拟合Critic。
Actor每帧仍只有本体93维及可选counts112/validity112，即93或317维；
不读取当前门状态9维。完整物理state73/context46、955维Critic输入和奖励定义保持不变。

两份配置使用单卡2048环境、WM/Actor batch各512、32步想象、64步Replay填充、
`wm_only_iters=0`，保留MLP bf16及编译。产物是带 `pi`、`wm`、`critic` 和优化器状态的
完整RL checkpoint；之后正常resume不依赖原始学生或教师文件。
当前学习率对照仅将Actor设为 `training.pi_lr=2e-5`，WM/Critic保持 `training.lr=2e-4`；
未加入KL/BC约束，也未冻结归一化统计。建议先按工作流命令运行500轮，确认早期自主评测再决定延长。
`tools/play_tactile.py` 同时支持蒸馏学生与新的RL checkpoint。
CUHK路径覆盖及Slurm命令见[背部开门工作流](../../../docs/open-door-workflow.md#蒸馏学生继续-supermimic-mbrl)。
