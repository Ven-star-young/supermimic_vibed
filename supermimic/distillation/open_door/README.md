# OpenDoor 本体＋触觉蒸馏

学生使用10帧本体观测和原motion goal，可选10帧112通道触觉、当前一帧depth，输出29维动作。
教师沿用已有 OpenDoor checkpoint 的完整状态观测，参数和归一化统计冻结。
学生不读取当前门的 9 维真值、物性、接触力或 Critic 输入；goal 仍是预先给定的参考轨迹。

## 运行

新实验默认使用 [counts-only 配置](../../../configs/open_door_dagger_tactile_counts_only.json)，在仓库根目录运行：

```bash
.venv/bin/python -m supermimic.scripts.distill_open_door --check
.venv/bin/python -m supermimic.scripts.distill_open_door \
  --config configs/open_door_dagger_tactile_counts_only.json
```

该配置沿用旧 model16 的全部采样和优化参数：3072环境、seed0、batch1024、LR2e-4，
BC采集64步并更新4×48=192次；16轮DAgger每轮采集32步并更新4×74=296次，
Replay容量300000、梯度裁剪1、weight decay 0、每轮保存。beta从1在8轮内降至0，
按从0起算的round_index计算，因此model9–16使用纯student采集，动作标签仍来自teacher。
新输出目录为 `logs/open_door_dagger_tactile_teacher7800_counts_only/`。

`tactile_include_valid=false` 表示触觉网络每帧仅输入112维counts。
有效性mask只用于预处理和诊断；无效读数或NaN/Inf填0，再除以255。
各区域首层从 `2×通道数 → 64` 改为 `通道数 → 64`；其余编码器和Actor结构保持一致。
触觉编码器及动作decoder随机初始化，由teacher-action MSE在BC和DAgger阶段共同优化，
没有触觉特征拷贝、特征对齐loss或expert触觉权重初始化。
旧实现也只复制goal编码器；新配置保留 `initialize_goal_from_teacher=true`，以保持这一对照条件一致。
日志增加 `tactile_invalid_fraction`，统计采集时无效通道比例；整个物理状态或动作非有限仍会报错停止。

旧配置 `open_door_dagger_observable.json` 和 `open_door_dagger_proprio.json` 保留原行为；
缺少 `tactile_include_valid` 字段按 `true` 解释，以兼容旧checkpoint和原实验。
新旧配置均默认读取触觉教师
`logs/rsl_rl/supermimic_open_door/2026-11-12_11-21-19_back_tracking_dr_rsi_tactile/model_7800.pt`，
旧配置使用3072环境、batch1024、16轮DAgger，触觉学生输出至
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

### 可选depth输入（与Sitting相同）

背部与侧面两份counts-only配置均提供独立的 `use_depth` 开关，默认 `false`。
修改 `use_tactile` / `use_depth` 可以选择四种学生；每次实验同时指定独立的 `output_dir`：

| use_tactile | use_depth | 学生输入（均含本体历史与goal） | Actor输入维度 |
|---|---|---|---:|
| false | false | 仅本体与goal | 962 |
| true | false | 加触觉 | 1026 |
| false | true | 加depth | 1026 |
| true | true | 加触觉与depth | 1090 |

例如仅使用depth，把相应任务的JSON修改为：

```json
"use_tactile": false,
"use_depth": true,
"output_dir": "logs/open_door_side_dagger_depth_teacher10000"
```

直接复用Sitting的 `DepthEncoder` 与共享HDMI相机，不另建视觉网络或训练入口。
相机为G1的 `d435_link` 上的160×120针孔相机，50Hz更新；输入为当前帧光轴Z深度
`[B,1,120,160]`、float32米。非正值与NaN/Inf映射到4m，裁剪到0.1–4m，无图像噪声、
额外normalizer或有效性输入维度。三层5×5、stride2卷积（1→8→8→8，Mish）后投影到64维。
depth CNN随机初始化，和触觉编码器、Actor共同由teacher-action MSE更新。

深度图与本体、触觉、teacher标签都在执行动作前读取；step/reset由原生传感器更新相机缓存。
CPU Replay存当前帧float16深度，取样时转回float32并重新通过当前CNN，避免存储过期latent。
`use_depth=false` 不创建策略相机、CNN或深度Replay存储；回放显式 `--camera-view` 仍可显示诊断相机。
深度学生回放会自动创建相机，不需要额外开关，`--camera-view`可同时显示RGB/depth。

沿用容量300000时，含深度的CPU Replay约13.27GiB（不含采样batch、渲染器、模型和优化器）；
`max_dataset_gib=16` 在配置解析时检查环形缓存预算，`--check`打印估计用量。
原3072环境的采样/优化参数保持原值；开启相机后全规模GPU渲染占用需另行测量，
本地小规模验证不代表3072相机已通过显存测试。

### Kimodo完整reference的10k教师：仅深度视觉学生

[`configs/open_door_dagger_kimodo_depth.json`](../../../configs/open_door_dagger_kimodo_depth.json)
使用CUHK作业1218的 `2026-11-22_21-53-59_back_door_cycle_walls_rsi/model_10000.pt`，
reference为 `data/motions/kimodo_back_door_cycle/kimodo_back_door_cycle.npz`。
教师没有触觉输入，因此 `expert_observation_dim=102`；学生设置
`use_depth=true`、`use_tactile=false`，只接收本体历史、goal及当前深度图。
沿用3072环境、16轮、batch1024、容量300000、学习率2e-4及原beta schedule。
输出目录为 `logs/open_door_dagger_kimodo_depth_teacher10000/`。

这份配置沿用当前DAgger的adaptive RSI与10秒采集episode，随机参考起点可覆盖完整
27.27秒reference；它不是每回合从首帧连续执行完整动作。触觉传感器仍保留用于现有采集
诊断，但学生无触觉编码器且不消费触觉输入。深度Replay估算13.26GiB。
已通过配置解析、teacher/门资产兼容、相机配置及CPU学生前向验证；未启动训练，
3072环境的实际GPU渲染显存尚未验证。

在CUHK项目根目录启动，清除旧teacher/reference环境覆盖后使用现有Slurm脚本：

```bash
cd /home/wangfeng/supermimic_vibed
env -u SUPERMIMIC_EXPERT_CHECKPOINT -u SUPERMIMIC_MOTION_DIR \
  sbatch --job-name=door-kimodo-depth tools/distill_open_door.sbatch \
  configs/open_door_dagger_kimodo_depth.json
```

脚本在所选输出目录下添加`slurm-<job_id>`；teacher路径按CUHK项目布局填写，
本地同源10k完整checkpoint已归档在Lexar。

### 侧面开门 counts-only 学生

使用 [侧面配置](../../../configs/open_door_side_dagger_tactile_counts_only.json)：

```bash
.venv/bin/python -m supermimic.scripts.distill_open_door \
  --config configs/open_door_side_dagger_tactile_counts_only.json --check
.venv/bin/python -u -m supermimic.scripts.distill_open_door \
  --config configs/open_door_side_dagger_tactile_counts_only.json
```

教师是已归档的侧面 `right_side_clearance_1cm_fresh_ac` 第10000轮Actor，参考为
`video6_g1_door_video45_right_side_asset.npz`。该教师无触觉输入，`expert_observation_dim=102`；
学生独立从Warp传感器读取112通道counts，随机初始化触觉编码器并用BC/DAgger优化。
保留goal编码器初始化，采样及优化参数与上述背部counts-only配置完全一致。
入口自动选择 `Mjlab-SuperMimic-OpenDoorSide-Unitree-G1`，恢复教师的侧面奖励、合法接触区域和物性配置，
学生仍不读取当前门的9维真值。输出至 `logs/open_door_side_dagger_tactile_teacher10000_counts_only/`。
这不需要开启侧面expert的 `task.use_tactile`，也不更改原expert的输入或权重。

回放使用同一个工具，并明确选择侧面任务：

```bash
.venv/bin/python tools/play_tactile.py --task open_door_side \
  --checkpoint logs/open_door_side_dagger_tactile_teacher10000_counts_only/model_16.pt \
  --reference data/motions/open_door_prepared/video6_g1_door_video45_right_side_asset.npz --paused
```

配置中的教师路径是本地Actor归档；CUHK可使用同源完整checkpoint
`logs/rsl_rl/supermimic_open_door_side/2026-11-01_01-14-42_right_side_clearance_1cm_fresh_ac/model_10000.pt`。
蒸馏只读取Actor与任务元数据；motion_dir须改为服务器上实际的video6文件路径。

### 在CUHK指定配置文件启动

在CUHK仓库根目录使用现有Slurm脚本，配置文件作为第一个参数：

```bash
cd /home/wangfeng/supermimic_vibed
mkdir -p logs/slurm
SUPERMIMIC_MOTION_DIR=/data/wangfeng/supermimic_open_door_tracking/data/motions/open_door_prepared/video1_g1_door_video45_centered_asset.npz \
sbatch --job-name=door-back-dagger tools/distill_open_door.sbatch \
  configs/open_door_dagger_tactile_counts_only.json
```

侧面配置的本地Actor归档可显式换成服务器同源的完整teacher：

```bash
SUPERMIMIC_EXPERT_CHECKPOINT=logs/rsl_rl/supermimic_open_door_side/2026-11-01_01-14-42_right_side_clearance_1cm_fresh_ac/model_10000.pt \
SUPERMIMIC_MOTION_DIR=data/motions/open_door_prepared/video6_g1_door_video45_right_side_asset.npz \
sbatch --job-name=door-side-dagger tools/distill_open_door.sbatch \
  configs/open_door_side_dagger_tactile_counts_only.json
```

替换最后的JSON即可使用自定义配置，`use_tactile`、`use_depth`及训练超参数均按该文件执行。
脚本默认申请1张A100、8核CPU、64GiB内存；Slurm选定GPU，进程使用可见设备 `cuda:0`。
`srun --cpu-bind=none`避免再次把训练主线程固定在单核，CPU集合仍由Slurm分配限制。
教师路径和motion路径默认尊重JSON；仅显式设置 `SUPERMIMIC_EXPERT_CHECKPOINT` 或
`SUPERMIMIC_MOTION_DIR` 时覆盖对应字段，不再强制替换为旧背部无触觉teacher/video1。
使用自定义配置前核对这两个环境变量，避免遗留的覆盖值影响本次作业。

每次作业把生效配置保存为 `logs/slurm/distill-<job_id>.json`，先运行 `--check`，
再训练；输出自动保存到JSON中 `output_dir/slurm-<job_id>/`，console为 `logs/slurm/<job_name>-<job_id>.out`。
脚本会检查teacher、motion和同名XML存在，并在训练初始化时核对任务元数据。
开启depth后的3072环境渲染显存仍需实测，CPU Replay预算检查不代表GPU显存验证。

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
2. counts-only学生每区只输入读数，经独立 `通道数 → 64 → 64` MLP。
   旧学生和已有expert保留读数与mask拼接的 `2×通道数 → 64 → 64` 路径，以原样加载权重。
3. 六个64维 learned queries 经4-head cross-attention和LayerNorm聚合区域特征。
4. 每个token的10帧加入时间嵌入，经一层64维、4-head Transformer（FFN128，dropout0）。
   只读取当前及过去窗口，取最后位置输出，与当前空间token残差相加。
5. 六个token展平，线性投影得到64维触觉latent。

全无效输入在新学生中清洗为全零counts且输出有限，不将mask拼入网络或用作注意力掩码。
固定通道顺序来自资产；demo背部图的显示旋转不会作用于网络输入。
仿真counts仍是未标定合成响应，参数和来源见[触觉文档](../../../docs/tactile.md)。

Actor保留原本体／goal归一化、goal编码器及decoder隐层，输入为：

```text
本体10×93 + goal latent32 + tactile latent64 = 1026 → Actor → 29
```

`STUDENT_INPUT_DIM=930` 仍表示本体历史长度，完整decoder输入是1026。
触觉latent只拼接一次；当前两组共用已训练的触觉expert7800。
启用depth时再拼接64维depth latent；侧面配置使用独立的侧面expert10000。
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
输入契约、教师来源与任务信息、触觉布局／通道摘要及counts映射参数。
counts-only触觉学生使用 `schema_version=2`，`policy_config.tactile_include_valid=false`；
未启用depth的旧格式和无触觉学生保持schema1。
depth学生使用schema3，kind为 `open_door_depth_student` 或 `open_door_tactile_depth_student`，
额外保存相机metadata、depth输入契约与初始化来源；加载时严格核对相机配置和模态开关。
`tactile_encoder`字段明确记录随机初始化、可训练及是否输入mask。
`student` 包含编码器、Actor与归一化统计；这是学生推理权重，不是完整训练续训快照。

```python
import torch
from supermimic.distillation.open_door import OpenDoorStudentPolicy

from supermimic.distillation.open_door import load_open_door_student

checkpoint = torch.load("logs/open_door_dagger_tactile_teacher7800/model_16.pt", weights_only=False)
student = load_open_door_student(checkpoint)
with torch.no_grad():
    action = student(proprio_history, goal, tactile_history, tactile_valid,
                     depth=current_depth if student.use_depth else None)
```

部署时必须使用同一布局、counts参数及10帧历史规则。现有触觉回放工具支持学生 checkpoint，
会恢复保存的触觉映射，并在重置环境时清空触觉历史：

```bash
.venv/bin/python tools/play_tactile.py --task open_door \
  --checkpoint logs/open_door_dagger_tactile_teacher7800/model_16.pt \
  --reference data/motions/open_door_prepared/video1_g1_door_video45_centered_asset.npz
```

新实验回放把上述checkpoint换成
`logs/open_door_dagger_tactile_teacher7800_counts_only/model_16.pt`；同一工具会校验schema并选择匹配的编码器。
旧评测脚本若只支持schema1，不能直接用于新模型；应按保存的 `tactile_include_valid` 构造学生并校验schema2。
以下已有student-MBRL入口仍使用原mask编码器，暂不接受counts-only的schema2或含depth的schema3学生。

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

首次从蒸馏新建训练时，Actor完整继承学生的 `enc`、`dec`、可选触觉编码器及归一化统计，保留已经学到的goal编码。
WM和Critic从教师初始化，包括各自的归一化统计，随后都继续更新；优化器和训练计数重新建立。
训练改为想象轨迹上的奖励优化：真实仿真转移训练WM，想象轨迹的λ-return用于拟合Critic。
Actor每帧仍只有本体93维及可选counts112/validity112，即93或317维；
不读取当前门状态9维。完整物理state73/context46、955维Critic输入和奖励定义保持不变。

两份配置使用单卡2048环境、WM/Actor batch各512、32步想象、64步Replay填充、
`wm_only_iters=0`，保留MLP bf16及编译。产物是带 `pi`、`wm`、`critic` 和优化器状态的
完整RL checkpoint；之后正常resume不依赖原始学生或教师文件。
当前MBRL配置沿用Actor `training.pi_lr=2e-5`、WM/Critic `training.lr=2e-4`，
增加 `pi_kl_coef=0.01`、`pi_kl_std=0.1` 的固定DAgger学生参考约束。参考包含原始归一化统计，
初始化后完全冻结；真实Replay的同一观测历史/goal分别输入当前与参考Actor，
以等方差高斯代理KL（动作维求和、batch求均值）作为额外Actor损失。
当前Actor统计仍按原逻辑更新，参考统计不更新。完整RL checkpoint内保存参考，恢复时不会改用当前Actor作参考。
设 `pi_kl_coef=0` 可关闭；旧无参考的MBRL checkpoint不能直接开启KL恢复。
当前有触觉配置从作业441的KL `model_500.pt` 恢复，追加1500轮至总计2000轮，每200轮保存。
完整恢复训练状态及原始冻结参考，Replay重新填充；无触觉配置仍从原DAgger起点新建训练。
`tools/play_tactile.py` 同时支持蒸馏学生与新的RL checkpoint。
CUHK路径覆盖及Slurm命令见[背部开门工作流](../../../docs/open-door-workflow.md#蒸馏学生继续-supermimic-mbrl)。

## 持续 teacher-MSE 的独立实验

[`open_door_student_mbrl_teacher.json`](../../../configs/open_door_student_mbrl_teacher.json)
从原触觉DAgger model16与teacher7800新建训练：先填充128步Replay，前300轮仅更新WM/Critic，
随后9700轮联合优化 `pi_mbrl_loss + 5 * pi_teacher_mse`，共10000轮。Actor统计全程冻结，student-KL关闭；
Actor初始LR2e-5，WM/Critic初始LR2e-4，每100轮保存。

teacher在执行前用同一份实际学生本体/触觉历史加门状态标注，student始终驱动环境。
Replay增加实际观测帧和teacher动作，监督采样覆盖末段并过滤无效/缺历史记录。
首版要求两者触觉契约一致。冻结teacher随完整checkpoint保存，resume无需原模型文件，
阶段按累计iteration继续；Replay重填，不能逐步复现不中断训练。

```bash
.venv/bin/python -m supermimic.scripts.train_open_door --config configs/open_door_student_mbrl_teacher.json --check
.venv/bin/python -m supermimic.scripts.train_open_door --config configs/open_door_student_mbrl_teacher.json
```

该设计借鉴[SUGARv2](https://arxiv.org/abs/2611.09266)的持续动作监督，将PPO改为MBRL属于工程适配。
完整参数、梯度诊断、恢复契约、CUHK同步后启动命令和本地评测约定见
[训练工作流](../../../docs/open-door-workflow.md#持续-teacher-mse-的学生-mbrl)。
