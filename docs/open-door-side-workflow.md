# OpenDoorSide：右侧躯干、肩部和前臂推门

任务 ID：`Mjlab-SuperMimic-OpenDoorSide-Unitree-G1`。独立目录
`supermimic/tasks/open_door_side/` 采用与其他任务相同的registration、env_cfgs、commands、
task、mdp、rl_cfg结构。共享OpenDoor的state、dynamics、网络、门资产、DR和RSI，
训练复用SuperMimicRunner。背部与侧面拥有独立参数、奖励配置、日志和checkpoint。

## 配置与数据

编辑 [`configs/open_door_right_side.json`](../configs/open_door_right_side.json)：

- `training`：每卡环境数、batch、迭代数和日志，格式与背部一致。默认1024环境、batch各512。
- `training.wm_checkpoint=null`：不加载背部或其他已有WM checkpoint。
- `training.wm_only_iters=500`：先从头训练500轮WM，再进入Actor/Critic联合训练，和背部Kimodo实验一致。
- `task.door_randomization_cfg.enabled=false`：门物性固定为标称值，与背部Kimodo配置一致；机器人原有随机化配置不变。
- `task.motion_dir`：默认 `data/motions/kimodo_side60_hug_elbow/side60_hug_elbow.npz`，自动使用同名XML。
- `task.episode_length_s=30`：容纳27.3秒的完整参考；参考结束仍正常终止。
- `task.door_reward_cfg`：独立侧面奖励，具体参数如下。
- 当前配对XML包含与背部Kimodo相同的两侧固定实体墙；过门根轨迹已直接修正，并通过门板、两墙和地面的120Hz全段穿透检查。

每个选定NPZ必须有标量 `door_contact_region="right_side"`；混入背部或未标注数据会报错。
背部任务接受旧版未标注参考，但也会拒绝明确标注为侧面的文件；环境变量覆盖目录不会绕过检查。
当前默认参考在Kimodo侧面动作上重新求解双臂：双手稳定留在胸前抱物位置，右肘近端外侧在推门段贴近门面。
沿门切向局部调整根路径以容纳抱物姿态通过门框。门开至60°，18.10秒开始回落，约18.61秒关合；
门角仍是独立参考。120Hz检查覆盖门、两墙、地面及补充手部网格；这是运动学参考，未验证耦合动力学。
旧侧面参考保留在 `data/motions/kimodo_side60_crossed_repaired/`，新旧模型评估应显式选取对应参考。
新版数据、回放与验证见 [肘部抱物参考](../results/kimodo_side_reference_hug_elbow/README.md)。
右上臂已加入合法接触集合，训练几何与实测接触分类同步生效；合法区域超过3mm的深穿透惩罚仍适用。
旧video6实验仍使用GMR转换器
及prepared文件；两类转换器均保留接触区域字段，不读取几何contact mask作为实测接触标签。
右侧实验仍使用原有RSI；本次未加入初始化穿模筛选。已有仿真数值失败隔离继续有效。

## 合法接触区域

同一份 `contacts.py` 定义训练几何和评估分类：

| 部位 | 允许范围 |
|---|---|
| 右侧上躯干 | torso/logo表面在torso坐标满足 `y < -0.04, abs(x) < 0.12, 0.02 < z < 0.35`（米） |
| 右肩 | `right_shoulder_pitch_link`、`right_shoulder_roll_link`碰撞壳 |
| 右上臂 | `right_shoulder_yaw_link`碰撞壳 |
| 右前臂 | `right_elbow_link`、`right_wrist_roll_link`碰撞壳 |
| 其余部位 | 非法，包含腕pitch/yaw与手部、背部、左侧、腿脚及头 |

各部位用实际碰撞资产的表面样本和各自FK变换，前臂距离随关节运动变化。
每geom最多32点，torso合法/非法区域分别采样；这是有限采样的几何近似，可能漏掉样本间碰撞。
允许接触不等于允许深穿透：合法区域超过3mm后也产生状态惩罚。

接触集合进入checkpoint元数据。旧侧面checkpoint若未允许右上臂，当前版本会拒绝其普通加载、resume及Actor/WM微调；
不会自动改写旧模型的接触语义。新定义可从头训练，或通过已有 `wm_checkpoint` 接口只迁移兼容WM。

## 奖励

机器人六项tracking、action-rate与joint-limit沿用现有MDP，不锁手臂，不添加固定姿态奖励。
门角和角速度继续跟踪视频reference曲线，不改成固定45°目标。

令 `d(S)` 为允许表面到门板正面有限矩形的最小距离；`a(S)` 为
`torso +Y · 门板 +X法线`，`g(a)=smoothstep((a-0.5)/0.5)`。
按参考帧预计算 `phase(f)=exp(-d(S_ref(f))² / reference_contact_std²)`。
参考接近门时奖励开启，离开时平滑减弱；不依赖门角是否已打开，因而也能覆盖推门前的靠近。

每步奖励（整个公式乘一次 `dt=0.02`，机器人项原本已包含dt）：

```text
robot_tracking
+ dt * angle_weight * exp(-(theta - theta_ref)^2 / angle_std^2)
+ dt * velocity_weight * exp(-(omega - omega_ref)^2 / velocity_std^2)
+ dt * contact_weight * phase(f) * exp(-d(S)^2 / contact_distance_std^2) * g(a(S))
- dt * illegal_weight * max_forbidden [intrusion^2 / (intrusion^2 + penetration_scale^2)]
- dt * contact_penetration_weight * max_allowed [excess^2 / (excess^2 + penetration_scale^2)]
```

非法区域使用 `intrusion=max(illegal_margin-signed_clearance,0)`；默认1cm内即开始惩罚，
避免脚贴门施力而采样点尚未穿透时奖励为零。合法区域仍使用
`depth=max(-signed_clearance,0)`，`excess=max(depth-contact_penetration_tolerance,0)`。
距离针对门正面，靠门背面不会获得相同接近奖励；深穿透惩罚不受phase关闭影响。
仿真和想象调用同一个SideDoorMDP.reward；MuJoCo接触力只用于评估。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| angle_weight / velocity_weight | 1 / 0.25 | 门角／速度跟踪权重 |
| angle_std / velocity_std | 0.12 rad / 1 rad/s | 跟踪尺度 |
| contact_weight | 0.25 | 允许区域接近奖励 |
| contact_distance_std | 0.05 m | 当前状态接近尺度 |
| reference_contact_std | 0.05 m | 参考阶段的平滑尺度 |
| illegal_weight | 2 | 禁止区域避让惩罚 |
| illegal_margin | 0.01 m | 禁止区域到门的避让距离；0恢复旧穿透惩罚 |
| contact_penetration_weight | 2 | 允许区域深穿透惩罚 |
| contact_penetration_tolerance | 0.003 m | 允许区域的小量穿透容差 |
| penetration_scale | 0.01 m | 有界惩罚尺度 |

终止沿用机器人跟踪失败、门铰链越界、参考结束和episode时限。默认
`penetration_limit=null`；该可选参数沿用背部任务的非法表面穿透终止边界。

## 启动、评估和日志

```bash
./run_open_door_side.sh --config configs/open_door_right_side.json --check
CUDA_VISIBLE_DEVICES=1,2 ./run_open_door_side.sh
```

`--check` 解析最终配置，不启动仿真。覆盖优先级仍为CLI > 显式环境变量 > JSON > Python默认。
旧命令若设置了 `SUPERMIMIC_MOTION_DIR`，它会覆盖JSON；服务器上应指定完整侧面NPZ路径：

```bash
cd /home/wangfeng/supermimic_vibed
CUDA_VISIBLE_DEVICES=1,2 \
SUPERMIMIC_MOTION_DIR=/absolute/path/side60_crossed.npz \
./run_open_door_side.sh
```

路径按侧面数据同步后的实际位置填写；没有自动向CUHK传输正在修改的数据或启动长训。
日志、TensorBoard、CSV和模型位于 `logs/rsl_rl/supermimic_open_door_side/`，遵循现有logs软链接。

```bash
.venv/bin/python -m supermimic.scripts.evaluate \
  --task Mjlab-SuperMimic-OpenDoorSide-Unitree-G1 \
  --motion-dir /absolute/path/side60_crossed.npz \
  --checkpoint-file /absolute/path/model_200.pt
```

评估恢复独立奖励和门物性配置，报告 `allowed_side_substep_rate`、非法接触率、
门轨迹RMSE与参考完成率。接触诊断按每物理子步的法向力合计>1N计算，不证明因果开门力矩。
checkpoint记录独立task name、允许区域、参考phase定义和奖励；不能将背部checkpoint
直接当作侧面任务resume。网络维度相同也不代表奖励语义相同。

## 蒸馏为本体＋触觉学生

侧面策略也可通过独立的BC/DAgger入口蒸馏为本体＋触觉学生，使用
[`configs/open_door_side_dagger_tactile_counts_only.json`](../configs/open_door_side_dagger_tactile_counts_only.json)。
该配置采用已完成的侧面fresh-AC第10000轮教师与video6，使用3072环境、batch1024、LR2e-4、
16轮DAgger；BC更新192次，每轮DAgger更新296次，teacher执行概率在8轮内降至0。
触觉编码器随机初始化，每帧仅输入112维counts，NaN/Inf或无效读数预处理填0；
valid不进入网络。goal编码器从侧面teacher初始化后继续训练。详见
[蒸馏配置与回放命令](../supermimic/distillation/open_door/README.md#侧面开门-counts-only-学生)。
同一JSON的 `use_depth` 可独立开启Sitting同款深度输入，与 `use_tactile` 组合为
本体、触觉、depth或触觉＋depth四种学生；默认关闭depth，开启时另设输出目录。
DAgger使用物理仿真采集和teacher动作监督，不使用下面的WM训练初始化流程。

## WM从头训练

当前侧面JSON不加载已有WM checkpoint，从头训练WM：

```json
"wm_checkpoint": null,
"wm_only_iters": 500,
"resume": false
```

`wm_checkpoint=null` 时不读取任何已有WM；`wm_only_iters=500` 会先进行500轮WM-only训练。

WM、Actor、Critic、三个优化器和迭代数均从头开始。先填充64个仿真步的Replay，随后完成500轮WM-only训练，再进行WM、Actor、Critic联合更新。

加载检查动力学schema、state/context语义、门资产、控制时间步、WM结构和所有张量的形状、
类型及有限性。不同任务的reward、参考和接触定义不进入WM，所以显式迁移时允许不同；
普通完整resume仍检查独立侧面任务的全部语义。尚未定义WM迁移接口的任务不会只凭形状相同接受文件。

以后继续侧面训练使用正常 `resume=true`；完整resume优先。

## 从已有侧面策略微调避脚行为

使用 [`configs/open_door_right_side_finetune.json`](../configs/open_door_right_side_finetune.json)：
从侧面第10000轮模型载入Actor、WM及各自归一化统计，Critic和全部优化器重新初始化，
迭代重新从0计数。保持非法权重2，仅增加1cm避让距离；合法接触区域和状态/网络维度不变。
默认每卡3072环境、WM/Actor batch各512、学习率5e-5、2000轮，无WM-only阶段。

```bash
cd /home/wangfeng/supermimic_vibed
mkdir -p logs/launch
set -o pipefail
CUDA_VISIBLE_DEVICES=1,2 ./run_open_door_side.sh \
  --config configs/open_door_right_side_finetune.json \
  --env.commands.motion.motion-dir data/motions/open_door_prepared/video6_g1_door_video45_right_side_asset.npz \
  2>&1 | tee "logs/launch/side_clearance_finetune_$(date +%Y%m%d_%H%M%S).log"
```

`finetune_checkpoint` 和 `wm_checkpoint` 不能同时设置。显式微调仅允许同一侧面任务修改
机器人/门奖励，仍检查状态、观测、接触定义、物性、网络和历史/goal长度；普通resume继续严格
检查奖励。新checkpoint记录 `finetune_initialization`，并保留最初背部WM的来源记录。
之后恢复微调结果用正常resume，不重新读取最初的侧面/背部文件。

旧侧面checkpoint没有 `illegal_margin` 时视为0；评估会恢复旧奖励，不会因代码更新而改变
旧模型的评分定义。新1cm配置不能直接完整resume旧奖励模型，应使用本节微调入口。
这项改动让接触附近有梯度，不保证完全消除脚接触；需复查真实部位接触率和开门轨迹。

本次微调入口已用实际侧面10000轮checkpoint验证：4环境、完整网络完成两轮eager训练，
另完成一轮BF16/torch.compile训练，三者均更新且权重/统计有限。双进程测试覆盖一致加载、
单卡Actor损坏时同步拒绝；回归测试覆盖旧奖励评估及新奖励微调后正常resume。

## 当前奖励下从头初始化Actor/Critic

使用 [`configs/open_door_right_side_fresh_ac.json`](../configs/open_door_right_side_fresh_ac.json)：
保留1cm避让奖励和背部第10000轮WM初始化，Actor、Critic及三个优化器从头初始化。
`resume=false`、`finetune_checkpoint=null`、`wm_only_iters=0`，先填充Replay，
随后第1轮共同更新WM/Actor/Critic。环境数3072/卡、batch各512、学习率2e-4、10000轮。
这组配置不加载已有侧面Actor，也不恢复先前训练迭代。

```bash
CUDA_VISIBLE_DEVICES=1,2 ./run_open_door_side.sh \
  --config configs/open_door_right_side_fresh_ac.json \
  --env.commands.motion.motion-dir data/motions/open_door_prepared/video6_g1_door_video45_right_side_asset.npz
```

实验目录后缀 `right_side_clearance_1cm_fresh_ac`，和微调实验分开。

## 验证

当前工作区全仓450项测试通过。CPU测试覆盖独立JSON和任务注册、实际碰撞表面/前臂FK、参考阶段、reward梯度和dt缩放、
合法区域穿透容差、双向checkpoint拒绝、参数恢复及接触力分类。原背部几何/奖励测试也保留。
本地4环境、完整网络完成2轮WM/Actor/Critic，保存后从新进程开启BF16/torch.compile续训到
第3轮，网络权重和归一化统计均有限。侧面评估和轨迹导出通过；短训策略在第38个控制步
结束，尚未完成参考。诊断使用冻结的数据副本，未覆盖正在编辑的源数据，也未启动CUHK长训。
输出位于 `logs/rsl_rl/supermimic_open_door_side/*side_task_smoke/`、
`*side_task_compile_resume_verified/` 和 `logs/open_door_side_validation/`。

WM迁移验证使用实际CUHK背部 `model_10000.pt`：完整网络、4环境完成2轮侧面联合更新，
第1轮即包含Actor/Critic。输出 `*back_wm10000_side_joint_smoke/model_2.pt` 的源迭代记录为10000，
侧面迭代为2，三个优化器均有新状态，网络/统计均有限。双进程CPU回归覆盖相同来源校验、
WM独立加载，以及单rank文件缺失/损坏时同步报错；正常侧面resume无需保留源文件。
