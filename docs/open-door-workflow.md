# OpenDoor：背部开门轨迹跟踪

任务 ID 为 `Mjlab-SuperMimic-OpenDoor-Unitree-G1`。使用同一套 Tracking/Sitting
Runner、Replay、PD 控制和 reference clock；网络复用 Tracking 的 MLP，分别训练参数。
控制 50 Hz、物理 200 Hz、10 帧历史、10 帧未来目标。门是被动铰链，没有 actuator。

## 数据适配

来源处理放在已有 `motion_data/converters/`，本次增加 `gmr.py`：

```bash
.venv/bin/python -m motion_data.converters.gmr \
  data/motions/open_door data/motions/open_door_prepared
```

原始数据保留不动。转换器校验 G1 29 关节、wxyz、资产坐标系，以及 NPZ 与同名 XML
的铰链位置/轴向/限位/尺寸/闭门姿态一致性。保留源 FPS 和诊断字段，不读取 reference/
中的单独视频角度文件作为机器人 motion。已存在且来源哈希相同的完整输出对默认跳过。
同名源数据/XML更新或旧输出缺少来源哈希时会明确报错，更新时用 `--overwrite`。
任务默认读取 `data/motions/open_door_prepared`，XML默认取第一条NPZ的同名文件。

当前prior为G1首帧heading对齐门面的版本（`door_alignment_reference` 标记）：
机器人root位置/朝向已更新，首帧水平yaw与关闭门的+X方向一致；保留原roll/pitch。
关节轨迹、门角曲线和245帧@30Hz时序未改变。转换保留用户的轨迹，包括末帧约10.88cm
的参考背点间隙，不重新拟合朝向或修改奖励。`archive/` 为历史副本，loader不递归加载。

```bash
.venv/bin/python -m motion_data.converters.gmr \
  data/motions/open_door data/motions/open_door_prepared --overwrite
```

这版是一个固定门资产：宽0.55m、厚0.04m、高1.8m、−Z铰链，闭门 yaw=0。
多条reference必须有相同门几何和位置。物性默认8kg、阻尼5、铰链摩擦0.5、
机器人–门摩擦0.7、弹簧刚度20、参考角−0.1；可通过command配置调整。
机器人摩擦、关节标定和COM随机化复用Tracking。

`DoorMotionLib` 使用机器人loader的同一个采样时钟：245帧@30Hz成为407帧@50Hz，
末控制时刻8.12s，不外推到源数据末时刻之外。门角插值后重新求角速度，各clip独立差分。
机器人和门在同一参考帧初始化，包含root/joint/hinge的速度；其后门由真实物理推动，
不会在每一步强制写入reference角度。诊断contact mask不作为接触力标签或奖励真值。

现有纯运动学回放器也会加载同名门XML并同步门角；去掉 `--headless` 可打开回放窗口：

```bash
.venv/bin/python -m tools.replay_motion data/motions/open_door_prepared --headless
```

## 网络

| 接口 | OpenDoor | Sitting |
|---|---:|---:|
| Replay state | 73 = G1 71 + θ/ω 2 | 84 |
| Context | 46 = robot33 + hinge xyz3 + 尺寸/物性10 | 46 |
| Actor 单帧 | 102 = robot93 + door9 | 102 |
| 每帧future goal | 75 = robot73 + θ + 0.1ω | 73 |
| Goal encoder | 750→2048→1024→512→256→32 | 730→同隐藏层→32 |
| Actor decoder | 1052→2048→1024→512→256→29 | 1116→同隐藏层→29 |
| WM | 152→2048→2048→2048→72 | 224→同隐藏层→82 |
| Critic | 955→256→128→1 | 1101→同隐藏层→1 |

Door9为实际机器人heading下的铰链相对位置3、门法向3、sinθ/cosθ/ω。
WM输入为G1特征71+door9+动作29+robot物性33+door尺寸/物性10。
铰链xyz已通过相对位置进入模型，不重复输入全局坐标。θ始终是MuJoCo正关节角，
门板世界yaw=−θ；数据不会为了套用旧+Z门而翻转角度。

Critic当前205 = robot184 + door9 + θ/ω参考误差2 + door尺寸/物性10，再拼750维目标。
Actor没有特权物性；没有PointNet、learned geometry、额外task memory或固定最终角度命令。
参考和奖励不输入WM。不同任务不共享checkpoint参数。

## 门物性随机化

训练时每个环境在每次reset独立、均匀采样以下参数；一个episode内保持不变。
门几何、铰链位置/轴和弹簧零位保持固定，不改变参考轨迹。

| 参数 | 训练采样范围 | 播放/评估名义值 |
|---|---:|---:|
| 门质量 | 5–15 kg | 8 kg |
| 铰链阻尼 | 2.5–7.5 N·m·s/rad | 5 N·m·s/rad |
| 铰链干摩擦 | 0.25–0.75 N·m | 0.5 N·m |
| 回位弹簧刚度 | 10–30 N·m/rad | 20 N·m/rad |
| 机器人—门滑动摩擦系数 | 0.3–1.2 | 0.7 |

门质量变化时，惯量按相同质量比例缩放。采样同时更新仿真物性和command的context，
由现有Replay交给WM/Critic；Actor不直接读取这些物性。各GPU使用独立随机种子。
弹簧零位仍为−0.1 rad，门尺寸仍为0.55×1.8×0.04 m。机器人原有随机化保持不变。

范围配置在 `configs/open_door.json` 的 `task.door_randomization_cfg`；设置其中的 `enabled=false`
可禁用门随机化，使用已有的 `door_mass`、`door_damping` 等名义参数。
`play=True`始终使用名义参数，保留训练分布配置供checkpoint校验。
例如只修改质量范围或关闭门随机化：

```bash
./run_open_door.sh --env.commands.motion.door-randomization-cfg.mass '(6, 12)'
./run_open_door.sh --env.commands.motion.door-randomization-cfg.enabled False
```

OpenDoor checkpoint schema升为2，保存采样范围及开关。续训严格要求分布一致；
评估恢复checkpoint中的范围和名义参数，但实际使用固定名义物性。
旧schema1固定物性checkpoint会明确拒绝加载，不自动变成新的随机化实验。
网络、state/context/goal维度及奖励公式没有变化。

加入随机化后的验证：全仓296项测试通过，其中19项专门检查范围、部分reset、
惯量与context同步、终止前Replay以及真实CUDA求解器中的力矩/加速度/接触摩擦。
本地4环境、完整网络完成2轮WM/Actor/Critic短训并保存schema2 checkpoint；
这验证随机化接入学习链路，不代表策略收敛。

## 同一状态奖励

真实仿真和想象均调用 `DoorMDP.reward(S, frame, action, previous_action)`；没有contacts参数。
`S`来自MuJoCo或WM，FK、门几何和奖励函数相同，dt只乘一次。

- 机器人：原Tracking的六个跟踪项、action-rate和joint-limit，完整复用、不重复扣除。
- 门角：`dt × 1.0 × exp(−(θ−θ_ref)² / 0.12²)`。
- 门速度：`dt × 0.25 × exp(−(ω−ω_ref)² / 1.0²)`。
- 背部：`dt × 0.25 × phase × exp(−d_face² / 0.05²) × alignment_gate`。
- 非法穿透：`−dt × 2.0 × max_i[d_i² / (d_i² + 0.01²)]`，
  `d_i=max(−clearance_i,0)`。深穿透仍保留梯度，没有1cm后的平坦饱和区。

`phase=smoothstep(θ_ref/0.05)`，`alignment_gate=smoothstep((cos_alignment−0.5)/0.5)`。
门角约2.86°后phase为1；这只是参考条件，不是假定视频contact mask是实测接触。
背部点是torso局部(−0.06861,0.00763,0.213)，距离到门+X的有限矩形面计算。
门法向与torso+X应同向（与背部外法向相反）。

非法几何使用同一G1碰撞资产的表面样本，每geom最多32点，包含固定child的变换。
只排除torso/logo上的合法背部区域；其余身体、头、手臂、脚的door碰撞几何参与。
这是几何近似，有限采样会漏掉样本间的小接触；它没有接触力或开门力矩的语义。

默认终止继承机器人tracking失败并增加铰链越界；真实hard与想象soft使用相同边界。
数据报告明确保留了参考穿模，因此默认 `penetration_limit=None`，穿透只产生有界惩罚。
若要启用穿透终止，可设 `door_reward_cfg.penetration_limit`；阈值须小于门盒最小半尺寸
（本门为0.02m），真实和想象都使用该状态边界。MuJoCo接触不会单独改变训练reward/done。

## 训练、播放、评估

### 配置文件

日常修改入口为 [`configs/open_door.json`](../configs/open_door.json)，与独立 `open_door`
项目一样按 `training` / `task` 分组。JSON覆盖Python默认值，仍调用原生mjlab的
`launch_training` 和现有SuperMimicRunner，不增加训练循环。

| 想修改的内容 | JSON位置 |
|---|---|
| 每卡环境数、WM/Actor batch | `training.num_envs`、`wm_batch`、`pi_batch` |
| 迭代数、预热、Replay、想象长度 | `training.iterations`、`wm_only_iters`、`buffer_steps`、`horizon` |
| 学习率、折扣、保存频率 | `training.lr`、`gamma`、`save_interval` |
| 日志和模型位置、实验名 | `training.log_root`、`run_name` |
| 轨迹目录、episode时限 | `task.motion_dir`、`episode_length_s` |
| RSI采样方式及混合权重 | `task.sampling_mode`、`uniform_weight`、`current_weight`、`history_weight` |
| 初始化扰动、观测噪声 | `task.reset_noise`、`reset_cfg`、`observation_cfg` |
| 机器人跟踪奖励、门奖励 | `task.reward_cfg`、`door_reward_cfg` |
| 机器人终止、可选穿透终止 | `task.termination_cfg`、`door_reward_cfg.penetration_limit` |
| 门质量、阻尼、摩擦、弹簧随机化 | `task.door_randomization_cfg` |
| 机器人摩擦、COM及关节标定随机化 | `task.physical_cfg` |

例如将 `training.num_envs` 改为1536、保持 `wm_batch=512`、`pi_batch=512`，
两卡合计3072环境；训练batch不会随环境数自动增加。
`training.gpu_ids="all"` 使用所有可见GPU；物理卡号由 `CUDA_VISIBLE_DEVICES` 选择。
列表形式的 `gpu_ids` 使用可见卡内的编号，例如物理卡1、2对应 `[0, 1]`。

```bash
./run_open_door.sh --config configs/open_door.json --check
CUDA_VISIBLE_DEVICES=1,2 ./run_open_door.sh --config configs/open_door.json
```

`--check` 检查配置解析并打印最终值，不启动仿真，也不验证数据文件或显存容量。
可以复制JSON保存不同实验；相对数据/日志路径始终相对于项目根目录
（启动脚本会切换目录），而不是JSON所在目录。最终生效值由原生启动器保存到
每次运行的 `params/env.yaml` 和 `params/agent.yaml`。

优先级为 **命令行 > 显式设置的兼容环境变量 > JSON > Python默认值**。
例如终端之前 `export NUM_ENVS=1024` 后会覆盖JSON，需先 `unset NUM_ENVS` 才以JSON为准。
不传 `--config` 时默认读取 `configs/open_door.json`。

配置入口验证：12项测试通过，覆盖JSON/CLI覆盖、嵌套数组类型、最终资产绑定和
配置导出后回载；另通过新入口完成本地4环境、完整网络的2轮WM/Actor/Critic短训，
保存模型、CSV、TensorBoard和生效配置。该短训不验证CUHK大规模显存容量。

### RSI 与多卡启动

训练已经使用与Sitting相同的Reference State Initialization（RSI），默认
`sampling_mode=adaptive`。每个环境reset选择一个reference帧，同时写入G1的root
位姿/速度、关节位置/速度，以及该帧的门角和角速度。机器人初态可以叠加训练扰动，
门仍从对应参考铰链状态开始，之后由物理推动。部署/完整reference评估默认从首帧开始。

采样按1秒的reference bins进行：50%均匀bin分布、40%当前低回报bin的softmax、
10%历史分布；在选中bin内随机选帧。最后一个短bin也参与，因而并非严格逐帧均匀。
各卡seed不同，分别采样RSI和物理环境；梯度、模型参数及归一化统计跨卡同步。

`run_open_door.sh` 读取上述JSON并封装原生mjlab入口：

```bash
CUDA_VISIBLE_DEVICES=1,2 ./run_open_door.sh
```

GPU编号在 `CUDA_VISIBLE_DEVICES` 中明确指定，JSON默认 `gpu_ids="all"`。
不要在这个脚本外再套torchrun；mjlab会用torchrunx启动一卡一进程。
环境数与WM/Actor batch均为每卡数值，上例全局2048环境、每次WM/Actor更新全局batch1024。
保留完整网络、8步WM窗口、32步想象、10次WM更新、5轮Critic更新、BF16和torch.compile。
默认前500轮只训练WM，Replay每卡256步、每200轮保存，学习率保持2e-4且不按卡数自动放大。
首次运行和续训均重新填充Replay；checkpoint恢复模型/优化器，不恢复RSI采样统计或仿真状态。

脚本也兼容已有的 `NUM_ENVS`、`WM_BATCH`、`PI_BATCH`、`ITERATIONS`、`WM_ONLY_ITERS`、
`SAVE_INTERVAL`、`LOG_EVERY`、`BUFFER_STEPS`、`LOG_ROOT`、`RUN_NAME` 环境变量覆盖规模；
后续CLI参数直接传递给mjlab。续训应明确旧目录名与checkpoint，避免匹配到新建空目录：

```bash
CUDA_VISIBLE_DEVICES=1,2 ITERATIONS=10000 ./run_open_door.sh \
  --agent.resume True --agent.load-run '<旧目录名>' --agent.load-checkpoint model_1000.pt
```

续训的 `ITERATIONS` 是额外迭代数。需按服务器空闲显存验证规模后再长期运行。

### CUHK 同步代码后的新训练

仓库 `/home/wangfeng/supermimic_vibed` 的 `logs` 已是软链接，指向
`/data/wangfeng/supermimic_vibed/logs`。下面的命令使用该软链接保存终端日志、
TensorBoard、CSV和checkpoint，数据则读取服务器上已同步的新版prepared目录。
`data/` 被Git忽略；仅同步代码不会复制轨迹，命令显式指定现有数据位置。

```bash
cd /home/wangfeng/supermimic_vibed
mkdir -p logs/launch
set -o pipefail
CUDA_VISIBLE_DEVICES=1,2 OMP_NUM_THREADS=4 \
SUPERMIMIC_MOTION_DIR=/data/wangfeng/supermimic_open_door_tracking/data/motions/open_door_prepared \
./run_open_door.sh --config configs/open_door.json \
  2>&1 | tee "logs/launch/open_door_dr_$(date +%Y%m%d_%H%M%S).log"
```

模型与训练曲线目录为 `logs/rsl_rl/supermimic_open_door/<时间戳>_back_tracking_dr_rsi/`，
实际位于 `/data/wangfeng/supermimic_vibed/logs/rsl_rl/supermimic_open_door/`。
默认JSON从头训练，使用schema2的门随机化和自适应RSI；不加载旧固定门参数checkpoint。
环境数、batch、总迭代等直接改JSON，无需修改这条启动命令。

旧运行的显存快照（每卡1024环境、WM/Actor batch各512、Replay256步、horizon32）：
初始化约10.6–10.9 GiB，编译阶段约14.3–14.4 GiB；WM-only第70轮时两卡分别
16921 MiB（16.5 GiB）与11223 MiB（11.0 GiB）。这些是采样时刻的占用，
**不是全程峰值**；没有记录Actor训练阶段的显存峰值，而且测量早于门随机化。
建议先改为每卡1536环境、batch各512；2048环境/卡需验证完整WM/Actor/Critic阶段。
仅Replay数据在1024/1536/2048环境时约254/381/509 MiB，仿真碰撞和求解器缓冲
还会随环境数增长，不能只按Replay估计显存。

CUHK 旧prior运行已按用户要求停止（2026-10-31）：最后日志2080轮，最新完整checkpoint
为 `model_2000.pt`。原运行使用物理GPU 1、2，两张A100 40GB，每卡1024环境、
WM/Actor batch各512。独立代码目录为 `/data/wangfeng/supermimic_open_door_tracking`，
复用原 `supermimic_vibed/.venv`，通过PYTHONPATH明确加载本次源码。
历史日志位于该目录的 `logs/launch/train.log`，训练进程和 `open_door_tracking` session已退出。
旧运行目录为 `logs/rsl_rl/supermimic_open_door/2026-10-31_15-16-31_back_tracking_rsi_2gpu/`。
该目录的 `reference_snapshot/` 保留停止时所用旧数据，`stopped_by_user.json` 记录停止状态。
`logs/launch/source_sha256.json` 为旧部署哈希，`source_sha256_current.json` 记录本次新prior适配。
默认数据目录已同步新版；之后由用户手动启动，不自动重启。

```bash
ssh CUHK 'tail -n 20 /data/wangfeng/supermimic_open_door_tracking/logs/launch/train.log'
ssh CUHK
cd /data/wangfeng/supermimic_open_door_tracking
CUDA_VISIBLE_DEVICES=1,2 RUN_NAME=back_tracking_rsi_aligned ./run_open_door.sh
```

上述命令默认新建运行，从头开始、目标10000轮、WM-only500轮、每200轮保存。
prior对齐本身不改变模型shape；后续加入门随机化后schema已升为2，旧固定物性checkpoint
不再直接兼容。上述启动命令不带 `--agent.resume`，从头建立新基线。旧checkpoint及其日志保留。
独立容量检查已在相同环境/batch/网络/想象长度下完成两轮完整WM/Actor/Critic更新，
并保存checkpoint；第2轮约0.8轮/秒，首轮包含约4分钟的编译和初始化。
该容量检查没有替代正式训练的WM预热，也不证明长训收敛。

使用现有mjlab入口，无独立OpenDoor Runner或重复训练循环：

```bash
OMP_NUM_THREADS=4 .venv/bin/train Mjlab-SuperMimic-OpenDoor-Unitree-G1 \
  --gpu-ids '[0]' --env.scene.num-envs 128 --agent.run-name back_tracking
```

输出在 `logs/rsl_rl/supermimic_open_door/`。通过
`--env.commands.motion.motion-dir` 指定其他转换目录或单个NPZ；单个XML可用
`--env.commands.motion.door-xml`指定。按GPU显存调小环境/WM/Actor batch。

```bash
.venv/bin/play Mjlab-SuperMimic-OpenDoor-Unitree-G1 \
  --checkpoint-file /absolute/path/model_200.pt --num-envs 1 --viewer native

.venv/bin/python -m supermimic.scripts.evaluate \
  --task Mjlab-SuperMimic-OpenDoor-Unitree-G1 \
  --checkpoint-file /absolute/path/model_200.pt \
  --record /absolute/path/door_rollout.npz
```

评估从每条reference首帧开始，统计失败与完整参考完成率，并输出门角/角速度RMSE。
RMSE只覆盖实际执行前缀，需结合完成率阅读；完成reference本身也不等于力验证的背推成功。
轨迹包含初态和终止末态的theta/omega，以及完整ref_theta/ref_omega。
评估同时在每个物理子步读取真实MuJoCo接触力，报告合法背部/非法门接触率及曾发生
非法接触的回合数（各类别法向力和>1N）。这些数据只用于诊断，不改变训练reward/done，
也不证明全部开门力矩由背部产生。评估会从checkpoint恢复奖励及门物性参数。

## 验证范围

CPU测试覆盖数据转换/时间对齐、状态与增量、轴向/坐标、模型梯度、几何与MuJoCo碰撞对照、
统一奖励和schema。CUDA测试覆盖原生环境、非零origins、中段reference reset、context物性、
真实reward与tensor reward一致性、终止前Replay。短程训练仅检查工程链路，不代表收敛。

2026-10-31 初版旧prior验证：全仓270项测试通过。实际数据、4环境、完整网络完成2轮eager训练，
保存后从新进程恢复，并启用BF16/torch.compile训练到第3轮；冻结Actor评估和轨迹导出通过。
本次短训策略在第58个控制步失败，未完成参考，也尚未学会开门。
诊断输出保存在 `logs/rsl_rl/supermimic_open_door/*open_door_smoke/`、
`*open_door_resume_compile/` 与 `logs/open_door_smoke_rollout.npz`。

首帧heading对齐版适配验证：55项相关CPU测试通过，覆盖同名源数据变更检测、
转换、全部reference状态/goal、几何惩罚及纯运动学回放。CUHK加载检查确认407控制帧、
每帧75维goal，本地/远端NPZ与XML哈希一致。本次适配没有启动训练。
