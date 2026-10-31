# OpenDoorSide：右侧躯干、肩部和前臂推门

任务 ID：`Mjlab-SuperMimic-OpenDoorSide-Unitree-G1`。独立目录
`supermimic/tasks/open_door_side/` 采用与其他任务相同的registration、env_cfgs、commands、
task、mdp、rl_cfg结构。共享OpenDoor的state、dynamics、网络、门资产、DR和RSI，
训练复用SuperMimicRunner。背部与侧面拥有独立参数、奖励配置、日志和checkpoint。

## 配置与数据

编辑 [`configs/open_door_right_side.json`](../configs/open_door_right_side.json)：

- `training`：每卡环境数、batch、迭代数和日志，格式与背部一致。默认1024环境、batch各512。
- `task.motion_dir`：默认单个 `video6_g1_door_video45_right_side_asset.npz`，自动使用同名XML。
- `task.episode_length_s=15`：适配约12.9秒参考；参考结束仍正常终止。
- `task.door_reward_cfg`：独立侧面奖励，具体参数如下。

每个选定NPZ必须有标量 `door_contact_region="right_side"`；混入背部或未标注数据会报错。
背部任务接受旧版未标注参考，但也会拒绝明确标注为侧面的文件；环境变量覆盖目录不会绕过检查。
继续使用已有GMR转换器，转换器保留该字段。更新源数据后重新转换prepared文件；本适配
不改轨迹、门角或开门时间，不读取几何contact mask作为实测接触标签。
右侧实验仍使用原有RSI；本次未加入初始化穿模筛选。已有仿真数值失败隔离继续有效。

## 合法接触区域

同一份 `contacts.py` 定义训练几何和评估分类：

| 部位 | 允许范围 |
|---|---|
| 右侧上躯干 | torso/logo表面在torso坐标满足 `y < -0.04, abs(x) < 0.12, 0.02 < z < 0.35`（米） |
| 右肩 | `right_shoulder_pitch_link`、`right_shoulder_roll_link`碰撞壳 |
| 右前臂 | `right_elbow_link`、`right_wrist_roll_link`碰撞壳 |
| 其余部位 | 非法，包含右上臂、腕pitch/yaw与手部、背部、左侧、腿脚及头 |

各部位用实际碰撞资产的表面样本和各自FK变换，前臂距离随关节运动变化。
每geom最多32点，torso合法/非法区域分别采样；这是有限采样的几何近似，可能漏掉样本间碰撞。
允许接触不等于允许深穿透：合法区域超过3mm后也产生状态惩罚。

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
- dt * illegal_weight * max_forbidden [depth^2 / (depth^2 + penetration_scale^2)]
- dt * contact_penetration_weight * max_allowed [excess^2 / (excess^2 + penetration_scale^2)]
```

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
| illegal_weight | 2 | 禁止区域穿透惩罚 |
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
SUPERMIMIC_MOTION_DIR=/absolute/path/video6_g1_door_video45_right_side_asset.npz \
./run_open_door_side.sh
```

路径按侧面数据同步后的实际位置填写；没有自动向CUHK传输正在修改的数据或启动长训。
日志、TensorBoard、CSV和模型位于 `logs/rsl_rl/supermimic_open_door_side/`，遵循现有logs软链接。

```bash
.venv/bin/python -m supermimic.scripts.evaluate \
  --task Mjlab-SuperMimic-OpenDoorSide-Unitree-G1 \
  --motion-dir /absolute/path/video6_g1_door_video45_right_side_asset.npz \
  --checkpoint-file /absolute/path/model_200.pt
```

评估恢复独立奖励和门物性配置，报告 `allowed_side_substep_rate`、非法接触率、
门轨迹RMSE与参考完成率。接触诊断按每物理子步的法向力合计>1N计算，不证明因果开门力矩。
checkpoint记录独立task name、允许区域、参考phase定义和奖励；不能将背部checkpoint
直接当作侧面任务resume。网络维度相同也不代表奖励语义相同。

## 验证

当前工作区全仓370项测试通过。CPU测试覆盖独立JSON和任务注册、实际碰撞表面/前臂FK、参考阶段、reward梯度和dt缩放、
合法区域穿透容差、双向checkpoint拒绝、参数恢复及接触力分类。原背部几何/奖励测试也保留。
本地4环境、完整网络完成2轮WM/Actor/Critic，保存后从新进程开启BF16/torch.compile续训到
第3轮，网络权重和归一化统计均有限。侧面评估和轨迹导出通过；短训策略在第38个控制步
结束，尚未完成参考。诊断使用冻结的数据副本，未覆盖正在编辑的源数据，也未启动CUHK长训。
输出位于 `logs/rsl_rl/supermimic_open_door_side/*side_task_smoke/`、
`*side_task_compile_resume_verified/` 和 `logs/open_door_side_validation/`。
