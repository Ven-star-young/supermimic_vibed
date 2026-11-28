# Ledge：video11 高台贴背稳定通行

任务 ID：`Mjlab-SuperMimic-Ledge-Unitree-G1`，checkpoint 名称 `g1_ledge`。
沿用现有任务的注册、`env_cfgs`、`commands`、`task`、`mdp`、`rl_cfg` 和 JSON 编排，
直接使用 `SuperMimicRunner` 训练独立 Actor、WM 和 Critic。

## 数据与场景

默认读取 `data/motions/ledge_video11/video11_g1_ledge.npz` 和同名 XML/JSON。
当前认可版本为1112帧、30Hz，训练重采样为1852帧、50Hz，末帧时刻37.02秒。
资产加载检查NPZ和XML哈希以及计划字段；不会重生成reference，也不读取archive中的候选动作。
动作修改和原始来源见 [reference说明](motion-and-assets.md#video11-加长贴背窄路仅局部修正)。

中央窄路长约1.263m、宽22cm、离地1.2m；+X沿通行方向、+Y离墙、墙面Y=0。
原XML作为固定几何挂载，移除Tracking默认平面以免重复地面。机器人使用完整交互碰撞，
并启用reference检查采用的左右脚显式碰撞对。脚底四球与地面碰撞，完整脚网格与墙碰撞；
脚尖网格悬空允许，后跟／前脚掌支撑由四球投影判断。

保留基础机器人摩擦、关节校准和躯干COM随机化。平台摩擦设置为训练摩擦范围下界，
使等优先级接触中实际采用的摩擦系数与机器人物理context一致；play使用标称摩擦。
墙面仍使用源资产固定摩擦0.7及其优先级。几何不随reset改变。
首版没有恒力或短时推力，任何网络输入均不含外力标签。

## 模型输入

| 网络 | 输入 | 输出 |
|---|---|---|
| Actor | 每帧105维×10历史＋未来机器人730维编码成32维；decoder1082维 | 29维关节位置动作 |
| WM | 机器人不变特征71＋场景XY／航向4＋动作29＋物理context33＝137维 | 70维机器人增量 |
| Critic | 当前特权特征204＋未来机器人730＝934维 | 状态价值 |

Actor新增12维按以下顺序拼到原93维本体观测：

1. 当前根位置X、Y、Z−台高，共3维。
2. 当前根航向的sin、cos，共2维。
3. 左右脚最小支撑余量、左右脚离台高度，共4维。
4. 当前根在场景坐标系中的线速度，共3维。

这些值按实际状态计算，减去并行环境origin；不使用当前reference姿态代替真实位置。
新增特权量默认无观测噪声，按既有RunningNorm归一化。未来reference保持原73维／帧。
Actor沿用已有MLP和10帧展平历史，没有新增PointNet、视觉、触觉或时序网络。

支撑余量使用三个台面并集的外边界，内部接缝不会产生假悬崖。
正值表示整块足底球投影在支撑区内；负值表示局部悬出，不单独作为跌落判据。
摆动脚与支撑脚由计划和实际几何高度共同约束，计划标签不是实测接触力。
WM补入场景XY和航向后，才可区分相同身体姿态在墙边、平台中央或台外的动力学。

## 奖励与终止

基础身体／根轨迹跟踪、动作变化和关节限位代价继续使用原实现。
新增各项为有界非正代价，按20ms乘一次dt，避免通过原地贴墙累积额外奖励：

- 仅在reference贴背阶段，约束合法背面到有限墙面的距离和朝向。
- 计划支撑脚约束支撑余量、参考足底高度和水平滑移；允许脚尖网格悬出。
- 相对当前reference的重力方向偏差约束摆动稳定性，不把正常转身误作失控。
- 非法身体墙面穿透受罚，所有实际碰撞表面超过3mm的深穿透另外受罚。

`task.ledge_reward_cfg` 中的初始失败边界：

| 判据 | 初值 |
|---|---:|
| 根高度低于当前reference | 19cm |
| 重力方向相对当前reference偏差 | 30° |
| 墙面或平台严重穿透 | 2cm |
| 额外跌落判据 | 双脚均低于台面6cm，且下降速度超过0.1m/s或根下降超过8cm |

小幅离墙、摆动脚抬起或单脚投影越界不立即终止，保留有限恢复机会。
真实仿真使用硬边界，想象使用相同边界的平滑终止概率；几何、奖励与失败均为同一套纯张量函数。
身体表面是有限采样近似，实际接触力与全程完成率仍应在物理评估中检查。

## 运行

```bash
# 查看有效配置，不启动仿真
./run_ledge.sh --check

# configs/ledge.json：每卡4096环境，WM batch1024、Actor batch512，前500轮仅训练WM
./run_ledge.sh

# 原生入口同样可用
.venv/bin/train Mjlab-SuperMimic-Ledge-Unitree-G1
.venv/bin/play Mjlab-SuperMimic-Ledge-Unitree-G1 --checkpoint-file /absolute/path/model.pt

# 从reference起点评估，并导出真实状态轨迹
.venv/bin/python -m supermimic.scripts.evaluate \
  --task Mjlab-SuperMimic-Ledge-Unitree-G1 \
  --checkpoint-file /absolute/path/model.pt --record /absolute/path/trajectory.npz
```

训练使用自适应reference初始化，但默认关闭姿态／位置reset噪声，避免开始时就穿墙或离台。
环境数和batch均为每卡数值：WM每批1024个8步窗口，Actor每批512条32步想象轨迹。
Replay保留256步，共1,048,576条转移；Critic每轮将16,384个想象状态拆成4个minibatch。
原生训练入口与JSON入口使用相同的环境数和batch默认值，play仍为1个环境。
原有本体观测噪声和机器人参数随机化仍开启。episode上限40秒，可容纳完整参考。
训练输出位于 `logs/rsl_rl/supermimic_ledge/`。checkpoint记录输入语义、物理参数、
奖励阈值和reference／场景哈希，拒绝不同场景或旧任务checkpoint直接加载。
定量评估会在创建场景前恢复checkpoint中的奖励、观测和物理配置；更换reference仍需显式指定。

任务接入验证包括全参考几何／终止检查、GPU物理步进、真实与张量奖励一致性、
短程WM／Actor／Critic更新及checkpoint往返。短程检查不代表已训练出完整通行策略；
后续需要报告从起点到终点的成功率、掉落率和贴背保持情况，再加入短时推力测试恢复能力。
