# Sitting / OpenDoor：从仿真状态策略到真机视觉策略

调研日期：2026-11-09（America/New_York）。范围：当前仓库输入契约，以及特权状态教师到视觉学生的代表性研究；不是穷尽系统综述。论文实现与实验结论以所链接版本为准，工程建议单独标注。本次只读检查训练代码，没有运行训练或验证真机性能。

## 结论

这条路线通常称为 privileged learning、asymmetric teacher–student learning 或 visuomotor policy distillation。典型流程是：**仿真训练 state teacher → 在仿真中渲染视觉并蒸馏 student → 仅部署视觉 student**。蒸馏通常在仿真内完成，无需先在真机取得同等质量的真实物体状态，也不要求真机重新做强化学习。

对当前项目，已有 SuperMimic checkpoint 可以直接充当 teacher。教师通过 PPO、可微世界模型或其他算法获得，并不改变动作蒸馏的基本接口。最小路线是保留 29 维动作、PD、本体观测与 motion goal，在渲染仿真中新增 BC/DAgger 学生训练。现有 state-only world model 无须先改成视觉世界模型。

但蒸馏不是删除输入即可：需要分别解决学生偏离教师轨迹、视觉域差异、相机视野/遮挡、传感器时延，以及参考轨迹的部署方式。尤其 sitting 的完整 CAD 几何和后方接触视野，不能被“用相机替代 pose”一句话覆盖。

## 1. 当前代码真正需要替换的输入

| 项目 | Sitting actor | OpenDoor / OpenDoorSide actor |
|---|---|---|
| 每帧本体信息 | 93 维：关节位置29、速度29、角速度3、projected gravity3、上一动作29 | 相同 |
| 每帧对象信息 | 椅子相对位置3＋朝向 rot6d 6 | 门轴相对位置3＋门面相对法向3＋sin/cos门角2＋门角速度1 |
| 对象几何 | 完整碰撞表面1024×6 xyz+normal及extent，经编码器得到64维latent | 没有几何编码器 |
| 历史 | 默认10帧，每帧102维 | 相同 |
| 命令 | 未来10帧机器人参考，每帧73维 | 未来10帧机器人及门参考，每帧75维 |
| 动作 | 29维归一化关节目标偏移，经scale/default/calibration转成PD目标 | 相同 |

代码定位：

- [本体actor观测](../supermimic/mdp/observations.py)：`proprioception`，第49行起；本体不直接读取机器人绝对位置、全局朝向、线速度或接触力。
- [Sitting对象观测](../supermimic/tasks/sitting/mdp.py)：第42行起，以当前机器人heading为坐标系计算椅子相对位置/姿态。
- [Sitting几何输入](../supermimic/tasks/sitting/models.py)：第28行起；[几何采样](../supermimic/tasks/sitting/assets.py)第91行起来自完整碰撞表面，**不是相机实际可见的局部点云**。
- [门观测](../supermimic/tasks/open_door/dynamics.py)：`door_features`第35行起；`read_sim_state`第56行起直接读取真实门关节角/角速度。
- [Side任务](../supermimic/tasks/open_door_side/task.py)：继承OpenDoorTask，改变交互奖励/几何，未改变这套actor输入。
- [未来参考](../supermimic/motion.py)：第114行起；[门参考](../supermimic/tasks/open_door/motion.py)第68行起。它们来自预先给定motion和软件时钟，不是未来真实状态，因此第一版视觉学生可以继续使用。
- [Sitting噪声](../supermimic/tasks/sitting/mdp.py)第27行、[门噪声](../supermimic/tasks/open_door/mdp.py)第50行：对象9维噪声均为零。教师目前习惯精确、50 Hz更新的对象信号。
- [训练runner](../supermimic/rl/runner.py)：真实模拟器转移拟合state world model，想象轨迹通过BPTT更新actor/critic；不是PPO runner。第121行附近的想象观测由预测state计算，没有相机图像生成。

**部署边界：** critic或world model需要的物体质量、摩擦、惯性、真实接触等，不必全部变成真机学生输入。actor依赖哪些信息才是审计重点。机器人局部对象观测可由相机外参、本体运动和重力方向构造，不天然要求外部全局定位；长时间遮挡后的相对定位漂移仍需处理。

## 2. 学术界主要怎么实现

### 2.1 视觉估计状态，再接已有state policy

RGB-D / 点云历史 → 物体位姿与速度估计 → 原state actor。

这是一条有用的工程基线，不一定包含policy distillation。优点是容易判断失败来自视觉还是控制，也最容易保留当前网络。缺点是状态估计误差、延迟和丢失会直接进入一个原本习惯精确观测的策略。

适用于已知椅型、有CAD、已知门轴/门框等场景。训练时应模拟实际估计器的偏差、低频更新、冻结观测和缺失，而不只是给每帧加独立小高斯噪声。Miki等人的感知运动控制体现了“几何前端＋有噪观测＋循环记忆”可以成功部署，但它不是原始RGB端到端策略。

### 2.2 直接动作蒸馏：最适合作为本项目第一版视觉student

教师读取本体、真实物体状态及目标；学生读取本体、相机历史及允许保留的命令；二者输出同一动作接口。

设物体特权量为`s`，可部署本体为`p`，图像为`I`，参考命令为`g`：

```text
teacher: a_T = π_T(p_history, s_history, g, optional_CAD_code)
student: a_S = π_S(p_history, I_history, g)
```

常用训练过程：

1. 冻结教师及归一化统计，采集一定量教师轨迹，为视觉学生做BC初始化。
2. 让学生实际驱动模拟器。视觉误差会把机器人带到教师成功轨迹之外。
3. 在学生访问的同一个模拟器状态查询教师动作，把它作为当前学生观测的标签。
4. 聚合这些数据，继续优化学生；重复采集与训练。这就是DAgger的核心。
5. 若模仿已饱和但任务仍失败，可在仿真中对学生做任务奖励微调，必要时保留动作监督。

重点是第3步：教师要回答“学生现在走偏到这里，该怎么做”，不是回放教师原轨迹在相同时间戳的动作。教师历史必须来自当前这条轨迹；`a_prev`应为真正执行的动作，不能使用教师未执行的动作。

常见目标为动作MSE，或高斯策略分布KL；也可增加位置、姿态、几何、接触等辅助监督。对当前确定性29维actor，归一化后的动作MSE是自然起点；没有必要为了使用KL重新制造随机教师。

### 2.3 蒸馏控制所需的latent，保留控制器

教师对象信息 → privileged encoder → latent → control decoder；学生视觉历史 → visual encoder → 同一latent接口 → control decoder。

可以监督latent、动作，或二者结合。Reflex提供了保留固定控制器、用视觉历史回归动态latent的直接实例；LessMimic用视觉latent替换distance-field表示并通过动作DAgger训练；ReGATE进一步监督几何、接触并使用循环门控。

优点是复用运动技能、把昂贵渲染集中在感知学习。限制是latent必须与控制器兼容，且不能要求视觉重建本来不可观测的信息。当前sitting的64维CAD latent不是现成的“视觉可识别物体状态”；完整形状、局部可见表面和控制相关几何是三个不同概念。

### 2.4 HDMI 的图像分支：不是 SigLIP / DINO，也不是论文默认部署路径

这里需要区分 HDMI 论文/默认训练和官方代码中的可选实验分支：

- 论文报告的真实 G1 部署依赖本体状态和动捕物体状态；视频只用于生成机器人/物体参考轨迹。它没有报告 RGB/DINO/SigLIP 视觉学生。
- 官方仓库默认 `enable_cameras=false`，`train_sequential.yaml` 的默认阶段只有 `ppo_roa_train` 和 `ppo_roa_finetune`。因此 README 中的默认 teacher/student 命令不是视觉训练命令。
- 代码另外提供 `ppo_roa_train_est` / `ppo_roa_adapt_est` 的可选 depth estimator。相机同时生成 RGB、depth 和 distance-to-image-plane，但 observation group 只取 depth。
- depth 先进入延迟缓冲（随机0–8帧），再加入约5 cm逐帧噪声和±15 cm episode bias，NaN/Inf替换并裁剪到0.1–4.0 m，输出为`[N,1,H,W]`。默认相机配置为160×120、50 Hz。
- 图像编码器是从零训练的小CNN：三层stride=2、kernel=5的`LazyConv2d`，通道8/8/8、Mish激活，Flatten后`LazyLinear(64)+LayerNorm`。没有加载预训练ResNet、DINO、SigLIP或CLIP。
- 本体与command先过MLP得到256维，再与64维depth特征拼接，经过`256→256` MLP得到预测的256维`priv_est`。student actor只读取`command + proprio + priv_est`，图像不会直接拼到动作头。
- 训练目标是先用特权encoder得到`priv_feature`，用适配模块预测`priv_pred`，再用depth estimator通过MSE拟合`priv_pred`；后续用PPO继续优化student。它属于“视觉预测控制latent”，而不是“RGB encoder直接回归动作”。

对应代码：[相机创建](https://github.com/LeCAR-Lab/HDMI/blob/32282f6dcf26cae70b814d585ceb12cc38aa1b60/active_adaptation/envs/locomotion.py#L119-L159)、[depth预处理](https://github.com/LeCAR-Lab/HDMI/blob/32282f6dcf26cae70b814d585ceb12cc38aa1b60/active_adaptation/envs/mdp/observations/common.py#L255-L343)、[depth CNN及latent estimator](https://github.com/LeCAR-Lab/HDMI/blob/32282f6dcf26cae70b814d585ceb12cc38aa1b60/active_adaptation/learning/ppo/ppo_roa.py#L274-L296)、[latent/action蒸馏](https://github.com/LeCAR-Lab/HDMI/blob/32282f6dcf26cae70b814d585ceb12cc38aa1b60/active_adaptation/learning/ppo/ppo_roa.py#L507-L589)。

### 2.5 非对称视觉RL：可以不做教师动作蒸馏

学生actor仅看视觉/本体，训练时critic看真实state，直接优化任务奖励。ViBe在已有tracker上训练视觉extractor和LoRA，属于这一路线。它说明教师蒸馏不是唯一方案；但已有可靠state teacher时，先DAgger通常是更直接的实验起点。

## 3. 最贴近两个任务的论文与证据

以下均核对全文方法和部署说明。2026近期论文按预印本证据理解，不把单篇实验当成所有平台上的保证。

### DoorMan：直接学习视觉开门

**Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer**，2025，读[2512.01061v1](https://arxiv.org/html/2512.01061v1)。

- 教师：PPO，输入门/把手相对变换、接触等特权量，使用已有全身控制器。
- 学生：RGB＋本体，视觉编码器＋两层LSTM＋动作头。
- 流程：state teacher → DAgger → **GRPO对视觉学生进行仿真微调**。
- [§2.4](https://arxiv.org/html/2512.01061v1#S2.SS4)指出：学生需要自己调整机器人位置，让交互区域留在视野；状态教师不一定有这种信息获取行为。单纯模仿存在上限。
- 真机：G1，D435i仅用RGB，推门/拉门/推杆等；推理使用桌面RTX4090，不能写成全部机载。
- [§3.3](https://arxiv.org/html/2512.01061v1#S3.SS3)的50–70%蒸馏后水平、80.8–85.8%微调后结果属于仿真子任务，不能当真机成功率。

对本项目：视觉开门的核心参考。但当前背部/侧身推门与带手操作把手不同，应借训练流程，不照搬其动作空间和全身控制器。

### ViLoMan：mjlab中由门状态/参考教师蒸馏到depth学生

**ViLoMan: Learning Visual-Proprioceptive Whole-Body Loco-Manipulation Skills for Humanoid Robots**，2026，[2609.19340v1](https://arxiv.org/html/2609.19340v1)。

- 教师：参考动作跟踪器＋交互残差策略，读取门状态、接触目标和参考。
- 学生：本体历史＋连续4帧36×64深度图，CNN视觉latent＋MLP，29维关节目标；**学生不输入motion reference或真实物体状态**。
- [§3.4](https://arxiv.org/html/2609.19340v1#S3.SS4)：online DAgger，学生控制仿真，教师在同状态标注，动作按维归一化后MSE；未声称学生PPO。
- 仿真使用mjlab；深度模糊及动力学随机化。
- [§4.4](https://arxiv.org/html/2609.19340v1#S4.SS4)：G1头部D435i深度、全部机载计算，**关门**32/40成功；BC-only为15/40。这是该论文同条件对照，不是对本项目成功率的预测。

对本项目：最值得参考的紧凑depth＋DAgger实现。任务是关门，不包含开锁/转把手。去掉参考能够成功依赖其任务与训练分布，不能直接推导你的多reference sitting也可无条件删掉goal。

### LessMimic：sitting视野问题的直接负面证据

**LessMimic: Versatile Humanoid–Object Interaction with Unified Distance Field Representations**，读2026-11-07更新的[2602.21723v2](https://arxiv.org/html/2602.21723v2)；v1题名不同。

- 非视觉策略使用物体distance-field距离/方向表示，通过模仿和RL训练；视觉版本用depth history预测控制latent。
- [§3.4](https://arxiv.org/html/2602.21723v2#S3.SS4)：冻结非视觉策略为teacher，以学生rollout上的动作MSE进行DAgger。不能把该视觉训练说成论文明确使用latent MSE。
- **[§4.4及附录](https://arxiv.org/html/2602.21723v2#S4.SS4)：前视depth的SitStand在全部五种尺寸条件下为0%；增加后视相机后，平均仿真成功率为31%。** 作者指出前视相机看不到背后的pelvis-chair接触区域。
- 后视相机没有完全解决问题：各尺寸结果0/1/31/85/38%，仍存在明显观测/泛化差距。
- **[真机SitStand使用MoCap版策略](https://arxiv.org/html/2602.21723v2#S4.SS6)，视觉版真机仅评估PickUp。不能引用其真机坐下结果来证明视觉坐下已经解决。**

对本项目：应先审查接近、转身、后退、下降、坐稳全过程的视野。增加网络容量或数据量无法恢复从未看见且没有其他测量约束的目标状态。先观察再记忆、侧/后视相机、几何跟踪均为可测试设计，不是该论文已证明能解决本任务的结论。

### VisualMimic：不一定要直接输出29维关节动作

**VisualMimic: Visual Humanoid Loco-Manipulation via Motion Tracking and Generation**，2025，读[2509.20322v2](https://arxiv.org/html/2509.20322v2)，[项目](https://visualmimic.github.io/)。

- 低层：带未来参考/特权信息的motion teacher，通过DAgger变成可部署tracker。
- 高层：物体state teacher生成身体keypoints；视觉student以depth＋本体预测keypoints，仍通过DAgger训练。
- 高层视觉策略＋低层运动tracker组成闭环，真实对象信息来自深度。
- G1 D435i，80×45深度；仿真随机遮挡块和相机姿态，真实深度做时空滤波；真机任务包括搬箱、踢球、踢箱和推箱。

对本项目：如果未来希望统一很多技能，可考虑视觉高层生成motion/goal，再复用低层SuperMimic。当前先验证两个技能，没有必要立即引入额外层级。

### Reflex：保留控制器，学习视觉动态latent

**Reflex: Visually Grounded Reactive Humanoid Control**，2026，[2610.12772v1](https://arxiv.org/html/2610.12772v1)。

- 先学习接箱子的state controller，再训练8维物体动态latent接口；通过动作DAgger初始化和PPO改进。
- 随后冻结latent encoder与controller，用25帧RGB-D＋ego-motion历史训练视觉模块回归latent，另监督物体位置/速度。
- 视觉模块也进行学生rollout和latent重标注的数据聚合。渲染集中于感知阶段，控制器保持固定。
- G1真机接人抛出的箱子；论文报告2.5m距离65%、5m距离40%，每条件20次，任务与本项目不同。
- 时序历史对于推断物体运动很关键；不能用单帧位置精度替代闭环动态表现。

对本项目：可作为“尽量复用现有控制器”的具体模板。当前actor没有同样的8维latent接口，不能只替换encoder文件就完成迁移。

### ReGATE：视觉遮挡、接触与时序融合

**ReGATE: Recurrent Gated Visual Fusion for Reactive Humanoid–Object Interaction**，2026-11，[2611.02871v1](https://arxiv.org/html/2611.02871v1)。

- 特权teacher → 深度/本体student；CNN、循环视觉记忆、门控融合和interaction latent。
- 监督包括latent、物体状态/几何、图像位置、接触标签；actor有动作模仿，并进行PPO改进。
- 先让学生学会使用干净深度，再逐步加入门控、遮挡、blackout，避免过早忽略视觉。
- control50Hz、depth15Hz；真机ZED2i输入，经保持最近帧与高频本体控制配合。
- G1搬箱等交互，明确部署不接收外部mocap、object pose或sim state；仍由用户命令指定移动与接触意图，未解决完整任务规划。

对本项目：值得借鉴缺帧/时延课程和视觉失效时的记忆；不建议第一版照搬全部复杂模块。

## 4. 通用实现和近期补充

| 论文 | 已核验的实现 | 应保留的边界 |
|---|---|---|
| [DextrAH-RGB, 2412.01791v2](https://arxiv.org/html/2412.01791v2) | 保留本体/动作接口，用双目RGB替换object pose和object ID；online DAgger，动作分布匹配＋物体位置辅助监督，ResNet18＋LSTM；全仿真训练后真机抓取 | Kuka＋Allegro，桌面双相机；不是人形任务。可借蒸馏代码，不必引入其fabric动作体系 |
| [LucidSim, 2411.00083v1](https://arxiv.org/html/2411.00083v1) | RGB＋本体，先BC，再学生采样/教师重标注；生成视觉增加外观多样性，并保持几何/时序一致 | Go1跑酷。实验中增加expert-only数据很快饱和；短历史不是背对椅子长遮挡的现成解 |
| [Miki等, 2201.08117v1](https://arxiv.org/html/2201.08117v1) | 理想地形/特权teacher→有噪height samples＋本体＋gated GRU student；动作模仿和重建辅助监督 | 输入是几何地图，不是原始RGB；ANYmal真机20Hz地图、50Hz控制 |
| [RialTo, 2403.03949v3](https://arxiv.org/html/2403.03949v3) | 场景数字孪生、state RL teacher、point-cloud student DAgger，并与少量真实演示共同训练 | 不是纯仿真无真实数据；论文明确假设不存在需要主动寻找信息的部分可观测难题 |
| [VIRAL, 2511.15200v2](https://arxiv.org/html/2511.15200v2) | RGB＋本体，教师轨迹BC和学生轨迹DAgger各50%，视觉/物理随机化，真机G1搬运 | 大规模算力配置不是第一版基线的必要条件；不要把无真实微调写成完全机载 |
| [ViBe, 2609.09918v1](https://arxiv.org/html/2609.09918v1) | 冻结tracker和视觉骨干，训练extractor/LoRA，非对称PPO；有真机迁移 | **不是教师动作蒸馏**，仍使用未来reference |
| [SUGARv2, 2611.09266v1](https://arxiv.org/html/2611.09266v1) | reference-conditioned teacher→RGB student动作蒸馏→PPO＋持续教师动作监督；五类搬运/放置，G1无真机微调 | 每任务一个策略；层级WBC命令。真机每任务10次，仅近期预印本证据，不证明sitting或door |
| [DreamMimic, 2608.22278v1](https://arxiv.org/html/2608.22278v1) | depth＋segmentation→RSSM；特权状态/接触/物体辅助监督＋动作蒸馏＋3步latent预测一致性，PPO辅助 | **只有仿真及定性sim-to-sim/跨形态证据，没有真机验证**；不是可直接继承当前SuperMimic state WM的模块 |

[RMA](https://arxiv.org/abs/2107.04034)常被一起提及，但其核心是通过本体历史预测动力学latent，不是视觉对象定位；不能靠它从未接触的本体历史推断身后未知椅子位置。[LocoTransformer](https://arxiv.org/abs/2107.03996)是直接视觉PPO，也不应列成teacher–student蒸馏案例。

## 5. 对本项目的建议：先做能定位问题的最小实验

以下是基于代码和论文的工程建议，尚未实现或实测。

### 第一步：定义真机可获得的信息

先固定相机布置、内外参、可用模态、帧率、延迟范围。渲染的是安装在机器人上的相机，不是录演示视频用的全局观察视角。若允许外部固定相机，方案会更容易，但那是另一种部署约束，应明确记录。

- **Sitting：** 已知型号椅子可保留CAD geometry code，只用视觉估计pose；新椅子泛化则必须面对partial depth/cloud与完整CAD的差异。重点观测座面高度、方向、边界以及机器人相对座面的关系。
- **OpenDoor：** 需要门轴/门框约束和门面法向；`theta`是相对闭合基准的角，不是图像中的门面方向。机器人自己转身时，只看门面法向不能直接恢复theta。角速度必须依赖时序和自运动补偿。
- **后方遮挡：** 当前sitting及背部/侧身推门可能让交互区离开前视相机。考虑看椅后转身的记忆、侧/后视相机，或视觉跟踪＋本体融合；完整遮挡期间仍可能发生椅子滑移或门转动，记忆不是永远可靠的真值。

### 第二步：先保留reference与动作接口

建议第一版学生：`proprio_history + depth/RGB-D history + 当前motion goal → 29-D action`。

保留已有归一化、动作尺度、关节顺序、PD和reference调度，先只检验对象感知替换。这样能把视觉迁移的影响与运动控制重构分开测量。

“有未来参考”不等于使用未来真实状态。预先存储的轨迹可以在真机使用；但若目标进一步变成“任意场景自动找椅子、对齐、选择动作并坐下”，还需要解决任务启动、reference选择/变形和进度控制。去掉reference是另一项实验，不应由state→vision蒸馏自动推定。

### 第三步：在真实模拟器rollout上做BC / DAgger

冻结现有state teacher。用相机渲染生成学生观测，用同一状态的teacher输出监督29维动作。可先BC热启动，再提高学生采样比例，混合保留教师数据和恢复数据。

对于当前世界模型，没有相机渲染器或视觉观测头，直接把视觉CNN接进state imagination loop并不能得到正确的未来视觉训练输入。第一版用模拟器产生视觉数据，不要求改写WM或可微渲染。渲染会降低吞吐量，环境数和视觉更新频率应独立测量，而非直接照搬state训练并行数。

可选辅助头：

- sitting：椅子/座面相对位置姿态、可观测的几何信息；已知CAD可尝试对齐teacher shape latent。
- door：hinge-relative、normal、sin/cos(theta)、omega，使用与当前actor一致的坐标与单位。

这些真值只在训练提供标签。辅助估计误差小不保证任务成功，动作和闭环任务指标仍是最终依据。也不应强迫从单个局部视图精确重建任意椅子的全部背面结构。

### 第四步：只在明确瓶颈后增加复杂度

- 学生模仿误差低但任务失败：先看DAgger覆盖、长时漂移、动作/观测归一化、相机时延。
- 物体长时间看不到：增加合适的时序记忆/状态估计或调整视野；当前10帧本体历史约0.2秒，未必足够。
- 需要转头/挪身保持视野：可以借DoorMan，在仿真中对视觉学生做任务RL，并加入必要的可见性设计；教师动作本身不一定包含主动感知。
- 外观迁移失败：RGB的灯光、纹理、材质、曝光、模糊与背景随机化；depth的孔洞、边缘飞点、截断、偏差与遮挡随机化。Depth仍存在sim-to-real gap。
- 真实物理差异：继续质量、摩擦、门阻尼、关节执行、延迟等随机化；外观随机化不能修复物理误差。
- 如果已有真实场景扫描或少量真实数据：可参考RialTo的数字孪生和real/sim co-training；不必把它当作第一版的硬前提。

## 6. 建议的对照实验

| 对照 | 输入/训练 | 回答的问题 |
|---|---|---|
| Oracle teacher | 当前真实state输入 | 现有控制能力上限是什么？ |
| Corrupted-state teacher | 模拟视觉估计噪声、偏差、延迟、丢帧 | 现有控制器对预期感知误差有多敏感？ |
| Visual estimator + teacher | 视觉估计state，接原actor | 显式感知能否支撑已有控制？ |
| Visual BC | 只模仿教师数据 | 单纯离线模仿的基线 |
| Visual DAgger | 学生采样＋教师同状态标注 | 是否解决学生偏离与恢复？ |
| 可选：DAgger + RL / latent / memory | 一次增加一个因素 | 改善来自任务优化、表示还是记忆？ |

固定同一初态/对象划分，记录任务成功、跌倒、接触/穿透、轨迹误差、动作平滑性、视野缺失时长及感知到控制延迟。未见椅型与同一椅型的新位姿应分别统计。

Sitting额外分前视/前后视、短历史/长记忆、已知CAD/未知椅子；门额外分门角、初始位置、侧身遮挡、门轴和阻尼变化。先验证仿真里的真实传感器约束，再把sim-to-real下降单独量化。

推荐阅读次序：**DoorMan → ViLoMan → LessMimic → Reflex / ReGATE → DextrAH-RGB实现**。VisualMimic适合评估分层扩展，DreamMimic适合后续研究视觉世界模型，不建议作为第一版必须新增的依赖。

## 7. 检索来源与范围

- 截止日期2026-11-09。arXiv API检索：`(all:humanoid AND (all:distillation OR all:privileged) AND (all:visual OR all:vision OR all:depth)) AND submittedDate:[202301010000 TO 202611092359]`，按submittedDate降序，读取30/77条命中，再对高相关候选核对全文。这77条不是领域完整论文数量。
- OpenAlex works分别检索`humanoid visual privileged teacher student distillation`与`humanoid sitting chair depth teacher student visual`，每组20条，日期上限相同；用于补充任务相关论文。LessMimic题名以实读v2全文为准。
- 主要证据来自本文链接的arXiv全文方法、实验和附录，辅以作者项目。DextrAH另核对[官方代码](https://github.com/NVlabs/DEXTRAH/tree/ebc08edafe5f7c5ad73dcb25ffa15e8d0353df50)的student rollout/teacher labeling与动作损失实现。
- Semantic Scholar对部分经典论文查询遇429，改用官方arXiv/项目来源；偶发TLS失败经重试取得全文。未把请求失败当作论文不存在。
- 未复现论文、未审计全部开源代码、未进行真机试验；数值均保留原论文的任务和评估范围。近期结果主要用于设计选择，不能直接外推本项目成功率。
