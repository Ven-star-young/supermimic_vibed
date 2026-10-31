# SuperMimic：按任务类别训练策略和世界模型

**Unofficial** reproduction of *SuperMimic: Scaling Humanoid Whole-Body Control with One
Differentiable World Model* ([arXiv:2610.28480](https://arxiv.org/abs/2610.28480)).
Original implementation: **Chong Zhang** (ETH Zurich). Code license: [MIT](LICENSE).

这个分支按任务类别组织网络。Tracking、Sitting、OpenDoor 和 OpenDoorSide 各自拥有 Actor、WM、Critic；
共同复用真实 Replay、WM 拟合、可微分想象和 Critic 更新代码。Sitting 的目标是
学习该类任务中的机器人–椅子动力学，帮助训练策略；收敛和未见椅型表现需要实验评价。

仿真使用 **mjlab 1.6.0 + MuJoCo-Warp**，控制50Hz、物理200Hz。
历史 Isaac Lab 训练结果见 [MILESTONES.md](MILESTONES.md)，不代表此分支的模型性能。

## 网络与代码

| 任务 | Actor 单帧动态观测 | Actor 解码器 | WM | Critic |
|---|---:|---:|---:|---:|
| Tracking | 93 | 962→29 | 133→70 | 914→1 |
| Sitting | 102 | 1116→29 | 224→82 | 1101→1 |
| OpenDoor（背部跟踪） | 102 | 1052→29 | 152→72 | 955→1 |
| OpenDoorSide（右侧跟踪） | 102 | 1052→29 | 152→72 | 955→1 |

默认历史和未来参考各10帧。Sitting 每个环境只有一把椅子；不同环境可以使用
不同椅型，采用固定平衡分配，不周期性关闭重建场景。1024×6的资产局部点云编码为
64维，Actor/WM/Critic共用同一编码器，只有WM更新其参数。没有通用物体槽位、
物体mask或关系注意力；参考轨迹和奖励不进入WM。

- [模型维度、坐标系和优化路径](docs/task-specific-sitting-design.md)
- [reference选择、训练配置、多卡、部署](docs/sitting-workflow.md)
- [OpenDoor数据适配、统一状态奖励与训练](docs/open-door-workflow.md)
- [OpenDoorSide独立配置、右侧接触奖励与训练](docs/open-door-side-workflow.md)
- [数据源转换、纯运动学回放、碰撞资产](docs/motion-and-assets.md)
- [实现契约](CONTEXT.md)

```text
motion_data/                  标准NPZ、原始数据转换、碰撞缓存、reference和split
supermimic/assets/g1/          G1资产与FK
supermimic/mdp/                共用机器人状态、动作、reward、reset
supermimic/rl/                 训练循环、Replay、多卡、归一化与几何组件
supermimic/tasks/g1_tracking/  Tracking网络与任务
supermimic/tasks/sitting/      Sitting网络、动力学、场景和混合椅型实验
supermimic/tasks/open_door/    G1＋被动铰链轨迹跟踪，复用Tracking网络与Runner
supermimic/tasks/open_door_side/ 右侧躯干／肩／前臂开门，独立奖励和checkpoint
supermimic/scripts/            train_sitting / play_policy / evaluate
```

新任务在自己的目录定义状态、观测、参考、网络和奖励，runner通过任务接口调用；
不要求它与Sitting共享参数或输入布局。旧多实体网络和跨任务Replay转移保留在
`codex/shared-entity-world-model` 分支。旧checkpoint不转换为新网络。

## 安装

需要Linux、uv和支持所安装PyTorch/Warp版本的NVIDIA驱动。

```bash
tools/setup.sh
tools/download_data.sh
```

下载器会记录不可变数据版本。G1重定向LAFAN1数据来自
[lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
（动作：Ubisoft La Forge CC BY-NC-ND 4.0；机器人资产：Unitree BSD-3-Clause）。
已有 `data/motions/sitting/` 和 `data/collisions/chairs/` 可直接使用，无需重新转换。

## 坐椅子训练

先检查所选reference和碰撞缓存，随后启动训练：

```bash
.venv/bin/python -m supermimic.scripts.train_sitting --check

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
.venv/bin/python -m supermimic.scripts.train_sitting \
  --split data/splits/sitting_back_references_seed0.json \
  --num-envs 128 --iterations 10000 --run-dir logs/sitting_task
```

双卡使用同一个入口，环境数和batch都是**每卡**设置：

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 \
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m supermimic.scripts.train_sitting \
  --split data/splits/sitting_back_references_seed0.json \
  --num-envs 128 --iterations 10000 --run-dir logs/sitting_task_2gpu
```

每卡环境数至少覆盖训练split的椅型数。现有选择为83把训练椅、22把测试椅。
默认WM/Actor batch128、Replay256步、前500迭代仅训练WM、每200迭代保存。
用 `--agent-json /path/to/config.json` 覆盖学习参数；用
`--resume /path/to/model_N.pt --iterations M` 追加M次迭代。显存随椅子碰撞复杂度、
环境数和batch变化；上述环境数不承诺适配所有GPU和资产组合。

训练目录保存完整checkpoint、split.json、experiment.json和train_log.csv。
恢复训练检查网络schema、训练split、几何和碰撞配置；Replay重新填充。
物理随机化保持椅子质量5–15kg、G1–椅子摩擦0.3–1.2，评估固定10kg和1.0。

## 背部开门跟踪

已放入 `data/motions/open_door/` 的 GMR NPZ/XML 由已有来源转换目录处理：

```bash
.venv/bin/python -m motion_data.converters.gmr \
  data/motions/open_door data/motions/open_door_prepared

./run_open_door.sh --check
CUDA_VISIBLE_DEVICES=1,2 ./run_open_door.sh --config configs/open_door.json
```

训练参数统一放在 [`configs/open_door.json`](configs/open_door.json)，按 `training` / `task`
组织。直接修改 `training.num_envs`、`wm_batch`、`pi_batch`、`iterations`；环境数和batch
均为每卡数值。奖励、RSI、初始化扰动、终止和门物性随机化在 `task` 中。
默认每卡1024环境、WM/Actor batch各512；`training.log_root` 为 `logs/rsl_rl`。
`--check` 只解析并打印最终配置，不启动仿真；运行时仍使用原生mjlab入口和现有Runner。

每帧未来目标为机器人73＋门角/角速度2，共75维；没有几何编码器。
真实与想象使用同一个state-based reward，MuJoCo接触只用于评估。
原始参考含穿模，保留源数据并默认使用穿透惩罚；硬穿透终止可配置。
完整命令、维度和奖励公式见[OpenDoor工作流](docs/open-door-workflow.md)。

## 右侧开门跟踪

独立任务 `Mjlab-SuperMimic-OpenDoorSide-Unitree-G1`，参数放在
[`configs/open_door_right_side.json`](configs/open_door_right_side.json)。默认只选video6，
允许右侧躯干、右肩和右前臂参与推门；参考接近程度控制接触奖励，合法部位的深穿透也受惩罚。

```bash
./run_open_door_side.sh --check
CUDA_VISIBLE_DEVICES=1,2 ./run_open_door_side.sh
```

与背部任务共享网络实现、RSI、门物性随机化和Runner，但独立训练参数与checkpoint。
输出在 `logs/rsl_rl/supermimic_open_door_side/`，不兼容背部任务直接续训。
侧面JSON通过 `training.wm_checkpoint` 只导入背部第10000轮WM及归一化统计，
Actor/Critic和优化器从头初始化。`wm_only_iters=0`，从第1轮共同训练三者。
已有侧面策略要减少脚贴门，使用 `configs/open_door_right_side_finetune.json`：
1cm非法区域避让，载入侧面Actor+WM，重置Critic/优化器；默认2000轮、学习率5e-5。
轨迹须包含 `door_contact_region="right_side"`；混入背部数据会报错。
使用环境变量指定服务器数据时，应给侧面单个NPZ路径。详见[侧面工作流](docs/open-door-side-workflow.md)。

## 部署与验证

```bash
.venv/bin/python -m supermimic.scripts.play_policy \
  --checkpoint logs/sitting_task/model_200.pt --partition train --index 0

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=4 .venv/bin/python -m pytest -q
```

部署只创建冻结Actor，保留当前椅子的点云；支持clean/perturbed初始化、随机参考帧、
所选reference和未见椅型。`--evaluate --partition all --seeds 0 1 2 3 4`生成完整执行与
严格稳定接触成功率；`--browse --port 8088`打开可切换reference及扰动的页面。
具体参数、成功判据和输出说明见[工作流](docs/sitting-workflow.md)。
回归覆盖状态/参考坐标、梯度、优化器归属、真实接触与reset、混合椅型、保存续训及
多进程同步。通过这些检查说明训练链路可运行，不代表已经得到收敛策略。

原Tracking仍使用原生mjlab入口：

```bash
.venv/bin/train Mjlab-SuperMimic-Tracking-Unitree-G1 \
  --env.commands.motion.motion-dir data/motions/lafan1 --gpu-ids '[0]'
```
