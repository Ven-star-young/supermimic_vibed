# 坐椅子任务：数据、训练与部署

入口统一为 `supermimic.scripts.train_sitting` 和 `supermimic.scripts.play_policy`。每个仿真环境只有 G1 和一把椅子；训练批次同时覆盖训练集中的各把椅子。这里的多椅子指不同环境使用不同资产。

## 准备 reference

原始数据继续通过 `motion_data/converters/` 转换为统一 NPZ。先用纯运动学回放筛选动作，再为每把椅子挑选一个 reference：

```bash
uv run python -m tools.select_motions
uv run python -m tools.review_chairs
uv run python -m tools.prepare_reference_training
```

默认读取 `data/selections/sitting_back.json` 和 `data/selections/sitting_back_references.json`，输出 `data/splits/sitting_back_references_seed0.json`。该 manifest 保持原 80/20 划分，只保留最终选择的 reference。相同网格的不同缩放版本归入同一组，避免同一椅子出现在训练集和测试集。不同的数据目录可以用 `--data-root` 指定；内容哈希仍须匹配。

数据第一次划分使用 `uv run python -m tools.split_chairs --help` 查看参数；已有划分不要覆盖。参考选择改变后应输出新的 split，避免对原实验的训练/测试成员做隐式修改。

## 训练

先检查训练集和碰撞缓存：

```bash
uv run python -m supermimic.scripts.train_sitting --check
```

单卡：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
uv run python -m supermimic.scripts.train_sitting \
  --num-envs 128 --iterations 10000 --run-dir logs/sitting_v1
```

双卡：

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 \
uv run python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m supermimic.scripts.train_sitting \
  --num-envs 128 --iterations 10000 --run-dir logs/sitting_v1
```

`--num-envs` 是每卡环境数；WM/Actor batch 也按每卡计算。环境数必须足以在每张 GPU 上覆盖训练椅子的各个几何/缩放变体。所有椅子持续参与收集和学习，没有“每椅训练 50 次后切换”的阶段。

训练参数定义在 `supermimic/rl/config.py`；用 `--agent-json path/to/agent.json` 覆写，例如：

```json
{"wm_batch": 128, "pi_batch": 128, "buffer_steps": 256, "save_interval": 200, "log_every": 10}
```

恢复训练时追加 `--resume logs/sitting_v1/model_2000.pt`，保持相同 split、椅子碰撞配置和模型维度。`--iterations` 表示本次额外执行的迭代数。训练目录记录 `split.json`、`experiment.json`、日志和 checkpoint；不应覆盖已有实验目录。

## 部署与验证

只加载新分支训练的 Actor 和共享点云编码器；旧多物体 checkpoint 结构不兼容。部署以 checkpoint 旁的 `split.json` 为数据清单，可用 `--split` 显式覆盖路径，但其 fingerprint 必须匹配。

```bash
# 列出训练 reference，然后选择第 0 个。
uv run python -m supermimic.scripts.play_policy \
  --checkpoint logs/sitting_v1/model_2000.pt --list
uv run python -m supermimic.scripts.play_policy \
  --checkpoint logs/sitting_v1/model_2000.pt --index 0 --init clean

# 从受扰动的 reference 状态部署，运行 5 个 episode。
uv run python -m supermimic.scripts.play_policy \
  --checkpoint logs/sitting_v1/model_2000.pt --init perturbed --episodes 5

# 冻结策略，在全部测试椅子上运行，保存逐 reference 结果。
uv run python -m supermimic.scripts.play_policy \
  --checkpoint logs/sitting_v1/model_2000.pt --partition test --all \
  --viewer none --episodes 3 --output results/sitting_test.json
```

`clean` 精确恢复 reference 的位置、朝向和速度。`perturbed` 只额外扰动 G1 初始状态；椅子质量固定 10 kg，G1–椅子摩擦固定 1.0，不开启训练时传感器/动作噪声。`--random-start` 独立控制 reference 起始帧；`--start-frame` 则指定重采样后 50 Hz 的帧号。

`--reference path/to/clip.npz` 指定单段动作；`--references path/to/choices.json` 或导出的 `manifest.json` 使用后续挑选的 reference。新椅子从自己的碰撞几何采样点云，通过已经训练的编码器获得特征；不会把训练资产的几何 ID 强行赋给它。

可视化可选 `--viewer native` 或 `--viewer viser --port 8087`；`--episodes 0` 持续回放直到关闭窗口。无窗口模式必须给出有限 episode 数。`--max-steps` 是额外截止条件；报告区分 reference 完成、跟踪失败和步数截止。完成 reference 尚不能单独证明稳定坐下。

部署过程检查策略参数与归一化统计不变，不分配 WM、Critic 或训练 ReplayBuffer。


## 完整 reference 与稳定接触评测

`f37c206` 的模型格式保持兼容。部署仍只运行冻结 Actor 和点云编码器，训练代码、
奖励、终止配置不因增加指标而改变。正式评测使用 `--evaluate`，始终从第0帧开始，
依次评测所选分组的全部 reference；默认同时运行 clean 和 perturbed。

```bash
.venv/bin/python -m supermimic.scripts.play_policy \
  --checkpoint logs/sitting_task_2gpu/model_10000.pt \
  --evaluate --partition all --seeds 0 1 2 3 4 \
  --output results/evaluation/model_10000.json
```

`--seeds` 显式决定每种条件的回合数量，覆盖 `--episodes`。每个 reference 使用同样的
seed列表，每次reset重新设定该seed，因此上一个回合执行多少步不会影响下一回合的
初始化。扰动范围沿用部署初始化：位置±(0.05,0.05,0.01)m、旋转向量±(0.1,0.1,0.2)rad、
线速度±(0.5,0.5,0.2)m/s、角速度±(0.52,0.52,0.78)rad/s、关节位置±0.1rad，
统一乘 `--noise-scale`，关节位置裁到限位。关节速度和椅子状态仍为reference值。
这是**初始化扰动**，不包含执行中的外部推力。clean重复seed主要检验重复性（GPU物理不承诺逐位一致），
不能当作独立物理变化样本或多次独立训练。

两个成功指标使用同一个预先声明的试验列表：

- **Full-reference success rate**：从第0帧一直执行到最后一帧；没有跟踪失败、
  超时、提前步数截断或手动中断。最后一帧若同时触发失败，仍算失败。随机起始帧的
  演示只记录segment完成，不计完整reference成功。
- **Stable-contact success rate**：先满足完整reference成功，再满足下面的严格
  Sitting判据。不是任意身体部位碰到椅子就成功。

默认稳定窗口固定为reference的**最后1秒**，不事后搜索表现最好的时段。按200Hz
实际物理步采样，至少95%的窗口采样同时满足全部条件，且连续不满足时间不超过40ms：

| 条件 | 默认阈值 |
|---|---|
| 支撑身体部位 | 骨盆、髋部和上段大腿；排除脚、手、躯干/背部 |
| 支撑接触法向 | 对机器人朝上，与竖直向上夹角≤45° |
| 有效向上承重 | ≥机器人总重的20% |
| 有效接触点切向滑移速度 | ≤0.10m/s |
| 椅子线速度／角速度 | ≤0.05m/s、≤0.15rad/s |
| 骨盆相对椅子的reference位置误差 | ≤0.10m |
| G1 root相对reference的旋转误差 | ≤20° |

默认支撑区域是reference骨盆下方2.5–25cm、水平半径22cm的几何区域，随reference
和椅子局部坐标对应。**它不是人工验证的语义座面标注**；这是一个可复现、严格的
稳定承重接触判据。脚踩、背靠或短暂擦碰不能单独通过此指标。特定椅型可提供人工
检查的chair-local包围盒替代区域；不同reference可预先指定各自坐姿窗口。

`--metrics-json metrics.json` 支持全局默认与按文件名/SHA256覆写，SHA256优先。
所有最终阈值、参考区域来源都写入报告。例：

```json
{
  "defaults": {"stable_window_s": 1.0, "min_support_bodyweight": 0.2},
  "references": {
    "sitting__example__000.npz": {"window_end_s": 4.0}
  }
}
```

显式座面边界使用 `seat_bounds_min` 和 `seat_bounds_max` 两个三维向量，单位米，
位于已应用资产缩放的椅子局部坐标系。两者必须同时提供。阈值和窗口应在比较模型前
固定；调整后必须作为不同评测协议保存。短于指定窗口的reference、过晚的随机起点
会标记为不可评价，不能靠丢弃这些样本抬高成功率。

报告结构为 `references[].conditions.clean/perturbed.episodes[]`，每回合记录seed、
完整初始化状态、结束原因、两种成功、评价资格、固定参考相位下的关节／机器人／
椅子误差（mean/max/final）、接触承重、滑移、稳定时长与失败判据。误差只在实际
执行过的前缀计算，明确标注其范围；固定reference长度归一化回报在失败后相当于补零。

`summary.references` 给逐reference/条件成功数、计划数、未执行数、无效数、
成功率及coverage。`summary.task` 分train/test/external和条件汇总：先合并同一
椅型的references，再对椅型等权平均，另给所有试验的合并率。主成功率分母为
**计划试验数**；中断或不可评价结果不会从分母消失。只有coverage完整且没有无效
试验时，主率才是完整执行该协议的策略成功率；否则应同时报告coverage和缺失数。

输出会在每回合后原子保存，完整计划在第一次仿真前就写入。已有输出文件不会被
覆盖。建议保留checkpoint哈希、split fingerprint、协议、原始逐回合数据一起比较。
接触力在每次实际Warp求解后立即捕获，避免mjlab后续forward重算导致力与时刻不一致。

## 交互式部署页面

```bash
.venv/bin/python -m supermimic.scripts.play_policy \
  --checkpoint logs/sitting_task_2gpu/model_10000.pt \
  --browse --partition all --port 8088 \
  --report results/evaluation/model_10000.json
```

打开 `http://127.0.0.1:8088`。`--report`可省略；它只展示已有评测结果，不自动开始
完整数据集评测。页面提供reference搜索/分组、初始化扰动开关、seed和扰动倍率，
并显示Viser中的策略执行、逐回合成功与失败原因、Sitting任务误差及稳定接触阈值。
开始/重播使用当前选择；停止会结束子进程并保留已完成和中断回合的结果。

页面使用8088端口，Viser使用8089端口，仅监听localhost。切换椅子时先等待旧进程退出，
再创建下一场景。回放结果存于checkpoint旁的 `policy_browser/`。可视化运行允许人工
交互，属于探索性检查；正式成功率使用上述headless `--evaluate`命令。

如果页面运行在CUHK，用SSH转发两个端口后在本机浏览器打开同一地址：

```bash
ssh -N -L 8088:127.0.0.1:8088 -L 8089:127.0.0.1:8089 CUHK
```
