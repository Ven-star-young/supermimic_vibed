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
