# 本地策略部署与 reference 验证

统一入口：`tools/deploy_policy.py`。它使用现有mjlab物理环境执行Actor生成的29维关节
位置动作，保留50 Hz控制与原任务终止规则。它是仿真策略运行，不是运动学reference
回放，也不向实机发送命令。单次只驻留一个环境，适合本地8 GB GPU。

## 已下载模型

`logs/cuhk_reference_deployment/` 含CUHK reference训练的 `model_2000.pt`（约173 MiB）、
对应 `split.json`、`experiment.json` 和下载时的 `train_log.csv`。
SHA-256为 `959306a101214ef3e9ba86402c8d67b9decaad789cf0bb788878a290530ce4a1`，
与服务器一致。选择清单共83个训练reference、22个测试reference，checkpoint实际访问80把
训练椅子。缺失的90份所选椅子碰撞缓存也已下载，本地105种所选几何缓存完整。
服务器训练继续运行，下载的是第2000次快照，不会随服务器后续checkpoint自动变化。

默认checkpoint及同目录split无需传参。更换模型时显式指定：

```bash
uv run python tools/deploy_policy.py \
  --checkpoint /path/to/model_N.pt --split /path/to/training_split.json \
  --data-root data/motions/sitting
```

checkpoint先加载到CPU；GPU只创建Actor（包括共享几何编码器），不创建WM、Critic、
Replay或优化器。运行使用FP32推理，全部参数及归一化buffer冻结，结束后验证摘要未变。
完整训练checkpoint的几何库只用于来源校验，推理使用当前场景的局部几何ID及原始点云。

## 三种使用方式

默认候选是训练清单，按文件名排序，`--index` 与 `--list` 的编号一致：

```bash
uv run python tools/deploy_policy.py --list
uv run python tools/deploy_policy.py --index 0 --init clean --episodes 0
uv run python tools/deploy_policy.py --index 0 --init perturbed --episodes 5 --seed 0
```

`clean` 从所选reference第一帧准确的机器人/椅子位姿和速度开始；“干净”不是零速度，
也不是从统一站立姿态开始。参考中已有速度时会照样初始化。

`perturbed` 保持同一reference及第一帧，给机器人初态叠加独立均匀扰动：

| 量 | 默认半宽（正负范围） |
|---|---|
| 根位置 x/y/z | 0.05 / 0.05 / 0.01 m |
| 根旋转增量 x/y/z | 0.1 / 0.1 / 0.2 rad |
| 根线速度 x/y/z | 0.5 / 0.5 / 0.2 m/s |
| 根角速度 x/y/z | 0.52 / 0.52 / 0.78 rad/s |
| 29个关节位置 | 0.1 rad，裁剪到关节范围 |

`--noise-scale 0.5` 将上述范围减半，0等同干净初始化。关节速度与椅子状态仍取reference。
扰动在重置命令中、观测历史生成前施加，历史因此匹配实际初态。两种模式都关闭观测和
动作噪声，固定椅子10 kg、G1—椅子摩擦1.0，机器人校准/COM偏移为0；保持变量可比较。
`--seed` 控制随机序列，同一设置重跑可复现初态；同一进程每个新回合采样新的扰动。

如果还要随机reference起始时刻，另加 `--random-start`。它与状态扰动是独立选择；
`--start-frame 30` 则固定从重采样后50 Hz的第30帧开始，不能与random-start同时指定。
末帧不能作为起点，因为必须至少剩下一步物理转移。

以后挑选reference直接用同一入口：

```bash
# 一个标准NPZ，或 --reference 文件名（相对data-root）。
uv run python tools/deploy_policy.py --reference /path/to/selected.npz --init clean

# 当前人工选择记录；index作用于该清单。
uv run python tools/deploy_policy.py \
  --references data/selections/sitting_back_references.json \
  --selection data/selections/sitting_back.json --data-root data/motions/sitting \
  --index 3 --init perturbed

# 导出reference目录里的manifest.json。
uv run python tools/deploy_policy.py --references /path/to/references-0001/manifest.json --list

# 原训练划分中保留的测试reference。
uv run python tools/deploy_policy.py --partition test --index 0 --init clean
```

始终传模型原来的训练split用于协议校验；新reference通过 `--reference` 或 `--references`
指定，不要用新选择清单冒充原训练split。未知几何可以执行，但需要完整标准NPZ对象网格
及对应碰撞缓存；缺失时脚本报错，离线用 `tools/prepare_collisions.py` 准备。加载沿用标准
数据检查、关节映射和50 Hz重采样。已有清单同名reference发生内容变化会被拒绝。

报告分别标注reference是否来自原train/test清单、几何是否来自train/test，以及该几何
是否已被checkpoint实际访问。使用新动作但相同椅子不等于未见几何测试。

## 窗口与批量验证

默认打开原生MuJoCo窗口。一次回合结束就关闭；`--episodes 5` 连续5回合，
`--episodes 0` 循环直到关窗。原生viewer也支持鼠标施力，可观察执行中的抗扰动表现。
浏览器方式：

```bash
uv run python tools/deploy_policy.py --viewer viser --port 8087 --episodes 0
```

打开 `http://127.0.0.1:8087`。服务仅绑定本机，支持暂停、单步、重置和相机控制。
浏览器标签关闭不会自动结束服务；在运行终端Ctrl+C结束。

无窗口运行或批量验证：

```bash
uv run python tools/deploy_policy.py --viewer none --init clean \
  --output results/analysis/deploy_clean.json
uv run python tools/deploy_policy.py --viewer none --init perturbed --episodes 10 \
  --output results/analysis/deploy_perturbed.json
uv run python tools/deploy_policy.py --viewer none --all --partition train --episodes 1 \
  --output results/analysis/deploy_all_train.json
```

`--all` 逐段重建物理场景，避免不同椅子同时驻留；不会把其它片段混入所选reference。
`--max-steps 100` 可限制单回合验证长度。报告包含初始状态、seed、执行步数、归一化回报、
结束原因及参数冻结检查。`reference_complete`仅表示未触发原tracking失败并到达参考末尾，
不是新的“稳定坐下成功率”；人工截断为 `step_limit`，不会计作完成。

## 本次验证

模型2000、默认训练reference（Adirondack chair）、seed0：干净初态完成248步/4.96秒，
归一化回报约0.685；扰动初态3回合分别162步失败、173步失败、248步完成。
首个test reference的干净初态也完成248步，回报约0.686。这些是流程与单例结果，
不是83/22总体性能。产物位于 `logs/cuhk_reference_deployment/`。

已验证原生窗口、Viser浏览器运行及回合重置。新增测试覆盖reference哈希与选择来源、
checkpoint维度、真实CUDA环境中的干净/扰动初态、历史一致性、固定物性、随机相位、
seed复现、边界帧拒绝以及连续回合重置。完整回归236项通过。
