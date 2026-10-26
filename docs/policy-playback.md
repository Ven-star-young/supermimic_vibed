# 本地策略回放

从项目根目录启动实时 MuJoCo 仿真窗口：

```bash
cd /home/frank/Robotics/Paper/Humanoid/supermimic_vibed
./play_policy.sh logs/2026-10-26_14-05-17/model_8000.pt
```

后续下载的模型保存在 `logs/` 下，替换上述 checkpoint 路径即可：

```bash
./play_policy.sh /绝对路径/model_4000.pt
```

启动脚本使用本项目的 `.venv` 和源码，不依赖 Codex 工作目录。
也可以从其它目录调用脚本的绝对路径；传入的相对路径始终相对于调用时的目录。
若重新安装环境，在项目根目录执行 `uv sync --extra motion`。

默认加载 Sitting 任务，从当前椅子对应的动作中随机选一段，再随机选非末帧起点。
机器人和椅子同步初始化到该参考帧，随后由策略输出动作、物理仿真推进。
回合结束自动重新采样；窗口一直运行，关闭窗口或 Ctrl+C 结束。
默认采用评估参数：椅子 10 kg、G1–椅子滑动摩擦 1.0，关闭训练噪声。

| 操作 | 功能 |
|---|---|
| 空格 | 暂停 / 继续 |
| Enter | 当前椅子重新随机选 motion 和起点 |
| 右箭头 | 暂停时单步 |
| `-` / `=` | 减速 / 加速 |
| 鼠标左键拖动 | 旋转视角 |
| 鼠标右键拖动 | 平移视角 |
| 滚轮 | 缩放 |

常用选项：

```bash
# 从随机抽中动作的第 0 帧开始；固定种子便于重复观察
./play_policy.sh /path/to/model.pt --start first --seed 42

# 使用训练时的域随机化、重置噪声和观测噪声
./play_policy.sh /path/to/model.pt --training-conditions

# 换球形舱椅：自动筛选相同几何和缩放的配套 motion
./play_policy.sh /path/to/model.pt \
  --object-motion-file data/motions/sitting/sitting__ball_pod_chair_white_fiberglass_retro_futuristic__000.npz \
  --start first

# 使用新的标准 motion 目录，并指定该目录内的椅子
./play_policy.sh /path/to/model.pt \
  --motion-dir /path/to/motions \
  --object-motion-file /path/to/motions/chair_motion.npz

# 浏览器实时可视化，访问启动日志中的地址
./play_policy.sh /path/to/model.pt --viewer viser

# 可选：导出视频而非实时窗口（需要系统 ffmpeg）
./play_policy.sh /path/to/model.pt --episodes 8 --video logs/policy_random.mp4
```

默认不会在回合间更换椅子几何；更换椅子请用 `--object-motion-file` 重新启动。
新椅子必须具备碰撞缓存，缺失时按报错提示运行 `tools/prepare_collisions.py`。
指定新 motion 目录时应同时指定椅子，否则会使用该目录排序后的第一个文件。

使用包含 actor 权重的完整训练 checkpoint（例如 `model_2000.pt`）；
`world_models/` 下的独立世界模型不能驱动策略回放。
回放自动读取 checkpoint 的历史窗口、目标窗口及推理精度，并校验任务结构。
其它兼容任务可用 `--task TASK_ID` 显式选择；新数据应使用已有转换器输出的标准 NPZ 格式。

实现位于 `supermimic/scripts/play_policy.py`。原始数据的纯运动学回放仍使用
`tools/replay_motion.py`。随机中途起点到达 motion 末尾不能当作完整坐下成功率。
