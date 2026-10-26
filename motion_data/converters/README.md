# 原始动作转换

每种数据源一个脚本，输出统一 NPZ；训练和回放只处理转换后的格式。

```bash
uv run --extra motion python -m motion_data.converters.grail data/grail/sitting data/motions/sitting
uv run python -m motion_data.converters.lafan1 data/lafan1_g1 data/motions/lafan1 --fps 30
```

两者均默认跳过已有文件，`--overwrite` 重新转换。GRAIL 额外支持
`--object-usd-dir PATH`；LAFAN1 的 `--fps` 指原始 CSV 的实际采样率。

新增来源时，在此目录添加 `<source>.py`，完成来源解析、单位与坐标系转换、
四元数顺序转换及目标机器人关节映射，再调用 `motion_data.save_motion(path, clip)`。
它会校验并原子写入文件。所需字段和可选物体字段见
[统一格式](../../README.md#统一动作数据与纯运动学回放)；所有姿态使用米、弧度、
Z 向上世界坐标和 wxyz 四元数，`joint_names` 与 `joint_pos` 的列一一对应。
保留真实采样率；50 Hz 重采样和速度计算由 `MotionLib` 统一完成。

当前训练要求数据已重定向到 G1 的完整 29 个关节，只调整列顺序不等于机器人重定向。
新来源满足该契约后，无需修改 `MotionLib`、策略、奖励或回放器。物体姿态和网格可保存在
NPZ 中供回放；它们目前不会自动加入训练场景或提供椅子碰撞支撑。
