# 原始动作转换

每种数据源一个脚本，输出统一 NPZ；训练和回放只处理转换后的格式。

```bash
uv run --extra motion python -m motion_data.converters.grail data/grail/sitting data/motions/sitting
uv run python -m motion_data.converters.lafan1 data/lafan1_g1 data/motions/lafan1 --fps 30
uv run python -m motion_data.converters.gmr data/motions/open_door data/motions/open_door_prepared
```

转换器默认跳过已有文件，`--overwrite` 重新转换；GMR只跳过来源哈希相同的完整输出对。
同名源数据/XML已更新或旧输出没有来源哈希时，GMR会要求显式 `--overwrite`，避免沿用旧prior。
GRAIL 额外支持
`--object-usd-dir PATH`；LAFAN1 的 `--fps` 指原始 CSV 的实际采样率。

GMR 接受一个 NPZ 或包含同名 NPZ/XML 的目录，输出到独立目录，禁止覆盖来源。
当前输入是 GVHMR → GMR 的 G1 背部开门数据，已使用米、弧度、Z-up 和 wxyz；
适配器保留机器人/门轨迹、源 FPS、门转轴方向、坐标系和全部诊断字段，
校验门角度范围以及 NPZ 中的铰链位置、门板尺寸、初始方向与 XML 是否一致。
输出记录 `source_name`、`source_sha256` 和 `source_xml_sha256`。
`door_contact_mask` 是几何诊断，不是物理接触标签。训练时使用 `DoorMotionLib`
在机器人同一时钟上重采样门角度并重新计算门角速度，形成每帧 75 维 future goal。

新增来源时，在此目录添加 `<source>.py`，完成来源解析、单位与坐标系转换、
四元数顺序转换及目标机器人关节映射，再调用 `motion_data.save_motion(path, clip)`。
它会校验并原子写入文件。所需字段和可选物体字段见
[统一格式](../../README.md#统一动作数据与纯运动学回放)；所有姿态使用米、弧度、
Z 向上世界坐标和 wxyz 四元数，`joint_names` 与 `joint_pos` 的列一一对应。
保留真实采样率；50 Hz 重采样和速度计算由 `MotionLib` 统一完成。

当前训练要求数据已重定向到 G1 的完整 29 个关节，只调整列顺序不等于机器人重定向。
新来源满足该契约后，无需修改 `MotionLib`、策略、奖励或回放器。物体姿态和网格可保存在
NPZ 中供回放；它们目前不会自动加入训练场景或提供椅子碰撞支撑。
