# G1 触觉传感器与部署查看器

Sitting、OpenDoor 和 OpenDoorSide 默认注册 `tactile` 传感器，查询 464 个机器人触点到
当前椅子或门的表面距离，再按原成员关系汇总为 112 通道。混合椅型场景按环境查询各自物体。
触觉按需读取，现有 Actor、WM、Critic 输入和 checkpoint 维度不变。
布局来源、区域覆盖、原文件哈希及腕部模型差异见[资产说明](../supermimic/assets/g1/tactile/README.md)。

## 部署与显示

在仓库根目录运行，浏览器打开终端打印的地址（默认 `http://127.0.0.1:8089`）：

```bash
.venv/bin/python tools/play_tactile.py --task sitting
.venv/bin/python tools/play_tactile.py --task open_door
.venv/bin/python tools/play_tactile.py --task open_door_side
```

每次运行一个任务，需具备对应模型、动作和碰撞资产。工具加载冻结 Actor，
在原物理场景中显示机器人、触点及通道热力图。

| 任务 | 默认 checkpoint |
| --- | --- |
| Sitting | `logs/cuhk_sitting_deployment/model_10000.pt` |
| OpenDoor | `logs/wm_transfer/back_model_10000.pt` |
| OpenDoorSide | `logs/side_policy_playback/2026-11-01_01-14-42_fresh_ac_model10000/model_10000_actor.pt` |

Sitting 默认读取模型旁的 `split.json`，选取 `partition=all` 排序后的第 0 条动作；
可用 `--partition train` / `test`、`--index N` 或 `--reference /path/to/clip.npz` 选择。
OpenDoor 和 OpenDoorSide 分别默认使用 `data/motions/open_door_prepared/` 下的
`video1_g1_door_video45_centered_asset.npz` 和 `video6_g1_door_video45_right_side_asset.npz`；
指定 `--reference` 时需有同名配套 XML。

`--checkpoint` 指定任务对应的模型。自定义 Sitting 模型还需匹配训练时的 `--split`；
`--data-root` 可重定位动作目录，加载时仍核验 reference 哈希和共享几何编码器。
`--layout inner_arms_v1` 切换历史内侧布局，默认是 `original_outer`；`--port` 更换端口。

```bash
.venv/bin/python tools/play_tactile.py --task sitting \
  --checkpoint /path/to/model.pt --split /path/to/split.json \
  --data-root /path/to/motions --index 0 --layout inner_arms_v1 --port 8090

.venv/bin/python tools/play_tactile.py --task open_door \
  --counts-full-scale-mm 10 --counts-exponent 1 --paused
```

页面的 `Play / Pause`、`Step`、`Reset` 分别控制播放、单步和从动作起点重置。
`Display → Loop policy` 默认开启，动作结束或回合终止后停留一秒再重播；关闭时停在终态。
`--paused` 暂停启动；终态下再次播放或单步会开启新回合。

`Display → Output` 提供以下模式，颜色从冷到暖表示响应增大或距离减小：

| 模式 | 格内数值 |
| --- | --- |
| `Counts (0-255)`（默认） | 未标定的合成 counts，固定 0–255 色标 |
| `Distance (mm)` | 通道成员的最小无符号表面距离 |
| `Contact` | 几何接触的 0 / 1 |

白框表示接触，青框表示选中通道，灰色及 `--` 表示查询无效。
`Channel inspection → Show raw channel IDs` 将格内标签改为原始帧编号，颜色仍显示所选输出；
`Highlight channel` 高亮机器人上对应的成员点。`Show 464 points` 控制全体点标记，
标记仅在显示时沿法向抬高 1 mm，测量位置不变。

`Counts full-scale depth (mm)` 默认 10 mm，越小越敏感；`Counts exponent` 默认 1，
小于 1 提高低响应，大于 1 压低低响应。两者换算为传感器单位后必须是 float32 正常范围内的有限正数。
暂停时修改也会刷新显示，无效输入恢复上次参数。
`Color range (mm)` 只控制距离模式颜色，不改变测量或 counts。

## 编号与空间方向

`state_index` 是稳定 112 维向量的 0 基索引；`raw_index_1based` 是 JQ 256 字节帧中的
1 基编号；热力图格子是显示位置。`load_tactile_layout().raw_indices` 提供 0 基 raw 索引。
显示变换不修改数据向量、触点或通道成员。

胸部按人体前视图显示。冻结背部 ABI 对应衣服展开图，背视图面板将其显示索引旋转 180°：

| 背部排列 | 左上 raw | 右上 raw | 左下 raw | 右下 raw |
| --- | ---: | ---: | ---: | ---: |
| 稳定 ABI／展开图 | 206 | 62 | 202 | 58 |
| 人体背视图面板 | 58 | 202 | 62 | 206 |

206/62 绑定下背，202/58 绑定上背；向量的 `state_index=48` 仍对应 raw 206，
背视图左上格对应 `state_index=87 → raw 58`。胸部四角为 195、51、200、56。
双臂列方向为肩到腕，两行表示两条皮肤带；网格不表示实际等距布点或皮肤尺寸。

来源为矩侨精密《织物电子皮肤（高频版）产品规格书》`260114-v1.0` 第 16–18 页，
并与冻结映射的 `construction.back` 一致。右臂采用第 17 页分区表：
`177,161,145,129 / 178,162,146,130`。第 18 页汇总表却为
`177,162,146,130 / 178,161,145,129`；保留已确认的第 17 页顺序及成员，不据此重排资产。

## 几何查询与有效性

每个传感器查询目标实体中的一个 geom，默认选唯一 visual mesh，否则选唯一 collision geom；
有多个候选时需设置实体内名称 `geom_name`，不加实体前缀。查询不包含地面、机器人自身或其他实体。

网格使用原三角面 BVH，而非采样点或凸分解碰撞壳；box、sphere、capsule、cylinder、plane
使用解析表面。空网格、退化面、Warp 会跳过的极细长面及不支持的 geom 类型会报错。
仅闭合且绕序一致的网格提供可靠符号，负值表示内部、正值表示外部；plane 仅提供无符号距离。
这些几何观测与 MuJoCo 求解器接触力不同。

默认感应范围 `contact_distance_m=0.004`。`near_surface` 为无符号距离不超过范围，
`inside_object` 为可靠判定的内部点，`geometric_contact` 为两者逻辑或。
因此深内部点在距离视图中可能为冷色，但仍带接触白框。
通道距离取成员最小值，接触取任意成员为真，`channel_contact_point_count` 为接触成员数。

`valid` 表示距离查询有效，`signed_distance_valid` 表示符号可靠；对应通道 mask 要求所有成员有效。
开放网格的符号无效不代表点位于物体外部。未命中或禁用网格输出无符号距离 `inf`、
有符号距离 `NaN`、`valid=False`、`nearest_face=-1`；符号无效的通道有符号距离也为 `NaN`。

## 合成 counts

符号有效时取 `d = signed_distance_m`，否则取无符号 `surface_distance_m`，距离单位为米：

```text
depth = max(0, contact_distance_m - d)
response = 255 × clamp(depth / counts_full_scale_depth_m, 0, 1) ^ counts_exponent
```

`point_counts` 对 response 作 round-to-even 舍入后存为 `uint8`。
`channel_counts` 对原成员的浮点 response 取最大值后作相同舍入。
默认感应范围 4 mm、满量程深度 10 mm、指数 1：

| 有效距离 d | 有效深度 | counts |
| ---: | ---: | ---: |
| 4 mm | 0 mm | 0 |
| 2 mm | 2 mm | 51 |
| 0 mm | 4 mm | 102 |
| −6 mm（可靠内部点） | 10 mm | 255 |

接触边界可以接触为真而 counts 为 0。开放网格仅表示表面接近程度，默认最高 102；
闭合网格负距离增加响应。有效深度不是测得的皮肤压缩量，counts 没有力、压力或硬件标定含义，
也不生成 256 字节硬件帧。无效点／通道填 0，使用时须同时读取 `valid`／`channel_valid`。

## 配置与读取 API

在环境创建前配置；OpenDoorSide 继承 OpenDoor 的传感器配置：

```python
from supermimic.tasks.sitting.env_cfgs import make_env_cfg

cfg = make_env_cfg(play=True)
tactile_cfg = next(sensor for sensor in cfg.scene.sensors if sensor.name == "tactile")
tactile_cfg.layout = "inner_arms_v1"
tactile_cfg.contact_distance_m = 0.004
tactile_cfg.counts_full_scale_depth_m = 0.010
tactile_cfg.counts_exponent = 1.0
```

直接构造时使用 `supermimic.tactile.G1TactileSensorCfg`，指定 `name`、`robot_entity`、
`target_entity`，可选 `geom_name`。读取结果在仿真设备上，保留环境维度：

```python
sensor = env.scene["tactile"]
data = sensor.data
valid_contact = data.channel_geometric_contact & data.channel_valid  # [num_envs, 112]
counts = data.channel_counts  # [num_envs, 112]，uint8，需配合 channel_valid
```

| 字段 | 每环境形状 | 含义 |
| --- | --- | --- |
| `point_positions_w`, `point_normals_w` | `[464, 3]` | 世界坐标触点及法向 |
| `surface_distance_m`, `signed_distance_m` | `[464]` | 无符号／有符号距离，米 |
| `signed_distance_valid`, `valid` | `[464]` | 符号／距离有效性 |
| `inside_object`, `near_surface`, `geometric_contact` | `[464]` | 内部、近表面、接触判定 |
| `nearest_point_w`, `nearest_normal_w` | `[464, 3]` | 最近表面点和面法向 |
| `nearest_face` | `[464]` | 编译后 mesh 局部面索引，非源 OBJ 编号；primitive 为 `-1` |
| `channel_surface_distance_m`, `channel_signed_distance_m` | `[112]` | 成员距离最小值 |
| `channel_signed_distance_valid`, `channel_valid` | `[112]` | 通道符号／距离有效性 |
| `channel_geometric_contact`, `channel_contact_point_count` | `[112]` | 接触判定／接触成员数 |
| `point_counts`, `channel_counts` | `[464]`, `[112]` | 0–255 合成响应，uint8；无效时填 0 |
| `compression_depth_m` | `[464]` | counts 映射的有效深度，米；无效时填 0 |

向 observation manager 导出可在创建环境前添加独立组：

```python
import torch
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

def tactile_contacts(env):
    data = env.scene["tactile"].data
    return (data.channel_geometric_contact & data.channel_valid).to(torch.float32)

cfg.observations["tactile"] = ObservationGroupCfg(
    terms={"contact": ObservationTermCfg(func=tactile_contacts)},
)
```

SuperMimic Runner 不会自动将新组拼入 Actor 或 WM。
已有的 [OpenDoor BC/DAgger 蒸馏](../supermimic/distillation/open_door/README.md) 单独读取
10帧 counts 和有效性 mask，编码为64维特征接入学生 Actor；教师及世界模型保持原输入。

## 更新与缓存

`scene.update(dt)` 和 `scene.reset(...)` 使缓存失效，下次读取 `.data` 才重新计算。
初始化仍会准备几何和 BVH。查询复用输出缓冲，保存跨步历史须对张量 `.clone()`。
直接修改机器人或物体 pose 后，先更新 FK，再使传感器缓存失效：

```python
env.sim.forward()
sensor.update(0.0)
data = sensor.data
```

仅修改 counts 参数时调用 `sensor.update(0.0)` 即可；查看器已在主线程处理参数更新与缓存失效。
