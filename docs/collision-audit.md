# G1 与 GRAIL 椅子碰撞审计

检查日期：2026-10-24。目标是验证碰撞几何覆盖、形状近似和接触过滤；不把参考姿态
回放中的“坐下”当成动力学稳定或训练成功的证据。

后续模型调整：按当前任务范围，已移除左右 `rubber_hand` 的视觉和碰撞几何，不再
建模手部接触。保留全部 29 个关节（包括腕部）及固定手部 body 的质量、惯量。
下文的原始模型统计、手部探针、穿透数值、GPU 接触数和测试数量均保留为移除手部
几何前的历史证据，不能视为当前模型的检查结果。移除几何没有修正参考动作或配准误差。

## 结论与修改

原审计发现 G1 的碰撞适合平面行走的简化模型，但手部和脚部的物体接触覆盖不足。
椅子视觉网格不能直接作为 MuJoCo 的单个碰撞 mesh：MuJoCo 实际用其凸包，
会填满腿间空隙、靠背空隙和球形椅的坐人空腔。

当前 `interaction_collisions=True` 配置沿用本项目 retargeting revision 的原 mesh，
仅补充整脚的物体碰撞；原审计中补充的左右 rubber hand 碰撞已移除。
关节、body frame、质量与惯量保持原值。原四球脚底接触保留为地面接触；整脚 mesh
仅与物体碰撞。tracking 默认使用基础碰撞配置。

| 几何 | contype | conaffinity |
|---|---:|---:|
| 地面 | 1 | 0 |
| 椅子凸块 | 2 | 0 |
| 机器人已有非足球碰撞 | 0 | 3 |
| 脚底四球 | 0 | 1 |
| 补充的整脚 mesh | 0 | 2 |
| 视觉网格 | 0 | 0 |

椅子经 CoACD 离线分解，使用多个凸块；共享 builder 以相同坐标系、姿态和一次缩放
挂到同一个物体 body。缓存完全独立于动作轨迹，同一几何的不同动作共用缓存。
输出逐块校验凸性、绕向、封闭性和正体积。CoACD 简化后的三角面可能有亚微米级非凸，
因此只对每个部件的已有顶点重新构建凸包拓扑，不移动顶点、不合并不同部件。
缓存缺失或无效时明确失败，禁止回退到整椅凸包。

## 原配置的证据

G1 有 35 个视觉 geom、36 个碰撞 geom（24 mesh、8 球、4 圆柱）。pelvis 本体虽无
collider，固定的 `pelvis_contour_link` 有碰撞，骨盆与大腿并非全部缺失碰撞。

- 两个 `rubber_hand` 只有视觉几何。腕部碰撞之后约 13.18 cm 的手部无覆盖。
  手掌内部的 5 mm 球探针原来没有 contact，补充手 mesh 后出现 contact。
- 每脚只有四个半径 5 mm 的球；中心和脚背缺少物体碰撞。足部视觉长度约 20.82 cm，
  球碰撞包络长约 18 cm。脚底中部球探针也由无 contact 变为有效接触。
- 骨盆/髋/膝/躯干的现有 mesh 凸包仍会填平内腔和孔洞。采样表面偏差可达厘米级，
  因此不能将所有参考姿态穿透都归因于椅子，也不能直接以凸包体积比例判定外壳不可用。

1748 段动作对应 189 个不同椅子网格。逐种拓扑检查：186 个封闭，3 个各缺一个
三角面（acadian、biedermeier、egg），没有发现非流形边、退化面、反向绕序或负体积。
3 个孔可用补洞处理，实际 CoACD 采用 `preprocess_mode=auto`；视觉网格保持原样。
整体凸包/原视觉有向体积中位数约 7.25，最大约 38.99；未闭合网格的有向体积仅供近似参考。

9 种椅子的原始最低点离地超过 1 cm，最大约 4.31 cm。检查保留原始物体位置，
没有单独把椅子移到地面，以免破坏动作与椅子的配准。
部分原始网格还带底部平板/平台，视觉与碰撞处理均保留；若它们是生成资产的底面伪影，
需要先确认物理含义，再修订资产，不能仅凭椅子的名称自动删除。

六种代表椅子按原动作全部 125 帧进行接触检查，单凸包最大重叠深度：

| 椅型 | 原 G1 + 整椅凸包最大重叠 |
|---|---:|
| acadian 木椅 | 8.51 cm |
| 布艺 armchair | 16.29 cm |
| egg 椅 | 22.19 cm |
| ball pod 椅 | 45.58 cm |
| aluminum bistro | 2.21 cm |
| 混凝土 armchair | 21.48 cm |

ball 椅末帧 pelvis 原点在单凸包内部约 24.7 cm，但到视觉表面距离约 8.3 cm，
处于真实空腔。egg 相应值约 15.2 cm 和 10.9 cm，证明整椅凸包产生了虚假阻挡。
body 原点检查并不代表整个机器人都没有真实穿透。

移除手部几何前的整段检查如下。此表同时切换了 G1 碰撞配置和椅子表示，是当时整体
修改前后的比较，不能把全部差值单独归因于椅子分解，也不代表当前无手部几何模型：

| 椅型 | 原 G1 + 单凸包 | 当时含手碰撞的交互 G1 + 分解凸块 |
|---|---:|---:|
| acadian 木椅 | 8.51 cm | 1.35 cm |
| aluminum bistro | 2.21 cm | 0 cm |
| egg 椅 | 22.19 cm | 3.60 cm |
| ball pod 椅 | 45.58 cm | 6.57 cm |

独立物体探针进一步确认空腔和支撑：

- 在 ball 末帧 pelvis 原点放半径 5 mm 的球，整椅凸包产生约 25.20 cm 的虚假穿透；
  细分解、64 块试验和 128 块试验均无碰撞。egg 细分解的相同探针也无碰撞。
  单个空腔探针通过不能保证压缩版全部表面满足误差要求。
- 在 acadian 座面上方释放半径 2 cm、质量 0.1 kg 的球，独立仿真 3 秒后有
  3 个 contact，法向支撑力约 0.990 N，与重力 0.981 N 接近；球没有穿过座面。

## 质量和性能边界

细分解并不自动等于适合批量训练：当前 1 cm 源坐标阈值下，ball pod 会生成
1036 个凸块。试验将其强制压到 64/128 块，CoACD 明确报告超出目标误差；这些试验
没有替换默认的细分解配方，也没有被标记为已通过训练精度验收。

历史审计加入手部 collider 后，ball 动作即使在细分解下仍出现约 6.57 cm 的手部
重叠；64 块试验增加到约 9.54 cm。当前按任务范围移除了手部视觉和碰撞几何，
这些手部接触不再参与检查；这不意味着原动作的手部穿透已修复。若以后重新建模
手部接触，仍需区分动作/机器人外形配准和近似网格误差。接触检查不会修正源动作。

当时的代码测试覆盖新旧 G1 的惯量/运动学一致、地面触点不变、手/脚物体探针、物体缩放、
凸缓存校验以及座面下空隙。MuJoCo-Warp 的两环境实际接触检查也已运行：bistro
92 个凸块，CPU 24 个 contact，GPU 两环境共 48 个 contact，状态有限。
圆柱与 mesh 的 MULTICCD 在当前 Warp 版本仅支持单接触，会给出相应提示。
此检查不等于 4096 环境的性能或稳定性基准。

验证覆盖范围：全部 189 种椅子已做拓扑检查；当前生成了 24 种几何的凸分解缓存，
其余 165 种未生成。批量处理在发现球形椅复杂度过高后停止，保留已完成资产；
碰撞形状、支撑和 GPU 接触检查覆盖代表样本，尚未对整库逐一做物理验收。
移除手部几何前的软件回归测试为 **78 passed**。

## 复现与产物

```bash
uv run --extra motion python tools/prepare_collisions.py /path/to/motion.npz
uv run python tools/check_collisions.py /path/to/motion.npz --render
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra motion pytest tests/test_g1_collision.py tests/test_object_collision.py -q
```

本地数值证据与图片位于 `results/analysis/collision/`：
`g1_original_audit.json`、`chair_mesh_inventory.json`、`chair_representative_audit.json`、
`gpu_smoke.json`、`ball_cavity_probe.json`、`egg_cavity_probe.json`、
`seat_support_probe.json`、`cache_inventory.json`，以及 `<clip>/contacts.json` 和 `<clip>/comparison.png`。
生成缓存位于 `data/collisions/chairs/`；缓存与审计产物按仓库规则不入 Git。

MuJoCo 的凸碰撞语义见 [官方说明](https://mujoco.readthedocs.io/en/stable/computation/index.html#convex-decomposition)。
