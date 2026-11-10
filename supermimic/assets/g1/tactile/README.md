# G1 触觉资产

来源为 `WBT-Genesis` 提交 `327e3d29b6f96042827c0eb47c1bb06c51053955`，
归档 `WBT-Genesis-20261023/WBT-Genesis-full.tar.zst`。两套点位／映射及公共通道 ABI
逐字节迁入，已与归档清单核验 SHA256。使用方法见[触觉传感器文档](../../../../docs/tactile.md)。

- `original_outer`：默认外侧布局，原物理触觉演示使用，15 个承载 link。
- `inner_arms_v1`：历史 HOI 抱持内侧布局，13 个承载 link；保留原 review 状态，
  不继承外侧布局的安装确认。

两套布局保留相同的 464 个稳定 ID、顺序和 112 通道成员。位置在承载 URDF link 局部系，
单位为米，法向为单位外法向；点位于源模型 visual 表面。4 mm 半径是未标定的虚拟感应范围。
区域及稳定向量顺序如下，不包含手掌或腿：

| 区域 | 点数 | 通道数 | 每通道点数 |
| --- | ---: | ---: | ---: |
| 胸部 | 192 | 48 | 4 |
| 背部 | 160 | 40 | 4 |
| 左臂／右臂（各） | 48 | 8 | 6 |
| 左肩／右肩（各） | 8 | 4 | 2 |

通道 ABI 来自 `jq-skin` 提交 `da8a2feb1b3be1ee3b694ad9ab3221d554010aa1`。
原 JSON 保留模型／网格哈希、来源、人工确认、派生关系和未标定说明；
加载器核验几何摘要、通道 ABI、模型身份一致性、ID、区域及成员覆盖。

## 模型相容性

源点位模型为 `g1_29dof_mode_15.urdf`，当前项目为 `g1_29dof_rev_1_0.urdf`。
胸、肩、上臂和肘的承载 link 坐标、visual origin 与网格 SHA256 相同。
左右 wrist roll/pitch/yaw 的 visual 网格不同，wrist yaw 关节 x 偏移从源模型的
0.051 m 变为当前的 0.046 m；外侧各有一个点绑定 wrist yaw，内侧没有。
点位保持原局部坐标，随当前 FK 变换，腕部并非完全贴合当前表面。

原腕部点至当前 visual 表面的最大距离约为外侧 1.94 mm、内侧 2.38 mm。
零姿态和保存姿态下，当前 MJCF 与当前 URDF 的触点 FK 最大差异小于 0.6 μm。

## 原文件 SHA256

| 文件 | SHA256 |
| --- | --- |
| original_outer/points.json | `2cb031fbffc4be611b5436c04e11dc5dc3cb5cdfb40a15b4927349bbd9ac844e` |
| original_outer/jq-channel-map.json | `12b79ed6754abf3b42cc64168f1b520d644966d8db0818458c5f136802f0b8a6` |
| inner_arms_v1/points.json | `1f2761dc852021e54fde1ff58b1035e351605a32d5141d060b80a384e43af96f` |
| inner_arms_v1/jq-channel-map.json | `ec4e0a6934af066b5dde9849a45628d12309a26fa29dfe401c64a34bca813996` |
| upper-body-channel-layout.json | `6ef676971d99296cb204cde2f237d11b2a866dc4d4fa26e5b14c2260655e2b2d` |
