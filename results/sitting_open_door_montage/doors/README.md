# 最终策略接触回放素材

## 简洁展示版

`back_clean.mp4`（9.20 秒）与 `side_clean.mp4`（13.92 秒）使用本目录已保存的 `scene.mjb`、`physics_states.npz` 直接重渲染，无需重新运行策略。均为 1920×1080、25 fps、H.264/yuv420p、无音轨，实际速度播放，结尾保持最终帧 1 秒。

每段仅保留一个完整机器人与门板的固定斜侧视角和任务名；不含近景分屏、接触点、圆环、引线、部位着色、力值或接触指标。门板完全不透明。地面与灯光只作视觉调整，物理状态不变。由 `render_clean.py back` 和 `render_clean.py side` 复现。`clean_video_*` 为正式渲染的首帧、中间帧和末帧。相机选取试图已清理。


## 本次实测结果

| 项目 | 背部 | 右侧肩部 |
|---|---:|---:|
| Checkpoint 迭代 | 10000 | 10000（fresh AC） |
| 参考动作完成 | 是 | 是 |
| 物理回放时长 | 8.12 s | 12.86 s |
| 成片时长（含保持） | 9.20 s | 13.92 s |
| 最终门角 | 42.866° | 44.779° |
| 门角 RMSE | 1.535° | 0.416° |
| 允许区域接触率 | 62.993% | 59.137% |
| 非允许区域接触率 | 2.401% | 0% |
| 允许区域法向力总和峰值 | 104.42 N | 152.39 N |
| 其他区域法向力总和峰值 | 57.00 N | 0 N |

接触率以每个物理子步内相应类别法向力总和大于 1 N 判定。背部此次还存在少量头部及背部区域之外的躯干接触，原始数据保留这些接触；侧面此次所有大于 1 N 的接触点均来自 `right_shoulder_pitch_link`。完成参考并不意味着全程只有目标区域接触。

## 数据来源与复现

- 背部模型：`logs/wm_transfer/back_model_10000.pt`。
- 侧面模型：`logs/side_policy_playback/2026-11-01_01-14-42_fresh_ac_model10000/model_10000_actor.pt`。
- 参考数据：`data/motions/open_door_prepared/video1_g1_door_video45_centered_asset.npz` 和 `video6_g1_door_video45_right_side_asset.npz`，分别使用配套 XML。
- `capture_contacts.py` 复用项目的原生评估与配置恢复逻辑，从第 0 帧、名义物性开始，每 0.005 秒在物理求解器完成后立即调用 `mujoco_warp.contact_force`，保存接触位置、法向力、部位、允许区域分类与同一步的 qpos。
- 每个任务目录保存原生 `scene.mjb`、`physics_states.npz`、`measured_contacts.json`、`rollout.npz`、`evaluation.json`；后者记录模型、参考数据和当前 `motion.py` 哈希。

本次侧面轨迹与先前保存的最终回放存在很小数值差异：门角最大差 0.264°，根位置最大差 2.54 mm。视频和记录的数值全部来自本次重新采集，未混用旧回放的 58.98% 指标。本次参考 NPZ 哈希与旧侧面报告一致；当前 `motion.py` 原有未提交修改未被改动。

运行目录为仓库根目录：

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 .venv/bin/python results/sitting_open_door_montage/doors/capture_contacts.py back
PYTHONPATH=. OMP_NUM_THREADS=4 .venv/bin/python results/sitting_open_door_montage/doors/capture_contacts.py side
PYTHONPATH=. .venv/bin/python results/sitting_open_door_montage/doors/render_clean.py back
PYTHONPATH=. .venv/bin/python results/sitting_open_door_montage/doors/render_clean.py side
```
