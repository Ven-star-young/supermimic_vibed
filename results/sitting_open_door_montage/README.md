# Sitting、背部开门和侧面开门策略展示

当前主视频：`sitting_back_side_clean.mp4`。1920 × 1080，25 fps，无音轨。开门采用单视角全身画面，仅保留任务标题，去除接触近景、彩色接触标记和力值。

视频顺序为九宫格 Sitting → 背部接触开门 → 右侧肩部接触开门。各段保持实际速度，动作结束后冻结记录的最终姿态；冻结区间是展示停留，不是额外物理仿真。

## Sitting

使用刚才浏览器部署页的 `logs/cuhk_sitting_deployment/model_8600.pt`，索引对应 **partition=all 的已排序列表**。9 个场景同时从 reference 第 0 帧开始，clean 初始化、seed 0。直接记录冻结策略在原生场景中的实际 qpos/qvel，并保留 MJB 模型、轨迹和逐回合评测。

画面中的 Complete 仅表示完整执行 reference。以下稳定接触列沿用项目原有判据，未据此筛除场景或修改动作。

| 索引 | 椅子 | 实际时长 | Reference | 稳定接触 |
|---:|---|---:|---|---|
| 0 | Adirondack | 4.96 s | 完成 | 通过 |
| 8 | Bauhaus armchair | 4.96 s | 完成 | 通过 |
| 10 | Berber chair | 4.96 s | 完成 | 通过 |
| 13 | Metal mesh chair | 4.96 s | 完成 | 通过 |
| 14 | Brass / velvet | 4.96 s | 完成 | 通过 |
| 19 | Hot dog chair | 4.96 s | 完成 | 未通过 |
| 28 | Chinese mahjong chair | 4.96 s | 完成 | 通过 |
| 34 | Curved concrete | 4.96 s | 完成 | 未通过 |
| 45 | Gold bamboo chair | 4.96 s | 完成 | 通过 |

## 开门接触

- 背部：第 10000 轮策略，本次完整执行 8.12 s，最终门角 42.87°；目标区域接触率 62.99%，其他区域接触率 2.40%。
- 右侧肩部：第 10000 轮策略，本次完整执行 12.86 s，最终门角 44.78°；目标区域接触率 59.14%，其他区域接触率 0.00%。

原始诊断数据仍保留在文件中，当前干净版视频不显示接触标记或力值。上述法向力统计来自每个 0.005 s 物理求解步之后立即调用的 `mujoco_warp.contact_force`。旧标记版视频和相机试图已在清理时移除。

## 复现与原始数据

- `sitting/build_sitting.py`：隔离子进程录制 9 把椅子、渲染同步九宫格。
- `sitting/chair_NNN.{mjb,npz,json}`：场景、实际物理轨迹、来源及评测。
- `doors/capture_contacts.py`：运行两个最终策略，采集每个物理子步的状态与接触。
- `doors/render_clean.py`：当前单视角干净版的渲染脚本。
- `doors/{back,side}/`：MJB、轨迹、逐步接触和评测。
- `assemble.py --clean`：无额外压缩地拼接当前三个片段，添加章节并完整解码验证。
- `video_manifest_clean.json`：当前成片的章节、时间、编码和验证信息。

场景二进制（`.mjb`）、轨迹（`.npz`）、视频和运行日志仅在本地保留，不随 Git 提交。
重建素材需先准备对应 checkpoint 和数据，再运行上述采集与渲染脚本。

本次只运行推理和渲染，未修改训练策略、核心源码或现有未提交修改。
