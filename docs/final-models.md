# 最终模型、日志与展示

2026-11-01 清理后，本地保留以下结果。三个任务的完整训练日志位于各模型目录的 `server_training/`，包含 train_log.csv、TensorBoard events 和原始训练配置/源码差异。

| 任务 | 本地 checkpoint | 说明 |
|---|---|---|
| Sitting 最终 | `logs/cuhk_sitting_deployment/model_10000.pt` | CUHK sitting_task_2gpu，最终10000轮；配套split.json保留 |
| Sitting 视频来源 | `logs/cuhk_sitting_deployment/model_8600.pt` | 已交付九宫格使用的8600轮策略，保留以便复现 |
| 背部开门 | `logs/wm_transfer/back_model_10000.pt` | 最终10000轮完整模型，也是侧面WM来源 |
| 侧面开门 | `logs/side_policy_playback/2026-11-01_01-14-42_fresh_ac_model10000/model_10000.pt` | 最终fresh AC 10000轮完整模型 |
| 侧面回放导出 | 同目录 `model_10000_actor.pt` | 当前视频来源，仅供Actor推理 |

Sitting10000训练分析：`logs/cuhk_sitting_10000_analysis/`。
Sitting10000全量扰动评估（105椅×10 seeds）：`logs/cuhk_eval_10000_perturbed_seed0_9_20261028/`。
侧面最终评估、训练CSV及来源：上述侧面模型目录。

最终干净版视频及全部重渲染素材：`results/sitting_open_door_montage/`，主视频为 `sitting_back_side_clean.mp4`。Sitting九宫格为8600轮策略；这不改变10000轮评估的模型来源。

## 本地播放

```bash
.venv/bin/python -m supermimic.scripts.play_policy \
  --checkpoint logs/cuhk_sitting_deployment/model_10000.pt \
  --split logs/cuhk_sitting_deployment/split.json \
  --partition all --browse --port 8088

.venv/bin/python results/sitting_open_door_montage/assemble.py --clean
```

原configs中的CUHK训练run路径保留其服务器语义。需要在本地初始化侧面WM时，使用上表 `logs/wm_transfer/back_model_10000.pt`；旧finetune配置对应历史服务器实验，不能以Actor导出替代完整checkpoint。

[清理清单、服务器来源和旧项目恢复说明](cleanup/2026-11-01/README.md)。
