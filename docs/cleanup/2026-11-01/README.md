# 2026-11-01 本地清理记录

按用户要求清理本地中间实验，只保留最终模型相关日志及最终展示产物。删除清单见 `deletion_plan.json`，实际执行结果和保留文件哈希见 `cleanup_result.json`。

- 删除121个目标，共44,348个文件，逻辑体积约11.90 GiB。
- 实际可用空间增加约5.80 GiB；虚拟环境存在共享硬链接，逻辑体积不等于释放空间。
- 保留最终Sitting10000、背部10000、新版侧面10000；Sitting8600保留为已交付九宫格视频来源。
- 已从CUHK补齐三组完整训练日志，以及缺失的最终Sitting10000和侧面10000完整checkpoint，并验证SHA256。
- 保留当前项目源码、数据、运行环境、两处用户原有未提交源码修改和干净版最终视频/复现素材。
- 移除旁边旧无reference MBRL/PPO项目 `/home/frank/Robotics/Paper/Humanoid/open_door`。

## 旧项目恢复依据

服务器源码：`CUHK:/home/wangfeng/open_door`，HEAD与本地一致：`77ff70f5e9ba9e7db06b49cf3bc9bd1816d9a41f`。
训练日志分别保存在 `/home/wangfeng/open_door/logs` 与 `/data/wangfeng/open_door/logs`。

删除前发现8个本地独有工作文件，已另外补存为：

`CUHK:/home/wangfeng/open_door/local_archives/frank_local_changes_20261101_025c26a48c25.tar.gz`

归档SHA256：`025c26a48c259058383c283cb1236e67520c7e69fdaf67574a6364e2a377acb8`，上传后已核验；内含8个文件、git diff及来源清单。见 `open_door_source_backup.json`。

CUHK主要旧run、最终run已核验；本地临时回放/诊断并非全部逐文件备份。`logs/2026-10-26_14-05-17`来自gnode01，未在CUHK找到，按用户“中间实验服务器已保存”的说明删除，未宣称独立验证该目录的远程副本。

服务器训练数据和源码未删除；仅新增了上述本地独有改动归档。其它项目及其它聊天的Git worktree未处理。
