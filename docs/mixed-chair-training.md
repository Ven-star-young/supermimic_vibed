# 同时混合多种椅子的训练

`--scene-mode mixed` 在每张GPU上建立一个原生mjlab变体场景，把全部训练椅子均衡分配给
并行环境。每个环境固定一把椅子，reset只改变参考起点和随机物性，不改变椅子身份。
例如83把训练椅子、每卡512环境，每把椅子占6或7个环境；两卡合计1024环境。
每卡环境数至少为训练场景数；尺度变体算独立物理场景，同一原始椅子的变体仍在同一划分。

原来 `--scene-mode rotate` 的逐场景轮换保留，仍是默认值。混合模式不使用
`--iterations-per-chair`，也不会每50次迭代重建场景或截断所有回合。正常tracking终止、
参考结束和10秒episode时间限制继续生效。因此混合训练的 `ep_len` 与旧轮换日志不能
直接当作同口径实验比较：旧模式每次访问仅约2.5秒，未结束回合被切换丢弃。

## 数据与网络

- 每环境分配由seed和rank确定，椅子数目差最多1；不同rank打乱分配及余数归属。
- 每次采样所有环境并行推进。各环境RSI概率只在自己椅子的reference片段与时间bin内
  归一化；显式传入其他椅子的参考帧会报错。训练全局帧编号不再需要局部offset。
- 原生variant模型、参考帧、Replay的geometry ID、1024点几何编码和每环境物性保持一致。
  reset按实际变体的质量/COM/惯量/惯量姿态写入context，不复用第一把椅子的物性。
- 原生variant的碰撞槽按角色补齐；G1接触副本与地面副本放在不同焊接body下，防止不同
  凸块数量使槽位混淆。空槽由后端 `geom_dataid=-1` 禁用；地面副本不额外增加自由度。
- Actor/WM/Critic结构、共享几何编码器、奖励、优化器、BPTT及双卡同步方法保持相同。
  几何编码器仍仅由WM更新。更多不同几何会增加编码和场景初始化开销，速度需实际测量。
- 每卡独立Replay；所有训练几何持续采样，窗口仍不跨环境或回合。持久环境让回合自然结束。

## 可选碰撞简化

`--collision-profile default` 使用原始缓存，`--collision-profile compact` 使用独立的
最多32凸块、每块最多32顶点近似。两种缓存同时存在，不修改原NPZ、视觉网格或原缓存。
网络的几何点云从实际使用的碰撞表面采样，geometry key包含碰撞配方，不能把旧几何ID
直接套到简化资产上。惯量继续从原始缩放视觉网格计算，质量随机化范围不变。

简化会改变接触物理；部分椅子在32块预算下会超过请求的凹度阈值。细杆、镂空、扶手和
座面支撑需检查，不能把显存下降理解为碰撞效果等价。两种资产的定量和可视化比较见
[compact-collisions.md](compact-collisions.md)。不会因为缺失简化缓存自动回退到原模型，
也不会跳过某把椅子。两种碰撞profile仍沿用相同的训练/测试归属。

```bash
# 仅需准备一次。默认缓存与compact缓存可共存。
OMP_NUM_THREADS=2 .venv/bin/python -m tools.prepare_collisions \
  --split-manifest data/splits/sitting_back_references_seed0.json \
  --partition all --profile compact --workers 6

.venv/bin/python -m supermimic.scripts.chair_experiment check \
  --split data/splits/sitting_back_references_seed0.json --collision-profile compact
```

## 双卡启动

为新模式建立新的运行目录。以下配置沿用每卡batch256，并缩短保存间隔：

```bash
mkdir -p logs
cat > logs/mixed_reference_agent.json <<'JSON'
{
  "wm_batch": 256,
  "pi_batch": 256,
  "buffer_steps": 256,
  "save_interval": 200,
  "log_every": 10
}
JSON

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 \
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m supermimic.scripts.chair_experiment train \
  --split data/splits/sitting_back_references_seed0.json \
  --scene-mode mixed --collision-profile compact \
  --agent-json logs/mixed_reference_agent.json \
  --num-envs 512 --max-objects 8 --seed 0 --iterations 10000 \
  --run-dir logs/sitting_back_refs_mixed_2gpu
```

环境数和batch均为每卡数量。每个环境当前仍是一把椅子，不是把83把椅子同时放进一个
物理世界。其他GPU正在训练时应选择空闲GPU；混合模式不会接管现有训练进程。

## 保存、续训与测试

混合checkpoint记录固定环境分配协议、卡数、每卡环境数、seed、split fingerprint和
碰撞profile。它和旧轮换checkpoint不兼容完整续训，应新开实验；切换profile也需新开
实验。混合任务自身续训保持这些设置一致，追加 `--resume .../model_N.pt`，
`--iterations` 仍表示追加迭代数。Replay重新填充，环境继续保持固定几何分配。

冻结评估继续使用原入口，自动从checkpoint读取碰撞profile：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m supermimic.scripts.chair_experiment evaluate \
  --split data/splits/sitting_back_references_seed0.json \
  --checkpoint-file logs/sitting_back_refs_mixed_2gpu/model_10000.pt \
  --output results/analysis/mixed_reference_zero_shot.json
```

训练只读取train成员，test资产离线预处理不更新模型。测试逐椅子运行，保持Actor和
归一化冻结；报告记录碰撞profile，避免将不同接触近似下的成绩混作一个实验。

## 检查工具

把训练入口的模块名替换为 `tools.check_distributed_training`，保留所有参数，即可在
每次保存时验证跨卡模型/归一化/Adam状态，并检查Replay参考与物理几何对应关系。
它会生成 `model_N.audit.json`，含每把椅子环境数、物理容量、PyTorch显存及整卡占用。
审计需要拷贝张量到CPU，不用于吞吐对比。

`python -m tools.measure_chair_scene --split ... --num-envs 128 --collision-profile compact
--output logs/mixed_scene.json` 只创建混合物理场景并step16次，测量不含学习网络的显存。
整卡显存采样包含Warp/CUDA graph；PyTorch allocator计数不包含这些工作区。

本地回归覆盖混合几何接触、禁用padding槽、每环境质量/COM/完整惯量、部分reset、
参考与geometry ID一致、旧50次边界不触发重置、训练/续训及默认/compact冻结评估。
本地完整回归252项通过。

## CUHK实测（2026-10-27）

使用空闲的GPU2/3（A100 40GB）测试双卡学习，GPU4做物理场景对照及冻结评估；已有
GPU0/1正式训练与GPU7作业未中断。代码只同步到独立
`logs/mixed_chair_validation_20261027/source` 快照，服务器主项目源码未覆盖。
83训练、22测试椅子的105份compact缓存均已独立生成并验证，原缓存保留。

| 测量 | 原碰撞 | Compact碰撞 |
|---|---:|---:|
| 83椅子、128环境，仅物理仿真 | 6.72 GiB | 1.98 GiB |
| 合并模板的geom数（含G1、视觉及补齐） | 1643 | 137 |
| 全部变体mesh库条目数 | 17801 | 2570 |
| 双卡各512环境、batch256，完整学习、编译开启 | 未测 | 每卡约12.95 GiB |

前两种显存是在16个控制步后逐步读取整卡占用的最大值，非瞬时峰值；512环境学习的
值是每秒一次 `nvidia-smi` 采样的最大值，包含CUDA/Warp/NCCL分配，不只是PyTorch。
相同128环境的物理仿真显存下降约70.5%。不要把128环境的物理值与512环境的训练值
当作仅碰撞profile不同的对照，也不能据此推断长训练绝对峰值。

512环境学习配置为完整默认网络、8物体槽、WM窗口8、想象长度32、WM/Actor batch256、
每迭代10次WM/1次Actor/20次Critic更新、BF16；诊断跳过WM-only阶段。先完成3次
不编译更新，再开启编译续训2次，最后采样器修正后又从checkpoint续训2次到7。
各次保存三网络/归一化/Adam状态跨卡逐字节相同且有限；每卡83椅子各6–7环境，
Replay参考帧与实际geometry ID校验通过。没有OOM、EPA或其他物理缓冲溢出。

采样器评审修复了两个实现细节：历史采样概率按每椅子的bin数初始化，EMA只更新本次
reset涉及的椅子；不兼容resume在改写旧实验JSON前被拒绝。对应CPU/CLI回归均通过。

模型7在全部22把测试椅子上完成单卡冻结评估，权重及统计摘要保持不变，报告自动选择
compact物理profile。七次短更新的轨迹完成率为0，macro奖励约0.140；这验证评估流程，
不代表泛化成功或算法改进。形状近似对坐姿接触的影响及长期训练效果仍需实验测量。

本地审计JSON、逐秒显存CSV、配置与日志保存在
`results/analysis/mixed_chair_memory_20261027/`。诊断结束后GPU2/3/4显存已释放。
