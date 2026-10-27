# CUHK 双卡训练检查（2026-10-27）

基线提交 `5e6ea43`，测试设备为 CUHK 的 GPU 0、1：两张 NVIDIA A100 PCIe 40GB。
后续512环境长训练出现的visit27显存问题及修复见[OOM复测记录](cuda-oom-visit27.md)；
下文的两倍容量策略为早期修复记录，当前实现已改为25%余量和小块对齐。
GPU 7 上已有作业，测试未使用它。服务器已有依赖配置修改，未覆盖；代码修订保留在
本地分支，服务器只运行 `logs/cuhk_dualgpu_validation_20261027/source` 下的测试快照。

环境：驱动 580.178.04、PyTorch 2.14.1+cu130、mjlab 1.6.0、MuJoCo/MuJoCo-Warp
3.11.0、Warp 1.18.0、torchrunx 0.4.0。两卡通信使用 NCCL。

## 检查方法

`tools/check_distributed_training.py` 包装现有多椅子入口，不改变网络或更新算法。
每次保存时比较各 rank 的三网络全部参数和 buffer、Adam 状态及参数组、迭代数和
全局采样计数；要求逐字节摘要一致且所有数值有限。另记录每卡 Replay 摘要、设备、
训练椅子访问记录和 PyTorch 峰值显存。输出 `model_N.audit.json`。

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m tools.check_distributed_training train \
  --split /path/to/diagnostic_split.json --data-root data/motions/sitting \
  --agent-json /path/to/diagnostic_agent.json \
  --num-envs 2 --iterations-per-chair 1 --iterations 4 \
  --run-dir logs/dual_gpu_diagnostic
```

该诊断划分复用已有真实椅子 smoke 数据：aluminum bistro 与 Adams oval-back 为训练
几何，Acadian dining chair 为测试几何。它与正式 151/38 划分分开，不作为泛化结果。
模型使用完整默认结构（约1487万参数、8物体槽），小规模测试只缩短 batch 和展开长度。

小规模设置：每卡2环境，Replay8步，WM/Actor batch2，WM窗口2，想象长度2，
每迭代1次WM更新、1次Actor更新、1次Critic更新，`utd=1`、warmup0、WM-only0，
不编译。实际预填充仍为2步，以满足WM窗口。检查版本每次迭代记录/保存，开启Replay导出。

## 已完成结果

- 原始多椅子CLI，FP32：双卡3次迭代正常结束。
- 审计入口，BF16：双卡4次迭代通过，每次保存模型、统计和优化器均完全一致且有限。
  实际发生场景切换，各卡Replay摘要不同；Actor/WM/Critic在训练过程中均有更新。
- 从 `model_4.pt` 恢复后追加2次迭代，迭代编号到6，全局采样步数到40；同步检查继续通过。
  checkpoint独占写入与每卡Replay导出正常；访问记录和训练覆盖并集得以保留。
- 默认规模：每卡128环境、WM/Actor batch128、WM窗口8、想象长度32、10次WM更新和
  5×4次Critic更新，BF16和编译开启。仅跳过500次WM-only阶段并提高记录/保存频率，
  完成2次迭代、17,664个全局采样步，所有一致性和有限值检查通过。PyTorch峰值张量
  显存约3.06 GiB/卡，不含Warp/NCCL/CUDA上下文等分配；第一次编译约数分钟。
- 原生 `train ... --gpu-ids '[0,1]'` 入口完成2次迭代并保存检查点，修复前后均验证。
- 双卡 `model_6.pt` 在单卡上完成未见几何冻结评估，参数/统计摘要不变。短训策略的
  轨迹完成率为0，回报约0.1435；这里只验证流程，不代表任务训练成功。

## 发现的问题与修复

正式151/38划分在修复前无法启动：rank0的 theater audience chair 需要853接触，
rank1的 tall wingback chair 需要891接触，超过固定512容量。堆栈来自
`mujoco_warp.put_data` 的 `nconmax overflow`，发生在模型上传GPU、首次训练之前。
容量只按物体数量估算，遗漏了同一个椅子可有很多凸分解部件的情况。

本地 `EntityEnvCfg` 增加场景级延迟容量检查：在全部几何组装后编译CPU模型并执行
初始forward，将接触与约束容量扩大到实测需求的两倍、向上取2的幂，不缩减显式设置。
现有自定义场景回调保留，配置重建与deepcopy重新绑定当前sim配置。
新增回归使用256个接地凸块，修复前真实Warp上传因1065个接触溢出而失败，修复后通过。
全套本地测试178项通过。容量余量不保证任意未来状态均安全；极端接触仍可能需要加大配置。

修复后使用相同正式151/38划分、全部1391训练片段和训练几何库重新运行：每卡128环境，
默认WM/Actor batch128、窗口8/想象32，BF16，关闭编译以单独验证场景容量修复，
`wm_only_iters=0`、每次迭代换椅子。两次迭代经过4种不同训练椅子，17,664全局采样步，
各次保存的模型、归一化和优化器均有限且逐字节相同，两卡Replay不同。
首轮容量自动提高到2048接触/8192约束；第二轮两场景分别为1024/4096和2048/8192。
PyTorch峰值张量显存约3.74 GiB/卡。结果位于 `full_split_fixed/`。
这验证完整数据加载和4个场景的物理更新，不代表已遍历训练完151把椅子。

原生torchrunx入口还会在成功结束后打印残缺的 `[rank1]: Traceback`。临时异常钩子捕获
的完整堆栈为Python `weakref._exitfunc → torch.library._del_library → SignalException(SIGTERM)`：
启动器结束进程时打断PyTorch退出清理。已多次复现，训练迭代、checkpoint和退出码正常。
它不是本项目梯度同步故障；没有修改服务器的第三方依赖来压制该日志。
多椅子 `torch.distributed.run` 入口在已完成测试中没有这个退出日志。

服务器正式151/38划分的碰撞缓存预检查当前全部通过：train/test缺失均为0。
该检查只证明缓存存在；完整实验的每个资产物理运行仍需单独验证。

检查产物位于服务器 `supermimic_vibed/logs/cuhk_dualgpu_validation_20261027/`，
审计JSON、日志与冻结评估报告已下载到本地同名日志目录。摘要检查会同步GPU并复制张量到CPU，
所以这些检查的迭代速率不能用作正式吞吐或双卡加速比。
全部测试结束后，GPU0/1的测试进程和显存已释放。服务器主项目仍是原提交，保留其原有
`uv.lock` 修改和 `uv.toml`；修复只同步到独立测试快照，未覆盖主项目源码。
