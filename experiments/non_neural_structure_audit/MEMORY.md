# 内存边界与复用模块

目标不是让任务管理器中的 RSS 每条样本后都回到零，而是消除原始 attention、完整 graph、null-replicate ensemble 和跨样本紧凑矩阵的 Python-heap 堆积。fit/score 的高维活对象只随当前单样本或明确有上限的 accumulator 增长；evaluation 的持久矩阵使用 memmap，heap 只长期保留 sample 元数据与磁盘切片。单个统计步骤仍可能临时物化 `O(T_total)`，exploratory A9 会物化 `O(T_total * M)` 的设计矩阵。

## 1. 原始 attention 生命周期

仓库级模块 [`attention_lifecycle.py`](../../attention_lifecycle.py) 提供：

```python
from attention_lifecycle import loaded_attention

with loaded_attention(sample) as attention:
    compact = build_compact_result(sample, attention)
```

无论正常返回还是异常，作用域退出都会调用 `sample.release_attention()`，但不会吞掉原异常。

`score` 中原始 CSR attention 只活到以下对象构造完成：

- independent sparse `RoutingEdges`；
- `RoutingState` 与 `LineageOperator`；
-真实 feature tensor 与 response token IDs。

随后立即退出 `loaded_attention`，再运行耗时的 endpoint null 和 layer shuffle。因此原始 attention 不会与所有 null scratch 同时驻留。

## 2. 单样本 score 峰值

设：

- `E`：当前样本 retained sparse edges；
- `T`：response token 数；
- `L`：layer 数；
- `F`：feature 数；
- `M`：relation 数；
- `R_e` / `R_l`：endpoint/layer replicates。

主要常驻量约为：

```text
routing + endpoint scratch                       O(E)
real and standardized features                   O(T * L * F)
preallocated endpoint relation scores            O(R_e * T * M)
preallocated layer-shuffle relation scores       O(R_l * T * M)
```

replicate 数组在开始前一次性预分配，循环直接写入，不再采用“list + stack + astype”产生三份瞬时副本。每个 replicate 的完整 graph/lineage trace 不会进入列表。

`StructureAudit._score_to_file()` 把“算一个样本、保存 NPZ、只返回小 manifest row”封装成独立作用域。上一样本的大 `arrays` 在下一样本加载前已不可达。

`EndpointSwapPlan` 只在单样本内保存一份不变 CPU edge geometry；不同 null replicate 复用它，避免重复做 GPU→CPU 几何复制。每轮仍需要排序当前 edge keys，所以它主要是 CPU 时间瓶颈，不是跨样本泄漏。

## 3. evaluation 保留 pooled AUPRC，但不堆积 replicate 维

旧的直接实现把所有样本的 `[replicate, token, relation]` 同时留在 RAM。简单改成正负均值差会改变预注册 pooled-AUPRC 统计量，因此当前实现使用独立模块 [`bounded_ensemble.py`](bounded_ensemble.py)：

```python
from experiments.non_neural_structure_audit.bounded_ensemble import DiskBackedAUPRC
```

它把 label、real score 与 null ensemble 写入 evaluation 输出目录下的临时 memmap，并逐 replicate 计算 pooled AUPRC。性质是：

- 数值定义与原始“拼接全部 token 后算 AUPRC”一致；
- RAM 只需单样本 NPZ、单 replicate score view 和小结果矩阵；
- 临时文件在 evaluation 正常结束后删除；
- 磁盘 scratch 上限约为 `(R_e + R_l) * T_total * M * 4 bytes`，另加很小的 real/label 文件。

`evaluation_data.py` 只从 NPZ 读取实际使用的字段，并在单样本 helper 内把完整 ensemble 折叠成 per-sample null mean。独立模块 [`bounded_samples.py`](bounded_samples.py) 将 real/final/null-mean/layer-mean 四个 `[T,M]` 矩阵以及 label/eligible 写入 memmap；`FrozenSample` 只持有对应磁盘切片。evaluation 持久 scratch 额外约为 `4 * T_total * M * 4 bytes` 加标签，结束后自动删除。

压缩 NPZ 仍要求把当前样本的一组 3D ensemble 解压到 RAM。loader 先处理 endpoint、写入磁盘并释放，再处理 layer；不会同时保留两组。`DiskBackedAUPRC.add_masked()` 逐 replicate 应用 content mask，避免 `null[:, mask]` 再复制一整组 ensemble。因此 evaluation 单样本该部分的峰值约为 `max(R_e, R_l) * T * M * 4 bytes` 加一个 `[T,M]` replicate/mean scratch，而不是两组 ensemble 与布尔索引副本相加。

relation audit 每次只拼接一个 relation，临时 heap 为 `O(T_total)`。A9 的 scikit-learn additive/interaction 比较需要一次完整紧凑设计矩阵，并可能产生 pairwise interaction 展开；它是 discovery-only、当前不授权模型，也是剩余的 evaluation 内存峰值。未来扩到远大于当前 449-sample test split 时，应把 A9 单独调度/分批或关闭，不能把 memmap 误解为所有下游算法都零拷贝。

## 4. 有界 train reference

`ReferenceAccumulator` 对每个 `(task, causal-position bucket)` 只保留 `reference_capacity` 条 reservoir rows：

```text
O(number_of_task_buckets * reference_capacity * L * F)
```

这可能仍是数百 MiB，但有显式上限。可通过一键脚本的 `REFERENCE_CAPACITY` 调整，不是随 train 样本数线性增长的泄漏。

## 5. 为什么不每条样本调用 empty_cache

`torch.cuda.empty_cache()` 只归还 allocator 中已无活引用的缓存块，不能释放仍被 tensor、NumPy view 或 kernel 引用的内存。逐样本调用会破坏 allocator 复用并降低速度。

判断泄漏应区分：

- `torch.cuda.memory_allocated()`：当前活 tensor；
- `torch.cuda.memory_reserved()`：allocator 保留池；
- RSS：还会包含 Python/NumPy allocator 的高水位和文件页缓存。

只有 `allocated/live objects` 随样本数单调增长才是主要泄漏证据；`reserved` 或 RSS 保持高位本身不是。

## 6. Windows 峰值监控

可按确切 PID 监控一键脚本启动的 Python 进程。更简单的做法是在另一个 PowerShell 窗口执行：

```powershell
$process = Get-Process -Id <PID>
while (-not $process.HasExited) {
    $process.Refresh()
    '{0:o},{1:F1},{2:F1}' -f (
        Get-Date,
        $process.WorkingSet64 / 1MB,
        $process.PrivateMemorySize64 / 1MB
    ) | Add-Content memory.csv
    Start-Sleep -Seconds 2
}
```

正式 profile 应至少记录：样本 ID、`E/T/L`、null replicate 数、RSS peak、CUDA allocated peak 和单样本耗时。`BLOCK_ROWS` 只约束 CSR 解码临时块；最终 `RoutingEdges` 仍含当前样本全部 retained edges，不能把它误解为整个 graph 的硬上限。

## 7. 已有回归测试

```powershell
python -m pytest `
  tests/test_attention_lifecycle.py `
  experiments/non_neural_structure_audit/tests/test_experiment.py `
  experiments/non_neural_structure_audit/tests/test_bounded_ensemble.py -q
```

测试覆盖正常/异常 attention 释放、前一 fit 样本 graph 的弱引用消失、score 在 null 前释放原始 attention、evaluation 每个样本只实际加载一次 attention、逐 replicate masked write、预分配 artifact 形状、compact sample memmap，以及 disk-backed ensemble 与原始 pooled-AUPRC 定义严格一致。
