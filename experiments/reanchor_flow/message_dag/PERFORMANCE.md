# 全量回看审计的工作量与执行优化

这不是一次普通生成。它先找出某个物理头从近处转向远处的位置，再重建该位置实际读取的
WV→WO消息，计算“定向增强这束消息，会怎样影响后续各token的候选logit差”。
远处可以是prompt或旧回答。三个路径版本是完整、固定Q/K、无MLP路径；
每个版本再保留0、1、2+次跨位置传递，所以一个事件有9组切向状态。
这是局部敏感性审计，尚不能直接证明正确约束被理解或自由生成进入自我强化。

例如日志中的242个事件位置、19016个读取点，表示读取点已按位置合并为242个事件。
原实现每批2个事件，需要121次逐层遍历；32层最多构建3872次层算子。
按日志中6.42秒/事件估算，该样本约26分钟；总进度的ETA还会受后续样本长度与事件数影响。
大量物理头中任意一个满足条件就能触发事件，因此事件可以很密集，不等于242个独立事实边界。

## 原来慢在哪里

计算已经由 `--device cuda:0` 放在GPU：Q/K/V/O投影、softmax导数、RMSNorm与SwiGLU响应都是
PyTorch设备张量运算。CPU负责读压缩缓存、处理结果和写文件。`tangent state alone`只是9组状态，
没有包含逐层权重、Q/K/V、score矩阵、MLP中间值、临时副本或CUDA分配器预留。

旧调度是：

```python
for batch in event_batches:
    direction = final_directions(cache)  # 逐批重复读出
    for layer in layers:
        op = DifferentialLayer(cache, layer)  # 重读权重、解压捕获、构建算子
        # 三个版本分别重建相同的原生 attention
```

`NpzFile`按字段惰性读取，不会替这段循环自动缓存已解压的数组。
此外，逐头 `float(cuda_tensor)` 和 `.cpu()` 会反复要求主机取得GPU结果。
依据：[NumPy NpzFile](https://numpy.org/doc/stable/reference/generated/numpy.lib.npyio.NpzFile.html)、
[PyTorch 性能指南](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)。

## 现在怎样执行

```python
# event_run.py：按1 GiB状态预算形成一组，所有事件仍会处理。
for group in event_groups:
    # event_trace.py：原生算子在一层内服务多批独立事件。
    for layer in layers:
        op = DifferentialLayer(cache, layer)
        op.cache_attention()
        for batch in gpu_batches(group):
            current = cpu_states[batch].to(device)
            advance_one_layer(op, current)
            cpu_states[batch].copy_(current)
```

上述最后三行是调度示意；真实计算见 `event_trace._step`。

- `--state-cache-gib 1` 限定用于共享的CPU切向状态预算。每次只将 `--event-batch` 个事件送入GPU。
  例如242个事件、约244个回答query、D=4096，1 GiB可形成约28个事件一组，约9次层遍历。
  总算子构建量从约3872降到约288；这是构建次数，不是整体加速倍数。
- 小于128 MiB的单层完整attention驻留设备并被所有版本、批次与V/K边复用；更大时保持分块重建。
  该缓存不改变head、来源、dtype或attention数值。单层Q/K也保留在设备上，避免反复传输。
- 相同候选对的最终读出只算一次/样本；明确候选变化使用不同缓存项。v2后缀读出每层每组读一次。
- 逐头远处消息与注入能量改为批量张量运算；仍保存物理head身份及完整来源质量。
  删除了微分算子初始化时不使用的旧分摊MLP计算和下一层状态读取。
- 结果NPZ使用已有 `--cpu-threads` 个线程并行压缩；GPU计算仍是一个进程。
  整组写完后更新索引；部分文件已写完而后续写入失败时，续跑仍按实际文件判断完成。

切向状态的CPU存取会增加主机与GPU之间的数据传输，这是用有限GPU工作空间换取算子共享的代价。
预算不包含其他CPU元数据；GPU峰值另行报告。`--state-cache-gib 0`可关闭跨批共享作运行对照。
分组、分块、线程数不进入科学参数身份；可以直接复用已完成v1/v2事件。未完成的一组需要重算，
内部层进度会持续更新，但总事件进度只在整组完成并保存后增加。

单卡开启多个完整进程会复制状态与算子，并竞争GPU、带宽和缓存文件。
因此这里优先共享原生数据并增大设备批次，CPU文件压缩用线程并行。
不同GPU若分开运行，仍须使用不同输出目录；同一输出目录保持单写入者。

## 核对过的数值与性能

CPU微型原生Llama，3层、4头、D=16、96个query、32个事件；参考代码是提交 `dce125b`。
同一缓存、同一事件，预热后轮换运行5次，下面是中位数。只计传播，不包括捕获、报告与文件压缩。
模型使用 `tests/test_message_lineage.py` 的 `capture_fixture`（seed=82、qk_scale=8）；
输入为 `(np.arange(100)*2)%29`，事件row为2到64的全部偶数，每个事件包含3层×4头。
CPU设置2个PyTorch线程；各配置分别预热一次，五轮中轮换执行顺序。

| 实现 | event-batch / query-chunk | 层算子构建次数 | 秒 |
|---|---:|---:|---:|
| 原代码、原默认参数 | 2 / 8 | 48 | 1.149 |
| 原代码、较大批次 | 4 / 16 | 24 | 0.563 |
| 优化代码、相同较大批次 | 4 / 16 | 3 | 0.420 |

同参数代码优化约1.34倍；连同批次调整，相比原默认约2.73倍。
所有保存字段均核对；候选响应最大绝对差约5.59e-9，来自浮点计算顺序。
没有修改FP32导数定义、候选、事件选择、head、跳数或三个路径版本。
这些是CPU实现基准；本环境没有CUDA/8B权重，不能据此承诺远端GPU加速倍数。
完整回归56项通过、CUDA专项1项跳过；Transformers 4.44.2的相关检查14项通过、CUDA专项1项跳过。
检查覆盖FP32/BF16捕获、缓存/分块等价、V/K边闭合、原生微扰，以及v1/v2改变执行参数后的续跑。

## 原地续跑与真实计时

先在原终端停止旧进程，再更新。下面继续已有v1任务：

```bash
git pull --ff-only origin main
conda run --no-capture-output -n research python -u -m experiments.reanchor_flow.message_dag.event_run \
  --audit experiments/reanchor_flow/outputs/attention_audit_v3 --legacy-v1 \
  --phase all --split all --task all --completed-only --device cuda:0 \
  --event-batch 4 --query-chunk 16 --state-cache-gib 1 --profile
```

运行v2时去掉 `--legacy-v1`；不要为了改变速度参数而另建科学实验输出目录。
更长序列或较少空闲显存时，可将批次/分块退回2/8，跨批共享仍生效。
`--profile`对CUDA阶段边界同步计时，会带来少量计时开销；需要纯吞吐时可去掉。

每个样本的 `index.json -> samples[*].event_execution` 保存设备、组大小、工作批次和总传播时间。
CUDA同时输出PyTorch峰值 allocated/reserved GiB，不包括其他进程及驱动的显存。
开启profile还包含以下字段：

| 字段 | 含义 |
|---|---|
| `state_and_readout_seconds` | 状态准备与候选读出，后续同候选命中缓存 |
| `operator_and_attention_seconds` | 权重/捕获读取、设备传输、层算子及可缓存attention |
| `native_seed_seconds` | 批量构造远处读取消息 |
| `propagation_and_edges_seconds` | 逐批切向传播、状态传输；v2还包含逐边计算和存盘 |
| `readout_and_finalize_seconds` | 目标投影；v2含边汇总与闭合检查 |
| `checkpoint_seconds` | 事件NPZ并行压缩与进度索引保存 |
| `layer_builds` | 实际层算子构建次数 |

v2每个样本一次的反向后缀准备计入总传播时间，未计入上述前向分组阶段。
完整逐边NPZ仍随目标范围二次增长；新增计时会显露这一成本，不将它隐藏在GPU算子时间里。
