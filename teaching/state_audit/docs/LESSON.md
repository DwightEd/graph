# 一次课堂可以讲清楚的流程

## 1. 回答的词与计算的位置不同

运行 demo，读取 `answer.json`，写出 prompt 长度 P。
回答第 0 个 token 由 query P−1 预测，第 t 个 token 由 P+t−1 预测。
修改一个未来回答词再回放：它之前的 query 状态不能变化。对应测试是 `test_future_changes_do_not_change_earlier_states`。

建议先读 `pipeline.py`，然后依次读 `generation.py → capture.py → audit.py`。
先理解每一步输入和输出，再读内部公式。

## 2. 地址、读取结果、实际写入是三个量

运行：

```bash
python examples/read_trace.py --run runs/demo --sample 0 --layer 0 --head 1 --target 3
```

固定一层、一个 head、一个 query，选输入位置 j：

1. attention `a[q,j]` 表示读取强度。
2. `a[q,j] * V[j]` 是该 head 读到的消息。
3. `W_O[h] @ (a[q,j] * V[j])` 是写回 residual 空间的消息。

让学生改变观察的 j、head 和 layer，对比注意力大小与写入范数是否同序。
然后把所有 key 的消息求和，与 `head_readout` 比较；再把所有 head 的投影结果相加，与 `attention_write` 比较。
这些是重建检查，不是幻觉证据。

## 3. 固定位置，移动内容

阅读 `roles.py` 中 `attention_from_keys` 和 `swap_probe`。
保持 query 不变，交换两组证据位置上的原始 key；用目的位置的 RoPE 重新旋转，再计算 attention。
两块均值向量记为 v=(左,右)，交换后为 v′：

```text
positional = cosine(v′, v)
symbolic   = cosine(v′, reverse(v))
```

先用 `[0.1,0.8] → [0.8,0.1]` 解释跟随内容，再用均匀 `[0.5,0.5]` 解释不可辨识。
这里没有把局部交换送入下一层，因而不能从该表直接推断最终回答会变。
同一 head 在不同 token 上也可能有不同结果。

## 4. 从候选到有对照的实验

先读 `measurements.py` 的候选条件；它比较同一个来源在当前 query 和过去窗口的质量。
再打开 `nodes.csv` 查具体 token，而不是只看统计数字。

最后才打开 `span_links.csv`：

- 幻觉之前有候选吗？前置窗口是否完整？
- 同长度、相近位置、相近重复率的正常 span 之前也有吗？
- 候选只在首词出现，还是进入 span 后才出现？
- 未匹配的 span 有多少？是否主要来自短回答或全错回答？

换阈值时换输出目录，保存每次假设。不因为某个阈值让图更好看就称它为检测方法。
随机模型只演示操作；这些问题的实证答案需要真实模型和足够多的独立来源。

## 5. 多 head 的条件作用

运行 README 中的 `intervene` 命令，手算四世界的 interaction。
如果删 A 的影响在有 B、没有 B 两个世界中不同，就观测到了该干预下的条件作用。
再用 `--source history --target 0` 做空来源对照：第一个回答词之前没有回答历史，四个世界必须相同。

接下来才讨论需要什么新实验：事实正确/错误候选的 margin、同题正负回答、同 token 位置的语义控制、不同来源上的重复验证。
不能仅凭目标词 logp 变化命名“纠错 head”或“幻觉 head”。

## 学生扩展作业

1. 转换自己的数据集为统一 JSONL，不改模型采集代码。
2. 加一个纯 NumPy 观测函数，输入原生状态，输出 `[token, head]` 数组。
3. 区分 `labels=null` 与 `labels=[]`，验证新生成回答不会继承旧标签。
4. 若做 LDA，单独写训练脚本，按 source_id 划分；分别报告 onset、continuation 和正常匹配片段的效果。

作业评价看索引是否正确、对照是否有效、结论是否超出证据，不以挑出的最高 AUROC 为唯一标准。
