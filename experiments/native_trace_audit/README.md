# 原始轨迹审计：读到的约束在哪里失去作用

只处理已核验的头饰、洋葱两个案例、四段保存的自然前缀。复用
`teaching/state_audit` 的 `ModelAdapter`、`CaptureSpec`、KV 回放与存储接口。
没有删除、置零、替换消息、自由生成、训练或全数据集检测。

## 一键运行

在原 research 环境、项目根目录执行：

```bash
git pull --ff-only origin main
bash experiments/native_trace_audit/run.sh
```

默认观察模型路径直接取自归档，仍为服务器原来的 Meta-Llama-3.1-8B-Instruct。
模型移动后使用 `MODEL=/新模型路径 bash experiments/native_trace_audit/run.sh`。
依赖为现有 `state_audit[model]`、NumPy、tqdm、matplotlib；模型接口沿用项目固定的
Transformers 4.57.x。默认 CUDA、BF16，分块预填充 + 单查询反传，共用同一前缀 KV。
不是整段前缀的反向传播。当前查询读取完整历史，未截断输入或合并头。

默认 **40 个查询**：每段自然前缀最后 9 个查询（决策前 8 步和决策步），以及各一个
新增的约束读出。仅决策步计算头/FFN 之间的局部依赖；其余位置记录真实轨迹、
逐边写入及对最终候选差的一阶敏感度。所有 32 层 × 32 heads 都保留。

支持断点续跑，脚本已带 `--resume`；每个查询 NPZ 原子写入。
改变语义设置使用新的 `--output`；工作内存参数 `--prefill-chunk-size 64` 可直接调整。
CPU 软件验证不保证用户 8B 检查点的实际显存峰值，运行中会记录 GPU 峰值。

只核验输入，不加载模型：

```bash
bash experiments/native_trace_audit/run.sh --stage inventory
```

从完成或部分完成的结果重新画图、汇总，不加载模型：

```bash
bash experiments/native_trace_audit/run.sh --stage report
```

完成后直接发送 `outputs/native_trace_audit_v1/review.tar.gz`。
完整原生向量保留在服务器逐查询 NPZ 中；review 包保留逐边量、全部局部依赖、分解、文本和图。
它不是可用于继续采集的完整缓存，不能解压覆盖服务器的完整 NPZ。

## 实验单位与读出

`examples/reviewed_prefixes.json` 保存上传 `reanchor_review.tar(1).gz` 的原始
`prefix_ids`、来源角色、解码片段与局部核验说明。启动时核对角色引文、互斥性、同 prompt
及 tokenizer 解码。没有重新采样，也没有把包含其他错误的历史当作完全正常对照。

| 面板 | 输入 | 候选 | 能解释什么 |
|---|---|---|---|
| 头饰 natural | 原始自然前缀 | caps / a… | 原始首词措辞偏好；冠词和数存在混杂 |
| 头饰 parallel_headwear | 原始前缀后强制追加 ` a` | cap / headdress | 同一头饰槽位的首个分歧 token；是新的诊断前缀 |
| 洋葱 natural | 原始自然前缀 | over / for | 火候与时长的措辞偏好；不是互斥的事实答案 |
| 洋葱 stage_binding | 原始前缀后追加关于已引述时长的诊断语句 | before / after removing… | 已引述 10–12 分钟与移除步骤的阶段关系 |

后两种新增读出不冒充自然生成时已经存在的决策。洋葱原文未给出移除后洋葱的确定时长，
因此不编造一个“正确分钟数”。诊断阶段关系不等同于恢复原始时长选择。
所有分数是**同一前缀的首个分歧 token**，不是不同长度完整序列的总概率。
强制诊断位置没有“实际生成 token”，其 observed surprisal 记缺失，不填正确候选的概率。

前 8 步的候选差只表示对同一后续候选方向的投影，不能当作当时应预测该词或提前发现幻觉。
真实下一词的熵与惊讶度独立记录。query=q 对应预测绝对位置 q+1。

## 三种量严格分开

### 1. 读取与实际写入

每层每头每条边保留原 attention 和

\[
m_{l,h,j}=a_{l,h,j}W^O_{l,h}v_{l,h,j}.
\]

来源分组为 scope、supported_value、value_source、其他 prompt、最近/较早 history、
query 自身及特殊 token。分组覆盖每个可见 key；特殊 token 不伪装成证据，但仍参与总量核算。
不重归一化 attention。分别保存各边能量之和、合成来源消息的范数；后者包含向量抵消。
来源位置只是消息的读取位置，不意味着该表示只编码那个词或那条证据。

### 2. 原始运行的分数分解

取本次实际最终 RMSNorm 的分母 s、权重 gamma，及输出词向量 w：

\[
d=\gamma\odot(w_c-w_e)/s,\qquad
F=z_c-z_e=d^Tr_0+\sum_{l,h,j}d^Tm_{l,h,j}+\sum_l d^Tf_l.
\]

候选 1 为核验的支持方向，候选 2 为另一方向；natural 面板仍有上表的措辞限制。
记录每层 attention 前、中间、FFN 后的残差，以及各头、来源与 FFN 的带符号写入。
该账本是一次真实运行的代数分解，不是删除效应，也不是每层自身的预测概率。

实际 BF16 的 A@V/W_O、残差相加、最终 RMSNorm、lm_head 存在舍入差异；代码显式保存
`attention_reconstruction_error`、两次 `add_roundoff`、`norm_roundoff`、
`unembedding_roundoff` 和偏置项。它们参与 `ledger_total` 的闭合检查，不能归到 FFN 的
“抑制”或某个头的“作用反转”。`numeric_absolute_sum` 同时展示误差量级。

### 3. 后续怎样使用：原始状态上的局部依赖

记录两种接收量 u：

- 后续 head 的 `(scope + supported_value) attention mass − value_source mass`；
- 每层 FFN 的实际输出在本次固定 d 上的投影。

对每个较早 head/来源消息及 FFN 写入，计算 `T = gradient(u) dot message`。
FFN 接收量也包括同层 attention sender。所有原生 Q/K/V、FFN、RMSNorm 路径保持不变。
这里没有关闭 attention 变化，也没有训练替代网络。

接收 head 默认最多 8 个：各层先按上述三类角色的**无符号读取总量**选一个 head，
再按读取量取前 8 个。不使用最终分数、正负号或新实验结果选接收点。
`--receiver-budget 31` 可检查每个非首层的一个接收 head；这不是全 head-pair Hessian。
所有 sender heads 与所有层 FFN 都保留。CSV/HTML 为方便审阅只展示较大的依赖，
完整带符号数组和结构可观测掩码保存在 NPZ 中；未测 receiver 不能被解释为没有作用。

`edge_margin_sensitivity` 另测各边对完整最终候选差的一阶作用，包含归一化和下游适应。
它与直接写入量不同，不能跨层加总，也不能把不同局部梯度相乘当成守恒路径流。
历史 KV 固定只对“当前查询内的消息方向”成立；此实验没有测跨 token 改变历史后的效应。
共现、方向相同和一阶依赖都不是已经证明的非线性协同。

## 看哪些具体表现

先检查读取的实际词与角色，再检查：对应写入是否支持候选、累计支持是否转向、
哪层 FFN/其他头写入相反方向，以及该约束方向与后续读取是否有局部依赖。
报告列出候选差跨零的位置，但不自动把跨零称为“信息删除”或“幻觉原因”。
直接投影为零也不能证明条件信息不存在；它可能只参与后续路由或非线性计算。

reanchor 使用连续前三个已采集查询，比较**当前查询固定的同一来源集合**：
旧来源增量和局部来源减量都至少 0.10，且从局部占优转为旧来源占优。
排除特殊 token，不因滚动窗口让 token 变老而制造切换。前 3 行缺乏观察历史，记未评估。
结构回看不自动等于读回当前事实的适用证据。

## 输出与代码

| 输出 | 用途 |
|---|---|
| `report.html` | 每案例的轨迹、层头图、来源词与局部依赖 |
| `queries.csv` | 每个查询的熵、惊讶度、候选差与数值闭合 |
| `ledger.csv` | attention / FFN 的逐层实际更新与跨零位置 |
| `head_sources.csv` | 决策步所有物理头的分来源读取、范数、正负写入和局部敏感度 |
| `dependencies.csv` | 每个接收量较大的 32 个 head/source 依赖及全部较早 FFN sender |
| `source_edges.csv` | 展示头所读取的具体来源词，保留正负作用 |
| `reanchor.csv` | 共同固定来源集上的逐头切换，含已测阴性 |
| `paired_decisions.csv` | 同案例同面板两种自然前缀的描述性差异，不是干预效应 |
| `cases/.../query_*.npz` | 全量数值及向量，物理层、头、key 身份不丢失 |

teaching 中 `native_trace.py` 管原生采集，`native_ledger.py` 做实际写入分解，
`native_dependencies.py` 做局部求导。实验目录只编译案例、安排查询和输出报告。
原有 `attribution.py`、检测分数和历史实验入口不改变。

软件验证见 `VALIDATION.md`。默认自然样本的新增 8B FFN/残差测量必须在有模型的服务器运行，
不能把随机小模型测试或已通过的输入核验当成这些自然样本的机制发现。
