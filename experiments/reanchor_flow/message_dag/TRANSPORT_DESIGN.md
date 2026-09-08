# 从跳数摘要到可核对的有符号传播图

本方案针对一个问题：模型从局部读取转向远处来源之后，哪些真实中继路径支持或反对后续候选？
远处来源包括 prompt 和旧回答。当前目标不是预先证明“回看后必然自我强化”，也不是给 DAG 换一个名字。
本文区分文献结论、本项目提出的可检验设计，以及已经运行的实现检查。

## 调研如何改变设计

| 原始研究及阅读范围 | 可以借鉴的内容 | 对本项目的限制及落实 |
|---|---|---|
| [Information Flow Reveals When to Trust Language Models](https://github.com/rxu0112/RAG-information-flow/blob/main/Information_Flow_Reveals_When_to_Trust_Language_Models.pdf)，§3–4、算法1、附录G | 实际 WV/WO 消息；从输出逆向找来源；沿合法路径聚合；区分相关性和贡献 | 式4将 L1 贡献截断为非负后归一化，路径乘积无法保留对候选的抑制方向。其最终置信度还使用外部 ranker 和训练的校准器，不能称为本项目要求的纯无监督检测。保留真实算子与逆向读出，改用有符号候选响应。 |
| [Attention as a Hypernetwork](https://arxiv.org/html/2406.05816v4)，§2、§4、局限 | 跨 head 的 attention 向量配置 query–key 专属的有效线性变换 | head 是运算基底，不能先平均后建图。HYLA 改变模型架构，不适合作为已有 Llama 的解释算子。保留每个物理 head 的 WV、WO。 |
| [Language Models Use Lookbacks to Track Beliefs](https://arxiv.org/html/2505.14685v3)，§2、§4–5、附录中的 pointer/payload 分析 | 区分 pointer、address、payload；粗粒度信息流不等于知道中间表征的含义；通过高层变量对齐和交换干预验证绑定 | 不能把任意远处读取叫作约束检索，也不能把 K 的控制来源叫作 V 的内容来源。我们的 V/K 边分别标识，语义绑定另设验证门槛。 |
| [Tracing Attention Computation Through Feature Interactions](https://transformer-circuits.pub/2025/attention-qk/index.html)，QK attribution、head loadings、局限 | 补齐“哪个头承载”和“为什么选择这个来源” | QK 得分还需经过跨来源 softmax 竞争。我们直接保留整行导数中的 V_s−平均V 项；不以单边 attention 作为独立运输概率。 |
| [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)，局部替代模型、归因图、验证与局限 | 节点必须有明确计算含义，图假设需用原模型核对 | CLT 需要另行训练，且该版本固定 attention 和归一化分母。当前不引入替代网络，使用原始 RMSNorm、RoPE、Q/K/V/O、SwiGLU 的解析导数。 |
| [AtP*](https://arxiv.org/abs/2403.00745)，§3.1.1–3.1.2 | 局部梯度的饱和和抵消会造成有限干预归因的假阴性 | 一阶响应只用于局部量化；不能用接近零的梯度证明模型不接收信息。数学测试包含增强 QK 的原生模型和有限差分；语义假设需要有限幅度验证。 |
| [Attention Illuminates LLM Reasoning](https://arxiv.org/html/2510.13554v1)，局部/全局 head、WAAD/FAI、干预 | 内部节律定位比标点或人工幻觉段首更贴近研究对象 | 节律本身不区分幻觉；FAI使用未来读取，不适合冒充在线起点检测。继续用当前及之前 query 的远处增量定位事件，未来只作结果。 |
| [Two Pathways to Truthfulness](https://arxiv.org/html/2601.07422v1)，§3–5 | 问题依赖和答案自身的真实性线索可以不同；需分辨信息来源与最终使用 | 答案自身路径不等于错误路径，该文检测使用 probes。我们的报告不能把 response 中继直接命名为幻觉；无监督分数不使用真实性 probe。 |
| [Grounding latent algorithm routing](https://arxiv.org/pdf/2607.24471)，§3、实验控制、结论 | 路由解释需要结构必要性、表面扰动不变性、定向可编辑性 | 原文结论限于受控任务和从头训练模型。借鉴验证原则，不宣称 RAGTruth 已具有同一算法路由机制。 |

上述调研不支持把“更局部”“多跳更强”直接作为统一幻觉机制。
它支持继续追问：读取的消息对输出有何方向的作用，作用由哪些合法路径产生，是否存在能够重复验证的失配。

## 当前实现缺口与本次取舍

v1 `event_trace.py` 维护完整/固定QK/无MLP三种路径版本，以及0/1/2+跳档位。
同一位置的读取点在各自原层注入真实消息，因此可以测整次回看的局部作用。
但“2+跳”没有告诉我们具体是哪个中继、哪个层/头在影响哪个输出；不同符号也容易在汇总时抵消。

本次实现**按最后一次严格跨位置传递划分路径**。
它补齐事件→中继→目标的有符号作用和物理边记录，不声称已经解耦源 token 内的事实、语法、指针。
读取束仍是事件级的共同增强方向；其不同原始来源没有完整的 source×carrier 联合归因。

不采用无依据的 GNN 聚合器、超图随机游走或把矩阵模长作为事实流量。
也不先枚举指数级路径，或给每个事件×目标都重新反向运行整张图。
把最后一次传递后的路径限制为同位置运算，允许所有输出位置同时计算各自的读出方向。

## 数学对象和图的实际用途

令 b 为事件载体，t=q+1 为被预测 token。各读取点在原 attention 加法后共同注入 εm_e：

    m_e = sum_{s remote} A[l,h,b,s] WO[l,h] V[l,h,s]

δx[l,s] 是完整前缀 DAG 将这个共同增强方向传至第 l 层输入位置 s 的响应。
它不是原始残差，也不是信息比特。最终目标 F_q 是固定候选的 logit 差。

先倒序计算 g0[l,q]：从该层 attention 加法处到 F_q，只允许后续同位置路径的读出导数。
设 B=I+J_MLP，A_qq=I+J_attention(q←q)，则：

    g_post[l,q] = B[l,q]^T g_input[l+1,q]
    g_input[l,q] = A_qq[l,q]^T g_post[l,q]

这是矩阵转置乘向量的解析计算，不训练参数，不调用生产模型 backward。
同位置 attention 包括 Q 对整行读取的控制，以及 self K 对整行 softmax 的影响；不是只取 attention 对角线。

然后正序传播完整 δx，对每条 s<q 的 attention 分支计算：

    c[l,h,s,q,V] = g0[l,q]^T WO[h] (a[q,s] δV[s])
    c[l,h,s,q,K] = g0[l,q]^T WO[h] (a[q,s] δscore_K[q,s] (V[s]−sum_j a[q,j]V[j]))

δscore_K 包含原 WK 和 RoPE；δV 包含原 WV；两者都包含 RMSNorm 分母导数。
Q 是同位置控制，在前缀或后缀中传递，不错误地当作来自被读取位置的内容。

每条曾跨位置的路径，存在唯一的最后一次跨位置边。其之前可以任意多跳，其之后只能同位置更新。
因此以上乘积是“所有前缀 × 当前边 × 同位置后缀”，每条路径恰好出现一次。
这是对**路径集合的分区**，不能简单用完整反向梯度给所有边打分再相加，后者会重复计数多跳路径。

因位置只能向前：最后来源 s=b 的路径恰好跨一次；s>b 的路径至少跨两次。

    sum_{l,h,type} c[l,h,b,q,type]       = response[1 hop,q]
    sum_{l,h,s>b,type} c[l,h,s,q,type]   = response[2+ hops,q]

程序对每个事件、每个可用目标检查这两个恒等式。不满足则报错，不能继续生成看似合理的图。
完全同位置的0跳作用单独保留，不和跨位置边混算。

注意：sum(c) 还原的是 dF/dε，不是 F 本身。不能用这些边重写“当前正确答案占多少，先验占多少”的完整分摊。
MLP 的影响已经在前缀及后缀中；多层 MLP 局部效应不能再全部加到 c 上，否则又重复计数。

## 一个预先固定的无监督候选分数

针对“读取束反对当前答案，但当前答案仍被选择”，定义：

    P(q) = sum_edges max(c, 0)
    N(q) = sum_edges max(-c, 0)
    opposition(q) = N(q)/(P(q)+N(q))

这是**回看之后，传播路径对 observed-vs-runner 的反对作用占比**，不是错误概率。
这里的正负按“最后边相同的路径族”计算；同一族内的前缀已经相加，不能称为所有基本路径的正负总量。
方向事先固定为越高越可疑，不根据 H/N 标签翻转或调参。
每个 query 只使用最近一个严格早于它的已检测事件；如果该事件尚未完成传播，不退回较早事件。
没有此前事件、没有非零路径作用、或使用了外部正确/错误候选的位置保持缺失，不填0，也不混入无监督检测。
该分数目前是条件检测原型，不能称为覆盖所有 token 的完整检测器。

对照均在同一批有效 token 上计算：

1. `direct_only`：只保留最后来源 s=b，检验多跳是否提供额外识别能力。
2. `value_only`：只保留 V 内容分支，检验路由控制的必要性。
3. `negative_logprob`：原候选置信度基线。
4. `relative_position`：原回答内位置基线。

以上前两个对照由同一张图直接求子集，不需要另外删除头或反复运行原模型。
所有样本、所有事件、所有可用后续目标仍然执行；条件覆盖率、H/N比例、无事件样本和未完成事件必须同时报告。
评估使用现有 `detection_metrics.py`：AUROC、AUPRC=AP、按 source_id 配对重抽样的差值区间。
train/test及三类任务分别报告。没有拟合检测器，也不使用 test 标签选层、选头或定分数方向。

## 实验顺序与失败标准

| 阶段 | 实验 | 支持标准与停止条件 |
|---|---|---|
| A：数学与结构 | 原生 tiny Llama 的 RMS/MLP/同位置 attention 对偶检查；所有 V/K 边之和还原1/2+，0跳另存；增强QK；FP32/BF16捕获；候选反转；不同分块 | 对偶及路径闭合必须通过。BF16检查的是捕获点上的平滑局部模型。不能把这些测试当作8B幻觉发现。 |
| A：图不可省略 | 比较一次与多次跨位置作用；故意用完整后缀给所有跨位置边归因，观察多跳被重复计算；同 attention 与前向 MLP 输出、相反消息导数的既有反例 | 必须显示路径次序、算子和读出影响结果。若只用局部边强度就能重现所有目标作用，则图结构没有新增价值。 |
| B：RAGTruth全覆盖 | 原模型、三任务、所有split、所有缓存完整样本；冻结事件定义和分数；相同有效token比较全图/直接路径/V-only/置信度/位置 | 若全图未超过直接路径，不能声称多跳有检测优势。若控制位置后差异消失，不能声称找到幻觉机制。任务失败需原样报告，不能用QA替代Data2txt结论。 |
| C：语义定向验证 | 在明确来源约束与固定候选对上，分别改变实体绑定/否定/时间条件；另做保持事实的改写与位置调整 | 相关约束变化应选择性改变对应边和候选，表面改写应相对稳定。随机方向、无关条件及等幅度控制不能出现同等恢复。当前没有自动将自然文本标成这些语义变量。 |
| C：有限幅度验证 | 对图提出的具体读取消息或载体做成对小幅微扰，比较预测的符号与幅度；必要时检查饱和区的大幅编辑 | 导数接近零但有限变化大时，只能报告局部模型失效。不能称为“模型拒收外部信息”。 |

阶段C用于检验“约束丢失/错误绑定”这样的机制命名，而不是先强行把所有头穷举消融。
一阶相加不能证明联合语义整合：J(u+v)=Ju+Jv。混合二阶响应可以作为后续工具，但非零交互本身仍不证明正确绑定。
Teacher forcing 没有自由生成的采样反馈，当前结构不能证明进入了闭环吸引子。

## 代码、运行和输出

- `differential.py`：原生导数与同位置解析伴随。
- `transport.py`：共享后缀读出、逐物理 V/K 边、路径分区闭合与流式NPZ。
- `event_trace.py`：原完整前缀传播，接入逐边记录；旧路径对照保持原含义。
- `transport_report.py`：固定候选分数、覆盖及同token配对检测比较。
- `event_view.py`：真实中继边的交互视图；显示裁剪不影响计算和存盘。

以下三处真实代码对应图的三个部分：

```python
# transport.prepare_local_readout：从目标倒着经过同位置 MLP/attention。
post = reader + op.mlp_vjp(reader)
write_array(archive, f'L{layer}', post.cpu().numpy())
reader = post + op.attention_same_vjp(post)

# event_trace.trace_events：此前0/1/2+跳响应先求和，允许任意合法前缀。
reader = op.tensor(cut_readout[f'L{layer}'])
for begin, end, a, effects in last_crossing_edges(op, state[0].sum(0), reader):
    for i, row in enumerate(event_rows):
        cut_recorders[int(row)].write(op, begin, end, a, effects[i])

# transport.CutRecorder.finish：按最后载体位置检验1跳和2+跳各自闭合。
net = (self.hop_positive - self.hop_negative).sum(-1)
expected = event['margin_response'][0, 1:]
error = net - expected
```

第一段约束了最后边以后的路径，第二段给具体边提供此前全部传播，第三段检验分区是否重计或漏计。
若去掉第一段的“同位置”条件而换成完整反向读出，两跳路径会在两条边上各被算一次。
这就是图次序在计算中的作用；仅画 attention 连线无法提供这一保证。

```bash
git pull --ff-only origin main
conda run --no-capture-output -n research python -u -m experiments.reanchor_flow.message_dag.event_run --audit experiments/reanchor_flow/outputs/attention_audit_v3 --completed-only --device cuda:0 --event-batch 2 --query-chunk 8
```

默认输出改为 `AUDIT/lookback_events_v2`，不混用v1的完成标记。
`--completed-only`明确跳过原生缓存缺失的样本；没有样本数、事件数、目标数的隐藏上限。
每个事件新增 `edges_<b>.npz`：`L<layer>Q<begin>`存[head,query,source,V/K]的有符号作用；
`AL<layer>Q<begin>`存实际 attention，来源从事件row开始，query/source绝对位置由`row_position`还原。
`value_energy_L*`保留 WV 值向量平方范数，`write_energy_L*`保留 WO 写入平方范数；实际消息能量另乘a²。
特殊token可以作为真实中继参与传播；没有被当成事实根或从原生计算中删掉。

`event_<b>.npz`保存逐层/头、逐中继、V/K、正/负、直接/多跳的完整边汇总及闭合误差。
`transport_scores.npz`保存每个样本每个token的分数及所选事件，标签不参与此计算。
`transport_detection.json/png`报告AUROC/AP、配对区间与覆盖，原gallery连接新结果。

完整边保存在CPU分块写出的压缩文件中，GPU没有L×H×R×R×event的常驻张量。
但完整存盘仍有二次增长：每个事件约3×4×L×H×n(n−1)/2字节的有效边数据，n为事件后的query范围。
例如L=H=32、n=512约1.50 GiB未压缩，另有块内零填充和小型汇总。程序显示估算；不会为省空间暗中丢边。
同位置后缀缓存暂用4×L×R×D字节磁盘，单样本完成后删除，不在GPU累积所有层。

## 当前证据状态

本次验证已完成：

- Transformers 4.57.6 / CPU：`message_dag/tests` 加 `test_message_lineage.py`，44项通过。
- Transformers 4.44.2：新传播模块的9项检查全部通过，包含原生 V/K 消息微扰；RoPE 使用显式 `config=`。
- 数学检查覆盖 FP32/BF16 捕获、QK增强、路径闭合、候选翻转、完整反向读出造成的重复计数，以及非有限值拒绝。
- 六个 train/test×task 分区的 tiny 模型流程检查覆盖全部事件、标签变化、缺失缓存披露、逐边文件缺失后补算，以及删除原模型/缓存后的离线报告。标签是合成测试标签，检测指标不能作为真实研究结果。
- 生成页面在 Node 的 DOM/Canvas 检查环境执行13个目标×3种分支选择及读取点切换，逐边SVG另行渲染核对。这不是完整浏览器兼容性测试。

本环境没有原8B权重、GPU及完整真实native缓存。新代码的数学/流程验证与原数据的机制结果分开报告。
已存在的“总体H更局部、起点附近较多远看”属于关联，不能直接拼成同一个样本的因果轨迹。
这次实现提供可检验的图方法及固定检测候选；是否优于直接路径、是否能命名为幻觉机制，仍由阶段B/C决定。
