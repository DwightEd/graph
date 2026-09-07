# 审稿意见与收敛后的实现：约束响应是否经过中继影响输出

评审日期：2026-09-07。评审对象为 `bd9abb6` 的 attention rhythm / native relay 方法及已报告实验。
本文件将审稿判断、已实施修正和剩余研究问题分开。没有新的 8B/RAGTruth 结果。

## 1. 审稿判断

**如果按“机制驱动的无监督幻觉检测”提交，当前倾向拒稿。** 研究问题有价值，逐 head 原生计算的实现方向合理，
但现在的证据既不足以认定事实机制，也没有支持新检测器的性能。代码复杂度或新命名不能补足这两项。

| 主要问题 | 为什么影响论文成立 | 本次修正／剩余工作 |
|---|---|---|
| 研究目标与测量不一致 | WAAD/FAI 测距离与后续读取；不能判定事实是否进入、绑定、保留或发挥作用 | 改用有明确候选判断的条件配对；自然节律保留为结构对照 |
| WAAD 是有选择偏差的唯一入口 | 持续读取、远处来源替换及无峰的有效更新会被漏掉 | 配对采集不筛 head/位置；从下游 q 的内容响应定位 carrier，不要求 WAAD 峰 |
| DAG 缺少内容联系 | 两条 attention 边存在，不等于第一条传入的信息被第二条使用 | 保存真实 AV/W_O 变化；冻结两跳路径后检验其 value 通道的实际输出效应 |
| observed−runner 不等于事实方向 | observed 可能错；不同节点各用自己的下一 token，不能串成同一事实链 | 同一实验固定一个事实候选差，贯穿 head、MLP、RMSNorm 与确认 |
| MLP 与归一化没有得到合适解释 | 负写入不能自动叫覆盖，MLP 也不等于参数先验 | 原生两端差分，单列最终归一化项；不把模块符号命名为四种失败 |
| 统计循环和混杂 | 按峰选点再证明峰附近变化大是循环；uniform-null 未保留自相关；head 数量多 | 配对条件与路径先冻结，控制条件沿用相同路径；新报告不发布逐 head 显著性结论 |
| 检测可用性不明 | FAI 使用未来；低敏感性不一定幻觉；已有来源异常弱于位置基线 | 本入口不使用未来 token，不凭条件敏感性造 detector；最终评分仍需独立验证 |
| 实验范围不支持普遍性 | 详细写入只在少量展示样本；RAGTruth test 已反复参与设计 | 首先用小型可控实验验证运算；之后必须转到真实、新来源和不同模型 |

另一个必须核查的问题是 **generator 与 observer 的身份**。如果 RAGTruth 答案由其他模型生成，只在 Llama 上
teacher-force，该实验研究的是 Llama 对给定答案的条件计算，不能直接声称发现了原生成者产生幻觉的机制。
原生轨迹实验须保留 `generator_model`；真正的生成机制结论需要身份相符的回答。

当前 ALL 的 routing_joint AUROC=0.5302、routing_static=0.5511，低于 position=0.6294；
pattern_distance=0.4488。它们支持停止当前异常评分主线，不能否定内部状态中存在可读信息。
没有 WAAD 的同口径优越性结果，也不能根据 test 把分数翻转后宣布成功。

## 2. 文献给出的要求，而不是借来的机制结论

- [Attention Illuminates LLM Reasoning, v2 §4](https://arxiv.org/html/2510.13554v2#S4)
  不只画锯齿，也比较高／低 FAI 位置的 token 替换对后续 rollout 的影响，并将信号用于 RL。
  本项目不把这些结果迁移成 RAGTruth 的事实因果证据，且继续保留逐 head 数据。
- [How do LLMs Compute Verbal Confidence?, v3](https://arxiv.org/html/2603.17839v3)
  结合路径阻断、状态替换和解码，对缓存后再读取的解释提供多种证据。它提示“可复用节点可能存置信度”，
  而不是允许把所有复用节点命名为事实 hub。要区分功能，必须定义相应输出目标与对照。
- [Have Faith in Faithfulness](https://arxiv.org/html/2403.17806)
  指出 circuit overlap 不能代替对模型行为的忠实性验证。类似地，本项目的峰值配对、图连通和向量相似不能替代功能检验。
- [AtP*](https://arxiv.org/html/2403.00745v1) 与
  [Towards Best Practices of Activation Patching](https://arxiv.org/html/2309.16042v2)
  提醒我们关注梯度饱和、效应抵消、干预方式与目标。逐边删除昂贵，但一阶梯度也不是自动可靠的答案。
  本实现用两条件精确运算记账筛出有限假设，再对冻结路径确认；不枚举每条边。

这些工作已包含 attribution、patching 和内容／路由分析思想。下文的乘积差分恒等式、logit 记账、路径替换
都不能单独声称为新算法发明。潜在贡献必须来自新且重复的功能失配规律，以及它在独立数据上的检测价值。

## 3. 收敛为一个问题

**在后续位置读取某个中继时，中继对输入约束的响应，是否通过该读取通道改变了同一个事实判断？**

第一阶段只检验一个可证伪预测：改变相关约束，应比改写措辞或改变无关事实更显著地改变注册的判断；
其中一部分变化能通过冻结的中继 value 路径传到输出。不假定每个样本都有 hub，也不假定幻觉都缺少这种路径。

“到达但未绑定”“绑定后覆盖”“仍在但沉默”需要额外实验，不能从本阶段的一张投影图直接命名。
尤其缺少事实可用性／需求判断时，正常 history 复用与 missed re-anchor 仍不可识别。

## 4. 已实现的核心算法

### 4.1 两个有效条件与一个固定判断

输入是两个 token 对齐的 prompt、同一个仍然有效的回答前缀 H，和预先登记的候选 a0/a1。
输入在 query q 截止，预测 q+1；没有 q 之后的回答泄漏。定义固定方向

\[
F(P,H)=z_q(a^1)-z_q(a^0),\qquad \Delta F=F(P^1,H)-F(P^0,H).
\]

例如材料写出 A=red、B=blue；P0 问 A，P1 问 B，H 只写“所问对象的颜色是”，尚未承诺颜色。
固定比较 blue−red。无关对照修改 C 的颜色；措辞对照把 Records 改为 Details，仍然问 A。
另外增加一致重命名对照：将 A/B 在材料和问题中的所有指代同时交换，正确颜色仍应保持。
它在问题处有与事实翻转相同的 A→B 文字变化，却不改变所指事实，可以暴露单纯的符号／词法响应。
四个比较共享 P0、H、候选与坐标。它们不是正确／幻觉标签，事实语义来自受控构造。

外部 JSONL 必须经过语义核验。不能只修改 prompt 后继续使用已写出旧事实的 H；后段冲突需要另行设计的 2×2 协议。
编码器拒绝长度不等的 prompt，不用 padding 假装位置对齐。多 token 候选取共同候选前缀之后的首个分歧 token，
因此测量的是该决策点的 logit 差，不是整个候选句的概率。

### 4.2 精确拆分 AV 变化

令 \(\Delta X=X^1-X^0\)，\(\bar X=(X^1+X^0)/2\)。每条边满足

\[
\Delta m_{s\to q}^{\ell h}
=W_{O,\ell h}\left(\underbrace{\bar A_{qs}^{\ell h}\Delta V_s^{\ell h}}_{内容项}
+\underbrace{\Delta A_{qs}^{\ell h}\bar V_s^{\ell h}}_{路由项}\right).
\]

`symmetric_av` 同时计算两个完整 pre-W_O head code；没有取模长替代功能方向。
所有 head、共享前缀位置都参与；q 的全部来源分别保存带符号两项。GQA 共享 V 只存一次，并保存 query→KV 映射，
这不是平均 head。

MLP 使用两条原生前向的输出差，不线性化 MLP，也不把 MLP 笼统归为“先验噪声”。
恒等式精确不代表两项是独立因果原因，更不等于得到唯一的事实神经元分解。

### 4.3 同一输出方向，最终 RMSNorm 变化单列

设 \(\kappa^i=(D^{-1}\|r^i_{L,q}\|^2+\epsilon)^{-1/2}\)，
\(w=\gamma\odot(W_U[a^1]-W_U[a^0])\)，\(u=\bar\kappa w\)。有

\[
\Delta F=u^\top\Delta r_{0,q}
+\sum_{\ell,h,s}u^\top\Delta m_{s\to q}^{\ell h}
+\sum_\ell u^\top\Delta MLP_{\ell,q}
+\Delta\kappa\,w^\top\bar r_{L,q}+\text{rounding}.
\]

w 在运行前已知，故可以流式先存其投影，最后乘上标量 \(\bar\kappa\)，不需要为读出额外重放一遍模型。
记录 native AV、head 求和、残差相加及最终 norm 的舍入余项。只有写到 q 的量进入输出账本；
不能把载体 b 的投影再加一遍，造成重复记账。

### 4.4 从下游定位候选，WAAD 不作必要条件

对 q 的所有实际 head/source 保留内容项读出

\[
C_{\ell h,s}=u^\top W_{O,\ell h}(\bar A_{qs}^{\ell h}\Delta V_s^{\ell h}).
\]

当前预算化原型取 response 中绝对 C 最大的一条读取边 b→q，且读取层 k>0、b<q。
这是有明确目标的候选筛查，仍不是因果重要性排名。响应为零时不强造候选。
选择相邻 carrier 作为位置邻近对照，固定同一个 root、writer/reader 层与两个 head，仅改变 b，
**未声称严格匹配了 attention 质量或词类**。

为了指定可确认的两跳路径，再在 ell<k 的“被编辑来源 s→b”中选择完整 post-W_O 消息变化最大的边。
这个入口选择仍是预算启发式，只服务于冻结一个可证伪路径；会漏掉分布式、多跳、更弱但关键的形成边。
其重要性取决于确认结果，不能把幅度最大写成最具事实意义。这也是当前原型尚需改进的部分。

每个源组只在 fact_flip 中选一次路径；无关／措辞对照沿用完全相同的 s、b、q、层与 head，
不会为每个对照另找最强路径。新条件的根位置即使未编辑，也仍采其真实状态和系数。

### 4.5 固定预算的两跳 value 路径确认

对冻结 e=(ell,wh,s,b)、f=(k,h,b,q)：

1. 在 P0 的原生前向中，将 e 的 pre-W_O 消息换成条件 1 的该边消息；其他输入不改。
2. 让该变化经真实 attention、残差、MLP 传播到 k 层输入，取得 b 的 patched value。
3. 另一次 P0 前向只在 f 注入 \(A^0_f(V_b^{patched}-V_b^0)\)，读取系数 A 保持原生。
4. 运行真实后续层，用同一个候选差计算

\[
\tau_{e\to b\to f}=F(P^0;f\leftarrow f+A_f^0\Delta V_b^{e\text{-patch}})-F(P^0).
\]

这才是本实验中**指定 value 通道的干预效应**；不是对所有路由逐个消融，也不是完整路径总效应。
从 s 的边写到 b，b 沿层更新，再经 b→q 的 value 通道进入 q，最终由原生 suffix 产生输出差。
不同路径的 tau 不能直接相加作为总间接效应，也不除以可能接近零的总条件变化来伪造“传递百分比”。
该实验有意排除 b 对其他接收者的支路和 reader Q/K 变化。零 tau 不排除其他通路或冗余载体。

每对输入最多 candidate 与 neighbor 两条路径，先落盘，再运行确认。确认失败、对照更强都照实保存，不补选。
是否确认由组编号预算决定，不依据模型是否答对或效应是否大。

## 5. 如何运行、如何读结果

先跑 **8 个程序生成的事实组 × 4 个条件比较**，其中前 2 组确认路径。它检验实现与可控绑定响应，
不等于跑了 RAGTruth，也不能凭它宣布发现真实幻觉机制。普通结构扫描可继续作为自然轨迹观察。

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph && \
git pull --ff-only origin main && \
conda run --no-capture-output -n research \
  python -m experiments.reanchor_flow.constraint_flow_run \
  --model /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct \
  --device cuda:0 --dtype bfloat16 --query-chunk 8 \
  --synthetic-sources 8 --confirm-sources 2 \
  --output experiments/reanchor_flow/outputs/constraint_flow_v1
```

已有结果可用同一入口 `--phase analyze --output ...` 离线重画，无需模型、tokenizer 或数据 cache。
同一命令可续跑；采集身份不一致会要求新输出目录。指定 `--pairs reviewed_pairs.jsonl` 可替换内置输入，
JSONL 每行的格式由 `synthetic_records` 返回值示范，必需字段包括：
`source_id, kind, condition0, condition1, response_prefix, negative, positive, prefix_validated`。
一个 `group_id`（默认 source_id）有一条 fact_flip 及其控制；多个 group 可以属于同一真实 source_id。
`prefix_validated=true` 是输入核验声明，程序并不能自动证明语言事实或历史有效性。

| 输出 | 能回答的问题 |
|---|---|
| `summary.md/json` | 每个事实／无关／措辞比较的固定候选变化、无路径与确认失败；按 source 的描述性区间 |
| `cohort.png` | 同一事实组的四条件连线，不隐藏行为失败 |
| `gallery.html` | 展开核对每一对真实输入文本、共同前缀、候选，再查看其图 |
| `pair_*.png` | 同一目标的内容／路由 head 图、阶段状态、MLP、最终 norm 与冻结路径效果 |
| `pair_*.npz` | 逐 head 完整 code、全部 q 来源记账、阶段向量和可核对输入，均不包含幻觉标签 |
| `pair_*.confirmation.npz` | 已冻结路径、精确 value-path 效应、额外基线及成本 |

先读三个事实问题：候选偏好是否按相关约束改变；变化是否超过控制；冻结路径是否确实传递了部分响应。
仅在这些问题有可重复证据后，才研究真实正常／幻觉样本在同一运算上是否失配。
`fact_delta_minus_absolute_control_delta` 是机制对照差，不是幻觉评分。
合成案例的 bootstrap 只描述此类程序生成案例，不能外推为自然来源或任务总体的置信区间。

## 6. 成本、验证和剩余门槛

一个 pair 使用一个模型、batch=2 的条件前向，无 autograd；每个比较仍重算共同 baseline，因此每组四个比较
实际是 8 个条件前向，不能宣传成 5 次。首次模型验证另计，trace 中有标志。
一个非空确认 pair 另需 1 次 baseline 与每条路径 2 次 forward 调用（其中一次只到 reader 层）。
有 tqdm；记录采集／确认时间、采集峰值显存、磁盘大小和确认调用次数。没有 8B 的实测性能保证。

每次 attention 仅驻留 [2,H,query_chunk,N]。保存的高维数据主要为共享前缀状态与 head code，
成本约 O(L R D + L H R d + L H N + L H R E)，R 是有效回答前缀长度、E 是少量根位置数。
没有保存所有 source/query 的完整向量图；q 全来源是带符号投影，其他边按登记根保存系数，不能冒称无损完整 DAG。

验证包括真实 tiny Hugging Face Llama 的 logits、两端 AV 恒等式、共同候选方向、GQA、chunk 一致性、
无 V 时无内容作用、零替换时零路径效应、层序、控制沿用路径、续跑与不加载模型的离线分析。
这些是数值和软件证据；没有把随机 tiny 模型的图当成研究结果。

后续进入论文确认实验仍需要：

1. **真实数据桥接**：从 research_dataset 的自然前缀构造并核验对照，保留无 hub、无响应与错误样本；
   不能把合成颜色任务当成 QA／Summary／Data2txt 的机制证据。
2. **功能特异性**：多表述、无关事实、位置／编辑幅度匹配；如声称置信度通道，另登记置信度 readout，不能仅看熵。
3. **路径范围**：当前只确认两跳 value 分支；Q/K 路由效应、冗余通路、后段 history 冲突和更长路径都未完成。
4. **正常与幻觉差异**：在独立来源中冻结机制统计，再连接标签；不把每个机制失败强行归为幻觉。
5. **检测迁移**：冻结只依赖部署时原生前缀的评分，再测全 token AUROC/AP、位置／logprob／旧来源基线与成簇差值区间。
   配对效应和人工事实候选不是单次无监督 detector 的可用输入，不能悄悄混进去。

这次修正把“结构图”推进到了可检查的约束响应和限定通道的功能实验；尚未解决最终评分，也没有证明达到 ICLR 的贡献标准。
