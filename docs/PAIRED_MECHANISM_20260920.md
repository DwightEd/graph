# 重采样正负配对：检测对象、机制检验与实现

2026-09-20。代码起点：graph `13acdbf`、reanchor `617dc70`。
本文件更新当前方案；既有发现见[证据台账](DETECTION_CONVERGENCE_20260919.md)，
前轮核查见[18 项原始文献](LITERATURE_TRANSPORT_20260919.md)。

## 1. 先明确要检测什么

研究对象是：**对当前陈述适用的证据，是否实际控制了内容选择；若未控制，偏离通过哪些
head、下游计算和历史路径产生并持续。** 模型内部冲突只是待检验的解释之一。
错误也可能表现为各路径一致支持一个不适用的值；正确生成也可能需要消解竞争解释。
因此不能将“冲突大”“状态变化大”“注意力低”直接定义为幻觉。

| 要检验的对象 | 可操作的观察和实验 | 必须排除的解释 | 本轮状态 |
|---|---|---|---|
| 证据是否适用、能否确定当前陈述 | 独立核验对象、阶段、范围及候选；区分适用来源和含目标词但不适用的来源 | 材料本身有歧义、缺少相关事实；不支持不等于世界中为假 | 两个局部事实有人工角色表；未穷尽全部候选或核验整个回答 |
| 适用证据是否控制选择 | 同一物理 head 分别删适用来源、不适用来源、回答历史；观察完整候选差和绝对概率 | 读到了值，但值已在来源位置被上下文化；直接限定词边小不表示条件未编码 | 已实现有限删除；不把删除解释为语义绑定证明 |
| 多头是在抵消、冗余还是有条件地改变作用 | 保存 full、−A、−B、−A−B，比较单独作用、条件作用及非加和余项 | 梯度符号误判；共同下游非线性；同层 head 没有直接顺序连接 | 已实现；正负两侧使用同一组 head 和配对 |
| 下游是在补偿还是扩大上游偏离 | 删上游后，将选定下游 head 写入或 MLP 写入恢复为其基线值 | 原值写回本身的数值扰动；旁路、其他中介仍在作用 | 已实现同分支写回和 sham；不估计唯一自然中介比例 |
| 起点影响是否延续到后半段 | 仅在起点 receiver 干预，后续固定原文本；另在当前步删除作比较 | 每一步反复干预；普通语法续写；标签连续性带来的分数平滑收益 | 已实现起点→后半段/子句后；后续读出是实际词 logp，尚不证明错误语义传递 |

先做标签辅助的机制审计，再决定是否将某种关系纳入检测。最终无监督评分不能使用
这里人工给出的真假候选、金标边界或依据本次测试结果选择的 head。

证据清晰度与模型置信度必须分开：材料可以明确支持一个陈述，模型仍犹豫；模型也可
对材料未提供的时间十分确信。扩展配对集时单独记录“明确支持 / 明确排除 / 信息不足 /
证据冲突或歧义”，将后两类作为独立层，不强行合并为内部冲突型幻觉。

## 2. 既有结果如何约束方案

graph 的既有监督分析中，CHARM current 与 past-EWMA 在共同覆盖词上 AUROC 为
.88415/.88243，但每答首错对正常为 .79470/.62585。这说明连续性对总体成绩有很大作用，
同时并未解释全部当前词信息。数字来自前轮[证据台账](DETECTION_CONVERGENCE_20260919.md)，
本轮没有重训，也没有把监督分数的平滑称作无监督方法。

reanchor 的服饰案例中，L31H14 两侧 self attention 都约 .996，作用方向却不同；
大部分直接“历史效应”来自当前 query。中层 evidence 可以仍然支持有依据候选，
最终选择却偏向无依据候选。因此本轮：保留 head 身份，拆开 self 与更早历史，
检查条件作用和下游重算，并将 onset 与后半段分开报告。

对上传 v3 结果的本轮实际重算，阈值固定为 .05 nats：

| 项目 | 实际结果 | 能说明什么 |
|---|---|---|
| 独立来源 / 面板 | 2 / 6 | 面板包括措辞控制，不是 6 个独立问题 |
| 完整候选、全剂量单删 | 60 条，其中 2 条梯度与有限效应显著异号 | 梯度只能筛候选，不能直接命名功能 |
| 完整候选、全剂量双删 | 24 组，其中 11 组两条单独作用显著异号 | 存在输出方向抵消的实例；不等于 11 个“幻觉冲突机制” |
| 标为 opposed 的候选配对 | 2 组未通过上述有限作用异号标准 | 梯度选点标签不能沿用为结果标签 |
| 原值写回误差 | 最大约 .073704 nats | 接近该幅度的恢复结果不能直接解释为机制 |

可复核表格在 [results/paired_audit_20260920](../results/paired_audit_20260920/README.md)。
它们属于已有 8B 输出的描述性重分析，既不是新增 8B 干预，也不是检测性能。

## 3. 文献带来的具体调整

本轮重新核对以下原文的方法段落；不以论文的总体结论代替本项目中的验证。

| 文献 | 可借用的内容 | 对本方案的具体影响 |
|---|---|---|
| [Graph of States, v2](https://arxiv.org/html/2603.21250v2) | 外部神经符号推理系统：症状、证据、假设图，支持/反驳关系与回溯状态机 | 保存证据来源、适用关系和未决项；它不是 Transformer token 的内部信息图，不据此强行设计两个潜在状态 |
| [Information Flow Reveals When to Trust](https://github.com/rxu0112/RAG-information-flow/blob/main/Information_Flow_Reveals_When_to_Trust_Language_Models.pdf) | 原生 AVWO 来源向量；区分来源贡献与外部相关性，并做答案可信度校准 | 保留真实消息和目标反查；不沿用先合并 head、非负裁剪后的量研究正负竞争，也不将校准器称为无监督 token 检测 |
| [Hallucination Begins Where Saliency Drops](https://arxiv.org/html/2601.20279v1) | 以注意力乘梯度衡量候选对上下文的敏感性；在视觉语言生成中研究局部上下文依赖 | 将绝对显著性与有符号作用区分；“较少依赖历史”是竞争假设，不将视觉模型结果当作自然 RAG 的普遍定律 |
| [RAUQ, v3](https://arxiv.org/html/2505.20045v3) | 选关注前一 token 的 head，递推置信度，最后聚合到答案级 | 加入熵、本地置信度、无 attention 的 EWMA、严格只读前缀的 RAUQ-style 对照，检验增量是否只来自平滑 |
| [The Hydra Effect](https://arxiv.org/html/2307.15771v1) | 删除后的下游补偿；直接作用与网络总作用不同 | 加入下游 head/MLP 基线写回；单删小不解释为不重要 |
| [AtP*](https://arxiv.org/html/2403.00745v1) | 单点梯度可受饱和与抵消影响 | 有限剂量验证；随机 head 从完整模型 head 集合抽取，不能只从梯度候选剩余项抽 |
| [EAP-IG](https://arxiv.org/html/2403.17806v2) | 行为保真不能由 head 集合重叠替代 | 以真实候选读出验证；本轮单点 VJP 和两剂量删除不是 EAP-IG 复现 |
| [Activation Patching Best Practices](https://arxiv.org/html/2309.16042) | 干预基线和读出指标会改变结论 | 增加候选各分支自己的 sham；零删除/随机方向是数值及幅度控制，不能替代语义反事实 |
| [How do Language Models Bind Entities in Context?](https://arxiv.org/html/2310.17191v2) | 通过因果交换分离内容和绑定结构 | 指针、地址、载荷必须接受独立交换检验；不直接把 Q、K、V 命名为语义角色 |

Saliency 论文的基本量为

\[
S^{l,h}=\operatorname{tril}\left(\left|A^{l,h}\odot\frac{\partial\mathcal L}{\partial A^{l,h}}\right|\right).
\]

论文再合并 head、归一化，在选定层和先前输出位置上聚合。它是对所定义候选损失的
局部敏感性大小，取绝对值后不能区分支持与反对。我们使用完整候选差的**有符号 gate
导数**定位消息，再运行有限干预；两者不是同一个分数，本轮也未复现其视觉拒采样实验。

RAUQ 的递推形式为 `c_t = α s_t + (1−α) a_t c_(t−1)`。
原文用完整回答选择 head 并报告答案级结果。本轮前缀对照为各 head 单独维护递推值，
只凭已出现位置选择 head，读取当前词分数；不将未来位置用于早期选头。
Instruct 对照采用 `s_t = log(V) − H(p_t)`、`α=.9`，另报去掉 attention 的 EWMA。
重采样缓存熵的单位是 bits，读取时转换为 nats。首词没有前一回答词，直接用 `s_0`。
这明确命名为 **prefix-RAUQ-style**，不是原论文的原样复现。

## 4. 配对样本与时间索引

目前核实的原始重采样是 4 题 × 4 seed，共 16 答。同题同模型不保证整答一正一负。
当前可用的审计标注是两个局部事实对：14315 的头饰范围、14375 的洋葱烹饪阶段。
两对共涉及 4 答。服饰 supported 的较早文字含另一个过度概括，不能充当完全正确历史。
洋葱“10–12 分钟”是对该阶段缺少来源支持，不是在现实世界中证明不可能。

标注写入 `paired_cases.json`，包含 source、seed、原文目标、候选、来源角色、历史核验状态。
扩展同一 JSON 列表即可加入新的核验事实对。不能把 RAGTruth 原回答的字符标签复制到
新生成回答，不能把同一道题的多个 seed 当作独立来源。

读原采样的 `settings.json`、`samples.jsonl`、`prompts.jsonl` 与逐答 NPZ，使用已存 token ID。
不重建聊天模板、不重采样、不覆盖原始 cache。生成器和观察模型必须相同；模型文件换路径
只能指向同一套权重，还要通过原始采样 logits 和 AVWO 重建回放检查。

令 prompt 长度为 P，待预测回答词为 t。NPZ 的第 t 个 attention 行对应 `q=P+t−1`，
其 self key 是最后一个**已输入** token。将回答目标映射为 `[start, stop)` 后，有限干预观察：

| 阶段 | 位置 | 读出与边界 |
|---|---|---|
| before_claim | start−1 | 实际下一词 logp；相邻位置尚未被核验为正常 |
| onset | start | 完整 supported / unsupported 候选 logp 差；另存各自概率、首词与长度归一化差 |
| back_half | start + max(1, floor(span_length/2)) | 实际下一词 logp；一词片段没有该阶段 |
| post_claim | stop | 实际下一词 logp；子句后不自动等于恢复正确 |

越界、EOS、缺少阶段直接记缺测，不补零。正负侧各保存自己有效的阶段；只有共同可观测
的相同 phase/head/source/readout 才配对。四个有限干预点是初轮预算，不代表逐词完整机制曲线。
轻量 confidence 对照另外保存完整 token 序列。不同阶段的读出不是同一语义量，不能连成
一条“事实支持反转”曲线。要检验功能转变，需要再核验各阶段同一事实的条件候选。
before 阶段也使用从 onset 事后选出的 head，因此该项是回溯机制审计，不是提前报警成绩。

## 5. 原生消息、单独作用和联合作用

每条消息保留实际 layer、head、receiver、source：

\[
m_{l,h,j\to q}=W_O^{l,h}\left(A^{l,h}_{qj}W_V^{l,g(h)}\tilde x_{l,j}\right).
\]

其中 `g(h)` 是 GQA 的 KV head 映射。删除是在原 attention 输出中减去指定来源的实际写入，
不重新归一化 attention；后续层正常重算。`head_total` 仅删除**该 head 在指定 receiver**
上的全部来源写入，不是全位置关闭这个 head。

每个 head 分别测量 evidence、inapplicable_source、history、query_self、prior_history、
claim_history、head_total。`history` 是已输入回答的全部位置；`prior_history` 排除当前
query，`query_self` 单独包括该位置。第一回答词的 query 仍在 prompt 内。
这些分组包含嵌套关系，不能相加当作互斥来源分解；联合干预只用预先指定的不相交组合。
`claim_history` 只含目标子句已输入部分，onset 时为空，记未测而非零效应。

onset 的读出为

\[
F=\log p(c^+\mid x_{<t})-\log p(c^-\mid x_{<t}).
\]

它是两个人工候选的对比，受长度、措辞影响，候选也未必穷尽合法回答。其他阶段的 F
为实际下一词的 logp，正值作用表示促进该词，不具有跨真假侧统一的“支持正确”方向。

对单位 A、B 保存：

\[
D_A=F_{full}-F_{-A},\qquad D_{A\mid\neg B}=F_{-B}-F_{-A,-B},
\]
\[
J_{AB}=F_{full}-F_{-A}-F_{-B}+F_{-A,-B}=D_A-D_{A\mid\neg B}.
\]

必须一起报告 D_A、D_B、D_AB、两个条件作用和 J。两条 D 异号表示对同一读出的作用方向
相反，并不证明 head 在物理上互相抑制。J 的符号也不自动决定协同：例如两个单删都接近
零，双删损失 2，则 J=−2，可能是冗余而非“负协同”。同层 head 的 J 可以全来自共同下游。

先对两侧 onset 的完整候选目标计算逐层 VJP，以**当前 receiver**的逐 head 来源导数
绝对值之和排序，取两侧合并的前 4 个物理 head；再从剩余全部模型 head 中随机取 1 个控制。
同一个 head 表用于两侧和所有阶段。head 配对、来源配对在有限干预前冻结。
这使用核验候选和同一发现集，属于探索性选头，不能当作留出来源上的确认性检验。
随机 1 个 head 也不能排除梯度筛选漏掉其他重要组件。

两个剂量为 .25 和 1。每个单位有同分支原值写回；每个 head_total 有等范数随机方向控制。
下游恢复分别作用于选定 head 消息和同位置 MLP。只在基线误差通过门槛后解释恢复量。
本轮将 prefix / correct / wrong 三种候选前向的基线分别缓存，避免用短前缀的 bf16 值
恢复到不同长度的候选分支。微型真实 Llama 验证了此修正；仍须在用户 8B 上重新确认
sham 是否消失，不能说已经解释了全部旧误差。

时间持续性只在 onset receiver 删除一次，保留后续原生成 token 不变。它检验给定文本路径
下的计算影响传递，不等于自由生成改变了错误持续时间，也不等于已找到唯一历史中继 head。

## 6. 指针、地址、载荷：下一项有语义判据的实验

自然配对先定位候选组件，再构造小规模控制组。三者以实验结果命名，不先假定 Q/K/V
与它们一一对应；上下文化 K/V、残差和 MLP 都可能携带多个因素。

| 假设角色 | 只改变什么 | 可检验预测 | 关键反例/控制 |
|---|---|---|---|
| 指针：当前要找的对象/阶段 | 交换目标身份或角色表示，接收样本内容不变 | 改读接收样本中新目标的值 | 第三答案：不能仅复制 donor 的值；同角色改写不应改变选择 |
| 地址：该角色对应哪个来源位置 | 调整角色—来源位置对应，内容值固定 | 读取应按新对应移动 | 交换物理位置、控制 RoPE/距离；相同词面不同关系 |
| 载荷：所选来源提供什么内容 | 固定目标与角色对应，只替换值 | 只在该值被选中时改变答案内容 | 换未选值不应同样生效；不应改变当前目标身份 |

采用对象/阶段 × 值的交叉控制，以及同关系换词面的控制。自然材料缺少唯一语义映射时
保留 unknown，不用一个高 attention 地址补成真值。当前本轮代码**没有实现或验证**这一
组语义交换，也没有训练绑定 probe；现有有限删除不能替代它。

## 7. 要不要两个状态，如何用图存储

可以保留“历史来源写入”和“当前外部证据写入”两个观察视图，但它们不是两个互斥、
语义纯净的内部状态：历史位置已经混合证据，prompt 的 value 也已上下文化。现在不新加
双状态网络。先比较条件作用，确定需要保留哪些关系，再选择学习结构。

当前存储为带类型和干预记录的关系表，保留转换到时序因子图的能力：

| 文件 | 记录的对象 |
|---|---|
| pair_inventory.csv / reviewed_case.json | 原题、seed、局部核验范围、来源引文和历史状态 |
| supported/unsupported_discovery.npz | 保留 layer/head/receiver 的梯度量、target 各 source 的量；CSV 保存稀疏候选边 |
| plan.json / 每阶段 context.json | 共用 head 表、配对、确切 token IDs、source/receiver 位置、读出 |
| 每阶段 baseline.npz | 不同候选分支的逐 head/来源组消息向量及 MLP 基线 |
| worlds/*.npz | 实际执行的干预和各候选评分所对应的缓存；world 名及冻结 context 定位实验 |
| effects / interactions / adaptation / persistence.csv | 单位作用、四世界关系、恢复、跨 token 干预结果 |
| paired_*.csv | 同题同组件的 unsupported−supported 差；缺测不补零 |
| screen/*.npz | 全词置信度对照、每步选择的 head 及前一词 attention |

图中应区分原生消息边与“两个组件共同影响某读出”的因子。J 是四世界对比，不能当作
head→head 的传输权重；标量 attention 连乘也不能声称精确分解原 Transformer。
现阶段 CSV/JSON/NPZ 足够，未增加图库、GNN 或额外训练任务。

若后续学习：先比较相同输入的独立 head 模型、head 关系模型和纯时间平滑；再加入有
端点的消息图，做端点置换、去时间、去来源角色的对照。只有在独立来源中超过这些对照，
才将增量归于图结构。使用人工真假标签训练的版本须明确称为监督诊断模型。

## 8. 验收、停止条件和运行

先同题同物理组件配对，再按 source 汇总。多个 head、剂量、token、seed 和措辞面板都
不是新的独立来源。当前仅两个来源，不报告自然总体比例；bootstrap 标为探索性，
单个 head 若仅一个来源覆盖则不计算区间。必须报告缺测和失败 sham，不能挑头后报总体。

若冲突同时见于正负，停止将冲突量当真假评分；若后半段增益不超过 EWMA，停止把它
解释为端点图的收益；若仅当前词概率变化、没有语义控制，停止命名错误载荷的持续。
后续确认需新增独立来源并冻结 head/关系假设，再覆盖完整回答评估首错、延续、子句后误报。

默认入口：

```bash
python -u main.py flow
```

默认读取原服务器的 `reanchor/outputs/samples_20260911_145421_235`，从其 settings 读取
原 Llama-3.1-8B-Instruct 路径，写入 `outputs/paired_head_transport_v1`。同命令复用完成
的逐 world NPZ；修改候选/剂量/选头设置需新输出目录。新配置拒绝混用旧协议。
默认两对、四阶段、5 个 head 的多种干预需要千次量级原模型前向，按 head/source 和
配对显示进度；不是一次低成本检测。单层 attention 使用后释放，不保留整模型 attention 栈。

```bash
python -u main.py flow --stage inventory
python -u main.py flow --stage screen
python -u main.py flow --stage report
```

inventory 仅核对文本、配对和 cache 可用性；screen 需要原 attention NPZ，不加载模型。
只有模型及原始缓存可访问时 run 才能完成原生干预。其他位置可显式传 `--samples`、`--model`。
可用 `--cases` 输入更多已核验事实对，不需要修改来源角色的 Python 代码。

只重算已有 v3 CSV：

```bash
python -u main.py flow --stage review \
  --review-input <解压后的v3结果目录> \
  --output results/paired_audit_20260920
```

旧目标消息边实验保留为 `python -u main.py flow-edges`，新写回协议写入
`outputs/target_transport_v4`。读取 v3 的旧报告使用
`python -u main.py flow-edges --stage report --output <原v3目录>`，无需重跑大模型。

本地已实际核对 16 答文本和两对目标、重算上传 v3 CSV，并用随机初始化的真实微型
Hugging Face Llama/GQA 前向测试索引、有限干预、候选分支恢复、时间因果方向及续跑。
当前环境没有用户的 8B 权重、CUDA 和这批逐答原始 attention NPZ；因此尚未执行新协议
的自然 8B 审计，也没有新 AUROC。软件测试与自然机制证据分别验收。

本轮相关回归共 82 项通过；新配对模块的 10 项在最终补充 FP32/bf16 的 head 与 MLP
分支写回检查后再次全部通过。测试环境为 torch 2.5.1+cpu、transformers 4.51.3。
