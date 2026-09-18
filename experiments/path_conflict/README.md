# 同题采样：适用证据与其他路径如何改变候选竞争

## 0. 已确认的数据，不重新生成

实际生成记录：`DwightEd/reanchor/results/routes_20260911_091806_237/` 的
`settings.json`、`prompts.jsonl`、`samples.jsonl`。

原始NPZ目录（来自settings，不由文件名猜测）：

```
/share/home/tm902089733300000/a903202310/lys/research/reanchor/outputs/samples_20260911_145421_235
```

| source_id | 问题 | seeds | trace | 记录中的生成token总数 |
|---|---|---|---|---:|
|14304|网络拓扑类型|0,1,2,3|00000–00003|541|
|14315|Inca服饰|0,1,2,3|00004–00007|289|
|14325|烤猪里脊问题|0,1,2,3|00008–00011|36|
|14375|bratwurst烹饪|0,1,2,3|00012–00015|666|

共16条，1532个生成token（原采样含终止token）。14325的四条是完全相同的拒答，不定为正确或幻觉。
model=本地 `Meta-Llama-3.1-8B-Instruct`；temperature=.7，top_p=.9；max_new_tokens=512；bfloat16。
不是CHARM使用的llama-2回答，不是RAGTruth原六模型的回答集合，也不是O6反事实九世界。
原记录没有官方train/test分区、整答真值或新回答的逐词幻觉标签。
`pasted_samples_20260911` 与该目录的samples.jsonl是同一个blob，不另算16条独立采样。

核验的Git blobs：samples `0b1cab5b2d9bda8e3b144c6b527d8a0a1c44250b`；prompts
`4b4d6fccc9f46a5cba9783d35377ee262591b920`；settings `eaea9f3bd656b767f3a8f117ce98f376f128d787`。
本轮确实在这些文字/配置上运行了inventory。服务器原NPZ/8B权重未挂载到本地，是否仍存在由远端inventory检查。

原NPZ保存token_ids、prompt_length、token_text/token_pieces、完整逐头attention（float16）、top5原生logits、
chosen_logit和log_normalizer（float32）；较新版有logit_entropy(bits)。
attention形状 `[L,H,T,P+T]`，第t行query为`P+t-1`，即采样token t之前。
没有V向量/残差。新的功能干预需要回放原模型，但无需重新采样答案或重建整个数据集。

## 1. 两个局部、可否定的验证案例

`cases.json`明确记录原文依据、目标片段、两侧seed和候选。

- 14315：seed1的caps陈述有来源；seed2把仅统治者可戴的特殊头饰用于一般服饰描述。
  特别注意：seed1前文另有tunic-length泛化，不能把它叫整篇正确或干净历史。
- 14375：seed1的继续小火烹饪洋葱有来源；seed0把移出香肠之前的10–12分钟，指定为之后洋葱阶段的时间。
  标记的是材料不支持这个阶段绑定，不声称现实烹饪一定不能如此。

这些是本次分析者依据来源做的局部标注，非RAGTruth官方标签，也非独立人工验证集。
不把原RAGTruth另一个回答的标签搬到新采样中。只有这两项局部陈述进入正负对照。
两侧使用完全相同的prompt IDs、相同模型和同一对候选；各自自然回答前缀不同，词面/历史混杂保留。
程序不按分数挑样本，也不寻找最好token。所有生成ID从原NPZ读取，不重套可能改变日期的chat模板。

## 2. 问题和可证伪判据

Q1：适用证据是否在当前决策处提供正确支持？
Q2：读到的不适用值或回答历史，是否将竞争推向错误？
Q3：这种作用是否经过后续head/MLP写入，能否被恢复实验阻断？

令M为正确候选首token logit减错误候选首token logit，另存完整候选序列logp差与每token均分差。

```
evidence_support = M(full) - M(cut_evidence)
rival_removal_gain = M(cut_value_or_history) - M(full)
interaction = M(full) - M(cut_evidence) - M(cut_rival) + M(cut_evidence_and_rival)
```

正的evidence_support和rival_removal_gain同时出现，才符合“证据支持与反向支持共存”的候选解释。
必须同时检查正确候选自己的概率是否提高、有依据侧是否受到普遍破坏、等幅度随机改动是否同样奏效。
联合效应不一定等于单组效应相加；interaction不是中介比例或语义冲突的自动证书。
若原生M已经偏正确，只是sampling抽中了错误词，单独标记；不能说模型的主导预测被修复。
相同prefix即使采样出了不同词，原模型分布仍应相同。seed不是回放输入。

本批只有两个已反复研究的来源，不能报告自然检测AUROC、总体显著性或普遍幻觉机制。
无干预效应不能证明没有证据：信息可能已通过其他层、来源状态和残差抵达。

## 3. 原生算子：保留全部头，删除实际消息，不改softmax

用原模型eager attention的A、v_proj输出V及o_proj权重计算：

```
head_value[h,q] = sum_{j in source_group} A[h,q,j] * V[h,j]
message[q] = sum_h W_O[h] @ head_value[h,q]
attention_output[q] -= message[q]
```

GQA按照query head到KV head的真实重复关系展开。候选尚未输入时，在真实`q=P+t-1`进行操作。
证据、错误值来源、剩余prompt和已生成历史是互不重叠的地址组；history包括最后一个已输入词。
attention值不归一化，不用平均attention冒充消息，不把W_OV裸矩阵当动态路由。

- query：只改变最终输入位置的该层写入，类似局部答案写入实验。
- prefix：改变该层所有已输入位置的所选来源写入，检验上游处理；候选后续位置不新增干预。
- cut_mlp：在相同query/prefix范围删除该层MLP更新，残差保留。
- 随机：每个受改query在残差空间匹配同样L2改变量，默认3个种子；不保证语义或分布匹配。
- restore：在上游删除后，将一个指定下游层的头输出恢复为同一prefix原始运行的值。
  恢复head之外的残差和MLP仍来自干预运行；它测试路径依赖，不是跨样本移植整份正确状态。

默认删除组内所有head，并保存每个head的attention质量、value范数、实际W_O写入范数和上下文logit-lens支持。
local lens是在当前残差背景扣掉某头消息后重新做最终norm/readout；它是观察，不自动等于最终因果效应。
真正的效应来自后续层完整重算。不能用全头删除代替单头/多头协同证明。
指定单头或固定多头集合时用`--layers L --heads H1 H2`，新output保持控制独立；不从本轮结果自动选“最佳头”。
这批Llama-3.1的head身份不能照搬另一模型Llama-2的LDA head编号。

## 4. 一键运行和复核

在graph根目录，原research环境：

```bash
python -u -m experiments.path_conflict.main
```

默认使用上面的真实samples目录，默认层8/16/24/31（0起始；预定四个深度位置），query及prefix两种范围。
只需盘点文件/原文，不加载torch或8B：

```bash
python -m experiments.path_conflict.main --stage inventory
```

输入迁移后显式`--samples /实际目录 --model /相同模型的新路径`。不自动换模型、重新生成、补标签。
输出默认`outputs/same_question_path_conflict/`，每次干预完成保存一个小NPZ，中断同配置可续跑。
`--stage report`只读取这些完成结果重新生成表格/图，无需模型；须完成当前预定干预集合。

下游恢复的明确实验（其余保持相同）示例：

```bash
python -u -m experiments.path_conflict.main --layers 8 16 24 --restore-layer 31 \
  --output outputs/same_question_path_conflict_restore
```

使用原checkpoint路径，强制eager+eval、无cache前向。先与采样时的top5 logit/log-normalizer核对。
默认允许bf16 cached-vs-prefill绝对误差0.25 logit，所有实际误差保存；A/V/W_O相对重建误差门限0.02。
这两个是显式数值容差，不是声称原回放逐位一致；可用`--replay-atol`/`--message-rtol`设置更严格门限。
误差过大停止，不拟合补偿原分数。文件对齐/候选边界检查集中在预检，不埋在每条消息计算中。

候选长度不同、开头词面不同；必须同时查看首token margin、序列总logp、均分和两个候选各自的变化。
保持候选字符串不变可以做同一前缀的干预比较；不能把长度归一化分数叫序列概率。
运行的是给定候选的teacher forcing和后续LLM层，不自动产生/评判新的自由续写。没有宣称自由生成已被修复。

## 5. 输出

- `inventory/`：16条样本身份、重复、拒答、trace可用性、原设置及仅两项局部标注。
- `decisions.csv`：原始step/query、目标前缀尾部、候选、source token数量。
- `replay.json`：原采样回放误差、实际argmax与sampled token是否一致、同prefix标记。
- `effects.csv`：每个原生干预的两候选分数和相对原模型的变化。
- `head_source_writes.csv.gz`：全部原生L/H的来源质量与功能读出，不先平均头。
- `trajectories.csv.gz`：每个干预后逐层attention/MLP残差读出，单列intervention_layer。
- `path_interactions.csv`、`same_question_effects.csv`：组间交互、两侧效应比较。
- `figures/`：逐层干预曲线、每头消息支持热图、原生残差轨迹。
- `path_conflict_review.tar.gz`：报告、图、紧凑结果；无模型、无原attention大数组。

## 6. 文献对应与边界

1. Scaling Reasoning Hop Exposes Weaknesses (arXiv:2601.21214v2)：借鉴局部答案写入、全位置处理干预、
   正误候选并行竞争及下游恢复；不复用其训练选头器，也不认定RAG错误与算术错误机制相同。
2. Two Pathways to Truthfulness (arXiv:2601.07422v2)：来源/回答历史通道分开验证；历史路径可能有助于真实性，
   不预设所有history都错误，也不将检测探针分数当生成概率。
3. 既有information-flow工作流：逐head、逐来源计算原生A V W_O，保留质量和写入能量，追到最终候选。
   本轮不复现某论文完整归因算法，不用算子统计为新样本自动贴真假标签。

这些是明确有边界的机制验证，不是最终无监督检测；自然同题样本作为发现案例，不作为无监督训练标签。
