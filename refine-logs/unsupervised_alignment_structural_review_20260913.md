# 无幻觉标签训练的图匹配路线审查

日期：2026-09-13  
模式：`research-lit` + `research-refine` 方法审查；真实 collaboration GPT-5.5 fallback，Codex MCP 不可用。阅读范围为 Hallucination Span Detection with Input-Side Evidence Alignment、SiGHT、FGWEA，并参考 layer-resolved OT 对 attention aggregation 的限制。未改实现、未启 GPU，未宣布方法成功。

## 结论

这三篇确实给出比“继续修 Qwen 多字段 sourceQA”更接近当前约束的结构线索，但不能直接采纳它们的训练目标。HalluSpan 解决的是“检测 span + 输入证据对齐”，但使用 RAGTruth hallucination labels 训练 confidence；SiGHT 号称 self-supervised，但用 LLM 合成负例训练 hallucination 二分类器；FGWEA 是真正无监督图匹配，但任务是两张已给定 KG 的实体对齐，不是事实真假或来源采用。

因此当前最小可用转向不是监督 detector，也不是新 GNN，而是一个 **frozen-feature partial/unbalanced event-graph matcher**：用冻结 source/response 指针图和冻结高维向量提出 source-event、role-span、history-span 和 unmatched 候选；只把这些候选交给现有 relation verification、contrast compilation 和 native auditor。匹配分数只能代表“在当前表示下可对齐/不可对齐”，不能代表 supported、conflict、not-stated，也不能替代 Qwen 关系核验或 native 干预。

如果 v3 继续 9/36 无 risk 或 native0，这个方向应作为下一步候选供给层，而不是把 0 机制调用解释为内部无信息。

## Primary 事实摘要

### 1. Hallucination Span Detection with Input-Side Evidence Alignment

论文提出同时做输出 hallucination span detection 与 input-side evidence alignment。方法假设 faithful output token 更容易从输入表示恢复；训练时按 SRL span mask 输出，encoder 从输入 token 表示检索最相关 span，用最大相似度作 confidence。loss 直接使用 hallucinated/faithful span 标签拉低/拉高 confidence；推理时逐 token mask，最高 confidence 的输入 token 是 evidence alignment，低于阈值判 hallucination。RAGTruth 无人工 alignment，但有 hallucination span label；阈值也在 dev 上调。论文报告 OTAlign 在长输入/输出集合对齐上很弱，且 conflict hallucination 因为有高度相关输入文本会混淆相似度 confidence。

对本项目的约束含义：它不是无幻觉标签训练路线；可借鉴的是“masked response span -> input span retrieval/null candidate”的接口，不是其 supervised confidence 判真。

### 2. SiGHT: Self-Supervised Graph-based Hallucination Detection

PMLR 页面和 PDF 显示 SiGHT 面向 domain-specific LLM hallucination detection，核心是两个阶段：先从领域文本生成训练数据，再用图模型检测。它把段落转成 word-level relational graph：POS 过滤保留名词、动词、形容词、数词等语义 token，Word2Vec 给静态词向量，双向相邻词边组成段落图。负例由 LLM 按 prompt 和近邻替换候选生成“流利但事实错误”的文本；原文本图为 y=0，合成图为 y=1。检测器是三层多头 GAT + global mean pooling + MLP，使用 focal loss，目标是低延迟二分类。它强调无人工标注和小参数量，但仍训练 hallucination binary classifier，只是标签来自合成过程。

对本项目的约束含义：SiGHT 的 word-level graph 和合成扰动可作诊断/消融参考；其 GAT 二分类训练违反当前主线“无幻觉标签训练”，不能进入主方法。

### 3. FGWEA

FGWEA 是 ACL Findings 2023 的无监督 KG entity alignment。它把实体名和属性用冻结 PLM 编成语义 cost，再用 OT 求初始 coupling；从高置信 entity links 得到 anchors，近似计算跨图 structural 与 relational similarity，然后反复做 multi-view OT；最后用 GW 做全局结构 refine。目标函数融合 Wasserstein 的语义匹配和 Gromov-Wasserstein 的图结构匹配。论文强调不观察预对齐实体、不调监督超参，且 progressive anchors 缓解直接 FGW 在大稀疏 KG 上的不稳和低效。关键限制是：它假设两侧 KG 节点/边已经存在，输出是等价实体候选，不定义 null fact、conflict、条件适用性或生成模型内部采用。

对本项目的约束含义：它最适合作为 source/response event graph 的无标签 alignment skeleton；必须改成 partial/unbalanced、有 null/dustbin，且只输出候选。

### 4. Layer-resolved OT caution

该论文把 cross-attention 分布到 uniform/reference 的 W1 距离作为无监督 hallucination 信号，发现 NMT 的 source disengagement 可测，但在摘要任务上明显弱于监督 MiniCheck。它给出的关键限制是：摘要不忠实时可能仍正确 attend 到相关 source tokens，只在内容聚合/表述阶段误用证据；这种错误对 attention concentration/OT metrics 不可见。

对本项目的约束含义：transport mass、attention 集中度或 source alignment 不能直接打真假；它们只能提出 candidate route/span，必须接 relation/contrast/native 证书。

## 对当前架构的差异判断

当前 v3/cloze 和 source_event_graph 路线仍把大量语义工作交给 Qwen：Qwen 先把 response/source 编成 role/condition/question，再用 sourceQA 产 answer/quote/condition table。指针图修复了 quote 编造，但没有解决“候选从哪里来”的脆弱性；如果 v3 的前 9 条仍无 risk，失败可能不是格式，而是 per-question sourceQA 没有稳定地产生可对比目标。

三篇 primary 支持一个更小的候选供给层：先对 source/response 事件图做无标签匹配，生成候选 evidence alignment、conflict alignment 和 unmatched spans，再让 Qwen 只验证少量已冻结候选关系。这样 Qwen 不再负责从开放问题中寻找 source answer，也不再决定全部候选；它只做 relation verifier。这个变化保留当前约束，因为没有用 RAGTruth 或合成 hallucination label 训练 detector，也没有让匹配分数变成真假分数。

## 最小算法：partial/unbalanced event-graph matching proposer

输入：

- source raw inventory 与 source pointer graph；Data2txt 用 AST literal whitelist 编成 field/record graph。
- response pointer graph；每个 response event/role/condition 有原文 span。
- 冻结 backbone 的只读向量：source span hidden pools、response span hidden pools、可选 generation-time query hidden、attention/message sensitivity summary。高维向量只作表示，不训练。

图：

- source 节点：field/text unit、mention、role value、predicate/action、condition、record/event。
- response 节点：claim/event、role value、predicate/action、condition、previous-link candidate。
- 边：same-record、predicate-role、role-condition、order/proximity、coreference/previous-link、Data2txt field path。边类型由程序或 pointer compiler 给出；Qwen 关系只作属性估计。

匹配目标：

```text
min_Pi  <C_sem + C_type + C_surface, Pi>
      + beta * FGW_relation_structure_loss(Pi; A_resp, A_src)
      + tau_r * KL(Pi 1 || a)
      + tau_s * KL(Pi^T 1 || b)
      + dustbin/null penalties
```

其中 `Pi` 是 response nodes 到 source nodes 的 partial/unbalanced coupling。行质量低或流向 dustbin 表示 `unmatched_under_matcher`，不是 N。`C_sem` 可由冻结 hidden cosine、lexical/number/unit equality、field-path compatibility 组成；`FGW_relation_structure_loss` 只比较邻接与边类型一致性。先用 exact numeric/string/date/coref 生成高置信 anchors，再在每个 response event 的 top-k source subgraph 内做局部 partial FGW，最后可做一次小范围全局 refine；不要直接对整篇所有 token 做 O(n^2m^2) GW。

输出：

```json
{
  "response_event_id": "...",
  "candidate_alignments": [
    {
      "source_event_ids": ["..."],
      "role_couplings": [{"response_role_id": "...", "source_role_id": "...", "mass": 0.0, "basis": ["hidden", "surface", "relation"]}],
      "identity_basis": ["..."],
      "changed_or_unmatched_roles": ["..."],
      "null_mass": 0.0,
      "ambiguity": "low|high|unresolved",
      "controls": ["same-type shuffled source event", "same-field unrelated event"]
    }
  ],
  "matcher_status": "candidate_matched|candidate_unmatched|ambiguous|coverage_unresolved"
}
```

这个 proposer 只替代“开放式找候选证据”的部分，不替代 `reference_integrity -> relation_verification -> contrast_validity -> native_audit`。

## Required 如果采用这条路线

1. **训练信号边界**：HalluSpan 的 RAGTruth label loss 和 SiGHT 的 synthetic hallucination label 都不能进入主方法。若做，只能单列为 supervised/weak diagnostic upper bound。

2. **null 定义**：`unmatched_under_matcher`、`low_confidence_alignment`、`dustbin_mass_high` 不能叫 `not_stated`。N 仍需完整 raw source 双盲核验和 source inventory coverage 检查。

3. **conflict 定义**：高匹配质量经常发生在 conflict hallucination 上，因为 source 有相关文本。必须在 matched source event 内做 typed role/value comparison；匹配质量高只能触发 conflict verification，不能判 faithful。

4. **关系失败定义**：若 event coupling 高但 relation_verification 失败，输出 `applicable_source_relation_failed` 或 `event_family_match_relation_unresolved`；不能用 FGW 结构分数覆盖条件/角色失败。

5. **Qwen 权限收窄**：Qwen 不能生成候选 source answer。它只看到 matcher 冻结的 candidate IDs/raw spans，返回 finite relation status 与失败原因。D 不能把 Qwen 新发现的 source span 回填进 matcher。

6. **partial FGW 可扩展性**：必须用 top-k retrieval + event-local subgraph + progressive anchors；记录未搜索 source nodes。全图 dense GW 对长 source 不可行，也会违反单卡预算。

7. **高维向量解释范围**：hidden/attention/message sensitivity 的匹配表示不能证明信息 origin。source key 的 V 仍可能含上下文混合；origin 结论仍需现有 donor-input -> selected V -> F 的嵌套介导。

8. **控制项**：每个 candidate alignment 要有 same-type/same-length/same-field-path 或 shuffled-structure controls。没有 matcher specificity 不能把 top-1 alignment 称为 source attribution。

9. **输出分母**：报告 response events、candidate_matched、candidate_unmatched、ambiguous、coverage_unresolved、relation_verified、contrast_valid、native_executed。invalid/unmatched 不得从分母删除。

## 对“能否直接提出证据归属和 span 候选”的回答

可以提出 **候选**，不能直接提出 **证据归属结论**。部分/非平衡匹配能减少 Qwen 多字段 sourceQA 的候选搜索负担，尤其适合：

- single-role mask 因非目标条件错误而 collapse 的 multi-role event；
- Data2txt 字典字段/记录结构比自然谓词更可靠的来源；
- sourceQA 找不到短 answer_quote 但图上有相关 event/field 的 cases；
- v3 没有 risk 导致 native 完全无供给的 cases。

但它不能解决：

- source graph 漏抽导致的 absence；
- relation/condition 语义错误；
- source attended correctly but MLP/content aggregation misuses evidence；
- 历史传播的方向与错误继承。

这些仍需要当前 strict auditor 的关系核验、A/B contrast、matched controls、origin mediation 和 MLP interaction。

## 推荐下一步，不改 v3

继续跑完 v3。若 v3 仍主要是 no-risk/native0，下一版不要再扩大 sourceQA prompt，而是加一个 **CPU/GPU-light matcher proposer**，顺序为：

1. CPU source/response pointer graph 冻结。
2. 冻结模型 observer 只读抽取 span hidden/message summaries。
3. event-local partial/unbalanced FGW 生成 top-k candidate source events 与 unmatched roles。
4. Qwen 只对 top-k IDs 做 relation verification。
5. 只有 verification + contrast validity 通过者进入 native auditor。

论文 claim language 必须保守：这是“无幻觉标签训练的候选对齐与机制审计”，不是“无监督事实真假检测”。若 matcher 提高 candidate supply，但 native 证书仍低，只能说自动候选覆盖改善，不能说归属/传播已经解决。

## Sources

- Hallucination Span Detection with Input-Side Evidence Alignment: https://arxiv.org/html/2608.15804v1
- SiGHT: https://proceedings.mlr.press/v300/chen26d.html and primary PDF linked there: https://raw.githubusercontent.com/mlresearch/v300/main/assets/chen26d/chen26d.pdf
- FGWEA: https://arxiv.org/abs/2305.06574 and https://arxiv.org/pdf/2305.06574
- Layer-resolved OT caution: https://arxiv.org/abs/2606.13216
