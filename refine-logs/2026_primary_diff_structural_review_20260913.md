# 2026 primary 差异补充：strict auditor 是否足够作为最终主线

日期：2026-09-13
模式：真实 collaboration GPT-5.5 fallback；不是 Codex MCP、不是 v3 实证通过。本补充不改 v3 协议、不触碰冻结代码、不启动 GPU。

参考 primary：

- SIRG / *Detecting Hallucinations in Retrieval-Augmented Generation via Semantic-level Internal Reasoning Graph*, arXiv:2601.03052v1，§§3.2–3.4 与 Appendix D：<https://arxiv.org/html/2601.03052v1>
- CORTEX / *Token-Level Hallucination Detection in RAG via Comparative Internal Representations*, arXiv:2606.31033v1，§§3.1–3.4：<https://arxiv.org/html/2606.31033v1>
- CausalGaze / *Unveiling Hallucinations via Counterfactual Graph Intervention in Large Language Models*, arXiv:2604.11087v1，§3.2 与 Appendix A.4：<https://arxiv.org/html/2604.11087v1>

## 每篇的可用信号与限制

**SIRG。** 它用 AttenLRP/LRP 得 token 贡献，再聚合到语义片段：target 只平均 substantive tokens，source/history 片段取 max，边用 top-k 或排序后最大离散梯度自适应选出。最后把 incoming edges 与 target 线性化，微调 AlignScore/RoBERTa 做二分类。对我们的提醒是：全流 attribution 可给自适应 span/edge 候选，不必等 QA anchor 成功。但它的检测来自有监督 CE 与阈值，且 max 聚合不能证明来源身份或事件条件归属。

**CORTEX。** 它构造同一 answer tokens 的 with-reference / no-reference 两个输入，取最终 hidden states 差值 `Δh_i`；再用 answer-history attention 加权前序 token 的 `Δh_j`，得到 `c_i = Δh_i - Σ α_ij Δh_j`，区分直接 reference sensitivity 与经历史 token 传递的影响。检测由 MLP + token labels 训练，后接固定 `p_stay` 链式平滑。对我们最有用的是一个低成本全覆盖 detector skeleton；限制是没有证据槽位、没有完整事件 B/A 干预，也没有来源条件归属。

**CausalGaze。** 它把 hidden states 与 attention map 作为图，用 detector loss 对 attention 的梯度形成 `S=|A ⊙ ∂L_detector/∂A|`，再由可学习 gate 精炼边，接两层 GAT 与 CE/稀疏正则训练。它提供“全流高维图 + 自适应边 + token saliency”的路线，但所谓 causal sensitivity 依赖已训练 detector loss，不是原生完整事件干预，也不是无标签因果发现。

## 对当前架构的差异判断

当前 v3/cloze 仍是一个严苛审计器：先用冻结 reader 建立可核查 slot/event target，再在 Llama replay 中用完整 B/A 事件 F 测实际消息效应。它的优势是证书语义清楚：若通过，可以说“这个条件化事件在这个消息组上有实测有限干预效应”。它的短板也很清楚：只要 anchor 失败，native 层就饥饿；v2 已经出现全批 native0。

三篇 2026 primary 说明，若用户最终要的是“图自适应 span + 可扩展检测器”，单靠 strict auditor 可能不是最终主线。它适合作为高精度机制证书与教师，不适合作为唯一全量检测前端。真正可扩展的主线需要一个不依赖 QA anchor 的全流 detector 分支，用 hidden-state / attention / attribution 的连续信号覆盖所有 answer tokens，然后只把高风险 span 送入 strict auditor 做机制证书。

但这条 detector 分支需要监督信号。RAGTruth human spans 可以作为 detection GT；v3 审计证书可作为少量 mechanism teacher；冻结 reader 的 E/C/N/U 只能叫 weak reader labels，不能叫无监督 GT。若用 reader pseudo-label 训练 detector，必须单独报告为 weak-supervised self-training，并用人工 labels 做外部评价。

## 最小结构建议

不改变正在跑的 v3。v3 完整跑完后，如果语义 anchor 仍低覆盖或 native starvation，下一轮加入一个并行 `full-flow_detector@1`，不要继续 prompt 修补：

1. **全覆盖特征。** 对每条 response 做两次 observer replay：`prompt+source+answer` 与 `prompt_without_source+answer`。为每个 answer token 保存若干层 `h_ref`、`h_no_ref`、`Δh`、`c_i`、source/history attention mass、attention entropy、距离分桶、可选 AttenLRP source/history 聚合边。先不用 GAT；MLP/linear probe + CORTEX residual 足够作为最小基线。

2. **自适应 span。** 训练 token risk 后接固定或验证集选择的 `p_stay` HMM/CRF-like smoothing。输出连续高风险 span，再映射到 cloze/event auditor。不要恢复以前自由 semi-CRF；这里只是 detector 分数的后处理，不负责生成事实目标。

3. **机制证书接口。** detector 只能提出 span/候选，不可直接声称 route/adoption。对每个高风险 span，仍需 v3 的 cloze/event alignment 构造 B/A；只有通过有限干预、matched controls、origin mediation 的子集才获得 native mechanism label。

4. **训练信号。** detection head 用 RAGTruth train span labels，按 source split 留验证/测试；class imbalance 用正类权重或 focal loss。机制 head 暂不训练，除非 v3 证书数量足够，并且 teacher 标签来自实测 Δ/Ω/origin，不来自 reader pseudo-label。

5. **图版本。** 若 MLP baseline + smoothing 在新 source split 上有稳定增益，再加轻量图层：节点为 answer tokens/segments，边为 source/history attention、CORTEX residual similarity、可选 LRP edge；训练目标仍是 human span BCE。不要直接上 CausalGaze 式 GAT 并宣称 causal discovery，因为其梯度边由 detector loss 定义。

## 失败触发条件

- v3 `semantic_scored_words < 50% assertion_words` 或 `native_selected_claims = 0`：启动 full-flow detector 分支；不得再把问题描述为单纯格式修复。
- v3 self/source 已大幅改善，但 `single_mask_scope=multi_role_event_mismatch` 覆盖 >20% assertion words：加入 event-level source alignment / joint event contrast；detector 只做 span prior。
- detector 在 source-held-out RAGTruth 上 token AP/AUROC 明显高于 entropy/NLL，但 strict auditor 证书少：可主张“全流检测有信号”，不可主张“机制归属已解决”。
- detector 高、auditor 低：用户核心的自动回看/归属/传播仍未解决，只说明需要另一个 mechanism bridge。
- reader pseudo-label 训练若超过 human-label baseline：只能说 reader-distilled detector，不能说无监督图发现。

## 必须报告的门槛

两条主线分开报告：

- **Strict auditor 门槛**：沿用 v3 的有效语义覆盖、A/B contrast、内部选择性证书覆盖、origin/continuous 分母。低于门槛，不能说自动回看/归属/传播已解决。
- **Full-flow detector 门槛**：source-held-out token AP/AUROC、answer-level AUROC、span boundary F1/IoU、coverage、false positive on supported recovery、跨任务分组。它只能支持检测/候选 span，不支持机制解释。
- **Bridge 门槛**：detector top-k spans 中有多少能被 cloze/event auditor 转成 valid B/A，多少获得 position/origin/propagation certificate。这个 bridge 才回答用户想要的“可扩展检测结构是否能连接到原生路由/采纳”。

## 审查结论

v3 应继续完整跑完，因为它检验了 v2 失败是否主要来自 cloze/self-recoverability 接口。若 v3 仍然 native 饥饿，最小可信转向不是再写更复杂的 QA prompt，而是双层结构：全流监督 detector 负责覆盖与自适应 span，strict causal auditor 负责少量可核查的原生 route/adoption 证书。这样会放弃“完全不依赖任何标签训练的全量检测器”这个说法；但它比把冻结 reader 伪标签包装成无监督 GT 更诚实，也更接近用户要的可扩展自动回看/归属/传播系统。

## Scope 限定补记

根代理不采纳将监督 detector 直接升级为当前主线的方法变更。当前 `FINAL_PROPOSAL` 的问题锚点仍是“不以幻觉标签训练”，用户偏好也是无监督图/原生路由审计；因此，上述监督 full-flow detector 只能保留为**改变约束后的条件替代方案**，不能默默进入当前无标签主方法，也不能用于 v3 或当前主线的候选选择。

在当前约束下，若额外训练 supervised probe，它的合格角色只是信息可解码上界诊断：证明 hidden/attention/attribution 连续信号中是否可线性或浅层读出 hallucination risk。它不得提供候选、不得作为机制证书、不得被称为无监督 GT，也不得把冻结 reader 伪标签包装成训练真值。若后续因 v3 覆盖失败而考虑这条路线，必须明确写成一个放宽约束后的 weak/supervised detection 分支，并和原生 strict auditor 的 route/adoption 证书分开报告。

保留原审查意见的理由是：三篇 2026 primary 的共同启发仍说明“全流高维信号”可能解决 anchor 饥饿和自适应 span 覆盖问题；但在当前任务边界内，它只构成风险提示和备选路线，不构成已经接受的主方法修订。
