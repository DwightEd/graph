# Attention Hallucination Mechanism Audit

这是一套**机制验证**代码，不是新的构图器、自编码器、VAE，也不是拿标签训练的幻觉检测器。它独立检验三个假设：

1. **Grounding drift**：生成过程中，被当前 chosen token 使用的注意力从 evidence/question/constraint 转向回答历史；
2. **Dispersion / cancellation**：路由变散、head 的角色分工分歧，或有符号贡献相互抵消；
3. **Counterfactual evidence bypass**：删除或替换证据后，冻结模型仍支持原回答。

三个轴分别报告，不存在手工加权的 hallucination score。第三轴只能称为“证据绕过/回答持续性”，不能单独证明知识一定来自参数。

## 核心机制

### 1. 严格 predecessor 对齐

prompt 长度为 `P`，第 `t` 个回答 token 是 `y_t=ids[P+t]`，只能由位置 `P-1+t` 的 logits 预测。原 sparse cache 只有 response-query rows，因此 cache query `i` 对齐回答 token `i+1`；token 0 的 routing/functional 特征必须是 `NaN`，不能伪装成 0。完整模型反事实仍覆盖 token 0。

### 2. 高效估计 token-local 功能贡献

冻结 LLM 并 teacher-force 缓存答案。每层 hook 实际 `v_proj` value state 和进入 `o_proj` 的 head context。令第 `t` 个 chosen log-probability 为 `s_t`；需要的是同一 predictor row 的 Jacobian 对角块 `∂s_t/∂c_t`。如果对整段答案均值只 backward 一次，当前 query 的梯度会混入未来 token，不能用于 onset 审计。

直接逐 token backward 需要 `R` 次反传。代码用 `K` 个确定性 Rademacher probe 做 Hutchinson 对角估计：

\[
\widehat{\frac{\partial s_t}{\partial c_t}}
=\frac1K\sum_{k=1}^K z_t^{(k)}
\frac{\partial\sum_u z_u^{(k)}s_u}{\partial c_t}.
\]

每个 probe 的 signed gradient 先平均，再计算绝对能量，避免先取绝对值造成正偏。对 cache-retained endpoint 和 exact diagonal 计算：

\[
\phi^{\ell}_{t,j,h}
=A^{\ell}_{t,j,h}
\left\langle
\widehat{\frac{\partial s_t}{\partial c^{\ell}_{t,h}}},
v^{\ell}_{j,kv(h)}
\right\rangle .
\]

这不是简单 attention mass：实际 value、GQA 映射、`W_O` 后的采用程度、下游 residual/MLP 和 chosen logits 都进入 local gradient。代码保存 signed contribution、absolute energy、signed role contribution 的 probe standard error 和 cancellation：

\[
\kappa=1-\frac{|\sum_e\phi_e|}{\sum_e|\phi_e|+\epsilon}.
\]

主漂移量为：

\[
g_t=\log\frac{E_{history}+\epsilon}
{E_{evidence}+E_{question}+E_{constraint}+\epsilon}.
\]

`other_prompt` 不冒充 grounding。现有 `operator_geometry.pt` 不会被动态路径读取；它继续用于静态 `W_OW_V` head-operator geometry 控制。

### 3. cache 与 replay 必须是同一数值路径

只匹配层数、head 数或模型名字不够；否则 cache 的 `A` 与另一 checkpoint 的 `V/gradient` 相乘会成为无效混合量。baseline replay 强制 `eager + float16`，并在接受功能特征前逐一比较：

- 每个 retained endpoint；
- 每个 exact response diagonal；
- retained-plus-diagonal row mass。

任一最大绝对误差超过固定阈值 `5e-3` 就中止。artifact 绑定全部 weight shards、config、tokenizer/chat template、依赖版本、probe seeds 及逐样本 attention-binding 摘要。

### 4. routing 的可观测边界

prompt roles 只从无标签 `source_info.jsonl` 构建，并复现历史 system+user chat template；重建 prefix 必须逐 token 等于 cache。

response-carrier 严格按 Transformer 层传播：当前层 endpoint 读取上一层 source state，不会错误地在同一层内串行传播。每层聚合所有 heads，并保留 head-role JSD、entropy/HHI censoring bounds 和 unresolved coverage。

formal cache 没有 prompt-query rows，因此代码不能拆出 prompt→prompt relay，也不能把 residual/MLP 分解成 attention-path mass。这里报告的是 response-query attention ancestry；functional gradient 吸收下游采用程度，但不声称“完整还原所有网络路径”。另提供距回答位置分箱、bin 内置换 role 的 label-free recency null。由于 QA evidence 是长连续 span，很多 bin 的 role 恒定，该 null 只作诊断，不作为 primary endpoint。

### 5. 七个冻结反事实与 donor ensemble

固定运行：

```text
full
no_evidence
no_history
no_evidence_no_history
swapped_evidence_0
swapped_evidence_1
swapped_evidence_2
```

- 所有分支始终 score 原 factual response token，不采用分支 argmax；
- `no_evidence` 屏蔽 evidence attention keys 对严格后续 query 的传播；
- `no_history` 只屏蔽 prior-response attention keys；当前 query embedding/residual 仍存在，因此它不是“删除全部回答历史”；
- 三个 donor 按 target-specific SHA-256 确定性选择，必须同任务、不同 source，并保持 evidence span 与序列位置长度不变；
- donor slot 不足时逐 slot unavailable；ensemble 仅对 available donor 求均值并报告 donor 间标准差，不 pad、repeat、复用或填 0。

主 evidence-bypass 为：

\[
B_t=\frac12[(\ell_t^{no\ evidence}-\ell_t^{full})
+(\overline{\ell_t^{swap}}-\ell_t^{full})].
\]

history necessity 与 evidence-history interaction 只作 attractor/control。history dependence 没有预注册为 hallucination-high，因为正确推理同样可能高度自洽。

## 一键运行 QA

先用小样本验证路径、显存、tokenizer 和 cache binding：

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph

LIMIT=2 BOOTSTRAP=50 GRADIENT_PROBES=2 \
bash experiments/attention_mechanism_audit/run_qa.sh
```

确认后运行完整审计：

```bash
bash experiments/attention_mechanism_audit/run_qa.sh \
2>&1 | tee experiments/attention_mechanism_audit/run_qa.log
```

正式默认 `GRADIENT_PROBES=8`。probe CPU 内存约为 `K×L×R×H×D×4` bytes；超过 2 GiB probe buffer 时会显式拒绝，需要降低 `GRADIENT_PROBES` 或拆分超长回答。formal cache 由 FP16 observer 提取，因此默认且要求 `TORCH_DTYPE=float16`；不要切换 bfloat16 后放宽 binding tolerance。

`K=8` 是可运行默认值，不是论文中自动成立的收敛保证。正式报告前应在相同长回答上分别运行 `GRADIENT_PROBES=8/16/32`（或至少两个 seed），比较六个 primary endpoint 的方向和排序稳定性。eager 全序列 replay 的 attention 激活按 `L×H×N²` 增长；先对最长 QA 样本做单样本 smoke，并记录 GPU peak memory，再开始全量运行。

覆盖默认路径：

```bash
SOURCE_INFO=/path/to/RAGTruth/source_info.jsonl \
MODEL_PATH=/path/to/Meta-Llama-3.1-8B-Instruct \
TEST_SPLIT=/path/to/attention/llama31_8b/test \
OUT=/path/to/new/audit_run \
bash experiments/attention_mechanism_audit/run_qa.sh
```

三个物理阶段：

```text
[1/3] source_info + exact cached prefix -> prompt_roles.jsonl
[2/3] labels sealed + frozen model replay -> mechanisms.npz
[3/3] freeze artifact SHA -> open labels -> evaluation.json
```

正式路径会逐文件校验 cache manifest 中记录的 SHA-256。由于既有 formal cache 把 attention 与 `y_token` 放在同一个 PT payload，底层反序列化无法做到“物理上从未载入标签张量”；前两阶段会立即丢弃该张量，既不保留也不调用 label API。这里严格主张的是 **labels are not exposed or used before artifact freeze**，而不是夸大为 labels were never deserialized。

断点重跑：

```bash
START_STAGE=2 OUT=/path/to/the/run \
bash experiments/attention_mechanism_audit/run_qa.sh

START_STAGE=3 OUT=/path/to/the/run \
bash experiments/attention_mechanism_audit/run_qa.sh
```

建议使用新的 `OUT`。旧超图、GCN、自编码器输出不会被本实验读取，也无需为本审计删除。

## 结果与可证伪性

`mechanisms.npz` 保存完整回答 token rows、逐层 trajectories、targets、predictor positions、cache-query shift、七分支 availability、probe 配置、role-position null 和完整 provenance。`evaluation.json` 包含：

- 每个机制锁定一个 primary endpoint，报告 frozen-direction AUROC/AUPRC、source-group bootstrap、source-level permutation 和 primary-family BH-FDR；
- prompt length、response length 及联合 source-group OOF baseline；
- 每个 primary feature 的 `length-only` 与 `length+feature` OOF 增量，并明确它只是 supervised readability，不是 detector；
- first hallucination onset 相对前一 token 的变化，以及 source-disjoint、同 response position 的 non-onset control；
- observer/generator provenance 和机制可观测边界。

如果 drift、dispersion、evidence bypass 不能稳定区分正负回答，结论就是这些假设在当前 observer/cache 上未获支持，不能通过翻转方向或给三轴调权来“修复”AUROC。

RAGTruth 原 generator 与当前 Llama-3.1 observer 通常不同，因此结果只能称为 **teacher-forced observer audit**，不能声称复原了原生成模型内部的幻觉形成过程。`A×gradient` 是有限 probe 估计的局部一阶 attribution；有限反事实是 evidence/history sensitivity；两者都不是完整因果证明。
