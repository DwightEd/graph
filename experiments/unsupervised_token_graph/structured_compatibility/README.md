# Structured Compatibility：把监督审计转成无标签结构先验

这不是新的密度异常器，也不把错误 basin 或 span 连续当机制创新。
方法只保留之前实验已经反复支持的结构：physical head identity、head contrast、source/prompt view、temporal context。

## 训练信号

每个预测位置保存 layer × head × 5 个 route views：
- exact source_info mass
- other prompt mass
- recent response history
- remote response history
- query-self

TRAIN 不使用 hallucination label values。元数据解析若为补身份而打开 response.jsonl，label 字段不会进入表示、corruption、训练或 calibration。真实 (x_t, x_(t-1)) 是 native compatibility，
并从同一个真实 token 构造四种人工 corruption：
- head_identity：整层 physical heads 置换；
- source_head：在每个head固定 prompt mass 下，只置换 source fraction 的 physical-head 身份；
- source_time：在固定当前head prompt mass 下，把 source fraction 换成同答同位置块 donor；
- previous_time：previous state 换成同答邻近位置 donor。

模型显式输入当前逐head contrast、当前减previous的逐head contrast，以及两者的layer-common均值。
因此不再像 GMM/Mahalanobis 一样只问离中心多远，也不先平均head。

## Continuity 只做第二阶段

primary score 是 compatibility。它必须单独过 onset / previous_gold_0。
随后才从无标签 calibration answers 的 compatibility lag-1 correlation 估计 rho，计算 causal sticky score。
sticky 只负责检验 persistence 能增加多少 continuation/all-error 性能，不能掩盖 base onset 失败。

## 一键运行

python -u -m experiments.unsupervised_token_graph.structured_compatibility --phase all --output outputs/structured_compatibility_v1 --resume

接口小检：

python -u -m experiments.unsupervised_token_graph.structured_compatibility --phase inspect

默认只加载原 attention cache 和 tokenizer，不加载 8B 权重。

## 输出

predictions/evaluation.json 同时报告：
- compatibility（主方法）
- head_identity
- source_head
- source_time
- previous_time
- sticky（连续性第二阶段）

除标准 RAGTruth views 外，还报告：
- previous_gold_0
- previous_gold_1
- previous_gold_0_sentence_start
- previous_gold_0_high_transition

如果 compatibility onset 不强而 sticky 总体很强，只能说明连续性先验提高了 continuation。
如果 source_head/source_time 在 onset 有增量，才说明 source/head relational consistency 值得继续做更细的 binding。

## 直接控制

同一次 evaluation 额外报告 self_jump 和 position。
source_head/source_time corruption 都保持每个head的 source+other_prompt 总量不变，
避免分类器只学习attention质量守恒被破坏。
head_identity在逐head标准化以后置换，避免只利用不同head的原始均值尺度。
