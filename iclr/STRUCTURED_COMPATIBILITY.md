# Structured Compatibility：实验注册与研究边界

## 已有证据如何进入方法

| 已有结果 | 设计约束 |
|---|---|
| CHARM/LDA 单token很强 | 当前 token 表示承载前缀状态，不能只看显式 past score |
| previous_gold / past_run 约0.98 | continuation 必须单独审计，不能把 pooled AUROC 当 onset 机制 |
| 固定 previous label 后 current 仍约0.76-0.78 | base detector 必须独立评价 previous_gold_0/1 |
| head_contrast 接近 full LDA，layer_mean弱 | 不平均head；保留physical head身份和within-layer contrast |
| unordered heads明显下降 | corruption 必须直接破坏 physical-head identity |
| prompt_only很强 | 单独保留source/prompt view并与当前状态做 compatibility |
| GMM/raw density失败 | 不再学习罕见点，改学关系一致性 |
| coarse grounding dynamics onset失败 | 不用 source/history 总质量守恒当机制 |
| headwear/onion 因果个例 | source-head/source-time mismatch 只作为 corruption 动机，不写入自然标签 |
| recovery状态跳变更大 | transition magnitude 只做控制，不作主分数 |

## 可证伪预测

P1. compatibility 应在 previous_gold_0 上优于简单 density / transition magnitude。
P2. head_identity 或 source_head 若有效，说明监督发现的 head structure 可以通过人工关系破坏迁移到无标签训练。
P3. source_time 若只对 QA 有效，则按 task 报告，不宣称通用机制。
P4. sticky 应主要提升 continuation / all-error；若 base onset 失败，不能用 sticky pooled AUROC 替代。
P5. 如果四种 corruption 都接近随机，则停止继续堆模型深度，说明现有无标签约束不足以恢复监督方向。

## 不允许的结论

- corruption classifier 高 AUROC 不证明对应 corruption 就是自然幻觉原因；
- source_info 是整段 source，不等于当前 claim 的适用证据；
- sticky 提升不叫机制创新；
- natural labels 只用于最终 evaluation，不能参与 epoch 选择、corruption 权重或 model selection；
- 当前 RAGTruth test 已反复研究，结果属于探索性验证。

## Corruption有效性约束

source相关corruption不得改变当前head的prompt总质量。
只替换 source/(source+other_prompt) 比例，避免学习attention质量守恒这种必然关系。
source donor限定在同一回答、局部位置块，减少位置和生成阶段的简单差异。
head_identity在逐physical-head标准化以后置换，目标是head身份/协同而不是原始尺度。


## 2026-09-20 natural TEST result: failed as a primary detector

Completed TEST scoring showed:
- ALL compatibility AUROC 0.5634, AP 0.0870
- ALL source_head AUROC 0.5822, AP 0.0962
- ALL sticky AUROC 0.5677, AP 0.0887
- position AUROC 0.6045
- QA source_head was the strongest component at AUROC 0.6100, AP 0.1385

Interpretation: the artificial corruption task learns generic internal inconsistency,
while natural hallucinations can remain internally coordinated. The pretext direction
does not recover the label-assisted LDA direction. Keep this package as a negative
control; do not tune corruption weights or add more corruption classes on TEST.
