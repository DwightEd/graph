# 研究约定：当前问题锚点

对冻结LLM，在不以幻觉标签训练的条件下，测量来源和历史的信息实际传递、聚合与输出采纳；区分疑似错误被继续沿用、被反对/纠正以及新陈述开始。事实区间和影响范围分别评价，熵只提供候选起点。

主线graph，reanchor保存机制实验与冻结全量。首版是离线机制教师：冻结语义读者锚定适用约束，observer内部图独立提出采用候选，真实同目标干预检验位置与输入介导。不是纯内部事实归属已成立，也不保证唯一回看点或在线提前检测。

自然标签只在冻结预测后用于独立评价；不供训练、选样或语义读者。图必须对同一语义读者的非图基线提供可验证增量。若仅少量解释案例、覆盖不足或没有质量增益，claim_supported=no。规格与预算以FINAL_PROPOSAL、EXPERIMENT_PLAN和冻结audit_protocol/settings为准。首批自然A–D尚待运行，代码和工程审查不是研究有效性结果。


## 2026-09-13T12:54:03+08:00 — population / typed results and SourceRel-Mini implementation

- Population COMPLETE17790/17790 failed0; `refine-logs/ragtruth_full_evaluation_review_20260913.{md,json}`: all36 groups, hashes verified; descriptive observer scope only.
- `docs/TYPED_HOURS_RESULTS_20260913.md`, `docs/TYPED_NATIVE_RESULTS_20260913.md`: full17790 compile2contrasts; soletrain native90forwards/raw/strong0. Freshdoc witnesses completed; oldpending entries historical.
- `outputs/typed_hours_owner_metadata_correction_20260913.json`, `next_iteration/typed_owner_metadata.py`: additive370 root/name-ID correction; independent impact review, no frozen rewrite or GPU rerun.
- `next_iteration/source_relation_{data,model,features,train,transfer}.py`; current plan `refine-logs/SOURCEREL_EXPERIMENT_PLAN.md`, method `docs/CURRENT_METHOD_20260913.md`.
- M0 fresh883sources/7064queries; fullCPUfeaturepreflight38541docs/3954619tokens/max1200; no truncation. Data/features/train independent reviews C0/R0 (scope engineering). TFIDF valtop1.8059816/top5.9900307. Sanity real21featureforward and2head epochs exit0; no effectiveness claim. Full features/20epochs/natural transfer pending at this entry.
