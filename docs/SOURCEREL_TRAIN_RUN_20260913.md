# SourceRel-Mini 来源重建训练与同池基线

既有 research@03909e02 环境，无安装/重建。全程CPU小头，冻结Llama辅助向量不回传梯度。必须先完成 source_relation_train 独立工程 Required 修复。数据目标仅为原source字段指针；不读取RAGTruth幻觉标签，不将自然回答候选当监督。

全量特征 CPU prepare 完成后，可先跑词面基线（无需等 GPU）：

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.source_relation_train lexical --features outputs/source_relation_features_v1_20260913 --output outputs/source_relation_lexical_v1_20260913
```

固定两个来源的特征编码完成后，运行一次两epoch接口 sanity：

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.source_relation_train train --features outputs/source_relation_features_sanity_v1_20260913 --output outputs/source_relation_train_sanity_v1_20260913 --epochs 2 --sanity
```

sanity只检验forward/backward/有限梯度/真checkpoint/分割与指标接口，不写研究结论。通过后才继续 feature 文档的全量 encode。全量完成 manifest 后，一次完整训练：

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.source_relation_train train --features outputs/source_relation_features_v1_20260913 --output outputs/source_relation_train_v1_20260913 --epochs 20
```

记录每一步实际 session/exit；原始逐query预测、原字段positive、同池TFIDF与frozen hidden baseline、source train/validation分母、homograph和same-field-other-record分组、excluded pools、20epoch轨迹和validation NCE所选checkpoint。不覆盖/重试已有目录，不根据official test选择超参。独立见证可按依赖分阶段执行，后阶段等待相应真实COMPLETE，不重做前阶段。报告 `refine-logs/source_relation_train_doc_witness_20260913.md`。自然迁移和完整检测器仍需单独验证。
