# SourceRel-Mini M2：完整旧36开发清单自然迁移

此处不训练模型、不用回答或另一LLM产生owner正例。query遮蔽原始surface目标span，保留完整当前parent和全部此前回答；candidate为原source字段。Data2txt以外的来源、无broad-type池、超长输入都显式留在完整槽位分母。候选top5、family mass只是归属提议，不是真假概率/SCNI，也不发native证书。

既有research@03909e02，无安装/重建。source_relation_transfer工程Required关闭后，独立fresh agent执行以下CPU prepare一次：

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.source_relation_transfer prepare --inputs outputs/native_audit_design_20260913/inputs.jsonl --model /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct --output outputs/source_relation_transfer_features_v1_20260913
```

等待全量source features编码实际exit0释放GPU，再执行以下一次；可以和CPU小头训练并行，不能和另一个GPU模型重叠。

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.source_relation_features encode --output outputs/source_relation_transfer_features_v1_20260913
```

M2 features完整manifest与全量20epoch头训练完整manifest均可核验后，执行一次CPU预测：

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.source_relation_transfer predict --features outputs/source_relation_transfer_features_v1_20260913 --training outputs/source_relation_train_v1_20260913 --output outputs/source_relation_transfer_v1_20260913
```

各输出必须新建，无覆盖/重试。记录实际session/exit、36回答及完整slot计数、域外/空池/不可编码分母、所有input/code/checkpoint/model/features/manifest hashes、真实训练source membership、逐slot同视图frozen cosine对比。报告 `refine-logs/source_relation_transfer_doc_witness_20260913.md`。旧36开发集已用于设计，不能宣称全新确认性测试；没有natural-owner ground truth就不报告owner accuracy。
