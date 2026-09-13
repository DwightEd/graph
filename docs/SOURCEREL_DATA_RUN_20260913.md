# SourceRel-Mini M0：来源字段训练数据

既有 research@03909e02 环境，Python/权重/包无变化，无安装。只从冻结的全量 annotation-free inputs 中选 official train / Data2txt，按 source ID 去重；source hash 划分内部训练/验证。source pointer 重建目标由原字段坐标构造，不读取 RAGTruth 幻觉标签，不加载 GPU 模型。该任务的得分不是自然回答归属准确率。

工程复核 Required 关闭后，一次执行：

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.source_relation_data --inputs ../reanchor/outputs/ragtruth_population_20260912/inputs.jsonl --output outputs/source_relation_data_v1_20260913
```

输出必须新建；失败保留目录和退出码，不重跑覆盖。独立 fresh agent 按命令执行，检查 source 数、划分互斥、query/candidate 分母、manifest/code/settings/input 哈希及 model forwards=0；写 `refine-logs/source_relation_data_doc_witness_20260913.md`。完整 tokenizer 长度预检属于后续 feature prepare，未通过前不执行编码。
