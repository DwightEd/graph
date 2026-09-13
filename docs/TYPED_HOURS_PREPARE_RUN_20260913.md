# Typed hours：完整自然清单CPU编译

范围：RAGTruth原冻结17790回答，全部保留；仅对Data2txt明确的七天营业时间语法做来源代数核验。输入不含labels/quality，输出不是效果评价。tokenizer在CPU离线使用，不加载模型、不暂停population。

来源侧要求单business记录、全部7天明确24h区间，跨夜/含糊区间拒绝；回答侧仅固定完整句关系，保留量词和WiFi等非目标约束，单个AM/PM端点改为来源值。QA/Summary、未解析语法、来源缺失、无单值修复、对齐失败均保留计数。协议写在 `next_iteration/typed_hours.py::PROTOCOL`。

执行前core/CLI独立工程Required必须已关闭。本运行使用已存在research@03909e02包环境，无安装/重建。以下是固定的一次性新输出命令：

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.typed_hours_prepare --inputs ../reanchor/outputs/ragtruth_population_20260912/inputs.jsonl --output outputs/typed_hours_population_prepare_20260913 --observer-model /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct
```

不要覆盖已有输出，不自动重跑或更换参数。settings/代码快照先发布，逐行rows、去重sources、合法contrasts随后发布，完整summary/manifest最后发布。若失败保留产物和退出码；不能把有目录叫完整完成。官方test只冻结未调参预测，迭代不读其错误内容/标签。下一原生实验仅从official train合格对照按固定hash每来源1条、最多12来源选择，不按observer正误偏好挑选。

独立文档见证记录：命令/session/实际退出码、17790行与分任务/划分分母、合格contrast数、代码/input/tokenizer hashes、reader/modelforward均0、population未暂停及实际增长。报告 `refine-logs/typed_hours_prepare_doc_witness_20260913.md`。不把工程fixture或source代数判定当RAGTruth人工标签准确率。
