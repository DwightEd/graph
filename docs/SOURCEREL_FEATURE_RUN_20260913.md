# SourceRel-Mini 冻结特征预检与编码

沿用 research@03909e02，不安装或重建。全量 population 已 COMPLETE 17790/17790 failed0，GPU 已释放。数据父产物为 `outputs/source_relation_data_v1_20260913`，883 个 official-train 来源、7064 query。训练和内部验证按来源隔离。

独立文档见证先依次执行以下三个命令。前两个只加载 tokenizer 并冻结精确输入；第三个才加载 GPU。每个输出必须新建，不覆盖、不自动重试。source_relation_features 工程 Required=0 后开始。头部训练的独立复核另见训练文档；此处 sanity 只验证真实小批特征接口，不作为研究结果。

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.source_relation_features prepare --data outputs/source_relation_data_v1_20260913 --model /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct --output outputs/source_relation_features_v1_20260913
```

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.source_relation_features prepare --data outputs/source_relation_data_v1_20260913 --model /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct --output outputs/source_relation_features_sanity_v1_20260913 --sanity
```

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.source_relation_features encode --output outputs/source_relation_features_sanity_v1_20260913
```

报告 `refine-logs/source_relation_features_doc_witness_20260913.md`：实际各 session/exit、全量 tokenizer min/max/总 tokens/超长分母；sanity 的实际 feature forwards、向量 shape/finite、按 source 选定的小批、时间与 CUDA 峰值；input/view/code/model/array hashes。确认是 causal 最后一层最后非pad token 的辅助视图表示，不是原生成隐藏状态；不产真假判定。

完成小头 sanity（训练文档）后才执行全量编码，一次命令如下。未完成 sanity 时保持待运行。按前面的预检和实际吞吐估计耗时，完整库存保留；超过4096的视图不截断，整个受影响 query pool 排除并计数。

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.source_relation_features encode --output outputs/source_relation_features_v1_20260913
```

同一个 fresh witness 可以在 root 核实 sanity 结果后继续此命令，不需重复前面已完成的步骤。现有锁阻止和 population interleave 同时加载。无自然归属或完整检测器有效性结论。
