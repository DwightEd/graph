# GroundedGraphAdapter 来源payload擦除对照

只在本轮source-validation的完整48来源运行，不看RAGTruth正误标签，不选自然结果好看的样本。目的：判断adapter的增益是否依赖所指向的来源内容。解释边界：同值或其它可替代来源可能仍存在，增益留存不自动证明格式捷径，更不等于原LLM路由机制已解决。

所有目标owner的source-token位置替换为单个ASCII空格的token，保持token坐标数量和回答前缀不变。实际重跑冻结Llama，source节点按原坐标池化新状态，绝不将旧上下文向量置零。三分支：原始；只替换source图特征但保持原query；同时采用擦除输入得到的新source和新pretoken query。输出每个坐标锚点的LM/adapter logp、增益和source-coordinate pointer概率、完整擦除状态数组与实际执行input IDs。

沿用 `.aris/compute/local.md` research@03909e02，无安装重建。主训练、自然预测与评价完成、GPU释放、erasure工程C0/R0后，由fresh文档见证执行一次；不覆盖、不自动重试，不另请用户批准。

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.grounded_graph_erasure --training outputs/grounded_graph_train_v1_20260913 --features outputs/grounded_graph_features_v1_20260913 --output outputs/grounded_graph_erasure_v1_20260913
```

报告 `refine-logs/grounded_graph_erasure_doc_witness_20260913.md`：48-source分母、实际forwards、source/input/response坐标一致、重新编码array hashes、三分支同目标、正负gain-drop均如实报告。不得称语义证书或幻觉二分类真值。仅当自然评价和此来源依赖对照都实际支持增益时，才可推动冻结官方test协议。
