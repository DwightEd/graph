# GroundedGraphAdapter 原文重建与真实预测前特征

本轮替换已停止的 Qwen 弱模板数据路径。原输出 `grounded_graph_data_v1_20260913` 保留：6/240来源，19/48机械可用但有确证错误归属，SIGINT退出130；不能供本轮训练。具体证据见 `refine-logs/grounded_graph_data_doc_witness_20260913.md`。

新数据采用同一冻结240个 official-train 来源及全文SHA划分。把完整来源原文作为重建文本，指针仅来自可重建原文坐标：Data2txt字段值首个词；QA/Summary原文context内部已有两个词之后的稀疏词锚点。开始8词不作指针目标，每源最多64个锚点。指针不是自然归属真值；该预训练可能只学复制，必须由自然全词/首错后评价与来源payload擦除对照判断。未验证的多字段AND/回看/传播仍未解决。

同一原始任务prompt、完整来源图、冻结Llama最终norm后的 h[t−1]。禁止截断、使用当前目标token状态或将回答token池化进source节点。特征kernel/runner与data工程Required=0后，新文档执行见证逐条执行以下命令一次；不得覆盖、自动重试、安装新环境。沿用 `.aris/compute/local.md` research@03909e02。

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.grounded_graph_reconstruction --inputs ../reanchor/outputs/ragtruth_population_20260912/inputs.jsonl --inventory outputs/constraint_inventory_v1_20260913 --output outputs/grounded_graph_reconstruction_v1_20260913
```

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.grounded_graph_feature_runner prepare --reconstruction outputs/grounded_graph_reconstruction_v1_20260913 --model /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct --output outputs/grounded_graph_features_v1_20260913
```

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.grounded_graph_feature_runner prepare --inputs outputs/native_audit_design_20260913/inputs.jsonl --inventory outputs/constraint_inventory_v1_20260913 --model /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct --output outputs/grounded_graph_natural_features_v1_20260913
```

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```

GPU空闲且准备完整，直接依次执行下面两条，无需再次请求用户批准。每次非阻塞会话、记录日志，报告实际PID/forward/进度。先完成训练来源特征，再完成36自然开发回答。自然回答既往已用于开发，不能作为全新test；不进入训练或epoch选择。

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.grounded_graph_feature_runner encode --output outputs/grounded_graph_features_v1_20260913
```

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.grounded_graph_feature_runner encode --output outputs/grounded_graph_natural_features_v1_20260913
```

报告写入 `refine-logs/grounded_graph_feature_doc_witness_20260913.md`：全部数据与feature manifest、240源/36答分母、train/val源SHA隔离、实际forward、数组bytes/shape/finiteness与packet/source row order核验、GPU峰值及各session/exit。遇错保留现场报告，不自行重新构造数据或放宽协议。此文档不授权同时加载两个LLM，现有独占锁继续使用。
