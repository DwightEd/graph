# GroundedGraphAdapter 来源弱监督数据执行

主方法与预先固定评分见 [GROUNDED_GRAPH_MODEL_20260913.md](GROUNDED_GRAPH_MODEL_20260913.md)。本步骤仅生成训练模板：来源提供 exact payload 和 owner 指针，Qwen 生成自然句模板，程序插入 payload。模板语义仍是弱监督，机械成功率不是幻觉检测指标。响应侧 Qwen reasoning graph 参照当前不运行。

沿用 `.aris/compute/local.md` 的 research@03909e02、现有本地 Qwen3-8B 和 RTX4090，不安装或重建。数据/执行工程 Required=0 后，由新文档执行见证按下面顺序各执行一次。只使用 official-train 来源，全文 SHA 决定 train/validation，每类64/16，共240来源，最多1920模板。相同七天值显式 bundle 优先保留；七个单日 scalar 共享抽样 family。要求模板在 VALUE 前提供实体/属性上下文，以对应真正的 pre-token query；这不是语义正确性的证明。

输出必须不存在，不覆盖或自动重试。预检命令无模型 forward。实际生成固定 thinking=true、temperature0.6/top_p0.95/top_k20、4096新token上限、batch2、16k总长度上限。超长/生成截断/JSON失败/null 全部保留并计入分母；不靠挑自然正误标签选例。当前无法精确预估吞吐，启动后用前几个真实 batch 报告预计时间。

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.grounded_graph_data --inputs ../reanchor/outputs/ragtruth_population_20260912/inputs.jsonl --inventory outputs/constraint_inventory_v1_20260913 --model /share/home/tm902089733300000/a903202310/lys/models/Qwen3-8B --output outputs/grounded_graph_data_v1_20260913
```

预检 source=240、来源 train/val 全文 SHA 无交集、输入/库存 settings/selected-source SHA 已绑定，工程测试通过且 GPU 空闲（下面命令）后直接执行生成，不另行申请用户批准。

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.grounded_graph_synthesize --output outputs/grounded_graph_data_v1_20260913
```

以非阻塞会话执行，保留输出日志，及时报告实际 session/PID、已完成来源/批次数、实际 model forward 数和失败分母。执行见证报告写入 `refine-logs/grounded_graph_data_doc_witness_20260913.md`，核验 code/model/input/inventory/prepared/results manifest。结束前后 settings/source/input/code 不得改变。此阶段没有 adapter 训练或自然回答评价，不得把模板生成数当方法效果。
