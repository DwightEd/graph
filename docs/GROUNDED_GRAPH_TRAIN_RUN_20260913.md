# GroundedGraphAdapter 训练和自然评价执行

沿用 research@03909e02，不安装或重建。完整方法见 GROUNDED_GRAPH_MODEL_20260913.md。训练/预测工程C0/R0、9项CPU通过；评价工程闭合后才能执行最后的标签join。两个feature输出必须manifest完整并由前一见证结束释放GPU；不用Qwen失败模板。原始数据/特征/所有训练依赖自启动起冻结。

固定10epoch、batch8来源、AdamW5e-4/decay0.01/clip1，128维2消息步，graph与no_edges相同初始化和每epoch样本顺序。epoch0也可按source-validation联合loss选择；不选自然回答指标、不调差分方向。只加载冻结LM head训练小适配器，无observer新forward。独占同一锁。

新执行见证在父feature完成和GPU空闲后，逐条执行一次。非阻塞会话、记录日志，不覆盖/自动重试。无需再次问用户批准。

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.grounded_graph_train --features outputs/grounded_graph_features_v1_20260913 --output outputs/grounded_graph_train_v1_20260913
```

训练 complete 后直接执行自然预测。三种图条件包括graph、独立训练no_edges、固定graph模型的边目标置乱。边置乱是破坏结构的诊断，不是语义等价反事实。保留所有token分数、全词、完整pointer数组、gate/residual轨迹。模型分配不声称原LLM原生采用路径。

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.grounded_graph_predict --training outputs/grounded_graph_train_v1_20260913 --features outputs/grounded_graph_natural_features_v1_20260913 --output outputs/grounded_graph_predictions_v1_20260913
```

预测complete且evaluator独立工程C0/R0，冻结评价代码后才读取真实label。原36回答仅开发诊断，6来源且部分可能进source-only预训练；训练重合单独报告，不称独立test。全词/through-first/strictpostfirst都报告原LM entropy/NLL、adapter NLL、固定差分与无边/破坏边对照。无available输出也保留词分母。

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.grounded_graph_evaluate --predictions outputs/grounded_graph_predictions_v1_20260913 --population ../reanchor/outputs/ragtruth_population_20260912 --output outputs/grounded_graph_evaluation_v1_20260913
```

报告 `refine-logs/grounded_graph_train_doc_witness_20260913.md`：各session/PID/exit、实际epoch/checkpoint/LMhead身份、source split、loss/梯度稳定性、完整manifest、自然词/token覆盖和固定方向指标。验证结果只支撑实际得到的检测排序，不支撑正确owner/回看/路由/连续因果范围。owner-payload source擦除对照由独立脚本和独立执行文档继续，不省略也不把未跑对照说成成功。
