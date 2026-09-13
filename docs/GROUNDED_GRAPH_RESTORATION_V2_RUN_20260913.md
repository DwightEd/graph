# A2-v2 来源恢复执行文档

沿用 .aris/compute/local.md 的 research@03909e02 / RTX4090，无安装、下载或环境重建。方案 GROUNDED_GRAPH_RESTORATION_V2_20260913.md。v1已冻结，全部新目录，已有输出不得覆盖/自动重试。所有GPU命令独占reanchor/runs/relation_interleave_20260913.lock，非阻塞会话保存日志。

新鲜执行见证先确认对应模块工程Critical/Required全部闭合。feature审查完成可先跑命令1、2；train/predict/evaluate审查完成才继续3、4、6；dependence审查完成才跑5。root负责通知相应闭合状态，不需再问用户批准。完整parent manifest成功后才进下一项，失败即保存并报告，不能伪造完成。

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```

## 命令1

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.grounded_graph_restoration_features --original outputs/grounded_graph_features_v1_20260913 --output outputs/grounded_graph_restoration_features_v2_20260913
```

## 命令2

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.grounded_graph_restoration_features --original outputs/grounded_graph_natural_features_v1_20260913 --output outputs/grounded_graph_restoration_natural_features_v2_20260913
```

## 命令3

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.grounded_graph_restoration_train --features outputs/grounded_graph_restoration_features_v2_20260913 --output outputs/grounded_graph_restoration_train_v2_20260913
```

## 命令4

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.grounded_graph_restoration_predict --training outputs/grounded_graph_restoration_train_v2_20260913 --features outputs/grounded_graph_restoration_natural_features_v2_20260913 --output outputs/grounded_graph_restoration_predictions_v2_20260913
```

## 命令5

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.grounded_graph_restoration_dependence --training outputs/grounded_graph_restoration_train_v2_20260913 --erasure outputs/grounded_graph_erasure_v1_20260913 --output outputs/grounded_graph_restoration_dependence_v2_20260913
```

## 命令6

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.grounded_graph_restoration_evaluate --predictions outputs/grounded_graph_restoration_predictions_v2_20260913 --population ../reanchor/outputs/ragtruth_population_20260912 --output outputs/grounded_graph_restoration_evaluation_v2_20260913
```

命令1/2实际新observer前向240+36次。新query是来源被实际擦除的H_empty；原始source X和full_query字节引用父特征。命令3保持240来源/192训练48验证/135519token/9220anchor、同参数同初始化10epoch两臂，选择不读RAGTruth标签。命令4保留4733全词，两个高风险差分显式分开：graph_difference=logp_full−logp_restore；graph_restoration_difference=logp_empty−logp_restore。不能互换/取反/挑一个冒称原方案。命令5复用48次真实v1 payload erasure缓存，不新造source向量，新H_empty在两分支完全相同；判断预先固定anchor恢复/内容依赖门控。命令6只在预测与评价代码冻结后join真实标签，36答为反复使用的开发集，6答来源进过source-only预训练，30答source-unseen单列。仍未运行官方2700test。

报告实际PID/session/exit、全部manifest+模型+source/node/query receipt及数组检查、两臂选定epoch、前向计数、全部词分母、来源重合与两种风险分数；失败/partial不得省略。工程见证不代替外部科学integrity audit。执行记录保存 refine-logs/restoration_v2_doc_witness_20260913.md。
