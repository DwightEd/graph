# 有限标签读出诊断：冻结自然请求的 CPU 回放

目的：核实首 token 在 S/C/N/U 等字母之间的条件分数是否掩盖模型实际输出格式。它不评价真假、不改变阈值、不替代原GPU结果，不是关系影响决策实验。

冻结 surface_owner_v1 的五类 strict request，各取 request digest 字典序最小的2个；不按分数、样本错误标签或已有人工判断挑选。保留完整原 instruction/payload/template，并复验模型文件与旧请求哈希。使用同研究环境、CPU bf16 SDPA、4线程；原全量GPU任务继续。CPU/GPU数值不保证逐位相同，结果并列保存。

记录全词表标签总质量、每字母绝对与条件概率、top12 token、最多48 token自然贪心续写，以及实际forward次数。不把生成解释作为真值，不改变旧缓存。新目录只写一次。

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.reader_output_audit --run outputs/surface_owner_v1_20260913 --output outputs/reader_output_cpu_audit_20260913 --per-kind 2 --max-new-tokens 48
```

解释分支：若大多自然输出是指定字母且标签总质量高，不能将语义失效归于格式/条件归一化；若字母总质量低、自然输出是别的格式，则首先修复读出可观测性，仍不推出修复后语义正确。若CPU与缓存差异很大，不能直接将该CPU观察归给原CUDA运行。无新效果声明。
