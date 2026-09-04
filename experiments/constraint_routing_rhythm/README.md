# Constraint Routing Rhythm

该目录保留为两部分基线：

1. 用 `A * ||W_O[head] V[kv(head), source]||` 描述 local/global route 与
   `evidence -> response carrier -> future query` 两跳 relay；
2. 真实删除 evidence-source Value messages，测量固定 target-versus-runner margin
   的变化：

```text
ConstraintDeficit = cut_margin - baseline_margin
```

路由量只用于解释与 carrier proposal，不与主分数组合。这个分数测量给定
teacher-forced prefix 下的 evidence-channel sensitivity，不等于无条件 factuality。

## 运行

```bash
bash experiments/constraint_routing_rhythm/run_all.sh --smoke

bash experiments/constraint_routing_rhythm/run_all.sh \
  --limit 20 --audit-limit 2 --plot-limit 2 \
  --output experiments/constraint_routing_rhythm/outputs/pilot
```

`--audit-limit 0` 关闭 direct-response、matched non-evidence 和 U/D/UD 额外分支。

## 模块责任

| 模块 | 责任 |
|---|---|
| `data.py` | 兼容导出；实现位于 `experiments/common/ragtruth_alignment.py` |
| `routes.py` | 流式归约 `A||W_OV||` 路由 |
| `rhythm.py` | local/global、uptake、delivery 与 carrier proposal |
| `capture.py` | baseline、evidence cut 和可选 U/D audit 编排 |
| `analyze.py` | label-free 遍历、释放、保存和画图 |
| `artifacts.py` | 轻量 NPZ 保存与恢复 |
| `evaluate.py` | artifacts 冻结后读取 hallucination labels |
| `run.py` | CLI |

真实消息观察和干预统一位于：

```text
experiments/common/llama_message_intervention.py
```

该模块直接运行 Llama decoder 的 q/k/v/o projections、RMSNorm、residual 与 MLP，
不依赖 Transformers 版本不稳定的 attention backend registry。Message gate 始终
位于 softmax 后、Value 求和前；删除质量不重新分配，后续层全部重算。

## 测试

```bash
python -m pytest -q experiments/common/tests
python -m pytest -q experiments/constraint_routing_rhythm/tests
python -m compileall -q experiments/common experiments/constraint_routing_rhythm
bash -n experiments/constraint_routing_rhythm/run_all.sh
```

正式机制主张仍要求 matched cut、evidence polarity、rhythm phase-null 和跨任务结果；
当前目录本身不证明 constraint integration 或 hallucination causality。
