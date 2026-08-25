# P-Cut

当前实现已经不再训练 HoloRoute 图掩码自编码器。旧方法的 QA 结果低于绝对位置基线，相关代码已删除，结果保存在 [`iclr/HOLOROUTE_BASELINE_QA_RESULT.md`](iclr/HOLOROUTE_BASELINE_QA_RESULT.md)。

P-Cut 研究一个更明确的问题：当前 token 的路由状态更依赖从 prompt 继承的路径，还是依赖回答内部已经闭合的路径。

## 代码入口

```text
graph.py       sparse attention -> multiplex token graph
pcut.py        prompt provenance -> matched graph cuts -> token embeddings
detection.py   closure score 的 train-only 条件校准
pipeline.py    fit reference, score, export per-sample graph data
evaluate.py    分数冻结后读取标签
run.py         命令行入口
```

核心计算路径：

```python
graph = build_graph(sample, config.graph)
result = compute_pcut(graph, config.pcut)

result.token_layer_embedding
result.token_embedding
result.prompt_necessity
result.response_closed_necessity
result.closure
```

每个测试样本都会在 `graphs/` 下保存一个 `.npz`，其中包含完整 sparse edges、prompt-rooted / response-closed / uncertain edge mass、逐层 prompt provenance、token-layer embedding 和最终 token embedding。

## 一键运行 QA

```bash
bash experiments/holoroute/run_qa.sh
```

默认路径：

```text
train: /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/train
test:  /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test
out:   experiments/holoroute/outputs/pcut_qa
```

## 研究文档

- [`iclr/LITERATURE_AND_EXPERIMENT_AUDIT.md`](iclr/LITERATURE_AND_EXPERIMENT_AUDIT.md)
- [`iclr/CORE_RESEARCH_PARADIGM.md`](iclr/CORE_RESEARCH_PARADIGM.md)
- [`iclr/HOLOROUTE_BASELINE_QA_RESULT.md`](iclr/HOLOROUTE_BASELINE_QA_RESULT.md)
