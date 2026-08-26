# Information Flow

这个实验验证一个比“某层 attention 是否异常”更具体的问题：

> 按 Transformer 层顺序连续传播的 token 信息流，是否比最后一层、层平均或错误层序更能表征幻觉节点？

代码以论文 **Information Flow Reveals When to Trust Language Models** 为主要依据，但当前数据只有 attention weight、diagonal 和 unresolved mass，没有 value、`W_O`、hidden state 或外部 relevance。因此这里实现的是一套 **attention-only information-flow proxy**，不是对原论文 contribution matrix 的等价复现。

详细方法、论文复盘、对照和停止条件见 [`METHOD.md`](METHOD.md)。

## 目录

```text
config.py       视图与实验配置
basis.py        不含标签的共享 source sketch
transport.py    逐层稀疏信息流传播
extract.py      构图、导出节点表征和逐样本图数据
evaluate.py     复用统一 node-only detector 与 probe
run.py          命令行入口
run_qa.sh       QA 一键运行
METHOD.md       论文、假设、算法、控制和停止条件
tests/          图算子与层序控制测试
```

## 一键运行 QA

```bash
bash experiments/information_flow/run_qa.sh
```

烟测：

```bash
OUT=experiments/information_flow/outputs/qa_smoke \
TRAIN_LIMIT=30 \
TEST_LIMIT=10 \
FOLDS=2 \
EPOCHS=3 \
BOOTSTRAP=100 \
bash experiments/information_flow/run_qa.sh
```

输出：

```text
experiments/information_flow/outputs/qa/
├── calibration/
│   ├── index_*.npz
│   ├── graphs/*.npz
│   └── run.json
├── test/
│   ├── index_*.npz
│   ├── graphs/*.npz
│   └── run.json
└── evaluation/
    ├── unsupervised_scores.npz
    ├── probe_scores.npz
    └── report.json
```

所有表征提取都不读取幻觉标签。标签只在 `evaluate.py` 中用于最终指标和明确标注为诊断的 source-disjoint linear/MLP probe。
