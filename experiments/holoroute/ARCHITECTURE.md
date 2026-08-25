# P-Cut 代码结构

```text
graph.py
    读取 sparse attention，构造 token graph，保存 diagonal 和 unresolved mass。

pcut.py
    传播 prompt provenance 上下界；拆分 P/R/Q edge mass；构造等质量 cuts；
    输出 token-layer embedding、token embedding 和 closure score。

detection.py
    只校准一个 closure residual，不再联合六个 reconstruction residual。

pipeline.py
    在 train 上拟合无标签 reference；在 test 上冻结分数；每样本导出 graph npz。

evaluate.py
    分数文件写完之后才读取 token labels。

run.py / run.sh / run_qa.sh
    fit -> score/export -> evaluate。
```

旧的 `model.py`、`learning.py` 和 `baseline.py` 已删除。它们对应的 HoloRoute masked-reconstruction 方法结果保存在 `iclr/HOLOROUTE_BASELINE_QA_RESULT.md`，代码仍可从 Git 历史恢复。
