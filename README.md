# Structural attention–entropy detector

主线是 RAGTruth 自然回答上的因果结构特征组合：当前预测熵、margin、来源/远历史注意力、跨头分歧及最近8步的衰减熵记忆。分别检测错误 token 与标注错误 span 的起点。标签不用于构造特征或定位输入节点。

[本轮结果](docs/S10_RESULTS_20260914.md)：150个来源留出测试，组合全token AUROC **0.7492**、**span 起点 0.7791**（不是每答第一次错误），优于本批熵/远历史位移单项；回答级误报仍高，尚非可靠报警器。

当前实现及数据边界见 [方法说明](docs/STRUCTURAL_METHOD.md)。本轮实验使用989条QA原始回答的既有 Llama3.1 observer 测量，按来源划分训练/校准/官方测试。它是有监督轻量读出，不是原生成器机制证明，也没有外部LLM核验。

```bash
# 在 reanchor 中从已完成的真实内部测量缓存导出无标签特征
bash ../reanchor/scripts/export_structural_features.sh full
# 在 graph 中一次训练、冻结预测、评价
bash scripts/run_structural_fusion.sh
# 单元测试
python -m pytest tests -q
```

新输出目录必须不存在；已有结果不得覆盖。参数化入口：

```bash
python main.py --features /path/to/export --annotations /path/to/RAGTruth/response.jsonl --output /new/output
```

S10 核心仍为 `structural_detector/features.py` 和 `structural_detector/experiment.py`。
新增 `structural_detector/audit.py` 对冻结分数补评整答首错、每段起点和延续，不重新训练。
原始测量及原生机制工具由 [reanchor](https://github.com/DwightEd/reanchor) 提供，不依赖本项目的旧实验代码。

## 绑定投影实验（未验证自然检测效果）

[方法、审查和完整命令](docs/BINDING_PROJECTION_20260914.md)。
`binding_detector/projection.py` 固定节点匹配分布，计算合法关系的联合概率质量和最小 KL 修正量。
支持未知关系上下界、关系身份置换、缓存向量匹配及与 S10 同队列比较。
**它需要额外的来源对应与关系数据包，不能由 S10 五列标量自动构造事实图。**
未实现通用自然文本事实抽取器；数学 demo 不是 RAGTruth 检测成绩，不替代当前 S10。

```bash
python -m pytest tests/test_binding_projection.py -q
python -m binding_detector.run --demo --output outputs/binding_projection_demo_v1
```

被替代的检测、外部核验、临时编排及对应测试已移除。完整旧代码保存在 `archive/pre-structural-20260914`（远端提交 `5186c4f`，原本地提交 `c3d2aad` 另有保留），[清理清单](docs/STRUCTURAL_REFACTOR_20260914.json)记录每条路径；历史 `outputs/`、`results/`、文档与执行快照保留。旧文档里的运行入口属于该归档版本。
