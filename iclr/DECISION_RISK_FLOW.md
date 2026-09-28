# 原生选择与预测风险：研究方案及反例审计

当前已实现分析入口和未训练的候选条件probe/响应收缩核心；8B原生采集、训练及完整检测器仍待实施。
用户允许轻量监督，本方案采用自然token/span标签，不称无监督；不预标prompt实体或证据。

共享研究文档：`/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/decision_risk_flow_20260928/`。
该目录保存完整方案、原文调研、逐轮评审、结果及执行状态。原始输出在本仓库
`outputs/decision_risk_flow_20260928_v3/`；前两版保留，v3统一使用原采样logit核验选择margin。

## 本轮已完成

`experiments.role_free_flow.diagnostics`复用旧冻结feature，在两题四个已审核局部声明上计算必要条件反例。
31个token中18个错误；错误中15个原生选择margin为正、12个熵低于1bit、9个不触发旧事件。
洋葱错误段6/7个token的prompt净直接选择写入为正；头饰完整message未测，不填0。
正确局部对照同样统计，不把局部正确当整答无错。事件覆盖、prompt净写入正负都不是检测正确率。

```bash
python -m experiments.role_free_flow.diagnostics \
  --readouts outputs/role_free_flow_20260928 \
  --output outputs/decision_risk_flow_recompute
```

输出目录须不存在。无新LLM前向，无分类器拟合，未计算新AUROC。
原生margin/熵、事件计数、message符号和缺测经另一段CSV/NumPy计算复算一致；这是同agent复核。

`experiments.role_free_flow.risk_response`实现15万参数以内的候选条件probe、给定两种原生
下游梯度后的逐消息响应收缩，以及每结构组8项统计。没有拟合权重或8B梯度数据。
`tests/test_risk_response.py`用非线性toy模型核验显式门梯度/有限剂量、候选条件化、
正负抵消和尺度行为；软件数值验证不等于检测有效性。运行：

```bash
python -m pytest tests/test_risk_response.py -q
```

## 研究主张

每个实际采样token均评分。候选条件内部probe提供统计错误风险，再在同一消息门上分别计算
原生选择margin响应和probe logit响应。两种响应的关系、结构来源及过去预测风险进入紧凑读出。
风险梯度是probe显著性，不能称独立真值或正确证据；message地址也不是语义根来源。

必须比较同输入强普通probe及原生响应基线，检验双响应关系的来源留出增量。
固定误报预算下逐项检查9个历史错误span与正常对照，再完整三任务探索复验。
当前没有新检测结果，不替换已有默认模型，不声称已解决全部漏检或达到顶会要求。

## 监督和运行状态

轻量指冻结8B模型、每折只训练147457参数的rank8风险probe，加80输入/81参数条件读出；
五折推理共有737285个probe参数，不能把单折数量当整个ensemble数量。训练标签来自官方
回答错误span映射的token，不需要人工prompt实体/证据，也不在测试评分时使用真值。
新方法没有训练过，尚未证明少量标签足够；全量梯度采集的算力成本也尚未实测。

`python main.py probabilistic-test`仅复评旧的已训练条件交互模型和HGB，不是本新方法入口。
新方法仍缺来源留出probe训练、原生梯度采集与最终读出训练/冻结；没有有效权重时不能给出
一键新方法测试命令。旧方法测试入口和参数见`experiments/probabilistic_detection/README.md`。
