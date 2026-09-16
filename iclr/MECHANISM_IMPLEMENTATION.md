# 代码实施与验证

## 当前目录

`experiments/unsupervised_token_graph/span_audit/`

| 文件 | 一个明确职责 |
|---|---|
| units.py | 原始回答、金标区间、匹配对的数据定义 |
| inputs.py | 原NPZ与RAGTruth身份/offset核验、按head迭代 |
| matching.py | 同答正常区间匹配、共同的边界观察位置 |
| readings.py | 一行attention的来源统计与条件端点基线 |
| measurements.py | 按入口、段内、段后三个问题计算同head配对测量 |
| statistics.py | source均值与source bootstrap |
| report.py | 保存逐答数值、文本配对和分组汇总 |
| experiment.py | 读答、配对、测量、保存的顺序主流程 |
| run.py | 命令参数、一次输入配置、运行与汇总 |

没有模型、优化器、损失、随机错接训练或预测片段解码。
没有新造一套attention格式。依赖父目录现有ResponseCache、CacheIndex、iter_channels、
EvaluationBinding；现有底层reader仍负责必要的原数据契约。
当前主数据明确允许读取RAGTruth labels。训练/评价隔离规范对未来检测器仍然有效，
但不能反过来阻止现在读取标签发现机制。

## 先实现什么，后面才能实现什么

A已实现：三种原attention格式、裸cache的身份与offset补齐、真实标签配对、
逐head的直接读取对照、结束后正常内容测量、缺失覆盖和配对source统计。
没有hidden时不伪造表征；不使用随机embedding“代替语义”。熵只读已有entropy字段。

B尚未实现：独立核验当前陈述的对象/阶段/范围，标记适用和不适用来源。
不拿target最大的attention当正标签。先在原烹饪/头饰案例中人工核验，然后固定格式。
B实现前，不在A报告中使用“正确证据被覆盖”“绑定读出正确”等结论。

C尚未实现：对B核验的同一事实，在关系承诺前比较表示与使用；小批量语义互换，
正确的接收者内容必须区别于donor答案，并包括同关系改写、随机子空间与有效读取位置对照。
同一层邻接递推不是真实多跳。新的跨层消息/子空间实验须有数据覆盖及明确功能假设。
保留reanchor的独立论文复现；本次不改它，也不把正常任务的指针结果当自然错误原因。

D尚未实施：只把复验稳定的机制用于完整回答的无监督图异常检测。
不得再自动以“观测边为正、人造错接为负”启动大规模训练。

## 退出旧主线

旧offline_span的训练模块不再作为可运行主线，两个入口给出迁移说明并退出。
与旧代码配套的训练测试退役，不将其历史通过数继承到新方法。
旧实现可从提交 8e0fcef778d3335f8836c87d1ba7a8be9176612a 查看。
保留旧data/metadata/evaluation读取工具，避免破坏历史结果读取及其他脚本。
旧SourceFlow、FlowTracer、CHARM和reanchor独立审计不修改。任何缓存/模型/结果均不删除。

## 本轮测试而非实验结果

测试覆盖：标签字符→token；同长度匹配与容差；缺熵；q→q+1与lag1；
条件端点期望对穷举；纯局部续写不产生伪超额；终点不强行清零；
缺测分母；source权重；真实父reader的dense/CSR/单head格式；三任务CLI及续跑。
这些只验算实现，不证明自然错误与正确存在差异。
测试数与运行日志在发布时另外列出，不沿用旧提交声称的通过数。
