# Offline span graph：设计与空接口

当前状态：**只有设计、数据字段和函数签名。没有实现、训练、评分或运行命令。**
所有算法函数只有说明和 `raise NotImplementedError`；这是待填实现的位置，不是输入防御分支。
不返回零分、随机分数或虚假完成状态。没有修改旧 `run.py`、`main.py` 或旧缓存。

先读：

- [模型设计](../../../iclr/OFFLINE_SPAN_GRAPH_DESIGN.md)
- [分阶段代码规划](../../../iclr/OFFLINE_SPAN_GRAPH_PLAN.md)
- [代码规范](../../../iclr/CODING_GUIDELINES.md)

模块顺序：`data → graph → regions → controls → model → learning → detection → evaluation`。
`run.py` 仅声明未来的三个流程入口，不执行任何模型操作。

关键变化：允许完整回答中的后文参与早期token和span的离线判读；
后文读取索引不是原模型的反向因果边；片段聚集是结构假设，不是幻觉真值。

本次只检查语法和空接口结构，不把这些检查称为方法测试通过。
