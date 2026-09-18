# Evidence ↔ Target：当前机制主线

当前只研究一个问题：**适用证据为什么没有控制最终答案，以及不同head在证据、错误来源和回答历史之间承担什么功能角色。**

完整定义见 [FLOW.md](FLOW.md)。

## 主实验

    python -u -m experiments.path_conflict.main --study flow

流程只有两步：

1. 基线前向扫描全部layer/head，记录每个head从 evidence / wrong_source / self+history / all context 的实际 A·V·W_O 写入，以及这些写入在当前层对 correct-vs-wrong 候选的局部支持。
2. 只对基线作用最大的少量head做真实删除，继续运行后续网络，得到最终支持、下游抵消和反转。

主要输出：
- baseline_head_sources.csv.gz
- head_roles.csv
- layer_competition.csv
- evidence_forward.csv
- propagation_delta.csv.gz
- same_question_head_deltas.csv
- flow_review.tar.gz

默认只用已有同题多采样的两个人工核验局部事实，不重新采样、不训练检测器。

## 监督检测器

LDA/CHARM使用的模型与本机制样本可能不是同一模型族，head编号不能直接对应。
监督模型自己的head统计角色单独运行：

    python -m experiments.charm_structure_audit.supervised_head_roles

这会解释哪些head在错误token中 self attention 上升/下降、prompt读取上升/下降，以及LDA怎样给这些通道加权。它是监督统计解释，不是原LLM因果head功能。

## 历史实验

固定L22H28/L23H6/L31H14/L31H21的定点实验保留：

    python -u -m experiments.path_conflict.main --study focused

更早的全层粗扫描保留：

    python -u -m experiments.path_conflict.main --study coarse

新结论以 flow 主线为准；旧结果不删除。
