# 最终候选 → 原生消息：当前机制实验

当前只研究一个问题：**适用证据为什么没有控制最终答案，以及不同head在证据、错误来源和回答历史之间承担什么功能角色。**

完整定义见 [v3 方法](../../docs/TRANSPORT_METHOD_20260919.md)，
研究依据见 [18 项文献核查](../../docs/LITERATURE_TRANSPORT_20260919.md)。

## 主实验

    python -u main.py flow

默认写入 `outputs/target_transport_v3`；同命令续跑，只重建报告用 `--stage report`。
流程分为三步：

1. 对完整候选 log 概率差，沿原模型逐层反向计算消息 gate 导数；保留 layer/head/receiver/source 身份。
2. 在干预前冻结正负消息候选和配对；执行单删、双删及同范数随机方向控制，记录首词和完整候选结果。
3. 对满足因果先后关系的消息，删除上游后恢复下游消息或同位置 MLP；用同世界恢复检查数值误差。

来源角色仅在自动选点后追加。候选仍由人工核验；该实验不是无标签自然检测。

主要输出：
- transport_edges.csv
- transport_interventions.csv
- transport_interactions.csv
- transport_mediation.csv
- REPORT_TRANSPORT_zh.md
- flow_review.tar.gz

默认只用已有同题多采样的两个人工核验局部事实，不重新采样、不训练检测器。

## 监督检测器

LDA/CHARM使用的模型与本机制样本可能不是同一模型族，head编号不能直接对应。
监督模型自己的head统计角色单独运行：

    python -m experiments.charm_structure_audit.supervised_head_roles

这会解释哪些head在错误token中 self attention 上升/下降、prompt读取上升/下降，以及LDA怎样给这些通道加权。它是监督统计解释，不是原LLM因果head功能。

## 历史实验

v2 的共享局部 lens 和来源分组实验见 [FLOW.md](FLOW.md)：

    python -u -m experiments.path_conflict.flow --output outputs/evidence_target_flow_v2

固定L22H28/L23H6/L31H14/L31H21的定点实验保留：

    python -u -m experiments.path_conflict.main --study focused

更早的全层粗扫描保留：

    python -u -m experiments.path_conflict.main --study coarse

不同协议结果分别保存；v3 尚无用户 8B 新结果，不用软件测试替代机制证据。
