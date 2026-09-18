# 主线：Evidence ↔ Target 功能流

当前研究只回答一件事：适用证据怎样进入候选决策，哪些head支持/抑制它，是否出现竞争、反转或下游丢失。

## 两端同时追

Target -> source：在候选分叉位置定义 M=logit(correct)-logit(wrong)。基线一次记录每层每头的实际 A·V·W_O 写入及局部候选支持，不预选head。按局部作用发现少量候选head，再做真实删除：

    final_support = M_full - M_without_path

正值表示路径帮助正确候选，负值表示推动错误候选。

Evidence -> target：人工核验来源只用于 scope / supported_value / wrong_source 的语义角色。每条证据同时记录：
1. attention mass：是否读取；
2. local_lens_support：当前层是否已经支持正确候选；
3. final_support：删除后跑完剩余网络，最终候选优势损失多少。

局部为正而最终接近0表示作用被后续抵消；局部和最终异号表示下游反转。

## Head角色

每个确认head输出 all_context、evidence、wrong_source、history 四种作用。一个head内部即可出现来源冲突，不预先命名“好头/坏头”。

同一层对全部head计算：

    balance = 2*min(sum positive, sum |negative|)/(sum positive + sum |negative|)

0表示同向，1表示支持正确和支持错误的作用接近平衡。这是候选条件下的局部竞争量，不是幻觉概率。

## 运行

    python -u -m experiments.path_conflict.main --study flow

默认先扫描全部head；每种来源角色取局部作用最大的少量head并集，再做最终干预。
flow-top-k和flow-max-heads控制成本，选择只依赖基线局部作用，不依赖干预后的结果。

输出：
- baseline_head_sources.csv.gz：全部head、全部来源组的局部作用；
- selected_heads.json：进入最终干预的head；
- head_roles.csv：局部与最终角色、冲突与反转；
- layer_competition.csv：逐层正/负head竞争；
- evidence_forward.csv：证据等来源在各层的整体局部支持；
- propagation_delta.csv.gz：删路径后影响怎样沿后续层传播；
- same_question_head_deltas.csv：同题有依据/无依据轨迹差异；
- flow_review.tar.gz：小型结果包。

这是两个局部案例的机制发现，不是总体检测评价。

## 监督模型

CHARM/LDA审计若使用另一模型族，head编号不能与本实验直接对应。
单独运行：

    python -m experiments.charm_structure_audit.supervised_head_roles

它输出该监督检测器自己的self/prompt差异、LDA权重和每个head的风险贡献。
