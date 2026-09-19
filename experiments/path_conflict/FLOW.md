# 主线：Evidence ↔ Target 功能流

2026-09-19 v2：当前判据和研究边界以
[收敛说明](../../docs/DETECTION_CONVERGENCE_20260919.md)为准。
推荐入口 `python -u main.py flow`，默认新目录 `outputs/evidence_target_flow_v2`。
增加query_self/recent_history/remote_history/past_history单独删除；history仅保留旧合并口径。
local_linear_support使用共享FP32局部lens梯度，逐头可加；旧local_lens_support是有限删除，
不可相加。新head_interactions.csv记录四世界交互，首词/整段候选并报。
同形模型不能按head编号对齐；完整checkpoint身份缺失时保持未对齐。

当前研究只回答一件事：适用证据怎样进入候选决策，哪些head支持/抑制它，是否出现竞争、反转或下游丢失。

## 两端同时追

Target -> source：在候选分叉位置定义 M=logit(correct)-logit(wrong)。基线一次记录每层每头的实际 A·V·W_O 写入及局部候选支持，不预选head。按局部作用发现少量候选head，再做真实删除：

    final_support = M_full - M_without_path

正值表示路径帮助正确候选，负值表示推动错误候选。

Evidence -> target：人工核验来源只用于 scope / supported_value / wrong_source 的语义角色。每条证据同时记录：
1. attention mass：是否读取；
2. local_lens_support：当前层是否已经支持正确候选；
3. final_support：删除后跑完剩余网络，最终候选优势损失多少。

局部读数与最终删除效应不同，只提示后续计算可能改变作用；不能凭两者相减量化信息损失。
异号还需排除局部lens不准、干预幅度和数值误差。

## Head角色

每个确认head输出 all_context、evidence、wrong_source、history 四种作用。一个head内部即可出现来源冲突，不预先命名“好头/坏头”。

同一层对全部head计算：

    balance = 2*min(sum positive, sum |negative|)/(sum positive + sum |negative|)

0表示同向，1表示候选方向相反的线性投影接近平衡。v2才具有线性可加口径；
旧finite-lens结果仅是删除敏感性分布。二者都不是幻觉概率。

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


## Binding completeness追加检验

原head发现规则保持不变，仍只依据all_context/evidence/wrong_source/history的基线局部作用。
对这批冻结head额外把人工核验的evidence拆成：

    condition = scope
    value = supported_value

并分别做真实A·V·W_O删除，再与joint evidence删除比较。
输出binding_completeness.csv：
- both_support_candidate：两个来源位置的删除效应都支持指定候选，不证明完整绑定；
- value_effect_only：值位置效应超过阈值，条件位置效应弱，不表示条件未被编码；
- condition_effect_only：相反的直接位置效应模式；
- condition_value_opposed：两条路径方向相反；
- joint_nonadditivity：joint cut与两个single cut之和的差；只表示后续网络非加性，不直接声称语义binding synergy。

这只能检查直接来源位置效应。V已经含上下文，尚不能单靠这些字段检验语义partial binding。
