# RAGTruth Evidence ↔ Target population audit

把两案例机制实验扩展到现有 RAGTruth，但严格区分缓存能测什么、原模型回放能测什么。

TRAIN screen：读取已有 Llama-3.1 attention cache，用金标错误 span 与同答等长正常区间比较逐 head 的 prompt / self / recent-history / remote-history 路由。它只用于筛选待确认 head，不把 attention mass 当因果支持。

TEST confirm：冻结 TRAIN 选出的 head，teacher-force 保存的原 token 前缀，并删除指定来源经原 A·V·W_O 写入的 message。

final_support = log p_full(y_t) - log p_without_message(y_t)

正值表示该路径帮助实际生成 token，负值表示抑制它。对错误 token 而言，正 history support 表示帮助已经生成的幻觉 token，不代表事实正确。

RAGTruth 没有逐 span 的正确替代文本，所以这里不伪造 correct-vs-wrong candidate。source 组只使用 source_info 在保存 prompt 中能够精确定位的 token；覆盖失败时 source 为空并报告。

一键运行：

    python -u -m experiments.ragtruth_flow.run --phase all --resume

先 CPU 全量筛查：

    python -u -m experiments.ragtruth_flow.run --phase screen --resume

再 GPU 确认：

    python -u -m experiments.ragtruth_flow.run --phase confirm --confirm-pairs 8 --confirm-heads 4 --resume
    python -u -m experiments.ragtruth_flow.run --phase report

默认 TRAIN 覆盖 QA / Summary / Data2txt。TEST 首次确认按 source 只取一对，默认 8 个 source；增加 confirm-pairs 时保持 TRAIN 冻结 head 不变。

主要输出：screen_head_effects.csv、selected_heads.csv、functional_effects.csv.gz、functional_pair_summary.csv、functional_competition.csv、ragtruth_flow_review.tar.gz。

这是标签辅助机制研究。被审计的是统一 Llama-3.1 observer 对既有 RAGTruth 回答的 teacher-forced 处理，不是六个原生成器自身的内部因果机制，也不是无监督检测成绩。

## Past-only route forecast（修正后的对照基线）

v1用当前Δsource/Δother_prompt/Δhistory预测Δself，完整行存在质量守恒恒等式。
旧成绩保留，但不解释成语义grounding失配。v2只用过去信息做路由预测，作为对照基线。
它不随`--phase all`自动启动。

```bash
python -u -m experiments.ragtruth_flow.run --phase grounding
```

TRAIN全程不读幻觉标签。对每层32个head，用

    Δself_t <- [self_(t-1), source_(t-1), other_prompt_(t-1), history_(t-1)]

拟合正常的内部动力学关系。每条回答等权，不让长回答主导。
然后对预测残差建立两种逐层无标签参考：
1. raw residual Mahalanobis；
2. 去掉层共同分量后的head-contrast residual Mahalanobis。

第二个保留head相对关系，但不假定无标签异常方向等于LDA的监督判别方向。
先写完所有无标签分数再附加TEST标签。previous_gold=0集合排除了每答第一词，
不能称完整首错评价；previous_gold=1用于比较延续错误与恢复正常。
新结果写`past_route_forecast_v2/`，拒绝混用v1回归/参考文件。

source只使用source_info在保存prompt中能精确定位的token；定位失败的样本不拿整个prompt替代。
