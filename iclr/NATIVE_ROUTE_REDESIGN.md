# 恢复路由信号并统一同样本比较

2026-09-22。此协议取代 NATIVE_SUPPORT_DESIGN 中将净写入传播指定为主检测器的部分。
保留原始采集、原结果和原评分定义；新的派生结果写入原输出目录的 route_comparison_v2。

## 为什么改

用户给出的四答结果：836 token，81 个错误，native support AUROC .57233、AP .10394；
direct prompt AUROC .56972；首错只有 2 个，AUROC .39205。没有证据支持历史传播的增益。
账本闭合只验证数值分解，不验证幻觉检测。此前未保留有效历史对照，是实施上的缺口。

历史依据分别保留，不把它们写成这四答的新结果：

| 方法 | 来源 | 已有结果与范围 |
|---|---|---|
| routing_imbalance | 079c33a 的 capture/evaluate；现存 run_all_qa.log | 971 答、215710 token；pooled AUROC .721235 |
| attention displacement | a2a40fd control_graph/audit.py；docs/ATTENTION_AUDIT_688.json | 688 来源、153513 token；equal-source AUROC .712162 |
| functional/attention route collapse | f7344e2 capture.py/detect.py；历史会话报告 | QA 约 .7337/.7333；本轮未取得原逐词结果，不能重新认证该成绩 |

这些实验的 prompt 模板、证据分区、总体和评价权重不全相同。不得直接把 .7337→.5723
解释为同数据上的损失。代码层面可以确认：新方法遗漏了旧方法的原始路由和来源/头结构信号。

## 一个采集底座，分开保存读取结构与输出作用

每条边保持物理 layer/head/source 身份：A 是读取质量，E=||A W_O V||²，M=sqrt(E)。
当前缓存已经保存 A/E/有符号输出投影 b，因此无需新的 8B 前向。
不能用 E 直接替代 M，也不能把各头向量先相加再取范数。

固定主基线为功能路由偏移：逐层 (sum M_history - sum M_source)/sum M_all，再平均层。
同一函数用 A 得到 attention 对照。官方来源区间由 prompt 结构及教学 tokenizer offsets
独立确定，标签不参与。普通 prompt 版本始终单独报告；不得把整个 prompt 冒名为事实证据。
self、严格过去历史、特殊 token 和其他 prompt 的贡献另存，以检查位置/控制词混杂。

来源—头—时间结构沿用 f7344e2：每头在 prompt 上归一得到 P，
N=exp(H(mean_head P))，rank=(tr(PPᵀ))²/||PPᵀ||²，
T=exp(H(当前及过去3行所有头的最强来源频数))；每层 log-volume=log N+log rank+log T。
Gram 在保留头轴后计算，不先把 attention 平均成一张图。它衡量路由冗余，不证明因果协同。

保留两种明确协议：历史 offline 校正使用整段长度/相对位置，仅作复现对照；
causal 校正只用已知 prompt 长度和当前绝对位置。两者均按 source 划分 fit/calibration/test，
只拟合路由量，不读自然标签；不够三个来源时显式缺测，不输出伪造的零分。
历史校正、普通 prompt/去特殊 token 的版本区别写入方法元数据，不继承旧 AUROC。

输出作用保持原 b/FFN/残差账本，并增加当前注意力峰窗口内的正、负、净写入，
从而对齐“读到了哪一片段”与“该片段支持了多少当前选择”。这些是解释字段，
不再通过未经验证的全局预算与风险传播替换路由主基线，也不把支持当前词等同于事实正确。
旧 support_graph/direct_prompt、熵、惊讶度和 observed negative margin 是固定对照。
不根据这四答的标签挑头、换符号、选择最佳方法或调整系数。

## 评价与停止条件

所有方法先冻结逐词分数，再读取同一 annotations；保留全错误、span onset、answer first、
continuation，并同时报告 pooled、equal-source 和同答 pair-weighted/macro AUROC。
报告同答前/后半段仅用于评估，不进入 causal 分数。首错逐条列出位置与各法排序。
输出高风险正常词列表，称为排序反例；没有校准阈值时不冒称 false positives。

原始缓存和 v1 分数文件只读；派生结果单独写入。只做针对性软件验证与可用缓存重算，
不运行全量大模型。新结构/作用指标只有在同样本、同权重比较中提供增量后才能升为主方法；
没有自然缓存时不声称恢复 .7，更不声称实现新 SOTA 或发现普遍幻觉机制。

## 文献边界

Information Flow Reveals When to Trust Language Models（作者代码：
https://github.com/rxu0112/RAG-information-flow ）提示分析内部信息流；不证明本项目每种读出都有效。
Attention Illuminates LLM Reasoning（https://arxiv.org/html/2510.13554v1 ）的 FAI 使用后续词，
不放入当前时刻检测；本实现的时间结构只用已经观察到的行。
