# 内部读取、残差整合与输出利用

用户2026-09-28纠正研究主线：先观察正确/错误生成的内部差异，不以来源删除或外部逻辑核验替代内部机制。新入口为 `python -m experiments.internal_flow.run`，复用path_conflict的原生A/V/W_O与残差观测；原算法/缓存不改。

## 测量与解释

7个已暴露局部声明、5个来源。原前缀下正确/错误候选共享同一个pretoken状态；不能人为产生两个“正确/错误前状态”。另测强制正确、错误、正确等价续写的末尾状态。后分叉状态含token身份/长度/位置差异，不是原生成者的自然因果机制。Llama3.1-8B是观察者，原始RAGTruth回答来自其他生成器。候选、证据位置与选头使用人工事实依据，只用于机制发现。

- 每层原始4096维残差：attention前、attention后、MLP后。
- 每个物理head和来源组：attention质量、AV范数、W_O投影消息范数、局部输出方向投影。
- 固定正确/错误候选的logit lens轨迹只是未校准的描述，不把跨层漂移等同知识变化。
- 保留重叠来源组；不能把source/evidence等组相加视为独立贡献。

来源子消息为 `m=W_O sum_j A[q,j] V[j]`。只改变一个head在原分叉query的来源分配：
`a'_S=(1-dose)*a_S + dose*mass(S)*(a_E/mass(E))`。
其他keys/heads不变，来源总质量固定；以等价的消息增量交回原生下游，并非删除来源文字。选点为绝对evidence局部读出最大的head，明确是同例探索选择，非已确认的因果选头。

剂量.25/.5、0剂量sham、同头其他来源、同层固定对照头、等写入范数随机方向。other_source与evidence的实际移动质量不同；control_head范数不匹配；random_direction是残差向量对照，没有真实attention搬移，CSV中的moved_mass仅为其生成参考量。不能将这些控制称为充分语义特异性检验。

分别评分正确与错误完整候选；续写阶段仍只在最初query注入一次，不反复强推每个词。报告两边logp及差值；不是自由生成修复。后续使用通路的更严格KV中介/反向诱导尚未实现。

## 运行

既有research Python，torch2.8.0+cu126 / transformers4.57.1；GPU0 RTX4090，bf16/eager。

```bash
python -m experiments.internal_flow.run --model /path/to/Meta-Llama-3.1-8B-Instruct --dataset /path/to/RAGTruth/dataset --output outputs/internal_flow_NEW --prepare-only
python -m experiments.internal_flow.run --model /path/to/Meta-Llama-3.1-8B-Instruct --dataset /path/to/RAGTruth/dataset --output outputs/internal_flow_NEW
python -m experiments.internal_flow.trace_keys --output outputs/internal_flow_NEW
python -m experiments.internal_flow.report --output outputs/internal_flow_NEW
```

同配置可从已完成case续跑；修改模型/候选/干预必须使用新目录。原已完成输出不改。单例失败不得跳过；同世界空干预误差>1e-5或消息重构相对误差>.01停止。

## 本轮实际证据

outputs/internal_flow_20260928：7例168次主前向实际exit0，随后7次逐key观测实际exit0；主计算51.17秒（不含加载），CUDA峰值约15.32GiB。35项相关测试通过。原生消息重构最大约0.00463，全部sham误差0。

时长L22H13对所列证据attention约0.91，其中days=.722656、four=.171875、more=.000877、than=.014282。intimate的L25H12对词片imate=.373047，对其后False仅.000033。注意V是上下文化向量：弱显式限定词attention不证明限定含义完全缺失；因果decoder的标识符V不能直接包含后面尚不可见的False，但目标Q/K与其他头可能间接利用它。

把来源质量移到evidence(.5剂量)的完整候选margin增量：WiFi+.3742、星期营业时间+.3738、hours+.1197、folding+.0102；pension-.0280、duration-.1063、intimate-.4494。4/7正向并非显著总体提升；其余为反向。WiFi/duration/folding原先偏错的完整候选margin均未翻正，未完成修复。2/7中正确/错误末层残差比正确/等价改写更近，整向量距离不可直接视为事实错误量。

12045两段未提供互斥且语法匹配的候选，本pilot未测；没有新检测AUROC、自然泛化或9span全检出。该限制不能在后续检测评价中转为剔除样本。详细中文结果/下一步在共享research的refine-logs/internal_flow_20260928。
