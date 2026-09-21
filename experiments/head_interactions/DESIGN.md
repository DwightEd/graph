# 头组条件作用审计

## 问题与结论边界

检验消息 A 的作用是否依赖消息 B、是否存在减弱正确候选的作用、以及删除上游后下游是否补偿。
单位是一个固定前缀、两个事先审核的候选答案、一组绝对位置及头/来源选择。
本实验不是无监督检测器；自然标签和人工来源角色只用于审计。高 attention 不等于正贡献，
负交互不自动等于抑制，删除后小效应不自动等于不重要。

## 固定协议

- 每个条件共享完全相同的 recipient prefix 与候选 token IDs。评分为两个候选的序列 logp 总和之差，
  同时保存逐 token 分数和长度归一化差；不把长度不同的候选假装成单 token 事实选择。
- 候选身份与 preferred 在审计前确定，同一 case 的所有干预使用相同候选顺序。
  同时保存固定候选方向与 supported−rival 方向，不按本例效果翻转分数。
- 在指定 query 删除 head_readout，或删除指定来源 attention 边；后者不重新归一化。
  来源质量使用原生 attention，排除 special tokens；不从压缩缓存补造 V 或隐藏状态。
- F11 原生；F01 删除 A；F10 删除 B；F00 同时删除 A/B。
  A_with_B=F11-F01；A_without_B=F10-F00；I=F11-F10-F01+F00。
- 下游恢复只用于严格跨层顺序：删除 A 后，将 B 的所选消息恢复成原生值，其他部分继续计算。
  多层 B 逐层重算并恢复，不能用未更新的静态增量假装恢复多个相互依赖的站点。
  记录自然重算与恢复结果；适应量是否缓冲删除影响需要看方向，不能只看向量变大。
- 每个 B 增益 1.5 的单组条件检验剂量响应；这是预注册扰动，不按错误样本寻找最有效增益。
- 可显式指定 donor case；对齐的是计划中的物理头与相对 query，来源角色在各自前缀中解析。
  保存 donor 语义类型。不同自然历史的 donor 只能作探索性替换，不能自动称为纯适用关系交换。
- 对照：删除强度 0；原生消息自替换；删除后逐层恢复；同层、同头数、同来源范围的随机头。
  随机头不宣称能量匹配。所有对照逐项保存是否通过，失败条件不进入有效结果汇总。

候选 0/1 固定为同一组值，preferred 单独记录支持哪一个。sum_margin 按支持方向定向；
candidate_sum_margin 始终为候选 0−1，跨条件的交互差只用后者，避免标签方向翻转的伪机制。
另提供两类程序构造题的条件交换、值交换、同义改写和双向 donor，对照自然配对的措辞混杂。

## 数据与统计

输入复用 teaching 的 answer.json，或把旧 paired contexts 转成同一个审计 case。
每个 case 保存 source、task、generator、history_status、候选文本、token IDs、实际干预 token。
旧 supported/unsupported 自然回答各自在自己的固定前缀中比较，不能把两边前缀当相同。
候选头来自已有探索，不作为新确认集；替换角色表不能由最大 attention 自动推断。

按 dataset/task/generator/panel/头组汇总；先对 source 内取均值，再按 source bootstrap。
缺少多来源时只给描述，不伪造置信区间。逐条件 margin 与逐头消息保存 NPZ/JSON，
报告保存四条件表、方向明确的恢复/增强结果和样本计数，不输出虚构 AUROC。

## 工程职责与实现顺序

1. 本设计及接口骨架。
2. teaching/capture.py：指定 Target 的轻量采集；继续复用 ModelAdapter.bind 和资源清理。
3. teaching/experiments/contrasts.py：固定前缀的候选对比；复用 score_targets。
4. teaching/experiments/messages.py：来源消息的读取、删除与恢复；复用 Target/Delete/Replace/ReplaceSource。
5. teaching/analysis/interactions.py：纯数值的四条件与补偿分解。
6. experiments/head_interactions：输入适配、实验条件、分来源报告和一键入口。

先写空接口再填充；交付不保留空实现。教学项目保持独立，不导入仓库 experiments。

## 验证门槛

软件测试：三种已支持模型、GQA、原生 logp 对齐、源消息删除等价、同状态恢复、
联合条件符号、跨层恢复、hook 清理、续跑与分来源统计。小模型结果只验证代码。
自然审计：需在用户 checkpoint 上运行；本地 CPU 不产生自然机制或检测成绩。

参考：[Copy Suppression](https://arxiv.org/abs/2310.04625)、
[Hydra Effect](https://arxiv.org/abs/2307.15771)、
[Activation Patching Best Practices](https://arxiv.org/abs/2309.16042)。
