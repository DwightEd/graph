# 方法演化与实现报告（2026-09-11）

## 最终方案

详见 [FINAL_PROPOSAL.md](FINAL_PROPOSAL.md)。当前主张被收紧为：在给定角色、来源单元、距离区间和自注意力分配后，候选状态与具体 attention 端点的对齐是否仍提供无监督异常信息。source selection、完整 enrichment/extraction 分类及因果 anchor 识别没有被证明。

## 从调研到方案

CHARM 支持图结构及激活共同建模的价值，但有监督成功不保证无监督迁移。TOHA、固定 graph scattering 和 attention rollout 均已存在，因此通用固定滤波器替代 AE 的创新叙事不足。仓库历史也提示高阶路径不天然有效，累计 rupture 可能退化为位置代理。

选择条件路径残差，因为其可证伪点更集中：固定条件质量后，只读取候选状态与边端点之间的关系。无需在冻结模型上重新做消融，N 是观测图内部的解析置换期望。

## 审查驱动的实质修改

1. 接受 role-only null 的距离／passage 混杂批评；改成 role × source unit × log-lag，并保留 self，singleton 不合并。
2. 明确定义候选来自预测位置的 top-K；用冻结 norm 和 unembedding 余弦差，不以真实响应补候选。
3. 明确 query 的预测位置和 source-through-history mask，排除 q 自身；承认二步量含一阶及交叉项，不称为纯 interaction。
4. 参考和测试按 source 隔离，条件匹配只用已知长度；四表征共享标准差／active mask，常数坐标不放大数值误差。
5. 完成 manifest、特征 digest、参考／测试完整流检查及最终 inactive reference 覆盖报告。

反驳并记录：初审声称 role-only null 不保留组内 cardinality 不准确，原定义已固定节点集合／数量。没有因为审查而引入额外神经训练、模型干预或把任务缩为可视化。

## 最终工程状态

默认 main 流程是 prepare → extract → detect → evaluate；旧 factorial/onset 有独立基线入口。代码通过真实随机 tiny Llama 的 float32/bfloat16 forward、未来不泄露、首 token、图算子构造性反例及端到端命令测试。32 项测试通过，详见 [核验记录](../docs/IMPLEMENTATION_REVIEW.md)。

## 剩余研究问题和停止条件

审查五轮到达配置上限，最终 scientific score 8.5/10、REVISE；实现一致性为 CONFORMANT。没有通过文字迭代伪造达到 9 分。只有自然 source-disjoint 结果能继续提升科学证据。

- residual 需在相同读出下超出 null/signal/observed 及 position/margin/entropy。
- 若强 null 下无增量，撤回端点条件对齐的检测贡献。
- 若二步不优于一阶，撤回中继贡献。
- 生成 token 覆盖不能替代正确答案覆盖；候选遗漏严重时不能声称区分富集与提取失败。
- 参考无变化／不足、source unit 分割、精确距离、probe/generator 不一致及稠密 attention 成本需要自然数据验证。
