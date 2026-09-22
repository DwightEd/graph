# 先优化排序：原生路由的因果状态滤波

2026-09-22。目标是检验同样本 AUROC/AP 增益，不宣称已解释幻觉机制。
本协议在实现前固定；自然幻觉标签不用于拟合、选头、选窗口、改变方向或选择获胜方法。
不启动程序关系训练、反向传播、模型消融或全量前向。

## 已有依据

- 当前四回答同缓存：功能路由 AUROC 0.754983/AP 0.199373；旧净支持传播 0.572333/0.103939。
- 历史 head_geometry QA：pair_state 0.6447，因果滑动均值 0.7404；pair_full 0.7047，
  并未超过 pair_diagonal 0.7071。来源见 head_geometry/CROSS_TERMS_REVIEW_20260921.md。
- LDA/CHARM 的监督结果支持保留物理 head 身份；不证明无标签状态距离有效。
- 三个上传起点说明高置信错误与读取增强可同时出现，不能用熵筛掉检测位置。

推断：优先测试有效路由分数的时间降噪，比较按物理 head 状态筛选历史是否超过普通均值。
历史平滑收益不等于本候选已有收益，更不保证首错收益；不能将不同总体的数字拼成新成绩。

## 单一候选及公式

固定底分 R_t 为现有 routing_imbalance，不改分母或符号：

    R_t = mean_layer ((sum_M_history - sum_M_source) / sum_M_all)
    M = sqrt(edge_value_energy) = ||A W_O V||

无官方来源块时明确使用 prompt_routing_imbalance，不冒称事实证据。
对每个 layer/head，把 M 分到 source、严格过去历史、response self、other 四区。
每层使用全部头的消息总范数作共同分母，得到 head×region 联合分布 P_t[l,h,r]，
保留各头的相对消息强度。全零层在各头置 inactive=1/H；不是观测缺失。
官方来源首行的 predictor self 归 other；没有来源分区时沿用普通 prompt 底分的定义。
因此 R_t 也等于 mean_layer sum_head(P_history + P_self - P_source)，状态与底分预算一致。

    d²(t,s) = mean_layer (sum_head,region (sqrt(P_t)-sqrt(P_s))² / 2)
    b_t = max(median(d²(u,u-1), u=max(1,t-W+1)..t), 1e-8)
    w(t,s) ∝ exp(-d²(t,s)/b_t), s=max(0,t-W+1)..t
    S_t = sum_s w(t,s) R_s

W=16 沿用历史代码默认窗口；旧日志缺少 settings，不能认证旧运行实际使用该窗口。
1e-8 仅为零距离尺度的数值下界。t=0 只用本行。
带宽只由当前和已观察邻接状态决定；不读取整段长度、其他回答、未来 token 或标签。
状态相同时严格退化为普通因果均值；新状态距离大时旧权重降低，但不保证零检测延迟。
窗口并不等于错误 span；相似状态不证明同一语义、同一错误或功能协同。

## 固定对照与评价

只新增一个候选 route_state_filter：

1. 原功能路由，不可删除；仍是已有实测有效的基线。
2. route_mean：同窗口普通因果均值，排除收益只来自平滑。
3. route_pooled_filter：先在每层合并 head，再用同一滤波，检查保留头身份的增量。
4. 原 attention 路由、熵，分别作读取与置信度对照。

保存所有分数后再打开 annotations。共同 token 上报告全错误、首错、span 起点、延续、
前/后半段、来源等权和同答 AUROC/AP；自动输出相对底分和普通均值的差值，不自动选方法。
每个错误段结束后第一个正常 token 另存同答正常词排名，检查旧高分滞留；没有阈值不叫误报率。
四答、两首错仅作诊断；不把这四答的正差当作稳定提升或方法确认。

## 运行与保存

main.py support --stage optimize --output outputs/native_support_ragtruth4

从原生 NPZ 一次提取紧凑的逐头路由表；后续 optimize 只读小缓存。
不重算 collapse 的 Gram/跨来源校正，不重算旧 signed risk，也不载入模型权重。
原始 NPZ、v1、route_comparison_v2 均保留。新目录 route_filter_v3/w16；其他显式窗口单独保存。
默认 run/score 使用本次排序比较；compare 保留完整 v2 历史比较入口。
新候选没有自然成绩前，不替换文档中 0.754983 的实测基线，不宣称创新机制已成立。
