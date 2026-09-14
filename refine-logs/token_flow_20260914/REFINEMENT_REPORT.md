# 方法收敛与发布说明

**工程状态：完成，可作为实验性基线发布。科学状态：REVISE，尚未验证自然幻觉定位有效性。**

## 问题锚点

在完整prompt＋response节点集合和可用实际响应连接上，无幻觉标签拟合、校准并输出token异常分数；仅在全部分数冻结后检验能否定位自然幻觉。必须避免自重构复制、未来拓扑泄漏和把图异常当事实错误；区分首次错误与后续延续。

## 结论

实现RoutingResidual：hidden节点经固定投影和参考集标准化；实际prompt/history邻居消息和位置/质量协变量预测当前节点状态；来源等权岭残差二次型为唯一主分数。score与evaluate分离，来源隔离和schema/hash冻结，正常数据标签不参与拟合。

当前query行已受当前token影响，所以不是历史innovation或因果信息流。diagonal只作显式工程sanity；其与行和的闭合不可用于语义解释。来源块校准只在可交换性下控制混合来源总体any-alarm，不保证正常条件FPR、FDR或幻觉后验。

主零模型是同维同质量mass_matched_uniform；graph−uniform首错midrank percentile配对区间上界须<0。支持性within-answer AUROC差区间下界须>0。无邻居与权重置换是辅助，不能挽救主门槛失败。

## 评审演变

| 轮次 | 加权分 | 判定 | 核心修正 |
|---|---:|---|---|
| 1 | 6.20 | RETHINK | query目标条件化、diagonal闭合、容量不匹配 |
| 2 | 6.85 | REVISE | hidden主输入、后验命名、uniform主对照；提出成对统计/schema要求 |
| 3 | 7.30 | REVISE | 主假设与主门槛一致、paired bootstrap、hidden schema已通过 |
| 4 | 7.30 | REVISE | 无新增代码阻塞，工程可发布，保留科学未验证边界 |
| 5 | 7.30 | REVISE | 最终报告与锚点复核通过；修正审查模型身份为请求配置而非已验证身份 |

全部评审使用同一Codex reviewer agent，首次独立上下文后连续复审；请求模型为gpt-5.6-sol，后端实际身份未独立验证。CALIBRATION:none；不是跨模型外部科学验收。

## 实际验证

全仓143测试actual exit0；新增主入口help正常。单真实训练CSR缓存976节点/1024维/84479边，1.02MB CSR，质量误差0，没有hidden，无标签评价。v2合成示例16参考+19校准+4测试来源，64测试token，四模型完整冻结后评价actual exit0。详见ENGINEERING_RESULTS.md。

## 未解决的研究证据

正式attention缓存缺hidden；自然hidden捕获、固定来源确认和真实graph-vs-uniform差异尚未执行。不报告新的自然AUROC，不声称方法科学成功或顶会READY。正式roster峰值RSS仍需测量，属性预算不是总内存保证。

## 文件与运行

- 方法：docs/TOKEN_GRAPH_INFORMATION_FLOW.md；本目录FINAL_PROPOSAL.md为最终规格副本。
- 算法：experiments/unsupervised_token_graph/information_flow.py。
- 生命周期：flow_run.py；标签后接评价：flow_evaluate.py。
- 主入口：python main.py token-graph {score,evaluate}。
- 完整例子：python -m examples.token_flow_demo --output outputs/token_flow_demo_new。
- 原有s10_reanchor.py未修改，不混入本轮提交。远端已有reanchor重构先快进继承，再完成用户授权的main合并与push。

达到5轮精炼上限后结束。本轮完成工程候选，不把自然证据缺失改写为READY；不存在剩余代码发布阻塞。
