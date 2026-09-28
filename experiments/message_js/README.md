# JS 与逐头原生消息作用实验

第二轮原生残差/FFN 算子在下方说明。第一轮来源链与风险递推保留为对照。
已完成三轮实际试验，均未达检测目标；详细数字与失败片段见 [结果](RESULTS_20260928.md)。

最终检测无自然标签训练、无标签挑头、无标签阈值选型。历史标签只用于研究先验，
本轮评分和阈值全部保存后才评价。它仍是探索性回归实验，不能称未见测试确认。

在 graph 根目录，用现有 research Python 运行：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m experiments.message_js.run --output outputs/message_js_new_run
```

完整入口执行 84 答：旧 36 fit 来源作为无标签尺度参考、24 dev 来源作为无标签阈值参考、
8 个来源排除的已知回归回答，以及 4 题 × 4 种子的 16 条自然采样文本。
读取缓存和本地 Llama3.1-8B 模型；真实 GPU 前向，绝非复跑旧监督分类器。
默认路径在 capture.py。已经完成的阶段保留；中断留下的单答目录不自动覆盖。

## 测量

每条消息的门导数为

`g[l,h,t,j] = d(z_actual - z_best_other) / d gate[l,h,t,j]`。

gate 作用在 softmax 后，不重归一化。actual 和 best-other 自动从原回答及未干预输出确定，
不用人工候选。当前 query 后续的 attention、MLP、残差和归一化均原生求导；原历史 K/V 固定。
`attention.npy`、`derivative.npy` 都保留 float32 的 `[layer, head, predicted_token, absolute_key]`，
无窗口合并、层带合并或随机低秩投影。导数有符号值不丢弃。
这是完整物理轴上的**一个选择方向的导数**，不是全词表 Jacobian，也不是模型全部内部信息。

分别用 attention 和 `abs(g)/sum(abs(g))` 构建逐头来源链：

`F = D + B F`，即 `(I-B)F=D`，`R=B F`。

D 保留各 prompt token 的地址，B 仅连接严格过去 query；self 为终端。
query `P-1+t` 的历史 key `j` 对应状态行 `j-P+1`。
在完全一致的 prompt 根坐标上计算用户给出的质量加权 JS；没有 relay 的位置保留 NaN。
两条路径的 JS、attention 与作用幅度分布的等权 JS、prompt 正向作用缺口分别保存，
并保留 prompt/history 的正负总量。分歧、正负和强弱都不是事实真假标签。

来源链是同一物理头的 token DAG 代理，**不是实际跨层 WV/WO 的守恒来源分解**。
原生 gate 导数与这个额外的 ancestry 假设明确分开。
原生采样文本重新用 FP32 重放，不能称捕获时 BF16 状态逐位不变；其他生成器回答是 observer。

## 无监督读出

每个任务、每个物理头、每种量用 12 个无标签参考来源建立来源等权经验 CDF。
头内等权组合 influence JS、read/use JS 和 prompt positive deficit 的百分位。
缺失 relay 时只组合剩余可测分量，不把缺失当零分歧。

风险继承是另一项待检验假设：

`r_t = (1 - sum_j w_tj) u_t + sum_j w_tj r_(j-P)`，其中 j 是已生成的回答 key。

这里 `w` 为消息门导数绝对值归一化权重；生成 token 风险的 parent 是 `j-P`，
不同于上面来源链的 query 状态 parent。prompt 作用越强，当前观测的权重越高。
保留全部头的 u、r，最后输出各头 r 的 90% 分位。
历史边按固定 lag band 打乱作为控制；未传播组合独立保存。

阈值用 8 个 dev 来源的全部可用 token、来源等权的 95% 分位，严格大于才报警。
不筛正确 token，因此只是混合参考报警预算，**不是保证正常 FPR 为 5%**。
旧 raw_route、同窗口离线均值独立比较；额外固定等权 route + propagated 候选也报告，
不按结果自动选择。旧缓存仅访问指定无标签分数，不读 selected_detector、risk 或 target。

在本轮标签评价前追加一个固定数学修正对照：`relative_influence_js = JS / H_binary(alpha)`。
它消除两分支质量不平衡引起的 JS 上限变化，保存 alpha 原值；小分支可能被放大，
所以独立报告且不自动替换主候选。`relative_propagated` 只替换组合中的这个 JS 分量。
它仍不是真假概率；缺分支/零上限保持缺测。

## 评价

8 答用官方逐 token 标签：报告错误召回、正常误报、答内 AUROC、9 段任意/全部/首词覆盖，
完整位置见 tokens.csv。JS 缺测另报覆盖。事后全召回所需误报只作诊断，不作部署阈值。
16 条自然采样全部测量，但只有两题四个局部声明已有审阅标签；未知区域不改成正确。
软件测试、原生重放数值吻合不等于检测有效，实际结果见输出 evaluation.json。

## 第二轮：原生算子与完整输出分布

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m experiments.message_js.operator_run --base outputs/message_js_20260928_v1 --output outputs/message_operator_new_run
```

需要已完成的第一轮测量。入口捕获 48 答：8 回归、16 自然采样、各任务原无标签清单前
4 fit/4 dev 来源；随后无标签评分、冻结阈值、评价、生成逐词 HTML、独立数值复算。
已有完成标记的阶段直接复用。中断的未完成单答目录需要先人工归档，避免静默覆盖。

`operator_capture.py` 将同一消息的选择作用拆开：对 attention 后状态 u，
`h = u + MLP(N(u))`，有 `B_u = B_h + J_(MLP∘N)^T B_h`。
消息 m 的总作用等于 `m^T B_h + m^T (B_u - B_h)`，分别是残差直通和 FFN 变换。
两项以 `[32,32,T,absolute_key]` float32 保存，没有丢弃有符号值或先合并头。
符号仍是实际词对自动竞争词的 margin 方向，**不是事实支持/反对**。

`operator.py` 对所有 prompt 消息门、所有历史消息门、所有 MLP 写入门这三个指定方向
计算原生 JVP；每个方向保留完整 4096 维最终归一化状态响应和完整基准状态。
用全词表概率与完整 unembedding 计算精确 Fisher Gram `K=J^T F J`，不使用随机 rank-8。
这只覆盖三个指定干预方向，**不是完整消息门 Jacobian**；组内相消仍可能存在。
同一输出概率空间内：

`JS(p(g+εe_P), p(g+εe_H))/ε² → (K_PP+K_HH−2K_PH)/8`（nats）。

它与 prompt 地址上的 D/R JS 分开保存；有符号 Jacobian 不是概率转移矩阵。
单头完整导数和三组输出切向量也不可互称同一对象。

`operator_score.py` 以无标签 fit 熵三分位 × 相对位置三等分匹配上下文，
每个物理头的条件 CDF 用 `n/(n+32)` 向全局 CDF 收缩，考察三个预先指定模式：

- 读取弱且 prompt 对当前词的正向作用弱。
- 读取强但当前词正向作用弱，并出现 FFN 抵消或读取/作用地址错位。
- 路径地址分歧、历史 Fisher 占优且来源/历史门的输出作用不同。

组合公式见 `pattern_score`；保留全部头分数，token 分数取头的 90% 分位。
这是一组可证伪的机制先验，不是真假概率。额外报告去条件、旧 raw_route 融合及各分量。
所有方法用同一 4 dev 来源/任务的无标签混合 95% 分位；与第一轮 8 dev 的阈值范围不同。
没有自然标签拟合、挑头、选权重、挑阈值；已暴露错例仍只构成探索性回归检验。

## 历史形成的独立干预

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m experiments.message_js.history_trace --output outputs/message_history_new_run
```

该模块在 16 条自然采样上，无标签选取前半回答的最大 JS 事件、中层最强 prompt 消息，
将该消息门改为 1±.01，原生重算全部后续位置。保存每个未来位置完整状态变化与输出 JS。
测试检查过去状态不变、中层可影响未来、末层不能影响未来，以及当前位置有限差分符合原导数。
原回答文本保持固定，因此测量历史表征传播，不是自由续写或跨离散采样的导数。
它是机制审计，不自动把后续受到影响的 token 标成错误。

## 第三轮：直接检查有符号候选竞争

`signed_score.py` 复用第二轮完整 Gram 与三个门方向的实际词/自动竞争词 margin 响应。
主候选固定为 `-margin_P/sqrt(K_PP)`；另报告原幅度、Fisher 标准化的历史减来源方向。
不拟合参数，零能量保留缺测；同样用无标签 dev 混合分布校准，不按结果选择最优列。

```bash
python -m experiments.message_js.signed_score --output outputs/message_signed_new
python -m experiments.message_js.evaluate --output outputs/message_signed_new
python -m experiments.message_js.report --output outputs/message_signed_new
python -m experiments.message_js.score_verify --output outputs/message_signed_new
```

`score_verify` 独立重算官方标签下的排名指标与来源等权无标签阈值，引用第二轮已验证的原始测量。
第三轮算子仍不是事实判据，实际平均答内 AUROC .5087，未改善总体检测。
