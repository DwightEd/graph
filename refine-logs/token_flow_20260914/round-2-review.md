## 第二轮独立方法复审：RoutingResidual

### 问题锚点核对

原问题锚点得到保留：任务仍是在完整 prompt＋response 图上，对已经观察到的 response token 产生无标签异常排序，并在分数冻结后检验能否定位幻觉。修订没有把任务偷换为生成前预警，也没有把 attention 解释成真实因果消息。

第一轮的两个理论阻塞已被实质解决：

1. 主方法已改名为 `RoutingResidual`，明确承认 query \(i\) 依赖当前 token，因此它是 post-observation、target-conditioned residual，而非 \(\mathcal F_{i-1}\)-measurable innovation。
2. 研究默认属性改为完整 frozen hidden，缺失即失败；attention diagonal 只能通过显式参数作为工程 sanity，且文档写明 softmax 行和闭合使其不能支持信息流解释。

这两项修改使方案从“中心统计对象解释错误”变为“统计对象明确、效果尚待验证”。`mass_matched_uniform` 也已在代码中保持与 graph 模式相同的 \(2d\) 消息维度和岭拟合规则；`no_neighbors` 被准确降级为删除对照。方向已不需要 RETHINK，但离 READY 仍有可落地的规格/评价闭环和自然证据缺口。

### 七维评分

| 维度 | 权重 | 评分 | 加权 |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 9 | 1.35 |
| Method Specificity | 25% | 8 | 2.00 |
| Contribution Quality | 25% | 6 | 1.50 |
| Frontier Leverage | 15% | 6 | 0.90 |
| Feasibility | 10% | 6 | 0.60 |
| Validation Focus | 5% | 6 | 0.30 |
| Venue Readiness | 5% | 4 | 0.20 |
| **加权总分** | **100%** |  | **6.85 / 10** |

**CALIBRATION: none**

**GAP：**没有人工提供的 known-good/known-bad 候选方案，故仍不能相对外部 exemplar 校准。相对第一轮自身，方案已从 6.20 提升到 6.85：Problem Fidelity 与 Method Specificity 的提升来自后验时间语义、hidden 主属性和同维零模型已经写进代码，而非文字回避。离 READY 的差距现在主要位于 Contribution Quality、Validation Focus 和 Venue Readiness：方法仍是经典岭条件残差，真正可能成为论文贡献的是“实际路由选择相对同质量均匀历史是否产生回答内首错增量”，但当前删除规则尚未把该对照设为硬门槛，评价实现也尚未对 graph-control 的来源配对差异给出区间；自然 hidden 结果仍不存在。

### 已解决事项

1. **时间语义与命名一致。** 文档、类 docstring、冻结清单都明确写出 `post-observation physical token query i`，没有继续暗示提前预测或严格创新。
2. **diagonal 闭合被隔离。** CLI 默认 `--attribute hidden`；hidden 缺失抛错；diagonal 需要显式选择。现有无 hidden 的真实 cache 只被报告为工程兼容性检查，没有被包装成自然有效性。
3. **主零模型已具备同维结构。** `mass_matched_uniform` 保留真实 prompt/history 总质量，分别均匀聚合全部严格历史节点，graph 与 uniform 的设计矩阵列数相同；测试也核对了相同 context 与不同 message。
4. **标签冻结和身份绑定基本成立。** fit/calibration/test source 不交叉，评分产物和代码哈希在标签读取前冻结；评价要求 annotation token IDs 与 frozen observer response token IDs 完全相同，缺标注直接失败。
5. **有限样本校准实现与公式一致。** 并列使用 \(M_b\ge a_i\) 的保守方向；\(B<19\) 时 5% 阈值为无穷；source 的多回答最大值按 source ID 合并。
6. **工程证据边界报告准确。** 22 项聚焦测试在当前复审环境通过；补充报告的全仓库测试、单真实 CSR cache 和 39 来源合成演示只支持工程正确性与流程可运行，不支持自然幻觉效果。该边界没有被越过。

### 仍需解决的核心问题

1. **规格中的核心删除门槛仍写错对象。**

   文档已经说明“真实图与 `mass_matched_uniform` 比较才检验连接选择的增量”，但问题锚点的成功条件仍写“同容量无邻居”，最小验证的删除判据仍是“未胜 no_neighbors/置换”。这与修订后的贡献定义不一致：graph 即使不胜 uniform，只要胜 no-neighbors，当前文字仍可能保留“图改善检测”的主张。

   必须把唯一主门槛改成：在预注册的回答内首错指标上，graph 必须胜 `mass_matched_uniform`。`no_neighbors` 只检验历史消息整体是否有用；`permuted_weights` 只在权重非退化时作为连接分配诊断。原锚点若必须逐字保留，应紧接一条 operational clarification，说明其中“同容量无邻居”由 `mass_matched_uniform` 实现，`no_neighbors` 不承担同容量含义。

2. **评价已有每方法 bootstrap，但缺少主张所需的配对差异推断。**

   新版 `flow_evaluate.py` 已加入首错 midrank/reciprocal-midrank、各 subset 重新来源→回答→token 等权及 source bootstrap，这是有效修正。但 bootstrap 目前分别对每个方法的 source-macro AUROC 取区间。分别的置信区间不能检验 graph 是否胜 uniform，也没有给首错定位差异做区间。

   应使用同一批 source 重采样，直接输出至少两项配对差：

   - \(\Delta_{\mathrm{onset}}=R_{\mathrm{graph}}-R_{\mathrm{uniform}}\)，其中 \(R\) 是每来源宏平均的首错 midrank percentile，负值更好；
   - \(\Delta_{\mathrm{AUROC}}=\mathrm{AUROC}_{\mathrm{graph}}-\mathrm{AUROC}_{\mathrm{uniform}}\)，使用预定的 through-first-error 或 within-answer source-macro 视图。

   同一 resample 同时计算两个方法，报告点估计和 95% 区间。no-neighbors/permuted 可作为次要配对结果。不要用“两个独立区间是否重叠”代替配对检验。

3. **区分假设仍过弱，需绑定预注册指标。**

   当前“某个 \(s\) 区间满足 \(P(a>s\mid H,C)>P(a>s\mid N,C)\)”允许事后挑阈值，而且某一小区间成立并不推出稳定排序或首错定位。

   建议直接以任务读出定义假设：在新来源和预定条件分层下，首错 token 在从回答开始至首错的候选集合中具有优于随机的 rank，且 graph 相对 mass-matched uniform 的来源宏平均 midrank 更低；或等价预注册一个条件 concordance/AUROC 大于 0.5 的命题。这样“可识别性假设—评价指标—删除门槛”三者一致。

4. **hidden 的 observer schema 尚未在代码中绑定。**

   `RoutingResidual.fit` 目前只检查属性名都是 `hidden` 且维度相同。来自不同模型、checkpoint、hidden layer、归一化位置或 tokenizer 的缓存，只要维数相同就会被静默混合。输入文件哈希能证明文件未变，却不能证明这些 hidden 表示属于同一统计坐标系。文档声称模型和配置冻结，但 `inputs.json` 没有显式保存或校验该 schema。

   roster 或 cache manifest 应提供稳定的 `observer_schema_id`，至少绑定 model/revision、tokenizer、hidden layer/位置、attention 层头定义和缓存版本；reference/calibration/test 必须一致，否则失败。若正式 roster 已由外部 manifest 保证，则把该 manifest 的 hash 和 schema 摘要写入 freeze，而不是只依赖约定。

5. **正式 hidden roster 的 CPU 内存可行性尚未闭合。**

   当前 `score_roster` 会先把 reference/calibration/test 的所有 graph 和原始 hidden 一次性载入内存，`build_flow_graph` 又把 hidden 转成默认 `float64`；fit 随后同时构造投影 target、全部 design 和拼接矩阵。单个无-hidden CSR cache 的 1 MB 级稀疏规模以及 39×64 合成演示没有覆盖这个瓶颈。

   本轮可用两种方式之一闭合：

   - 在正式 roster 冻结前给出按 \(N,D\) 计算的峰值内存估计并设置硬上限，证明目标 roster 可由当前实现运行；或
   - hidden 载入后立即以 float32 投影至固定 64 维并释放原数组，reference 用流式充分统计量拟合，calibration/test 按响应评分，不同时保留所有高维 graph。

   这是工程可行性问题，不要求改变方法。

6. **若要声称精确定位，首错统计还应形成明确的主比较表。**

   当前实现已有 `first_error_localization`，但字段 `midrank_percentile` 是“越低越好”，应在输出或文档中明确；同时需要报告 eligible sources/answers 和 ties。`within_answer_source_macro_auroc` 目前基于完整回答的全部错误 token，并不等于首错定位。论文主读出应明确选用 first-error midrank 或 through-first-error AUROC 之一，另一个作为辅助，避免结果出来后选择更好看的视图。

### 低于 7 分维度的具体修正

- **Contribution Quality — 6/10，IMPORTANT**

  弱点：中心机制已经诚实且单一，但岭残差本身不新；贡献是否成立完全取决于真实 attention 连接是否稳定胜过同质量均匀历史，而当前硬删除规则尚未锁住这一点。

  修正：把 graph-vs-mass-matched-uniform 的回答内首错配对差设为唯一主机制检验。成功时贡献是“目标条件化实际路由选择提供增量”，失败时删除图贡献，不再用 no-neighbors 的较弱胜利挽救。

- **Frontier Leverage — 6/10，IMPORTANT**

  弱点：不堆新模型是正确选择，但文献边界仍停在 2025，尚未纳入 ACL 2026 的 attention topology/divergence 邻近工作。当前 novelty 不能仅依赖“attention 转图”。

  修正：加入 TOHA 和 attention divergence 的定位，明确它们与本方案在粒度、监督、统计对象和标签冻结方面的差异。无需实现新模块；若其公开特征可在同一缓存上无标签计算，可作为次要基线，否则只做设定清晰的文献比较。

- **Feasibility — 6/10，IMPORTANT**

  弱点：hidden 是研究必要输入，但现有正式 cache 没有 hidden；新捕获成本尚未知，现实现还可能因 float64 全量载入在正式 roster 上超内存。

  修正：冻结 hidden schema、来源规模、总节点数和峰值内存预算；在运行自然实验前完成小规模真实 hidden cache 的端到端 smoke。当前无 hidden 的真实 cache 继续只算工程证据。

- **Validation Focus — 6/10，CRITICAL**

  弱点：验证菜单已经精简，但主零模型没有进入硬删除规则，bootstrap 也没有估计 graph-control 的配对差；这两点会使自然结果无法直接裁决核心 claim。

  修正：预注册一个首错主指标、一个 graph-uniform 主比较、同 source 配对 bootstrap，以及失败即删除的方向。其余 all-token、strict-post-first、entropy/NLL 和 permutation 都作为辅助诊断。

- **Venue Readiness — 4/10，CRITICAL**

  弱点：没有任何 frozen natural hidden 结果，因此尚不知道图区分假设是否成立；经典残差方法也尚无结果层面的新发现。工程测试再多也不能补足这一项。

  修正：完成上述规格和评价闭环后，运行一次预固定的 source-disjoint hidden roster。只有 graph 在首错回答内指标上胜 uniform 且来源配对区间支持正向增量，才进入论文贡献判断；否则保留严谨负结果并回退为基线/审计工具。

### Simplification Opportunities

1. 将主结果收敛为一条：`RoutingResidual(graph)` 对 `RoutingResidual(mass_matched_uniform)` 的首错回答内配对差。no-neighbors 和 permutation 不再共同充当硬门槛。
2. `coding_nats_per_dimension`、routing entropy 和 Dirichlet energy 可以继续落盘作诊断，但不进入核心贡献或模型选择；主分数始终只有 residual energy。
3. 在文档中只指定一个首错主指标和一个辅助指标，避免 all-token、through-first、within-answer 等多个视图产生事后选择空间。

### Modernization Opportunities

1. **不增加新的 trainable component。** 当前 frozen hidden＋线性残差已经符合最小机制原则。
2. 用 observer schema fingerprint 和成对来源统计提升现代基础模型实验的可复现性，比添加 critic/LLM judge 更契合当前瓶颈。
3. 将 2026 attention topology/divergence 工作纳入最近邻定位；本方法的差异应落在 post-observation token 粒度、无标签拟合、严格冻结和 mass-matched falsification，而不是“首次使用图”。

### Drift Warning

**NONE。** 修订明确拒绝将问题改成严格前缀预警，并保留看到 token 后的定位任务。`RoutingResidual` 与代码时序一致；只要后续不把异常分数写成幻觉概率、事实似然比或因果信息流，问题锚点保持稳定。

### Verdict

**REVISE**

第一轮要求重想的中心对象已经修正，因而不再是 RETHINK。当前剩余问题都可以通过合理的本轮规格/代码修订解决：统一成功条件与 mass-matched hard gate、加入 graph-control 来源配对差、把可识别性假设绑定首错指标、校验 observer schema，并给正式 hidden roster 一个可执行的内存方案。

自然有效性仍完全未知，因此不能 READY。真实 CSR cache、22 项聚焦测试、全仓库测试和 39 来源合成演示共同证明的是工程路线可运行且主要泄漏边界得到约束；它们没有证明 RoutingResidual 能定位自然幻觉。完成上述闭环后，应冻结一次自然 hidden 评价，让结果决定该工程基线能否发展为研究贡献。
