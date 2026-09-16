# CHARM 高 AUROC 从哪里来：深入审计

本目录接在 main 的 `6136f816`（上一会话的 `learned_audit`）之后，并继续 `charm_structure_audit/diagnose_saved.py` 的工作，**不替换检测器，不改原 checkpoint、预测、标签或阈值**。先把高分拆清楚，再检查模型用了什么。新增代码仅放在本目录和 `tests/test_charm_deep_audit.py`。

上一会话的 `learned_audit` 已实现层内/层间头配对、真实消息MLP中的双头交互删除、邻居状态替换和原logit解释。本轮**保留它的全部代码**。存在 `test/learned_audit/samples/` 时，`learned.py`自动把已有控制并入本轮token/回答/span审计，核对原分数、标签、offset、身份与原文，不重新运行这些干预；缺文件明确记覆盖不足。自定义位置可对单一预测目录使用`--learned-root`。

两者用途不同：上一会话的小回归器解释“原模型为什么给这个分”；本轮线性读出检验“输入/图编码后还有多少错误与正常可分信息”。两者都不是新提交的无监督检测器。


## 一键运行

在 graph 仓库根目录、原 research 环境运行：

```bash
python -m experiments.charm_structure_audit.deep_audit --root outputs/charm_structure_audit_qa --stage all
```

`--root` 可填某个具体的 `.../charm_in/test` / `.../external_checkpoint`，也可以是这些结果的共同父目录。它按实际存在的 `prediction_settings.json` 找模型，不猜高分对应哪次运行。上面的路径是原实验默认路径；使用自定义 OUTPUT 的实验应传该目录。

默认前台执行，逐回答显示进度；复用已存图与分数，不加载 8B 模型、不重新提取 attention、不重新训练原 GNN。`all` 会额外训练小型**有监督线性诊断读出**；只在实际的 fit 来源上拟合，再评价原 test 来源。没有原 `training.json` 时明确跳过读出，不临时随机拆 token。

先只看统计、逐词命中和已有对照，不运行神经网络：

```bash
python -m experiments.charm_structure_audit.deep_audit --root outputs/charm_structure_audit_qa --stage scores
```

只回放冻结 GNN、不拟合线性读出：将 `--stage` 改成 `graph`。`--bootstrap 0` 关闭来源重采样区间，`--repeats 1` 做一次随机对照；默认分别为100和3。`--completed-only` 允许明确标为未完成子集的预览，不把它冒充全量，也不和完整重训结果比较。

搬过目录时使用 `--prepared /现在的prepared目录`；搬过模型时对**单一预测目录**传 `--checkpoint /原模型的新路径`。保存分数的旧绝对路径自动按 samples 下的文件名定位；不会用新模型重训练来修复回放不一致。

每个预测目录生成独立 `deep_audit/`。最先打开 `summary.md`，再看 `saved_scores/tokens.html` 或 `frozen_scores/tokens.html`。原文件不改写。

## 问题、设计、代码、输出一一对应

| 问题 | 实际做什么 | 代码 | 主要输出 |
|---|---|---|---|
| 正负样本与首错分母是多少？ | 统计实际评价集；另读prepared索引统计官方train/test、任务、生成器；不把缓存子集当全RAGTruth | `scores.population`、`run.prepared_inventory` | `dataset_inventory.json`、`scores.json` |
| 高AUROC由首错还是续写贡献？ | 每个回答首错、后续标注起点、片段延续三类正例；使用相同正常负例精确拆分排序对 | `scores.auc_accounting` | `auc_accounting.cells` |
| 只是整篇回答分高低吗？ | 分解同回答/跨回答配对；另报回答内宏平均与配对数加权AUROC | `scores.within_answers` | `answer_rankings.csv` |
| 具体哪些token报对？ | 固定原阈值，逐token保存TP/FN/FP/TN、原文、位置、角色、片段、排序贡献、所有冻结对照分数 | `scores.token_ledger` | `tokens.csv`、`tokens.html` |
| 是不是只会识别一片高分区域？ | span覆盖、首词漏检/延迟、段内分数差异；和同答全部/附近正常词比较；单独检查两错误段之间正常词 | `scores.span_comparisons` | `spans.csv`、`normal_gaps.csv` |
| 同span的相似性是否只是距离或重复造成？ | 同一回答、完全相同token距离、是否有边、是否同token ID下，比较同错误span与不同错误span；同时导出正常-正常和混合对 | `representations.relation_rows`、`matched_relation_contrasts` | `relation_pairs.csv`、`matched_span_relations.csv`、`representation_summary.json` |
| 学到单头强弱，还是多头共同指向？ | 对每个query/来源角色/距离组保留每头的全部权重；共同置换整个多头向量 vs 每头独立置换 | `controls.shuffle_endpoints` | `paired_controls.json`、`intervention_integrity.json` |
| 是固定head身份有用吗？ | 每层固定置换head编号；联合置换节点和边，另做只置换边 | `controls.permute_head_identity` | `score_head_identity_*` |
| 真正的端点连接有用吗？ | 因果RR双边交换；同时保持入度、出度、RP边、目标边向量和粗距离带；报告真正改动多少边 | `controls.degree_preserving_rewire` | `score_degree_rewire_*`、改边比例 |
| 是边属性，还是邻居状态/多跳信息有用？ | 不接收消息；消息中邻居状态清零但保留边属性；禁止邻居携带继承自更早邻居的信息 | `capture.capture_controls`、`representations.trace_model` | `score_no_messages`、`score_edge_only_messages`、`score_no_relay` |
| 错误span内的边是否专门抬高区域分数？ | 金标定义段内RR删边；和同目标、同距离带、同数量随机删边比较；报告两种删除重合多少 | `controls.oracle_span_cut` | `oracle_minus_matched_random_*` |
| 分数是否用了后续结构？ | 保留已存prefix分数；缺失时补均匀位置+标注起点的前缀回放，重新计算degree；只在共同位置比较 | `capture.missing_prefix_control`、`scores.control_comparison` | `score_prefix`及其覆盖数 |
| 图编码到底增加了多少可分类信息？ | 冻结输入投影和每层GNN状态，用相同线性读出规则比较；所有尺度拟合只用fit集 | `probes.py` | `probes.json`、`probe_coefficients.csv` |
| 图是否确实提高泛化成绩？ | 自动对齐已存在的独立重训兄弟实验；同recipe/seed/来源划分/逐token对齐才比较 | `run.compare_retrained` | `independently_retrained.json` |

## 先理解三个统计问题

**首错不是每个span起点。** 一个有错误的回答只有一个回答级首错，但可以有多个标注起点。后续起点与片段延续分开统计。原始标注可以在token空间重叠：统计原始标注数量，同时合并重叠来定义片段成员；相邻但不重叠的片段仍保留。连续的二值标签段又是另一统计单位。

**AUROC不是“检测对了多少token”。** 对某个错误token，计算它的分数超过多少正常token，打平算半个。设错误token数P、正常token数N：

```
credit(t) = [#正常分数小于s(t) + 0.5 * #正常分数等于s(t)] / N
AUROC = 所有错误token的credit平均值
每token的超随机排序贡献 = (credit(t) - 0.5) / P
```

按“正例角色×正常上下文×同答/跨答”分组，这些互不重叠配对单元精确还原原AUROC。不是事后换阈值，不是因果贡献比例；AP不能按这个方式相加。代码的AP使用`average_precision_score`，不混称梯形积分版PR曲线面积。

**错误span内没有正常负例。** 因此不能在它自身计算错误/正常AUROC。段内看分数标准差、范围、命中覆盖、首词和后续分数、检测延迟；定位能力要拿附近正常词和段间正常间隙来检验。对两个错误span的比较是“同为错误的token为什么在本span更相似”，不是把另一错误span硬充负例。

只含一种标签的回答不进入回答内AUROC平均，分母明确保存；它们仍参与总体评价。附近没有正常词、没有同距离匹配、对照分数缺失时输出缺测，不填0.5或0。字节/子词可能有重复offset，逐token表不擅自合并；另报有实际文本长度的token指标。

## 多头检验为什么不只是清零实验

把一条边看成向量`[head0权重, head1权重, ...]`。共同置换是把整个向量移到另一邻居；独立置换是每个头各选置换。二者都限定在同一个目标、同一prompt/history角色、同一位置/距离组。

每个头在组里的非零数量、权重列表、总质量及保留权重条件熵不变；共同置换还保持头之间在同一端点的组合，独立置换则破坏它。比较**独立减共同**，而不只看独立减原图。额外的`head_marginals`探针只看逐头总质量和条件熵，`head_overlap`看每层头之间共同指向的概括量。

这仍是冻结输入干预：不是自然语言反事实，也不自动证明学习了语义事实关系。独立置换可能出现全零边向量，但为了不混入度数变化，仍保留原union图；全零边数明确报告。两种置换改变的单元数量也可能不同，需结合`changed_cells`看。共同置换本身改变“头向量与具体邻居状态的配合”，不能把其下降全部叫多头协同。

`edge_only_messages`进一步去掉消息里的源节点状态。如果仅看边向量就接近原分数，不能把效果归给多跳图推理。固定模型的清零存在分布偏移；独立训练对照是另一层证据。

## 图编码和区域聚集怎么看

`node`是输入的逐头attention对角线；`projected`是原in_proj之后、还没接收邻居信息的状态；`layer_1...`是每次消息更新后的状态。`local_causal`是入边数、局部连接比例、保留熵等简单统计；`local_with_future_degree`故意额外加入未来出度，作为排查未来结构捷径的诊断对照。所有探针是有监督诊断，不称无监督检测，不在test挑最好的头或反转评分方向。

`no_relay`保留每个源节点自身逐层MLP更新，但发送时不让它带上从别的节点接收到的信息。因此一层GNN时应与原模型一致；没有边时也应一致。测试覆盖这两项，避免把“禁止多跳”错误实现成“取消所有深层节点变换”。

同span和不同错误span的比较匹配回答、距离、有无边以及token ID是否相同。短span/远隔span可能没有共同距离，此时明确缺匹配，不拉远处对照凑结论。token ID匹配不是完整词、同义词或语义控制；高相似性仍不是事实依赖的证明。

独立重训比较只读现有结果。没有node_only/local_in/rewire_in的同配方结果就写不可用，**不会偷偷新训练一个检测器**。不同模型二值指标使用各自在独立calibration集保存的阈值；原阈值应用到另一模型只标为敏感性比较。优先看charm_in与其他变体；charm_out自身可能用到未来出度。

## 有哪些不能由现有缓存回答

准备图丢弃了阈值以下attention，不能恢复完整attention熵，更没有词表预测熵。`retained_entropy`只是在已保存边和对角线上归一化的条件熵；不能据此断言“首错就是完整熵检测”。本轮不重新提取LLM状态，不用补零假装完整数据。

原模型是post-token检测；前缀裁剪也包含当前token，不是生成前预警。前缀默认只查已披露的子集，包含金标起点，只能用于后验诊断。源码中的未来出度依赖可由代数和测试确认；它究竟解释多少真实AUROC必须看用户原预测的回放差异。

标签只在统计、配对、金标删边和明确标记的监督探针中使用，不进入label-blind图变换。所有统计关联和冻结干预均针对**检测器**，不是对LLM真实生成原因的证明。

## 文件与阅读顺序

`run.py`先读结果、写统计，再回放冻结模型，最后做可用的线性读出。纯统计看`common.py → scores.py`；图干预看`controls.py → representations.py → capture.py`；读出看`probes.py`。

`summary.md`写中文解释及本次实际数字。`saved_scores/`保留原结果的只读拆解；`frozen_scores/`包含新增对照。`captures/`每回答保存新增分数与各层特征，`fit_features/`保存拟合来源特征；一次只加载一张图，但单张图本身和新增特征仍占CPU内存与磁盘。

复用capture前检查原checkpoint、配置、图文件和原分数。回放误差超过默认2e-5会停止归因，不重拟合阈值，不覆盖旧分数。报告是可重算的，来源bootstrap不是训练seed方差；逐头/逐层探索没有多重比较校正。

测试：

```bash
python -m pytest tests/test_charm_deep_audit.py -q
```

依赖沿用已有numpy、scipy、scikit-learn、tqdm、pytest；图阶段另用torch。不要为了本审计升级服务器的CUDA/PyTorch。

## 本次实现验证

在CPU上23项合成测试通过，包含完整`--stage all`命令、实际CHARM checkpoint回放、线性读出、缓存复用、原文件字节与mtime不变，以及置换/度数/统计恒等式。完整熵缺失、历史fit成员缺失都不会被虚构。

没有用户实际高分checkpoint或自然数据预测在本地，**没有运行真实RAGTruth审计，没有新的真实AUROC，也尚不能判定高分主要来源**。软件验证和机制结论分开记录。

## 已有QA结果的对照起点（不是本轮重算）

上一会话提交的 `../learned_audit/BASELINE_ANALYSIS.md @6136f816` 摘录了用户贴出的结果：
30,619个token，2,307个标错；AUROC 0.884927、AP 0.538455；回答首错52个只命中1个；非起点错误2,223个，命中861个。
85个标注span仅1个全覆盖，35个完全没报。因而不能把当前表现概括成“漏首错但后面整段都对”。
旧RR删边比旧同距离重连影响大，但前者不是公平重训收益，后者没有破坏多头向量内部配对。
本轮以实际文件重新核对分母与这些关系，不把这份文字摘录当逐token原始数据。
