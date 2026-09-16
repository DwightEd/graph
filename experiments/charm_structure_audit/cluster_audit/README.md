# 正确片段也聚集时，CHARM 还能区分吗？

本轮重点针对正在运行的 **charm_in**。不再围绕出度展开，也不重新训练它。
这是独立的 `cluster_audit` 增量；不覆盖 `deep_audit`、`learned_audit`、原模型、图或分数。

## 运行

在 graph 仓库根目录，使用原 research 环境：

```bash
git pull --ff-only origin main
python -m pytest tests/test_charm_cluster_audit.py -q
python -u -m experiments.charm_structure_audit.cluster_audit --root outputs/charm_structure_audit_qa
```

`--root` 可以是 `.../charm_in/test`，也可以是含多个任务/seed结果的父目录。
默认只选 `prediction_settings.json` 中 variant=charm_in 的模型。
需要定位迁移过的图时，增加 `--prepared /现在的prepared目录`（内含 `graphs/test/*.npz`）。
程序不会猜 checkpoint，不加载 LLM、tokenizer、torch，不产生新训练任务。

尚在运行的实验只能读取已正式保存的结果文件；预览用：

```bash
python -u -m experiments.charm_structure_audit.cluster_audit \
  --root outputs/charm_structure_audit_qa --completed-only --output-name cluster_preview
```

预览明确标为不完整子集。完整结束后用默认 `cluster_audit`，不要混用同一输出目录。
若训练尚未开始保存预测文件，没有可审计的样本时会直接报错，不制造结果。

## 为什么上一版不够

“同错误span内的词更相似”或“比附近正常词得分高”，不能排除模型只是在识别普通聚集结构。
本模块先在同一回答里找与错误span长度、位置和聚集结构相近的正常区间，再读取模型分数。
正常指**数据中没有幻觉标注**，不等于已经人工核验的独立正确命题。候选是等长评估区间，
不是模型自动发现的claim，更不是标注完整正确span的新数据集。

## 三层对照

- `context`：同回答、同长度、相近相对位置、相同首token表面类别、相近token-ID重复比例，
  且两段开始之前“是否已经出现标注错误”的状态相同。
- `cluster`：以上条件，再匹配段内连接密度、段内边比例/权重比例、lag=1比例、
  来源hub集中程度、平均边质量、平均入度。错误/正常段都必须有实际段内边。
- `cluster_heads`：以上条件，再匹配**每一个layer/head**的段均值：self-attention、
  保留入边总质量、保留权重条件熵。不把head平均掉，也不匹配模型输出/embedding。

第三层匹配的是逐头统计的**区间均值**，不是逐token逐head分布完全相同。
窗口之间的具体来源角色、跨head端点配合、邻居状态仍可能不同，留作模型区分的候选信息。
匹配描述符不是图同构，也没有控制完整词义/句法/事实角色。

候选不看分数、最终embedding或错误类型强弱。每层内正常区间不重复、不重叠使用；
按候选较少的错误span优先的确定性贪心匹配。它不保证全局最大配对数量。
找不到相近结构时记录缺匹配，不自动放宽标准或拿普通正常词补足。
单token错误没有可研究的内部聚集，在总体统计计数，但不硬造配对。

默认相对位置差≤0.25、重复比例差≤0.15；各结构阈值见 DESIGN.md。
配置在程序顶部固定，**不能看test效果调宽窄以制造结论**。

## 先看哪些结果

每个原预测目录下新增 `cluster_audit/`：

| 文件 | 直接回答的问题 |
|---|---|
| `summary.md` | 错误与同样聚集的正常片段，是否还能区分？ |
| `pairs.json` / `pairs.html` | 每一对具体是哪两个区间？图结构相近到什么程度？得分多少？ |
| `balance.csv` / `unmatched.csv` | 是否真的匹配上了？排除了哪些错误？ |
| `pair_scores.csv` | 配对内token AUROC、分数差、首词命中、正确聚集段误报 |
| `tokens.csv.gz` | **配对片段内**每个token的TP/FN/FP/TN；不是全数据token清单 |
| `head_differences.csv` | 哪些逐头读取统计在匹配后仍有差异；只是关联，不是模型重要性 |
| `representation_geometry.csv` | 错误段和正确段是不是都聚集？输入/投影/各层编码怎样变化？ |
| `report.json` | 原总体指标、配对覆盖、来源置信区间、同一配对上的原模型/控制比较 |

首要指标是**配对内token AUROC**、**正常聚集区间的token FPR**和错误token recall。
配对平均分胜率只作区间诊断，不能替代token定位；不把原token阈值用于阈值化span平均分。
正常片段any-alarm也有长度效应，因此每对等长，同时保留token FPR。

`pooled_matched_token` 的正负比例人为配成1:1，不是原测试集prevalence。
它的AP不能拿来直接宣称原全数据AUPRC提高。主要看配对内AUC和固定阈值两侧结果。
同一tier里错误和正常token都不会因重叠片段而重复计数。
原AUC和匹配AUC不是同一个估计对象；不把二者差值说成“聚集贡献了X%”。

置信区间以source为单位，先做source内平均再重采样；不把token/head当独立样本。
第三层可能只有少数难匹配例子。`common_error_cohort` 另在三层共同错误span上比较，
但正常对照仍不同，因此依旧不是因果贡献分解。

## 复用已有消融：一定评价同一批配对

原 `samples/*.npz` 中的 `score_*` 会自动加入。main 的
`deep_audit/captures/*.npz` 存在时，核对原分数回放和图文件记录后再读额外对照与各层表征。
自定义路径用 `--captures /实际/deep_audit/captures`（只允许单一预测目录）。
没有这些结果就明确记 `not_available`，本入口不会自动回放或重新训练模型。

每个控制只和原模型在**双方分数完整的同一批配对**上比较，输出control-minus-base；
prefix通常只有少数位置，完整span结果可能缺测。首词另用双方都可见的首位置比较，
不把首词结果冒充整个span结果。

已有独立训练的node_only/local_in/rewire_in等结果可直接接入：

```bash
python -u -m experiments.charm_structure_audit.cluster_audit \
  --root /实际/charm_in/test \
  --compare node_only=/实际/node_only/test \
  --output-name cluster_with_node_comparison
```

对齐回答ID/source/原文/标签/offset/span；原 `training.json` 存在时核对除variant外训练配方。
配方不一致或缺失，只标记预测比较，不称为已隔离的图增益。
独立模型使用各自在calibration保存的阈值；冻结干预沿用原阈值。

## 结论边界

在匹配后的正常聚集片段上仍大量误报，才是“可能只认聚集”的直接风险证据。
匹配后还能区分，只能说明存在超出**已匹配统计**的信号，不能直接说“读懂了真假”。
逐头差异不等于模型确实依赖该头。需结合相同配对上的coupled/independent重排、
邻居状态/边信息/中继控制与独立重训，并检查改动是否真正发生及其分布偏移。
当前模块复用这些已有实验，不把正常和错误一起降分误报成区分机制消失。

本次本地执行25项合成数据/CLI测试；没有运行真实RAGTruth样本、用户checkpoint或服务器实验。
