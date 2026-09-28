# 原生上下文补全响应与逐头稀疏检测

`measure.py` 复用完整source Q/K、回答source attention和message-gate梯度：每个物理头构造来源严格向前关联 B 与行归一化 Bᵀ，以 `E = normalize(D B)`、`deltaA = M E - A` 保持来源总读取量。当前原词相对固定最强竞争词的响应为 `sum((g/A) * deltaA)`。反向邻接不是原生因果流；响应符号也不是真假标签。

所有32层×32头分别保存两向响应、来源身份打乱对照、相对响应、特异响应、两向JS及source质量/绝对门效应。`finite.py` 在原始past KV下，对单头作0/.01/.05/.25剂量凸组合并重算下游。实际560个重放点：0.01剂量140项符号一致，中位相对误差0.2%；0.25剂量4项变号。没有有限重放sham，打乱图仅在一阶测量中完成。

## 三轮曝光样本实验

- v1：有符号抑制逐头fitCDF再q90；条件距离和固定75/25融合。直接抑制主候选1/134错误、97/1353误报，失败。
- v2：不按符号判错；每头signed-log响应与JS按无标签fit median/IQR标准化，双侧异常，同头几何均值，token内top8头读出。所有头参与，无标签挑固定头。纯sparse_joint为7/134、80/1353；固定融合63/134、74/1353，答内AUC .7364（同口径基线20/71/.7377）。召回增加不等于整体排序提高。
- v3：保持v2头读出，按entropy/surprisal/位置/prompt长度/词项复现/表面类别匹配每来源16个fit位置，校准条件秩。原阈值版本与结果独立保留。

fit/dev每任务4来源，各来源等权；混合dev95分位不保证正常FPR为5%。曝光回归8答、自然16答；自然仅31个局部token有人审标签，且缺旧source_pair，融合方法在那里不可评分，不能把缺分称为零误报。标签用于冻结后评价与机制发现，不拟合检测器。

## 实际完整三任务入口

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
-m experiments.context_response.full --output outputs/context_response_full_new --stage all
```

全2700答/450来源/424408原token；模型来自pilot manifest。默认复用本项目本轮pilot v2的**无标签拟合参考**，因此须保留相应输出。分阶段为 `freeze`、`capture`、`evaluate`；`resume` 跳过已评分记录并在全部采集后自动评价。中断后 `capture` 跳过已经有scores的回答，若当前答只存在部分目录，先保留该目录并使用新的目录位置，不能冒充完成。

全量重新执行观察者原生固定past replay；全部1024头保留。`stream.py`仅把dense source-key A/g临时放RAM，磁盘保存全头12字段与原token IDs，而非完整测试集所有key图。GPU按key-query乘积自动分批，不删除query。3回答21score数组已与原实现核对，报警一致；原始响应/JS坐标相对误差<3.3e-6。除极小source效应的相对响应数值较不稳定，未被全量主方法使用。

主方法`sparse_joint_fused`，6个对照包括同参考source_route_fixed、response/address/联合异常及分量融合。所有分数冻结后才评价官方token标签，报告AUROC/AP、答内/单元内AUROC、token召回/FPR、首错召回、正常回答any-alarm及来源簇bootstrap。已有测试总体此前暴露，属于探索复验。不能用较小参考集的基线成绩冒充此前全训练参考QA .890665；两者应分报。

## 原始产物和核验

- `outputs/context_response_20260928_v1`：完整48答新测量、有限干预及失败对照。
- `outputs/context_response_20260928_v2`：逐头稀疏、所有FP/FN/TP位置、`TOKEN_AUDIT.html`、`mechanism_token_audit.csv`。
- `outputs/context_response_20260928_v3`：条件校准，独立冻结和评价。
- `outputs/context_response_full_20260928_v1`：本轮全测试，只有`evaluation_complete.json`才表示评价完成；`progress.json`表示运行进度。

```bash
python -m pytest experiments/context_response/test_response.py tests/test_decision_risk_native.py -q
python -m experiments.message_js.score_verify --output outputs/context_response_20260928_v2
python -m experiments.head_state_readout.audit --output outputs/context_response_20260928_v2
```

本轮验证由同一个agent以另一实现复算，不能称独立外审。AtP*原文说明梯度归因近似可能漏掉非线性与抵消（https://arxiv.org/abs/2403.00745）；这里只借鉴有限干预核验思想，不声称实现AtP*。

本轮另冻条件校准全量v2，可复用v1新内部测量而不重跑模型：

```bash
python -m experiments.context_response.conditional_full --stage score-evaluate \
  --previous outputs/context_response_full_20260928_v1 \
  --output outputs/context_response_full_20260928_v2
python -m experiments.context_response.full_audit --output outputs/context_response_full_20260928_v1
python -m experiments.context_response.full_audit --output outputs/context_response_full_20260928_v2
```

全量v2的参考/阈值已在v1全测试评价前冻结，`conditional_full --stage freeze`用于创建新输出。局部v3融合64/134、78/1353、答内AUC .7425，没有明显胜过v2；完整比较保留两个版本，不以测试结果选择赢家。

## 完整参考恢复与请求参照对照

`restore.py`保持原固定无监督基线的完整fit分布、逐token分数及原dev阈值；4来源小参考并非原QA .890665方法。局部v4 strong_fused为54/134错、87/1353误报、答内AUC .7581。

`request_anchor.py`从来源段后的原prompt query和生成前query建立逐头source读取参照，不人工标注实体。v5原生参照融合54/71/.7707；v6按相同lag桶和copy状态作端点均值的匹配对照60/101/.7669。请求query可能关注模板和主题，不能称正确证据；自然配对结果不一致。`request_audit.py`保存205个相关token的top8评分头和实际地址，Q/K重建条件注意力最大误差3.06e-6。该方向仍只有一阶响应，原560次有限干预验证的是来源关联方向。

## 重复状态传播与逐token路径审计（2026-09-29）

`recurrence.py`保留1024物理头的两向signed-log响应与两向JS，以fit每来源64位置median/IQR标准化、asinh后完整坐标余弦比较。每个lag分别用来源等权fit分布校准；相似度分位低于.95不连边。max-min路径固定3跳、每跳最多8token：

`r_next[t] = max(r0[t], max_j min(edge[t,j], r[j]))`。

这是模糊逻辑路径容量，不是幻觉概率、因果传播证明或span真值；主方法双向离线，过去向另列。无需人工实体或边界。原始输入/完整头轴保留，top头仅在事后路径解释展示。

`recurrence_barrier.py`另测相邻状态最低5%处禁止跨越，以及多个不同原始起点的第二大路径支持。环路不能把同一风险起点重复计数。无监督混合dev95重新校准阈值；即使分数max保留，阈值升高仍会丢失旧报警。

8答/134错/1353正常：strong_fused 54/87，重复路径64/81且3/9整段覆盖，两个起点63/70且3/9，边界+两个起点59/68且2/9。原生重复边对照：头身份独立打乱67/173，仅距离74/98。仍漏养老金、时长限定和intimate；没有近乎全检。自然融合分数缺失不计零误报。具体回归是开发案例，不能当盲测。

```bash
python -m experiments.context_response.recurrence --stage pilot \
  --previous outputs/context_response_20260928_v4 --output outputs/recurrence_new
python -m experiments.context_response.recurrence_barrier \
  --previous outputs/recurrence_new --output outputs/recurrence_barrier_new
python -m experiments.message_js.evaluate --output outputs/recurrence_barrier_new
python -m experiments.context_response.verify_restore --output outputs/recurrence_barrier_new
python -m experiments.head_state_readout.audit --output outputs/recurrence_barrier_new
python -m experiments.context_response.recurrence_audit --output outputs/recurrence_barrier_new
python -m experiments.context_response.recurrence_paths --output outputs/recurrence_barrier_new
# 完整旧三版完成后，复用全量新内部信号，评分结束才评价；不重跑模型
python -m experiments.context_response.recurrence --stage full \
  --previous outputs/context_response_full_20260928_v3 \
  --pilot outputs/context_response_recurrence_20260929_v2 \
  --output outputs/context_response_recurrence_full_new
python -m experiments.context_response.full_audit --output outputs/context_response_recurrence_full_new
```

全量新入口仍是探索复验。`propagation_tokens.csv`列出新增/丢失TP、FP和路径起点；`propagation_head_paths.json`保存每条路径的边可靠度及逐头贡献。`propagation_audit.json`中的相同FP预算阈值是事后排序诊断，未写回检测器。官方RAGTruth内部量为观察器重放，不能把养老金`taxed`原词margin −6.543误说成该观察器仍选择了taxed：它更偏好`not`，但原回答来自别的生成器。
