# 首错、熵与局部连续性：2026-10-08 实际验证

当前连续性惩罚不是首错判别器。开发验证发现，把熵作为逐 token 分数输入有收益；按高熵削弱连接的预设主候选反而低于不削弱连接的对照。本轮不修改旧检测默认值，也不把这项标量改动称为原生因果机制或完整论文方法。

## 当前惩罚到底做什么

代码在 [smooth.py](../experiments/flow_latent/local_graph_transport/smooth.py)、[unlabeled.py](../experiments/flow_latent/local_graph_transport/unlabeled.py)。图权重由 posttoken 原生 32 头 attention 的均值构造，连接前 8 个历史位置；每 node 的 incident degree 预算不超过 1。保留每词独立分数，不按标点/金标 span 共分。

目标为

\[
\tfrac12\|r-s\|^2+\lambda\sum_{j<t}w_{jt}\operatorname{Huber}_{\delta}(r_t-r_j),\qquad\lambda=.5,\quad\delta=1.
\]

本轮所有 unary 是 [0,1] CDF 排名的凸组合。非负连接的极值原理保证解仍在 unary 的最小值和最大值之间，边差不超过 1。因此 Huber 始终处于二次段，严格等价于

\[r=(I+.5L_w)^{-1}s.\]

该逆矩阵每行非负、行和为 1，作用是依图加权平均。它不能自行创造真假方向，也没有实际启用鲁棒线性段来保护入口。虽原边指向历史，对称目标和完整回答求解仍可让未来分数修正过去；这是离线 observer 检测，不是首错预防。

## 固定修改与对照

S=.5(rank(source_local)+rank(source_full))，R=rank(raw_route)，H=rank(next-token 全词表熵)。CDF 只用 fit 来源，来源等权，不用自然幻觉标签。

- 旧逐词输入：.75S+.25R。
- 固定加熵输入：.5S+.25R+.25H。
- 预设熵门：E_t=max(0,2H_t-1)，g_jt=exp[-4 max_{j<i≤t}E_i]。原边先做预算，再乘门，不放大弱边。

门只能减弱跨过高不确定性位置的平均，不能认证语义边界或首错。它还会保留正常高熵尖峰。九个预设对照全体评分冻结后评价；主候选保持为 entropy_unary_gated_huber，没有把效果更高的对照事后改名成预设主方法。

## QA 开发集实际结果

168 来源、1,008 回答、147,981 有效 token、307 首错。严格首错只与 115,308 个正常回答及首错前正常 token 比较；不包含错后正常词。阈值为 fit 来源等权混合分布 95 分位，不能解释成正常 token 5% FPR。

| 方法 | 全词 AUROC | AP | 严格首错 AUROC | 首错检出 | 检出且此前无误报 | 正常回答报警 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧 unary | .753137 | .197782 | .677036 | 49/307 | 11/307 | 488/701 |
| 旧 unary + native 惩罚 | .772523 | .218085 | .690893 | 46/307 | 9/307 | 438/701 |
| 原始熵 | .631208 | .132741 | .776113 | 55/307 | 10/307 | 574/701 |
| 旧 unary + 熵门惩罚 | .759507 | .204616 | .681653 | 52/307 | 10/307 | 473/701 |
| 加熵 unary | .792115 | .215742 | .763017 | 54/307 | 9/307 | 478/701 |
| 加熵 unary + 原惩罚 | .808426 | .232091 | .773376 | 56/307 | 12/307 | 432/701 |
| 加熵 unary + 熵门（预设主） | .796944 | .219920 | .765628 | 56/307 | 10/307 | 465/701 |

旧 native 对比 unary 的首错排序增量来源 bootstrap CI [.010733,.017008]；但与重连惩罚的差 CI [-.000341,.001641]，没有证明原生拓扑必要性。

加熵普通惩罚对旧普通惩罚的严格首错差 CI [.06749,.09545]；熵门对相同加熵输入普通惩罚的差 CI [-.01001,-.00523]。门全词差 CI [-.012122,-.010838]，明确退化。所有区间是 300 次按来源重采样的探索性区间。

归因不能忽略 S/R 权重也变了。冻结后借助二次算子线性做同系数去熵诊断：.5S+.25R 的普通惩罚全词/首错为 .790041/.695868；加 .25H 后为 .808426/.773376。因此全词 +.035903 中约 +.017518 来自 S/R 权重变化，另 +.018384 来自同系数加入熵。这项是事后归因诊断，不是重新选型。

## 为什么还不能稳定识别首错

熵对首错排名更强，却不是必要或充分条件。307 首错有 47 个 H 排名低于 fit 中位数；低熵例包含 Fruia 的首子词 ` F` 和 `300°F` 的 `300`。某些官方 span 起点是列表符号、句号或编号，具体错误内容随后才出现，不能删掉这些起点美化指标。

严格正常前缀也有高熵：标题 ` Tips`、转折 `However`、实体 ` Hydro`、数字 `34`。旧 native 的正常 FP 不是都在开头：5,414/135,858 正常词误报，其中标点 1,074/16,998、数字 92/4,966、前 16 词 18/15,645。加熵后仍有 432/701 正常回答报警，尚未成为可用的首错报警方案。

路由与熵擅长的阶段不同：原始 route 错误 span 内部 AUROC .790208，熵 .625265；严格首错分别 .630516/.776113。当前结果支持保留两种逐词读出，不能由持续状态相似性给整个片段赋同一真假标签。来源内容是否适用、是否主导最终选择仍未被本轮标量改动解决。

## 运行与证据

新算法：[renewal.py](../experiments/flow_latent/local_graph_transport/renewal.py)；完整 train 冻结与 DEV 验证：[run_renewal_validation.py](../experiments/flow_latent/local_graph_transport/run_renewal_validation.py)；独立冻结后的 QA test 入口：[run_renewal_test.py](../experiments/flow_latent/local_graph_transport/run_renewal_test.py)。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=.:teaching/state_audit/src /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.flow_latent.local_graph_transport.run_renewal_validation all --output outputs/first_penalty_rerun/renewal
```

使用新输出目录以保留旧冻结结果。对应 test 命令：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=.:teaching/state_audit/src /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.flow_latent.local_graph_transport.run_renewal_test all --dev outputs/first_penalty_rerun/renewal --output outputs/first_penalty_rerun/renewal_test
```

原始产物在 outputs/first_penalty_validation_20261008；研究计划、反例逐词上下文、数学说明、完整报告、执行追踪和 fresh 审计在共享 codex/research/refine-logs/first_penalty_validation_20261008。

已有 19 项新 gate 与旧 smooth 检查通过，包括空边、跨入口长边、无未来事件、归一预算、逐词 .5 改动界以及独立 Laplacian 求解等价。根代理用 scipy rank-sum 独立复算九项开发 AUC 和报警计数一致。0 新 LLM forward/自然标签拟合；既有 DEV/test 已暴露，结果为探索性复验。使用的是 Llama3.1-8B 对六种生成器回答的 observer 重放，标量 prechoice、局部图 posttoken；本轮不宣称原生成器因果轨迹、GSM8K 或其他两任务的新结果。

## 完整 QA test：冻结参数后的实际复验

150 来源、900 回答、124,817 有效 token、6,751 错误 token、160 首错；首错负例为 108,472 个严格正常前缀 token。九个预设读出全体先冻结，再加入官方 test 标签；参考 CDF、权重及阈值完全复用 fit 设置。

| 方法 | 全词 AUROC | AP | 严格首错 AUROC | 首错检出 | 检出且此前无误报 | 正常回答报警 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧 unary + native 惩罚 | .779725 | .172794 | .651677 | 18/160 | 2/160 | 447/740 |
| 纯熵 | .637397 | .101584 | .790523 | 36/160 | 5/160 | 615/740 |
| 加熵 unary + 原惩罚 | .815225 | .184809 | .755875 | 23/160 | 6/160 | 441/740 |
| 加熵 unary + 熵门（预设主） | .803321 | .170736 | .749955 | 21/160 | 7/160 | 489/740 |

普通加熵对照对旧普通惩罚的全词 AUROC 增量来源 CI [.027656,.044254]；熵门对相同加熵输入普通惩罚的差 CI [-.012515,-.011168]。纯熵的首错排序仍高于融合，因此不能说当前平滑提高了纯熵首错识别。

同系数去熵事后诊断：.5S+.25R 的普通惩罚全词/首错 .796454/.665592；加入熵后 .815225/.755875。S/R 比例调整贡献全词 +.016729，保持该比例加入熵贡献 +.018771（CI [.010053,.028235]）。但是同系数全词 AP 从 .189588 降到 .184809；首错 AP 从 .005692 降到 .005544，低于旧基线 .006201。这次收益主要是 AUROC 排序，不等于顶部报警已经可用。

旧漏检仍在：12297 在旧普通惩罚、加熵普通惩罚、熵门三者中均只检出 1/49 错误词；12219 加熵普通惩罚检出 4/13，正常词误报 27。按现有混合95阈值逐词评价，不能宣称已经解决这些错误或几乎检尽。

根代理在验证冻结 hash 后，用独立 scipy rank-sum / 直接首错前缀计数复算 test 18 个 AUROC 与 36 个报警计数，全部一致。fresh 同家族代理审计 DEV 官方标签指标、数学/直接解、源码与冻结；test 只核查评分时序、冻结及聚合一致性，未打开 test 标签重算，不冒认 fresh test 指标复核。审计为 provisional WARN，涉及历史暴露、有限任务、执行依赖快照不完整等边界，不是外部科学通过。

计划中的所有 span 起点指标另外保存 renewal/ONSET_METRICS.json 和 renewal_test/ONSET_METRICS.json，不改冻结九分数/阈值；不同于仅每答第一次错误。七旧例逐词 CSV、位置类型误报与持续错误指标在 renewal_test/ 下。
