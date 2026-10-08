# 来源、路由与局部图：2026-10-09验证

当前目标：无自然标签训练的逐token AUROC≥.8。完整QA旧DEV .772523，尚未达到；旧片段共分成绩不计作逐token。

## 现有图的实际作用

每词U=.375(q_local+q_full)+.25*q_route，各q来自fit-source等权CDF，非事实概率。原图取L15 native posttoken全部32物理头平均、过去8个回答token（含标点）的attention边，不选择最大来源token、不做标点切段。所有token独立U，完整answer离线推断。

J(r)=.5||r−U||²+.5ΣwHuberδ(r_t−r_j)，incident degree≤1。旧δ=1全在二次区间，等价(I+.5L)^{-1}U。高分会被低分邻词拉低；同阈值完整DEV图移除2470FP、损409TP、补4TP。主要问题仍在unary：95.90%漏检与99.96%误报在图前已有。

## 已新增并验证的状态图

`experiments/flow_latent/local_graph_transport/state_graph.py`：把分数差惩罚替换为二元状态切换惩罚：

P(z|x) ∝ exp[Σlogit(U_t)z_t−κΣw_tj1(z_t≠z_j)]，κ=log(.993/.007)。保留最大lag8图，用last8bits/256frontier精确前后向；不是把带环图当2状态链。返回state1和相邻joint的enter/continue_state/exit，都是未校准风险相容度。

`gated_weights`：读取现成native/所有层source-key-blocked L15 posttoken group_heads，各头Δm保留5×128全部坐标，32头分别计算历史边cos正部，再与同头attention相乘后形成边权。gate只调连续性，不能把相似称同一真假或原生信息传输。全32头gate也保留用于审计；不挑最大的head/token。

`run_state_graph.py`：固定9方法，纯输入→完整评分→FREEZE→官方标签评价。历史4来源8答1139词/77错误：旧图AUC .430271、state_gated .414080；同57正常词FP均7TP，gate置乱.418776。此候选失败，没有扩展DEV或替换旧默认。paper_chain只是CORTEX监督MLP之后的链平滑控制，未复制CORTEX检测器。

无边退化、全配置枚举、frontier淘汰、CORTEX转移等价与物理头gate有5项CPU验证，独立60随机图枚举复核通过。fresh审计WARN/same-family provisional：分数/指标/哈希通过；历史暴露/2答fit重叠/阈值未校准/运行日志范围保留。

原始：`outputs/source_route_state_20261009_v2`，包括完整token表、全部状态和gate。旧准备失败与15401旧DEV FP/FN诊断保留于`outputs/source_route_state_20261009`。

## 来源校准候选的实际结果

`likelihood_calibration.py`：原效应Δ=ell_absent−ell_present≤−ell_present。全局CDF混合不同native生成概率的效应范围，改为source_local条件matching with_source.local actual logp、source_full条件matching with_source.full。固定16个等源质量桶，tie不拆分；先每来源等权，再在桶内归一化同一weights；四科学CPU测试核对手算权重/constant/ties/shuffle。实际词logp只分参考桶，不独立变成熵/错误risk。

`run_likelihood_calibration.py`：完整fit4026回答612083有效词拟合reference及混合fit95，各法评分每个原token，之后按原valid过滤；固定7场旧unary/native、双条件unary/native(primary)、各单视图native、源内打乱reference native。旧route/fusion/δ1图保持。

纯identity pack不打开标签；source+native logp及完整attention字段保存消费身份/哈希，代码含评价与GT对齐依赖、官方GT字节绑定。全部分数freeze完成后才读官方GT；各split报告完整AUROC/AP/正常答alarm/严格首错/匹配FP oracle/来源bootstrap/各generator/逐词。

该改动检验统计测量噪声，不能认证来源帮助的词汇适用于主体/范围，不能声称native因果机制。历史same-position source_raw、CAD及route|attention负结果不重复。

运行（现有research环境，不安装包）：

```bash
export PYTHONPATH=.:teaching/state_audit/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PYTHON=/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python
$PYTHON -m experiments.flow_latent.local_graph_transport.run_likelihood_calibration --stage score --phase fit --output outputs/source_likelihood_calibration_fresh
$PYTHON -m experiments.flow_latent.local_graph_transport.run_likelihood_calibration --stage score --phase pilot --output outputs/source_likelihood_calibration_fresh
$PYTHON -m experiments.flow_latent.local_graph_transport.run_likelihood_calibration --stage score --phase dev --output outputs/source_likelihood_calibration_fresh
$PYTHON -m experiments.flow_latent.local_graph_transport.run_likelihood_calibration --stage evaluate --phase pilot --output outputs/source_likelihood_calibration_fresh
$PYTHON -m experiments.flow_latent.local_graph_transport.run_likelihood_calibration --stage evaluate --phase dev --output outputs/source_likelihood_calibration_fresh
```

输出目录必须新建，评分/CSV不覆盖已有证据。完整fit/pilot/DEV均已实际结束。DEV primary AUROC .773897757、AP .234860398，对旧 .772523119/.218084651；AUROC增量来源CI[-.002892361,.006134702]跨0，同5414FP下TP2136→2421，但答内 .699416→.693904、首错46/307→13/307。目标未达；预设full-only对照 .781478564不能事后升为primary。完整test未运行，扩展门未通过。不能把test已完成写成进行中，也不把QA冒称三任务。

完整共享计划、原文调研、所有结果/审计：`/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/{source_route_state_20261009,source_likelihood_calibration_20261009}`。

完整原始7场评价见 `outputs/source_likelihood_calibration_20261009/dev_evaluation.json`，所有147981逐词分数/变化见dev_tokens.csv；报告与独立审计保存在共享目录。条件CDF桶内方向单调，但Δ=0在高置信参考q≈.95、低置信参考q≈.3–.47，可能将正常续写抬高、将低置信首错降分，不能把该校准误读为真假认证。
