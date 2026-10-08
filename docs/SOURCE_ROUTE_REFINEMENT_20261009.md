# 逐词来源与路由融合后的图平滑：固定修订验证

主线固定为“逐词来源删除敏感度 → 来源/路由融合 → 图连续性正则”。不引入新的分类器、候选解析、熵错误项或标点span平均。全部完整回答离线求解，每个token保留自己的初始分数；当前图是L15 post-token注意力32头均值/历史8lag，不能称其已保留完整头身份机制。缓存来自Llama3.1观察器重放混合生成器的回答。旧source_local测量仍在按标点/换行/64-token文本单元裁回prompt后重放；新评分不做单元均值，但没有消除旧测量的文本边界依赖。

## 原实现及两个问题

`unlabeled.py` 在671个fit来源的混合有效token上等source拟合mid-CDF：qL、qF、qR。S=(qL+qF)/2，U=.75S+.25qR。`smooth.py` 求

\[
\min_r\ \tfrac12\|r-U\|_2^2+.5\sum_{j<t}w_{jt}\rho_\delta(r_t-r_j).
\]

边权按两端incident degree预算归一到每节点总质量≤1。旧δ=1而U∈[0,1]，实际一直是二次图平滑。一个来源视图的高风险可能被均值抵消；孤立的错误实体高分可能被邻近低分词稀释。图也会降低大量正常误报，故不能直接去掉正则。

## 只验证两个修订

来源修订保留现有干预的两种上下文视图：M=max(qL,qF)，Q=F_max(M)，U_union=.75Q+.25qR。F_max仍只用fit有效token等source拟合。对照F_mean((qL+qF)/2)使用完全相同的重标程序，排除只有尺度变化的收益。max表示两种来源删除敏感度中强信号保留，不表示任一路认证了正确事实。

图修订使用真正的Huber线性段：δ取fit完整回答旧U的边差|U_t−U_j|的加权中位数。权重是实际图目标的normalized边权，各source先总质量归一后等权。δ为0立即失败，无额外分位数/epsilon选择。λ=.5、β=.25、图边不改。

最优条件为 r_t−U_t=−.5∑_j w_tj clip(r_t−r_j,−δ,δ)，因而|r_t−U_t|≤.5δ。它保护任何尖峰，包括正常尖峰；它限制图从低初始分数补检的能力。实现保存最终线性段边数/质量比例，不能只因参数δ<1便声称机制有效。

## 文件与入口

| 文件 | 职责 |
|---|---|
| `experiments/flow_latent/local_graph_transport/source_route_refine.py` | max/mean参考、固定unary、边差参考、11个固定场和线性段审计 |
| `.../run_source_route_refine.py` | 无标签fit/score、原token对齐、旧baseline逐词再现、源码与产物冻结 |
| `.../source_route_refine_eval.py` | 冻结后才读取官方标签；全词/首错/误报预算及逐词CSV |
| `.../test_source_route_refine.py` | 融合抵消、旧场身份、Huber保护上界/零边科学检查 |
| `.../test_source_route_refine_eval.py` | strict预算ties、健康prefix首错、alarm转移检查 |

旧 `unlabeled.py`/`smooth.py` 与默认入口不修改。预定primary是`union_robust`，不能评价后把其他对照改名为primary。共11字段包含2×2、无图、meanCDF同δ、chain/部分重连控制，不是参数搜索。

在repo根执行（0 GPU/新LLM，CPU四线程）：

```bash
export PYTHONPATH=.:teaching/state_audit/src
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PYTHON=/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python
$PYTHON -m unittest experiments.flow_latent.local_graph_transport.test_source_route_refine experiments.flow_latent.local_graph_transport.test_source_route_refine_eval
$PYTHON -m experiments.flow_latent.local_graph_transport.run_source_route_refine fit
$PYTHON -m experiments.flow_latent.local_graph_transport.run_source_route_refine score --phase pilot
$PYTHON -m experiments.flow_latent.local_graph_transport.source_route_refine_eval --directory outputs/source_route_refine_20261009 --phase pilot
$PYTHON -m experiments.flow_latent.local_graph_transport.run_source_route_refine score --phase dev
$PYTHON -m experiments.flow_latent.local_graph_transport.source_route_refine_eval --directory outputs/source_route_refine_20261009 --phase dev
```

输出已经存在时拒绝覆盖；复现实验应在每条命令统一传新的`--output`/`--directory`目录。`--fit-limit 4`只作软件见证，输出标记incomplete并禁止科学cohort评分。

fit混合95阈值不保证正常5%FPR；两种同误报预算使用评价标签，只是oracle诊断。8答中15604/15600进入旧fit参考，其他6答是已暴露test；完整DEV也经历历史研究。新分数没有自然标签fit/选型，研究迭代受历史标签启发，不能声称全研究过程完全盲无监督。

## 实际验证结果：不替换旧默认

已完整运行671 fit来源/4026回答/612083有效token参考估计，δ=.1757035788157119；8历史回答1139token/77错误，以及完整DEV168来源/1008回答/147981token/12123错误，全部先冻分数后评价。旧native逐词再现差0.0，8项科学软件检查通过。

| DEV字段 | AUROC | AP | 同5414正常token FP预算的TP |
|---|---:|---:|---:|
| old_native | .772523 | .218085 | 2136 |
| old_robust | .764864 | .207240 | 1974 |
| union_native | .765811 | .201177 | 1736 |
| union_robust（预设primary） | .756879 | .190652 | 1558 |
| mean_cdf_native | .765901 | .220096 | 2289 |
| mean_cdf_robust | .757348 | .208201 | 2159 |

primary对旧native AUROC增量source-bootstrap95%CI[-.0200763,-.0112580]。新Huber线性段实际覆盖47.36% DEV边质量，机制启用但效果下降。八答TP/FP7/57→8/46、正常答报警3/5→4/5，引用2和养老金首错仍漏；not从检出变漏检。max遗漏两路共同偏高的信息，统一减小δ也保护正常尖峰。完整DEV丢失1037TP全部在来源rank差<.25区域；来源差≥.25新增570TP同时新增2540FP。这些只是后置解释分组，不进入评分。

7062/12123 DEV错误满足U_union+.5δ≤阈值，当前尺度图无法把它们抬过阈值。养老金private/normal两来源风险均低，不能继续仅靠图传播保证修复。meanCDF_native只改善低误报工作点、all-AUROC仍下降，不事后改名primary。

本轮未过门，完整test/其他任务不启动。下一设计保持本主线，优先保留来源的联合状态、检验route抵消与图边降噪/传信可靠性；尚无验证过的下一版，不承诺顶会创新或关键样例近乎全检。

结果/逐词CSV/冻结记录在`outputs/source_route_refine_20261009`；共享计划、评审、RESULTS.md/MAINLINE_CODE_MAP.md/EXPERIMENT_AUDIT与交接在`/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/source_route_refine_20261009`。独立审计状态以该目录报告为准；主线有效性需来自同误报预算召回与强控增量，不能以算子检查通过替代。
