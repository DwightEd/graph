# 来源排序与 token 内部细化

2026-09-25。入口 `main.py transport-refine`，只复用已经完成的 source-first 标量缓存。
新实验不调用大模型、不修改已有输出，不包含原生 KV 中介实验。

## 设计来自哪些实测

用户上传的全量 source-first 轻量包：17,790 答、2,903,263 token，零排除/截断；
test 为2,700答、450来源、424,408 token。官方train/test来源不重叠。
开发集分别选择 Data2txt/QA 的 pair 来源均值，Summary 的 full 来源均值，未选路由融合。

| Test | local均值 AUROC | full均值 | pair均值 | 原始route单元内 | window16 route单元内 |
|---|---:|---:|---:|---:|---:|
| Data2txt | .774199 | .738940 | .784388 | .451474 | .560711 |
| QA | .857607 | .869365 | .878533 | .699867 | .831193 |
| Summary | .700458 | .759659 | .746843 | .495480 | .555733 |

三个任务原始token local来源的单元内AUROC分别为 .573231/.611729/.584031。
混合单元分别891/119/185个；内部指标按正负token对数加权，不是单元宏平均。
QA窗口路由在全部train上也从 .673156 提高到 .742987；不是独立开发集指标。
这些test结果已经用于设计，本轮属于探索性复验，不能称完全未触及的最终确认集。

旧 `A_u + lambda*(R_t-mean_u R)` 对任意正lambda的单元内排序都与R完全相同。
增大lambda只会改变跨单元排序，无法修复不合适的内部观测。
新设计同时保留较强来源条件、测量尺度与零修正控制，不恢复已失败的图风险传播。

## 一个分层读出，而非新的大模型测量

1. 三个来源观测：local/full，以及二者token分数的平均pair。
   来源分数方向仍为 `logp_without_source - logp_with_source`。
2. 每个来源条件得到单元平均 `a_u`，作为粗粒度风险。
3. 对每个条件独立考察三个细粒度观测：该条件的原始token来源、window16来源、window16路由。
   窗口跨文本单元但不跨回答，包含未来token；不将连续性当作事实正确性的证明。
4. 来源锚点与细粒度观测分别做source等权mid-CDF。每个task/split独立拟合；
   train只使用固定开发来源，test用完整未标注test。这仍是transductive离线读出。
   与v1的全部train尺度不同，导出的开发/test测量足以精确重现本轮评分。

令 `r_t` 为细粒度观测的秩，`d_t = r_t - mean_u(r)`。
统一比较两种约束：

**只细化同分：**

\[
s_t = k(a_u) + \tfrac14 d_t,
\]

其中 `k(a)` 是同一task/split内不同原始锚点数值的有序整数编号，相同数值共享编号。
`d`位于[-1,1]，每块宽度至多0.5，相邻块中心相距1，因此任何严格的原锚点排序不改变。
保留的是锚点的**排序**，不是原始均值的数值/单位。它不是错误概率。
计算锚点与建立整数块时必须使用完全相同的均值浮点运算，不能用不同归约制造伪排序。

**允许有限跨单元修正：**

\[
s_t=F_a(a_u)+\lambda d_t,\qquad \lambda\in\{.025,.1,.25\}.
\]

严格排序保护可能限制总体AUROC改善，所以保留小幅残差修正，实际收益由开发集选择检验。
三个anchor×三个channel×四种细化=36个固定候选。
另保留8个v1基线（local/full/pair均值、local token、route、route窗口、attention、entropy），
若原输入存在selected_detector，原样读入为previous_selected。
固定无标签候选是 `pair_source_window_tie`，不据新test成绩替换。

## 可检验的AUROC分解

令B遍历相同锚点的分数块，包括不同单元恰好同分的情况。严格细化有恒等式：

\[
\Delta\mathrm{AUC}
=\sum_B\frac{n_B^+n_B^-}{N^+N^-}
 \left[\mathrm{AUC}_B(d)-\tfrac12\right].
\]

因此不能保证细化必然提升：内部排序错误会降低AUROC；相同锚点跨单元的对也参与计算。
同分正负对占比还给出全局提升空间上限，即其占比的一半。
`tie_accounting.json`保存分母、块内AUROC和预计增益；与整体AUROC差核对。
这是排序代数，不声称是一项新的因果理论或已解决证据适用性。

## 选择与评价

`--select-on-train`保持原seed=42、官方train来源hash前20%的开发划分，整组保留同来源生成器。
只按开发token pooled AUROC选择一个方法，每task一个；并列按冻结顺序优先基线。
选择比较精确的半整数正负对一致数，避免AUROC积分约1e-16的尾差错误选中更复杂方法。
选择池包含3种来源均值、raw/window route及36个细化候选，共41种。
没有符号翻转、窗口搜索、test生成器单独选择，也没有新分类器训练。
它仍是标签辅助方法选择，应与固定无标签候选分开报告。

先冻结全部候选分数，再读开发标签；固定方法与选择结果随后在三个任务test评价。
按task×generator报告AUROC/AP、答内/单元内，以及onset/first-error/continuation/前后半段。
每次只加载一个任务的测试分数矩阵，不把不同任务的整数块分数混合计算总体AUROC。
`summary.json`保留基线、固定候选、旧选择结果、新选择结果，不按test选出“最佳方法”。

## 一键运行、断点与复查

```bash
git pull --ff-only origin main
bash experiments/native_support/run_ragtruth_refine.sh
```

默认输入 `outputs/native_support_ragtruth_all/source_first_v1`，输出独立目录
`outputs/native_support_ragtruth_all/source_refine_v2`。无需GPU，不重复捕获、tokenize或prefill。
默认三任务全部test + 各任务20% train开发来源，不处理其余80%train。
本次完整输入对应5724个回答、918307个token；实际数目以manifest为准。

```bash
REFINE_INPUT=/path/to/source_first_v1 REFINE_OUTPUT=/path/to/source_refine_v2 \
  bash experiments/native_support/run_ragtruth_refine.sh
```

原命令续跑。逐回答NPZ原子写入；完成文件跳过，部分task会重建无标签尺度但只补缺失回答。
已完成task完全跳过评分，选择/评价仍可重算。参数或输入manifest改变要求新输出目录。
`--stage score`仅冻结分数；后续 `--stage select/evaluate/export/pack`可以只指定output。

自动产物：

- `_review_light.zip`：汇总、全部候选指标、选择过程、同分增益分解和图。
- `_QA_cache.zip`、`_Summary_cache.zip`、`_Data2txt_cache.zip`：分别保存开发/test原始标量、
  原token身份、文本/单元和独立对齐标签。无模型权重/KV/逐头大数组，也无重复候选矩阵。
  这些包支持在本环境真正重算新读出；原轻量包不含测量，无法承担这个用途。

单个便携包重现：

```bash
python main.py transport-refine --input source_refine_v2_QA_cache.zip \
  --output outputs/refine_QA_replay --tasks QA --select-on-train --resume
```

只有四答test缓存时显式 `--tasks QA` 且不加 `--select-on-train`；不得伪造开发集或全量成绩。

## 验证记录

本环境目前仅有全量汇总包和旧四答完整测量，缺少全量逐token观测。
三任务真实新AUROC需用户服务器CPU复用现有缓存后返回；不能从旧AUROC合成新成绩。
已在旧四答完整缓存运行：836 token、81标错、41个单元，零混合单元。

| 四答候选 | AUROC | AP | 答内AUROC |
|---|---:|---:|---:|
| local来源均值 | .914856 | .451496 | .957736 |
| local＋来源窗口同分细化 | .914856 | .370258 | .957736 |
| local＋路由窗口同分细化 | .914856 | .370258 | .957736 |
| local＋路由窗口残差 .25 | .915771 | .374252 | .951688 |
| 固定pair＋来源窗口同分细化 | .893582 | .308774 | .936305 |

同分正负对为0，理论AUROC增益也为0，与实测一致；纯正例块打破同分仍可改变AP。
残差 .25 的微小AUROC增益伴随AP与答内下降，不能据此认定优化成功或改选默认方法。
这四答反复用于设计，不做开发选择，不冒充三任务test验证。

8项针对性软件测试通过：严格锚点排序与AUROC恒等式、float32/64均值归约一致、
三任务合成全流程、修改test标签不影响选参、同分精确择优、完整/部分断点、
便携包逐数组/指标复现、轻量汇总缺测时明确拒绝。CLI与shell入口也已检查。
