# 来源与路由的条件耦合检测：代码与验证范围

三个固定 CPU 实验命令已实际各执行一次、退出 0。预定主方案在八个历史配对回答、1,139 token 上的 AUROC 为 0.505295082、AP 为 0.067653055；同 57 个正常 token 误报预算下仅检出 3/77 错词，旧方案为 7/77，正常五个回答全部报警，干净首错 0/3。六项联合门只有 AUROC 一项通过，状态 `PILOT_FAIL_STOP`；不扩 DEV/test、不替换 default、不调整方向或提升对照。12 项合成检查和 CLI help 均实际退出 0。之前 FIT-only 统计相关性通过，没有转化为有效错误判别。独立完整数值复算已通过；审计整体 WARN 保留八答、四来源、历史暴露及 FIT 交叠的范围限制，同系列审计为暂定结论。实验自身的联合门仍失败。

这轮继续使用“逐 token 来源与路由读出 → 图”的主线，改动的是融合读出。旧融合把两个来源风险和路由风险作固定加权求和；新读出检验：在相似的当前词生成概率下，这个 token 的来源作用和路由状态是否符合 FIT 中观察到的联合关系。图、原始观测及目标 token 保持原有实现。

## 每个 token 读什么

输入五个已有缓存量，不选择 attention 最大的历史词，也不标注 prompt 实体：

| 输入 | 含义 | 新模块中的用途 |
|---|---|---|
| `source_local` | local 语境中，来源移除后的实际词 logp 减去来源保留时的 logp | 来源作用坐标 |
| `source_full` | full 语境中的同一实际词来源作用 | 与 local 保持成对的来源坐标 |
| `raw_route` | 旧实现已经保存的消息路由标量 | 与来源作用共同建模 |
| `local_native_logp` | local 原生语境中实际词的 logp | 条件变量 |
| `full_native_logp` | full 原生语境中实际词的 logp | 条件变量 |

后两项控制词本身的生成难度。它们没有独立变成错误分数，也不是分布熵。正负来源作用表示来源对当前词生成的帮助或抑制；这种帮助本身不认证主体、关系或适用范围正确。

前三个坐标复用旧 FIT 的来源等权 empirical midCDF。后两个坐标只用同一 FIT 的原生 logp 建立来源等权 midCDF。每个来源总权重相同，再把其权重均分给该来源全部有效 token。相同值保留相同 midrank：

\[
q(x)=P_{FIT}(X<x)+\tfrac12P_{FIT}(X=x),\qquad z(x)=\Phi^{-1}(q(x)).
\]

边缘饱和点使用 FIT 实际支持的最小和最大 midrank，不设置 epsilon，不外推未观测尾部。本轮固定完整 FIT：4,026 个回答、671 个来源、612,083 个有效 token；不读取自然幻觉标签来拟合。

## 条件密度比如何计算

在五个正态坐标上拟合一组来源等权均值和协方差；协方差固定采用 `1/672` 的对角收缩。令前三个坐标为 \(X=(S_{local},S_{full},R)\)，后两个为 \(C=(C_{local},C_{full})\)。先消除原生生成难度条件对应的平均关系：

\[
B=\Omega_{XC}\Omega_{CC}^{-1},\qquad
(s,r)=X-\mu_X-B(C-\mu_C),
\]

\[
\Sigma=\Omega_{XX}-B\Omega_{CX}
=\begin{bmatrix}A&b\\b^T&d\end{bmatrix},\qquad
\beta=A^{-1}b,\quad v=d-b^T\beta.
\]

其中 \(s\) 有两个来源坐标，\(r\) 是路由残差。来源对预测的路由为 \(\beta^Ts\)；路由偏离预测的量为 \(r-\beta^Ts\)。逐 token 读出为：

\[
D=\log\frac{p(s\mid C)p(r\mid C)}{p(s,r\mid C)}
=\tfrac12\log\frac vd+
\tfrac12\left[\frac{(r-\beta^Ts)^2}{v}-\frac{r^2}{d}\right].
\]

这里的概率密度是固定高斯正态坐标模型的密度。两个来源坐标在分子中仍作为一个联合块；只打破“来源对 ↔ 路由”的依赖。减去 \(r^2/d\) 是为了扣除路由本身罕见的贡献，而非用联合距离把所有罕见词一律当错。它检验条件耦合关系，不能认证事实关系。

这个读出有必须保留的限制：\(D(s,r)=D(-s,-r)\)，因为两项都是平方。它无法区分条件残差同时反号后的绝对方向，仍保留来源与路由的相对符号交叉项；在拟合的联合模型下 \(E[D]=\tfrac12\log(v/d)=-I(S;R\mid C)\)。有相关性及互信息并不能推出“大 D 就是错误”。因此旧有符号来源方法保留为固定对照，新模块的错误方向要靠预先声明的自然样本门检验。

## 如何接回逐 token 图

只用 FIT 的 \(D\) 再建立来源等权 midCDF，得到每个 token 自己的 unary \(u_t\)。没有句子平均、16-token 窗口平均、标点硬重置或标签 span 广播。

继承的图读取 `local_attention[0, 1:]`：这是 **post-token 行**，每行的 32 个物理头取均值，连接前 8 个回答 token。它是标量连续性先验；此实现没有声称保留完整多头因果消息，也没有解决 predictor 行与 post-token 行的机制差异。边权除以 `max(1, source 的 incident degree, target 的 incident degree)`，再求解：

\[
\min_x\ \tfrac12\sum_t(x_t-u_t)^2+
0.5\sum_{j<t}\bar w_{jt}\operatorname{Huber}_{1}(x_t-x_j).
\]

因为 unary 在 \([0,1]\) 且 Huber 阈值为 1，这个设置的最优解处于二次平滑区间。图先在完整回答节点上求解，再筛选原有有效目标 token。全回答平滑可以回头改变早期节点，所以这是事后检测，不是在线报警或因果传播追踪。每个 token 的原始效应、rank、正态坐标、条件残差、\(D\)、unary 和最终图分数都会保留。

## 三个代码文件

| 文件 | 实现 |
|---|---|
| `experiments/flow_latent/local_graph_transport/source_coupling.py` | CDF 支持端点、正态映射、五坐标 FIT 模型、条件密度比和风险参考 |
| `experiments/flow_latent/local_graph_transport/run_source_coupling.py` | 完整 FIT/pilot 评分、输入与快照绑定、冻结后官方评价、全部 FP/FN 导出 |
| `experiments/flow_latent/local_graph_transport/test_source_coupling.py` | 独立 logpdf 对照、符号对称、来源置换、条件平移、来源等权重复、CDF ties 和完整节点图的 12 项检查 |

核心计算按 `fit_normal_model` → `answer_coordinates` → `dependence_ratio` → `coupling_rank` → 旧 `graph_fields` 执行。固定输出五个场：`old_native`、`likelihood_native`、`null_native`、`coupling_unary`、预定主方案 `coupling_native`。前三个必须逐值复现旧冻结分数，不根据结果切换主方案。

## 精确运行命令

每个命令的工作目录均为：

`/share/home/tm902089733300000/a903202310/lys/research/graph`

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:teaching/state_audit/src /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.flow_latent.local_graph_transport.run_source_coupling scorefit
```

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:teaching/state_audit/src /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.flow_latent.local_graph_transport.run_source_coupling scorepilot
```

```bash
env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:teaching/state_audit/src /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.flow_latent.local_graph_transport.run_source_coupling evaluatepilot
```

固定新输出目录为 `outputs/source_coupling_20261009`；已有完整或部分结果会拒绝覆盖。按上述顺序各执行一次，失败保留报错和部分产物。本轮不提供 DEV/test 入口、不启动新模型或 GPU、不改变旧 default。

`scorefit` 在首个自然 \(D\) 前复制并直接绑定实际执行代码及 evaluator，绑定官方标签字节但不解析；再复现先前 FIT-only 模型参数和 CDF 字节。`scorepilot` 冻结八个历史配对回答全部 1,139 个有效 token；只有 `evaluatepilot` 在验证两阶段冻结后解析官方标签。

评价导出 `pilot_tokens.csv` 的全部 token、`pilot_failures.csv` 的主方案 FP/FN、`pilot_changes.csv` 的每处变化和上下文，同时报告答内、首错及来源 bootstrap。FIT95 是未知正确/错误混合参考的 95% 分位，不是“正常 token 的 5% FPR”；匹配 57 个正常 token FP 和 3 个正常回答报警的点明确为标签 oracle 分析，不是部署阈值。

## 什么结果才允许继续

预定 pilot 门要求全部成立：AUROC 高于旧方案；同至多 57 FP 时 TP 严格大于旧 7；正常回答报警至多 3/5；至少一个没有提前误报的首错；保留 `not@12219:223`，并恢复 `2@12297:106`、`private@15604:96`、`normal@15604:103` 至少一处。失败就停止这个固定模型，不翻转方向、不混合旧来源权重、不把较好对照改名为主方案。

八回答来源及错误模式已有历史暴露，且部分来源与 FIT 重叠。即使通过，结论也只是这个探索 pilot 通过，需另行冻结完整 DEV 方案；不能宣称全部幻觉检出、原生成器因果机制成立或顶会主张已经得到验证。

共享计划和实际工程见证位于 `lys/codex/research/refine-logs/source_null_calibration_20261009/COUPLING_DETECTOR_PLAN.md` 与 `COUPLING_TEST_WITNESS.json`。首次 staged 合成检查的两处图手算参考错误已保留；修正的是 incident-degree 手算和匹配原有优化停止精度的测试要求，没有修改密度比或旧图算法。

## 已测到的阶段差异

引用 `2@12297:106` 的原始 coupling unary 为 .957031569，图降为 .884631702，低于同 57 FP 阈值 .886912151，也低于 FIT95 阈值 .898032881。这里读出已经形成异常峰，图确实造成漏检。`not@12219:223` 的 unary 仅 .044710280，图分数 .082427368；它的条件路由误差虽然为 1.688033274，但密度比减去较大的 native-only 路由成本，最终 D 为 -.471543933，故错误信号在读出阶段就消失。`private@15604:96` / `normal@15604:103` 最终分数 .285862010 / .670189284，两处也漏检。

全部旧 7 个 TP 丢失，新增 3 个 TP 仅为旗帜错误 span 中的 from@79、proper@88、句点@90。所有 57 个当前 FP 与旧 57 个 FP 不重合。全部 74 个当前 FN、57 个 FP 和 124 个两政策变化 token 均保存；不只分析成功位置。以相同 primary FIT95 cut 比较 unary→图，图去掉 52 FP，也损失 3 TP，新增 0 FP/0 TP。图有降噪收益，不能把全部失败都归因于图。

本轮保留 D 及源/路由残差作分析工具，Gaussian detector 不通过。另一个独立的 source-null 校准方案在完整 QA DEV 获得 .803398309，但同误报预算检出退化；这个数字不属于本 coupling 方案，也不是未见测试集结果。详细记录见共享 `COUPLING_RESULTS.md`、`COUPLING_TOKEN_DIAGNOSTICS.md`、`COUPLING_EXECUTION_WITNESS.json` 与 `COUPLING_EXPERIMENT_AUDIT.md`（以实际存在与状态为准）。
