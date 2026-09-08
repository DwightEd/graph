# 正常／幻觉的完整结构审计 v3

本次主线：**读取较早材料 → 在回答位置形成载体 → 更深层后续读取 → 预测目标及原生写入**。
全部输入、全部回答 token、全部物理 head 参与；标签只用于事后比较和展示。
不再先筛 WAAD/FAI 峰的交集，也不把强 attention 自动叫作相关事实。

这是一个有明确覆盖表的观察性审计，不是承诺穷尽所有幻觉机制。新方法仍需由实际结果决定，不能预先保证发现机制或得到更高 AUROC。

## 为什么替换 v2 的运行路径

用户提供的 v2 结果覆盖 2,946 个样本、509,834 个 token，42,587 个标为幻觉。
跨层节律和部分逐 head 差异可在 train/test 复现，但详细中继 36/36 都从 BOS 出发；测试 QA 的三个展示样本都正常。
这说明实例选择与正负展示没有回答研究问题。另一个重要结果是同一 head 的 WAAD 可以下降、未截断距离上升、峰率也上升，不能预设“幻觉就是回看减少”。

v3 的唯一推荐全量入口是 `attention_audit_run`。旧 `attention_rhythm_run` 与兼容入口保留用于复现旧结果，旧输出不删除、不覆盖。
配对条件的 `constraint_flow_run` 是另一个功能验证工具，不替代这里的真实数据比较。

## 运行前已经确定的覆盖表

| 要回答的问题 | 连续观测与对照 | 对比范围 |
|---|---|---|
| 是否只是 BOS/控制符吸走 attention？ | 特殊来源绝对质量；排除后条件分布；保留含特殊来源的距离对照 | 全 head；N/H；原始与位置配对 |
| 正常是否普遍从 prompt 转向 history？ | prompt/history 来源份额、实际可见普通来源数量基线、逐 head 斜率 | 正常 token、幻觉 token、整段已知正常回答分别报告 |
| 读取变近、变远还是只改变尾部？ | 截断尺度 4/10/32、未截断距离；attention 与实际 W_O V 范数加权版本 | 全部位置，包含稳定读取、平台和没有峰的变化 |
| 是否集中于少数来源？ | 熵、归一化熵、最大权重、top-k 保留质量、来源单位熵与最大份额 | 不用稀疏 top-k 代替完整行计算 |
| 是否突然改变读取对象？ | 完整普通来源分布的相邻 TV、材料/远 history 份额增量 | 无需句子边界或 WAAD 峰；阈值 .05/.10/.20 作敏感性对照 |
| 是否只读取重复文本？ | 当前 query 同 token 的过去来源质量；目标 token 与材料/history 字面相同的质量 | 仅字面匹配，不当作语义正确性 |
| 载体是否被随后复用？ | 完整 history 边；普通来源条件 FAI 与均匀机会富集；实际可用 query 数 | 默认 1–16、17–64、65–末尾、1–末尾；有限窗口主统计要求完整时间覆盖 |
| 回读后是否真的更常复用？ | 进入/未进入 × 复用/未复用的四种状态，全部严格跨层 head 对 | 以载体 b 标签分 N/H；保留失败和空支持，不只画成功路径 |
| 最终幻觉 token 的两跳来源是否不同？ | 每对 head 的材料→载体→目标系数总量；与同位置块内循环移动载体来源份额的对照比较 | **以目标 q+1 的标签**分 N/H，并做位置及 token 类型配对 |
| 具体材料单位的路径是否改变？ | 保存全部材料单位质量；指定 writer/reader 可离线得到每单位的直接与两跳路径曲线 | QA passage、Summary source sentence、Data2txt field；这些只是来源单位，不是预设回答边界 |
| 原生写入如何变化？ | 全 head AV、真实 attention/MLP/residual，固定 observed-vs-runner 方向的带符号读出和舍入余项 | 每个实际 head、每层 MLP 独立保留，不把正向支持等同于事实正确 |
| 变化发生在幻觉起点之前还是之后？ | 起点前后 ±8 token，匹配两侧的完整正常窗口 | 标签仅定位事后窗口；无合格对照时缺失，不补成“无效应” |
| 总体差异是否只是位置、词法、回答构成？ | 同回答、同粗 token 类型、两侧正常位置插值；最近正常位置作次要对照；logprob/entropy/position 控制量 | N/H token、整段正常/含幻觉回答、train/test、三个任务及每个 split 的 ALL |

所有主统计是固定物理 head 或固定 head 对。对样本和 source 汇总不等于对 head 平均。

## 1. 特殊 token 和原生输入

`special_mask = isin(token_ids, tokenizer.all_special_ids ∪ extra_ids)`。
可重复传入 `--exclude-token-id ID` 加入额外明确的 token ID。普通标点和空白不是自动排除对象。

特殊 token 从主要来源分布、候选边、载体、query 和标注目标的比较中排除；被排除的质量和目标数量单独记录。
**不删除模型输入 token，不改 position，不重新对模型 attention 做 softmax。**

令 S 是特殊来源集合。保留原始质量 `A(q,s)`，普通来源的条件分布为：

\[
A^*(q,s)=\frac{A(q,s)\mathbf1[s\notin S]}{\sum_{u\notin S}A(q,u)}.
\]

这是统计口径，不是一次模型干预。分母为零时统计缺失；不会把 BOS-only 的行解释为“完全局部”或“没有熵”。
材料质量、特殊质量和重复 token 质量等绝对量也保存，避免排除后的归一化掩盖实际读取量。

## 2. 一次前向保存什么

复用 `experiments/common/llama_message_intervention.py` 的真实 Llama/GQA/RoPE 前向，无梯度。
共享代码只增加一个可选的 post-RoPE Q/K 观察回调，模型计算不改变。

每样本包括位置 `P−1..N−1`：前者预测第一个 response token，后者可作为已输入的末尾载体。预测统计不使用最后一行。

| 文件 | 内容与用途 |
|---|---|
| `<id>.npz` | 全 head 的完整行统计、所有材料单位质量、显示用 top-k 边、token 文本/坐标/来源/特殊 mask、带符号写入及舍入余项 |
| `<id>.history.npz` | 每层 `L0/L1/...`，形状 `[H,R,R]`；原生完整回答内部 attention，包含初始预测位置列；可重新计算任何后续窗口和目标标签条件 |
| `<id>.qk.npz` | post-RoPE `query_l[H,R,d]`、`key_l[Hkv,N,d]`、缩放、原始计算 dtype；可在 CPU 重建任意 head 的完整 prompt+response 来源行，无需加载模型权重 |
| `<id>.states.npz` | 默认保存：各层回答位置完整 residual、attention、MLP、全部 head 的原生 AV code；全序列原生 KV value；最终回答 residual |
| `<id>.labels.npz` | 独立标签文件；不会传入采集 observer |
| `<id>.audit.npz` | 窗口 FAI/富集/暴露数、载体进入状态、标签、配对坐标、逐 head 对比 |
| `<id>.review.html` | **每个样本**的完整 prompt/response 文本，幻觉标红、正常标绿、特殊 token 标灰；悬停查看坐标 |
| `<id>.review.png`、`.paths.json` | 每任务每 split 的正常和含幻觉展示样本，或 notebook 请求的任意样本/head；路径从目标向前回溯，不要求有峰 |

`R = response_tokens + 1`。Q/K 保存实际值；不同硬件的矩阵乘内核可能有小的舍入差异，因此原生 history 独立保留作核对。
**Q/K 重建覆盖所有 response query 的所有来源，不是仅重建 top-k。**
可加 `--full-attention` 直接保存完整 `[H,R,N]` 原生 attention；一般无需为画 prompt 图开启它。

默认 full states 使后续可以检查原生向量，不需要再为这些位置重跑模型。若磁盘确实不足，可用 `--discard-states`；仍保留 Q/K、history 和全部带符号统计，但完整 value/residual/MLP 向量不再保留。
不会用 PCA、随机投影或归一化后的 W_O W_V 代替原生消息。

带符号量围绕当前已观察 token 相对 runner 的固定方向，包含 head 求和、残差加法与最终读出的舍入余项。
这个方向不是独立的事实真值方向，不能把负 MLP 写入自动命名为证据覆盖。

## 3. 正负比较及位置控制

- token 层面：H=标签 1，N=标签 0，未知=-1；特殊目标和未知目标不混进 N。
- 回答层面：positive=存在已知 H；negative=所有普通 response token 均有已知 N 标签；其余为 unknown。
- 普通读取的 query q 对齐标签 `y[q+1]`。FAI 与四状态表使用载体标签 `y[b]`。
- 目标两跳统计使用 `y[q+1]`，不能与载体标签表混为一个对比。

主配对在同一个回答内，为每个 H 位置寻找 **同 token 类型、最多相距 32 的前后两个正常位置**。
类型仅分空白、包含数字、字母文本、标点，不使用内容正确性分类器。
若 H 在 r，两侧正常位置为 a<r<b：

\[
\Delta g(r)=g(r)-\frac{b-r}{b-a}g(a)-\frac{r-a}{b-a}g(b).
\]

该对照对纯线性位置趋势给出零差异；最近单个正常 token 仍保存为次要对照。
不把单边近邻偏差误报为机制。没有两侧同类型正常位置时，此 H 不贡献主配对，但仍贡献原始 H 描述；覆盖数明确保存。
正常对照可以被复用，因此所有区间与检验在 **source 层面**计算，不把 token 对当独立实验。

起点要求从已知 N 进入 H，且两侧窗口有完整普通已知标签。正常控制窗口须完全正常。
这些起点只是事后比较坐标，不作为内部节点或图边的定义。

## 4. 完整四状态与目标两跳矩阵

四状态使用固定 writer i、reader j、同一个载体 b，且 `layer(i)<layer(j)`。
主“进入”约定是材料条件份额上升至少 .10，local+self 份额下降至少 .10；连续值同时保存。
“复用”约定是主未来窗口中的平均普通来源 attention 至少为均匀普通来源机会的 2 倍。
四种状态的分母为全部具备测量支持的载体位置，包含：

1. 未进入、未复用；
2. 进入、未复用；
3. 未进入、复用；
4. 进入、复用。

空支持保持缺失。由四状态表可得到 `P(reuse|entry)` 与 `P(reuse|no entry)`；它们也以明确字段保存。
这些从 source 平衡表推导的条件率没有另行制造置信区间。

另一条主统计直接从目标回溯：

\[
C_{ij}(q)=\sum_{P\le b<q}A_j^*(q,b)\,E_i(b),\qquad
E_i(b)=\sum_{s\in\mathrm{material}}A_i^*(b,s).
\]

每个 i,j 单独保存，包含所有合法载体，没有取最强路径，没有对 head 平均。
对 C 的 N/H 与配对差异使用最终目标 `q+1` 的标签。
`chain_excess_matched` 另外减去 E 在 16 个位置块内循环移动后的同一对比，检查是否只由来源份额的缓慢变化解释。
这是保留块内轨迹形状的一种描述性错位对照，不是完整的自相关显著性检验。

对指定 i,j，将 E 换成每个材料单位的份额，就得到每个单位的两跳系数曲线；notebook 可以直接查看。
这些乘积描述真实 attention 图的两跳连接，**并不等于真正传递了该事实的向量分量**。

## 5. 汇总与推断

同 source 的多回答先等权汇总，再对 sources 等权汇总。原始 N/H 均值、配对均值、有效 source 数分别保存。
source 层面使用 Student 区间和双侧检验，属于源级均值近似；少于 3 个 source 或零经验方差时不虚构 p=0。
每个 cohort/contrast 的全部 metric/head 或 head-pair 采用 Benjamini–Yekutieli 校正，明确允许 head 间依赖。
不同任务和不同对比仍是不同检验族，不把“搜遍所有表之后挑一个低 p”称为全局验证。

head 与两跳 head-pair 的进一步展示选择只按 train 效应大小固定，再查看 test 的相同 ID。
test 缺失或方向相反仍然保留，不重新选择。相关系数是效应轮廓的一致性，不是 AUROC。
本轮没有新训练器、没有按 test 调整分数方向，也没有声称产生新的检测性能。

## 6. 完整运行及进度

先验证所有 token、prompt 材料对齐和标签覆盖，再加载模型。
可用旧 v2 的原始 trace 核对 token 身份后复用标签，避免再次读大 attention cache。
预检读取标签只是检查与报告覆盖；任何标签、类别或标签决定的选择都不会进入采集 observer。

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph && \
git pull --ff-only origin main && \
conda run --no-capture-output -n research \
  python -u -m experiments.reanchor_flow.attention_audit_run \
  --phase all --split all --task all --samples-per-task 0 \
  --scans experiments/reanchor_flow/outputs/mechanism_all_v3 \
  --labels-from experiments/reanchor_flow/outputs/attention_rhythm_v2 \
  --model /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct \
  --cache /share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876 \
  --source-info /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/source_info.jsonl \
  --device cuda:0 --dtype bfloat16 --query-chunk 32 \
  --output experiments/reanchor_flow/outputs/attention_audit_v3
```

如果旧输出目录名不同，修改 `--labels-from`；也可移除此参数，从原 cache 读取标签。
没有 response 截断参数。若输入 scan 已截断回答，预检直接指出该样本，避免把部分覆盖称为全量。

**只做预检：同一命令末尾加 `--plan-only`。** 它会列出各组样本/token/N/H/未知/特殊数量、未压缩存储估算与当前空闲空间，不加载模型权重。
完整向量和 history 比 v2 的曲线摘要占用更多磁盘。以上一批回答长度计算，仅 history 未压缩约 397.4 GiB；NPZ 实际压缩大小不同，程序不会把估计压缩率当保证。
最大样本的单层 history 约 23.9 MiB，history 按层写入，不在 GPU/CPU 堆积所有层的完整矩阵。

运行中显示：

```text
preflight train/test: 当前输入 / 总输入
join N/H labels: 标签覆盖核对
native audit capture: 当前样本 / 总样本，任务/ID，forward layer=x/L
native audit capture: 当前样本，readout layer=x/L
native audit capture: 当前样本，saving
audit train/QA: 当前 source / 总 sources
```

每个未完成样本一次原生前向；首次仅多一次 8-token 的共享实现校验。不为画图做第二次模型前向，不做逐路由消融。
完成的主 NPZ 最后原子写入；同命令续跑会核对输入/设置并跳过已完成样本。每个样本完成后更新 index。

纯 CPU 离线分析：

```bash
conda run --no-capture-output -n research \
  python -u -m experiments.reanchor_flow.attention_audit_run \
  --phase analyze \
  --output experiments/reanchor_flow/outputs/attention_audit_v3
```

**采集被中断，只评估已完成样本：**

```bash
conda run --no-capture-output -n research \
  python -u -m experiments.reanchor_flow.attention_audit_run \
  --phase analyze --completed-only \
  --output experiments/reanchor_flow/outputs/attention_audit_v3
```

以最终 `<id>.npz` 及配套 history、Q/K（按采集设置还包括 states/full attention）文件判断完成，
不依赖可能尚未来得及更新的 `resumed` 标志。临时文件和未完成样本不参加分析，也不会触发补采。
已有标签会直接复用；仅为实际分析样本补读缺失的标签，不加载模型或 tokenizer。
终端打印每个 split/task 的已完成／计划样本数与 token 数；`summary.md/json` 和 `gallery.html`
明确标为部分结果，JSON 保留跳过样本及缺失文件清单。未采集分组不伪造零效应或 test 验证。
`index.json` 保留完整续跑清单；以后仍可使用原全量命令继续采集。报告会重新生成，原采集数据保留。
不加 `--completed-only` 时发现未完成样本，会在统计之前提示如何分析子集或续跑。
如果没有任何完整样本，明确报错；这些评估仍是正常／幻觉结构比较，不是新检测 AUROC。

可离线调整 `--horizon LO:HI`（可重复，HI=0 为全部可见未来，第一组必须有限）、`--match-window`、`--onset-radius` 与展示数量。
改变排除的 token IDs 或来源定义会改变采集统计，不能把旧统计直接重命名；Q/K、原生 V 和完整来源单位数据为后续重分析保留了原始信息。

## 7. 查看与后续决策

先看 `summary.md` 的正负覆盖与匹配覆盖，再看 `gallery.html`。默认每个 split/task 至多两例完全正常与两例含幻觉回答；不足时如实少显示。
所有样本都有全文标签 HTML，`review_index.json` 可筛选正负类别。
输出目录的 `view_attention_audit.ipynb` 可以切换任意 sample/layer/head、所有指标、幻觉起点曲线和具体来源单位路径，完全离线。

判断顺序：排除特殊来源和位置后，读取变化是否仍在；载体标签与目标标签的两跳差异是否分别成立；具体来源是否能通过文本核查；同一 head/pair 是否在 test 保持方向；最后才讨论事实内容与功能验证。
如果只有原始总体差异而配对后消失，结论应是位置或构成能够解释这一观察。如果条件路径不成立，应报告否定结果，不再追加一个命名更复杂的异常分数。

当前代码还不能自动给出“事实未进入／未整合／被覆盖／读出沉默”四类语义机制标签。完整向量和结构数据为进一步检查保留证据，但独立事实存在性与因果作用仍需专门验证。

## 代码组织与验证

| 模块 | 职责 |
|---|---|
| `attention_audit.py` | 原生采集、Q/K 与完整向量存储、带符号读出及 CPU 重建 |
| `attention_audit_stats.py` | 标签坐标、位置插值、复用暴露、四状态、目标两跳、source 统计与 BY |
| `attention_audit_report.py` | 按 source 流式汇总、全部 head/pair 存盘、train→test 固定 ID |
| `attention_audit_plot.py` | 标红全文、逐 head 原图、总体三列对照、起点与材料单位路径 |
| `attention_audit_run.py` | 全量预检、label 复用、一次前向、进度、续跑与离线入口 |

测试覆盖 BOS 主导、普通质量为零、chunk 不变、未来窗口与 q→q+1 对齐、四种状态、正确载体/层序、source 统计、线性位置反例，以及真实 tiny-Llama 的原生 attention、完整来源 Q/K 重建与写入记账。
完整入口测试覆盖三个任务、两个 split、正负回答、预检不加载模型、续跑不加载模型、修改窗口后的纯离线分析。
这些是计算与流程验证，不是已经跑出的真实 8B 机制结果。
