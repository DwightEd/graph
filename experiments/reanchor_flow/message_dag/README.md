# 消息 DAG：来源、聚合与同目标路径贡献

本目录是新的核心实现。研究主线是 **消息路由模型 → 图上的生成结构 → 正常／幻觉差异**。
不通过 WAAD 峰挑节点，不训练 GNN，不把某个高流量节点或低材料分数直接命名为幻觉机制。

**当前方法与九项原始研究的设计对应见 [TRANSPORT_DESIGN.md](TRANSPORT_DESIGN.md)。**
`event_run` 默认扫描所有可用样本和内部事件，远处来源包括旧回答。v2将0/1/2+跳进一步展开为
事件→真实中继→目标的有符号V/K边，并验证所有路径按最后跨位置边只计一次。
新增固定、无标签的 `opposition` 候选分数，以及同token上的直接路径/V-only/置信度/位置对照AUROC/AP。
默认输出 `AUDIT/lookback_events_v2`；没有此前事件的位置不填0，条件覆盖率与检测结果同时报告。
完整物理边流式保存，HTML只裁剪显示；计算和存盘不做top-k筛选。
原事件定义与v1审计说明保留在 [LOOKBACK_EVENTS.md](LOOKBACK_EVENTS.md)。
续跑已有 `lookback_events_v1` 请加 `--legacy-v1`，保持原算法并跳过已保存事件；默认v2另存。
进度与NPZ用独立临时文件保存，同一输出目录只允许一个运行中的写入进程（包含离线evaluate）。
执行开销、按层共享、CPU/GPU分工和真实计时见 [PERFORMANCE.md](PERFORMANCE.md)。
默认以1 GiB CPU状态预算共享层算子；`--profile`报告阶段耗时和CUDA峰值显存，兼容已有事件续跑。
下文 `run` 保留原有来源加法分摊；新事件实验不使用该分摊规则解释 MLP 导数。

## 目录与依赖

| 文件 | 单一职责 |
|---|---|
| `cache.py` | 轻量读取既有 v3 缓存、补齐控制 token mask；划分材料单位和其他边界来源 |
| `selection.py` | 内容目标与 N/H 覆盖预检，冻结比较样本和同回答对照 |
| `operators.py` | 原生 attention/value、残差与 SwiGLU 条件算子，及其伴随算子 |
| `graph.py` | 来源前向传播、独立目标反向传播、逐边贡献与节点平衡 |
| `structure.py` | 从完整图读出来源、复用、集中和抵消；不把这些量当模型本身 |
| `report.py` | source 等权 N/H 比较、逐 head 统计；均匀抽样模式另输出固定检测对照 |
| `view.py` | 同目标图、来源/head 选择、带标签全文和节点跨层轨迹 |
| `run.py` | 小批／全量入口、成本估算、按目标保存与续跑 |
| `view.ipynb` | 按样本编号和目标即时查看已完成图，无需模型/GPU |
| `events.py`、`event_run.py` | 不使用标签的全量内部回看扫描、传播调度、续跑与覆盖清单 |
| `differential.py`、`event_trace.py` | 原生 RMSNorm、QK/OV、SwiGLU 局部导数和0/1/2+跳消息响应 |
| `transport.py` | 共享同位置后缀读出、最后跨位置V/K边、完整流式NPZ和逐目标闭合 |
| `transport_report.py` | 无标签分数、条件覆盖、同token/source配对的AUROC/AP与对照 |
| `event_report.py`、`event_view.py` | 离线 source 等权比较、原始 token 与逐事件传播界面 |

缓存和读出工具复用父目录的 `attention_audit`、`message_lineage`；统计复用 source 分组和 tied-score AUROC/AP 实现。
父目录旧实验暂作历史复现与数值对照，不再新增方法分支。新算法在本目录维护。
不删除已有输出或原始采集代码；后者仍负责一次 teacher-forced LLM 前向采集。

## 1. 图和边算子

节点 `v=(token position, layer, stage)` 保留 input residual、post-attention 和下一层 input。
一个 head 的边是

\[
B_{(l,h,s\to q)}=W_{O,l,h} A_{l,h}(q,s) W_{V,l,g(h)} N_l(s).
\]

`g(h)` 是真实 GQA 映射。这里的 `N_l(s)` 使用原生 RMSNorm 分母；attention 固定在已观察序列上。
代码从不显式构造 D×D 矩阵，也不先平均 heads。残差边为恒等算子。
材料向量通过 WV 可以被保留、旋转或消去；两跳 attention 乘积无法替代该运算。

SwiGLU 的参考归一化输入为 z，g=Gz，u=Uz，a=SiLU(g)，k=a/g（g=0 时 k=1/2）。
对归一化前的来源分量 p，用

\[
M(p)=D\left[\tfrac12 a\odot U(Np)+\tfrac12 u\odot k\odot G(Np)\right].
\]

它在线性分配意义下重构参考 MLP。`--mlp-rule up` 使用固定 gate、只分配 up 支路，供规则敏感性比较。
这些不是任意新输入下的等价 Transformer，也不是删边后的模型；Q/K 的来源归因未包含。
线性分配／伴随传播本身不能仅凭“公式新写了一遍”宣称创新；研究贡献要看该图是否读出了可复现、具有内容区分力的结构。
乘积分配与固定 attention 归因的相关依据见 [AttnLRP](https://arxiv.org/html/2402.05602v2)。

## 2. 来源传播与边界

每个材料单位独立成组，另保留 `other_prompt`、`special`、`response_initial`、`rounding`。
未分配 unit ID 的材料仍作为独立的 `material:-1`，不丢弃。

\[
x_v^{(c)}=b_v^{(c)}+\sum_{u\to v}B_{u\to v}x_u^{(c)}.
\]

所有来源之和重构原生状态。原生 BF16 residual add 舍入和算子回放残差作为显式 `rounding` 注入，
不归给材料；模块回放误差与残差加法舍入分别处理。

当前是 **response 计算子图**：每层 prompt V 作为外部边界，response 初始 embedding 作为初始来源。
`P-1` 参与第一个回答的预测，但作为被读取位置时仍是 prompt 边界，不伪装成 response 中继。
prompt V 已含上下文混合，所以材料单位表示边界位置来源，不是纯净事实编码。
`response_initial` 不等于错误历史或参数先验。相关材料的语义身份仍需独立来源对应／约束实验。

## 3. 同一个目标的全部路径

目标 token t 使用 predictor q=t-1，固定 observed-vs-native-runner 的最终读出方向 d_t：

\[
\lambda_T=d_t,\quad
\lambda_u=\sum_{u\to v}B_{u\to v}^{T}\lambda_v,\quad
C_{u\to v}^{(c,t)}=\lambda_v^{T}B_{u\to v}x_u^{(c)}.
\]

所有后续路径由 DAG 动态传播合成，不枚举路径，也不逐边消融。
每个 target 保持自己的 adjoint，目标分组只共享层权重、缓存解压和来源 WV 计算，**不把多个目标求和**。
这是条件算子的伴随，不训练参数，也没有调用 autograd/backward。

带符号节点平衡为

\[
\lambda_v^T b_v^{(c)}+\sum_{u\to v}C_{u\to v}^{(c,t)}
=\sum_{v\to w}C_{v\to w}^{(c,t)}+\text{target exit}_v.
\]

输出保存每个节点的平衡误差、每个来源的源边界贡献与目标出口贡献。
**绝对通量不守恒**；沿所有边求和会按路径长度重复计数，不能当作一次最终输出贡献。
原生 logit 舍入与固定最终读出方向下的分解也不是同一个数值对象，闭合针对后者。

## 4. 从图中看什么

先记录完整节点／边计算，再比较 N/H：

| 输出 | 含义 |
|---|---|
| `node_input`, `node_post` | 每个来源×层×位置，对同一目标的净贡献 |
| `node_out_absolute` | 节点出边的绝对贡献总量，可定位高通量／抵消；非概率 |
| `mlp_credit` | 指定来源经过该 MLP 转换边的同目标贡献 |
| `head_boundary` | 每个实际 head 从 prompt 边界向普通接收状态注入的贡献 |
| `head_relay`, `head_relay_absolute` | 每个实际 head 的普通严格过去 response 中继净／绝对贡献 |
| `carrier_absolute` | 材料或其他来源通过该 carrier 向后读取边传播的绝对贡献 |
| `root_position_credit` | prompt 边界和初始状态按实际位置的入口贡献；数值注入另列 |
| `edge_index`, `edge_source_credit` | 显示边的 (layer,head,source,receiver) 及每个来源贡献 |
| `edge_attention`, `edge_value_norm` | 显示边的原始 attention 与原生 WV 输出范数，不代替有符号贡献 |

六个紧凑结构读出分别是：材料输出有符号份额、材料中继绝对份额、局部中继份额、carrier 通量 HHI、
中继正负抵消、材料单位出口绝对贡献 HHI。份额与 HHI 没有被宣称为事实真值或路径概率。
它们用于定位差异，不能代替来源×节点×目标的图数据。无材料／无中继等空支持为缺失。

节点不需要出现回看峰。特殊输入仍参与原生计算和平衡，但不会成为展示中继、材料根或 N/H 目标。
schema 2 会在原 capture mask 上补入已知 Llama 控制 token 的完整拼写，包括旧 tokenizer 元数据漏标的
`<|start_header_id|>`、`<|end_header_id|>`；原始 token ID、位置和状态文件不变。
控制 token 的消息统一进入 `special`，随后按该来源重新传播，不只改展示标签。
普通标点和 `assistant` 等角色文字不自动当成控制 token。新采集同时读取 tokenizer 的 special added-token 元数据。
“局部循环”只能解释为 DAG 上持续向前复用；图中不存在真实有向环。
高流量、低 prompt 读取也可出现在正常压缩中；“捷径未约束”“约束被覆盖”等语义机制还要独立验证。

报告包含 source 等权 N/H 均值、同回答／同粗 token 类型／32 token 内两侧正常位置插值、区间与 BY 校正。
train/test 相同物理 head 的同向重复性另外报告；稀疏目标预算常缺少配对支持，不能解释为零效应。
均匀抽样模式保留固定 `material_deficit` 与直接边界读出、logprob、位置的 AUROC/AP 诊断，不作为图模型的定义。
配对模式用标签挑审计案例，不能用这批富集目标估计总体检测表现，因此不计算 AUROC/AP。
来源细分会影响绝对值总和、HHI 和抵消，跨任务比较须同时考虑材料单位数，不以 pooled AUROC 代替分任务结论。

## 5. 运行与成本

要比较正常与幻觉，先运行下面的配对小批。每个已有 split 的 QA 最多八条回答：四条全正常、四条含幻觉。
程序先仅解压文本、坐标和标签等小字段，打印 N/H 回答数、内容目标数、可配对来源数和待修复控制位置数；
缺少任一类的 split/task 跳过，全部缺乏支持则在读取模型配置／权重之前报错。
预算内按来源 ID 均匀选择，每个来源每类至多一条回答，选择过程不读取图贡献。

```bash
conda run --no-capture-output -n research \
  python -u -m experiments.reanchor_flow.message_dag.run \
  --audit experiments/reanchor_flow/outputs/attention_audit_v3 \
  --completed-only --split all --task QA \
  --selection paired --samples-per-group 8 --targets-per-sample 6 \
  --device cuda:0 --source-chunk 2 --target-chunk 2 --query-chunk 8 \
  --output experiments/reanchor_flow/outputs/attention_audit_v3/message_dag_compare_v2
```

这里的“内容”是公开的词法筛选：拼接原回答文本，取实词／数字的首 token，跳过常见功能词、回答套话、
纯空格、纯标点、列表编号和显式 passage/source 引用编号。不把词内的 `cher` 当作 `Butcher` 的词首。
这不是事实识别器，也不保证选到的每个词都承载待检验事实；界面和 index 保留实际文本供核查。
不删除输入中的这些普通 token，它们仍参与模型和图计算，只是不作为本批输出目标。

含幻觉回答中的每个 H 内容目标，预先选同回答、同粗 token 类别、两侧各 32 token 内的最近正常内容目标，
组成 N-H-N 三元组。六目标预算最多选两组三元组；共享正常对照只计算一次。全正常回答另选六个内容目标。
可用来源不足四个时等量缩小两类；没有两侧内容对照的 H 不进入本次配对小批，并在预检中报告覆盖缺口。
这是局部比较设计，不代表所有类型幻觉；统计先在回答和来源内汇总，不能把重复对照当独立样本。

第一次运行即将样本 ID、目标和对照写入 `index.json`。重复同一命令使用原计划；即使后来又采集了新样本，
也不重新选择。标签在每条回答开始构图前复制到输出，保证中断后的离线报告可用。
某个对照图未完成时，报告记为 `missing_pairs`，不重新匹配另一个对照。
标签用于选择审计案例与比较，**不进入来源传播、伴随传播或边算子**。

终端分开显示 `comparison inventory`、每样本计划、来源传播层、目标反向层和完成文件，
最后打印 `N=… H=… matched=…`。报告同时列出内容目标数量和缺失图数量。
先看同回答配对 H−N 和逐 layer/head 结果，再看跨回答的原始 N/H 均值，避免把位置差直接解释成机制。

若原生采集被中断，test/QA 可能尚未完成。`--completed-only` 仅跳过缺失缓存，仍遵守 split/task 筛选；
不会自动把 train 样本当 test。先看实际可用范围：

```bash
conda run --no-capture-output -n research \
  python -u -m experiments.reanchor_flow.message_dag.run \
  --audit experiments/reanchor_flow/outputs/attention_audit_v3 --list-available
```

它按 split/task 列出已完成／计划数量，即使全部尚未完成也能列出；不读取模型配置或权重，不生成图。
构图只要求 `.npz`、`.history.npz`、`.qk.npz`、`.states.npz` 四个文件，不要求旧审计完成或保存可选的 `.attention.npz`。
显式指定无可用样本的范围时，错误会列出其他已完成范围，不悄悄切换数据。

只看计算计划在上述命令末尾加 `--plan-only`。它不读取权重张量，不运行 LLM；缺失标签时可复用原标签接口补齐。
另保留 `--selection uniform`（默认），用于不依赖标签的抽样或完整固定评估；两个 0 表示全部普通目标：

```bash
conda run --no-capture-output -n research \
  python -u -m experiments.reanchor_flow.message_dag.run \
  --audit experiments/reanchor_flow/outputs/attention_audit_v3 \
  --completed-only --selection uniform --samples-per-group 0 --targets-per-sample 0 \
  --device cuda:0 --source-chunk 2 --target-chunk 2 --query-chunk 8 \
  --output experiments/reanchor_flow/outputs/attention_audit_v3/message_dag_uniform_v2
```

先看小批真实耗时再安排全量。当前环境没有用户的真实 8B/GPU，不能承诺全量时长。
每样本来源传播一次，每个目标各有一次伴随遍历；按目标块共用层加载。
临时 source tape 为 `2 × L × G × R × D × 4` bytes（float32 input/post）；逐样本处理后删除。
计划也打印派生数组未压缩体积和层加载次数估计；压缩率与实际耗时需看小批运行。
CPU 当前来源数组为 `G × R × D × 4` bytes；GPU 分来源块处理大 MLP 激活。
边贡献分块张量还随 `G×H×query_chunk×R` 增长，source_chunk 并不是总 GPU 内存的硬上限。
需要更多临时空间可用 `--scratch /有空间的目录`，不保存所有路径的向量副本。

目标文件各自原子保存。重复同一命令跳过完成目标；中断后只需为尚未完成目标重新生成该样本的临时 tape，
不重跑 LLM、不重新做旧版全部 head 对审计。已有 Python 进程不会因 git pull 自动更换实现。

默认输出 `AUDIT/message_dag_v2/`；上述配对命令显式使用 `message_dag_compare_v2/`。
旧 schema 1 的 `message_dag/` 不能与新结果混用。复用四类完整原生缓存，重新传播所选目标即可，**不重跑 LLM 采集**。
若只有旧派生图，无法补算内部来源分量；原输出继续保留作对照。

| 路径 | 用途 |
|---|---|
| `index.json` | 原生来源、固定样本与 N-H-N 目标、覆盖清单、配置和临时空间估算 |
| `samples/<split>/<task>/<id>/meta.npz` | 文本、坐标、原／修复特殊 mask、修复位置、source unit IDs 与读出控制量 |
| 同目录 `target_<t>.npz` | 单目标图、来源／节点／head 贡献、误差和结构读出 |
| 同目录 `labels.npz` | 独立标签，可与图一起下载；不作为图输入 |
| `gallery.html`、`summary.md/json` | 总体结构图、N/H／配对覆盖、逐 head 图和实际抽样方式 |
| `cohorts/*.npz` | 完整逐 head N/H 统计，不仅是示例图 |
| `view.ipynb` | 选择样本和已完成目标，按来源／head 查看图 |

每样本至多 24 个目标时，HTML 为每个目标生成页面，优先打开 H 内容目标，再选非首位置的内容目标；
更多目标仍可在 notebook 中查看。notebook 从实际可用数据设置初始 split/task，不再固定要求 test。

每层最多 `--edge-budget` 条边用于显示。**全边参与传播和统计，NPZ 不保存全部 G×H×R² 边标量**；
保存的完整节点统计和稀疏边明确分开，并报告各来源显示覆盖。任意未存边的精确值仍需原缓存与算子重算，
不能从稀疏显示图推断“这里没有连接”。目标 head 的完整总量始终在 NPZ 和 cohort 中。

报告可完全离线，不需要原模型或大缓存：

```bash
conda run --no-capture-output -n research \
  python -u -m experiments.reanchor_flow.message_dag.run \
  --phase evaluate \
  --output experiments/reanchor_flow/outputs/attention_audit_v3/message_dag_compare_v2
```

`--phase trace` 只构图；配对选择始终需要标签。`all` 在构图后生成报告。
正常的 v3 采集已有标签；缺失时不需要先完成旧 `audit train/QA` 配对计算。
换 `--mlp-rule up` 用另一 `--output`，比较来源与节点结论是否依赖分配规则。

## 6. 当前验证范围

测试覆盖原生 tiny Llama 的 float32/BF16 重构，GQA 伴随双线性关系，源／汇及内部节点平衡，材料单位可加性，
目标块不混合读出、未来不影响当前目标、显示预算不改变结果，以及中断／续跑／无大缓存离线评估。
schema 2 另验证 header 归属在完整传播中的变化、词首筛选、全正常数据在 GPU 前拒绝配对、
N/H 小批完整运行、固定选择续跑和对照缺失不重新匹配。
反例用 attention 完全相同的两份真实小模型计算，检查 MLP 转换是否让材料分量进入后续 reader 的 value 子空间。
这些是算法实现验证。没有真实 8B 的新机制发现或性能增益结论。
若回答由其他模型生成，归因描述的是当前捕获模型对该序列的条件计算，不是原生成模型的计算过程。

上传的八目标 pilot 及可离线核对的修复前后比较见 [PILOT_REVIEW.md](PILOT_REVIEW.md)。
