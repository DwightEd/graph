# 消息 DAG：来源、聚合与同目标路径贡献

本目录是新的核心实现。研究主线是 **消息路由模型 → 图上的生成结构 → 正常／幻觉差异**。
不通过 WAAD 峰挑节点，不训练 GNN，不把某个高流量节点或低材料分数直接命名为幻觉机制。

## 目录与依赖

| 文件 | 单一职责 |
|---|---|
| `cache.py` | 读取既有 v3 缓存；划分材料单位和其他边界来源 |
| `operators.py` | 原生 attention/value、残差与 SwiGLU 条件算子，及其伴随算子 |
| `graph.py` | 来源前向传播、独立目标反向传播、逐边贡献与节点平衡 |
| `structure.py` | 从完整图读出来源、复用、集中和抵消；不把这些量当模型本身 |
| `report.py` | 标签事后连接、source 等权 N/H 比较、逐 head 统计及固定检测对照 |
| `view.py` | 同目标图、来源/head 选择、带标签全文和节点跨层轨迹 |
| `run.py` | 小批／全量入口、成本估算、按目标保存与续跑 |
| `view.ipynb` | 按样本编号和目标即时查看已完成图，无需模型/GPU |

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
“局部循环”只能解释为 DAG 上持续向前复用；图中不存在真实有向环。
高流量、低 prompt 读取也可出现在正常压缩中；“捷径未约束”“约束被覆盖”等语义机制还要独立验证。

报告包含 source 等权 N/H 均值、同回答／同粗 token 类型／32 token 内两侧正常位置插值、区间与 BY 校正。
train/test 相同物理 head 的同向重复性另外报告；稀疏目标预算常缺少配对支持，不能解释为零效应。
固定 `material_deficit` 与直接边界读出、logprob、位置的 AUROC/AP 保留为诊断，不作为图模型的定义。
来源细分会影响绝对值总和、HHI 和抵消，跨任务比较须同时考虑材料单位数，不以 pooled AUROC 代替分任务结论。

## 5. 运行与成本

先用已完成 QA 缓存运行小批。默认每 split/task 一个样本、每样本四个普通目标；选择不读取标签。
下面选择各个已有 split 的 QA 两个样本，程序在前台打印来源传播层、目标反向层、耗时与每个完成文件：

```bash
conda run --no-capture-output -n research \
  python -u -m experiments.reanchor_flow.message_dag.run \
  --audit experiments/reanchor_flow/outputs/attention_audit_v3 \
  --completed-only --split all --task QA \
  --samples-per-group 2 --targets-per-sample 4 \
  --device cuda:0 --source-chunk 2 --target-chunk 2 --query-chunk 8
```

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

只看计算计划加 `--plan-only`。它不读取权重张量，不运行 LLM。
所有任务／split／普通目标用同一入口；两个 0 表示全部，不按事件挑目标：

```bash
conda run --no-capture-output -n research \
  python -u -m experiments.reanchor_flow.message_dag.run \
  --audit experiments/reanchor_flow/outputs/attention_audit_v3 \
  --completed-only --samples-per-group 0 --targets-per-sample 0 \
  --device cuda:0 --source-chunk 2 --target-chunk 2 --query-chunk 8
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

默认输出 `AUDIT/message_dag/`：

| 路径 | 用途 |
|---|---|
| `index.json` | 原生来源、选中样本、计划 target 位置、配置和临时空间估算 |
| `samples/<split>/<task>/<id>/meta.npz` | 文本、坐标、特殊 mask、source unit IDs 与读出控制量 |
| 同目录 `target_<t>.npz` | 单目标图、来源／节点／head 贡献、误差和结构读出 |
| 同目录 `labels.npz` | 独立事后标签，可与图一起下载 |
| `gallery.html`、`summary.md/json` | 总体结构图、N/H 覆盖、逐 head 图和固定检测诊断 |
| `cohorts/*.npz` | 完整逐 head N/H 统计，不仅是示例图 |
| `view.ipynb` | 选择样本和已完成目标，按来源／head 查看图 |

每层最多 `--edge-budget` 条边用于显示。**全边参与传播和统计，NPZ 不保存全部 G×H×R² 边标量**；
保存的完整节点统计和稀疏边明确分开，并报告各来源显示覆盖。任意未存边的精确值仍需原缓存与算子重算，
不能从稀疏显示图推断“这里没有连接”。目标 head 的完整总量始终在 NPZ 和 cohort 中。

报告可完全离线，不需要原模型或大缓存：

```bash
conda run --no-capture-output -n research \
  python -u -m experiments.reanchor_flow.message_dag.run \
  --phase evaluate \
  --output experiments/reanchor_flow/outputs/attention_audit_v3/message_dag
```

`--phase trace` 只构图；`all` 会在构图完成后复用原数据接口连接标签，再生成报告。
正常的 v3 采集已有标签；缺失时不需要先完成旧 `audit train/QA` 配对计算。
换 `--mlp-rule up` 用另一 `--output`，比较来源与节点结论是否依赖分配规则。

## 6. 当前验证范围

测试覆盖原生 tiny Llama 的 float32/BF16 重构，GQA 伴随双线性关系，源／汇及内部节点平衡，材料单位可加性，
目标块不混合读出、未来不影响当前目标、显示预算不改变结果，以及中断／续跑／无大缓存离线评估。
反例用 attention 完全相同的两份真实小模型计算，检查 MLP 转换是否让材料分量进入后续 reader 的 value 子空间。
这些是算法实现验证。没有真实 8B 的新机制发现或性能增益结论。
若回答由其他模型生成，归因描述的是当前捕获模型对该序列的条件计算，不是原生成模型的计算过程。
