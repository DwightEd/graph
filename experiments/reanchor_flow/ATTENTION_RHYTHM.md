# 原始 attention 与原生载体审计 v2

**本页保留作旧结果复现。新的全量运行请使用 [正常／幻觉结构审计 v3](ATTENTION_AUDIT_V3.md)。**
v3 排除特殊来源、区分正负、保留完整进入/复用四状态和目标两跳，保存 Q/K 供任意 head 离线重建；不需要为新图再次运行模型。

本入口完成**结构审计与有限原生运算检查**：逐 head 读取 → 同一载体的跨层复用 → 真实消息、残差/MLP 与固定读出。
它不再把模式稀有度当幻觉分数，也不预设“正常一定转向 history”或“WAAD 峰一定是事实重锚定”。
**尚未完成事实语义配对、hub 因果中介或四类 missed-reanchor 的验证；不输出新的机制检测 AUROC。**

本入口现作为结构对照保留。新增 [约束配对与 value 路径确认](ICLR_REVIEW.md) 直接使用固定事实判断，
不把 WAAD 峰作为候选的必要条件；代码与小型先导指令见该文档。两个入口的测量范围不同，不混报为全量事实审计。

## 对最新版实现的审核与修正

| 要求 | 原入口的情况 | v2 的实现及边界 |
|---|---|---|
| 不平均 head | 原始曲线保留 head，但总体耦合只剩跨 head 对均值 | 主统计保存所有实际 head 对，按 source 汇总；local/global 排名只作描述对照 |
| 看完整原图与时间轴 | 有完整来源列，示例曲线局限于展示窗口 | 原图保留完整来源列；另画整个 response 的逐 head 时间轴 |
| 检验正常生成是否 prompt→history | 没有直接检验趋势 | 分开估计全部 token、正常 token、幻觉 token、完全正常回答的逐 head 趋势与来源区间 |
| 读取之后怎样传递 | 同位/邻位 WAAD–FAI 耦合，没有层序限定 | 单独检验同一载体、读取层严格更深的关系；所有 head 参与，不先筛 30% |
| 真实消息与模块运算 | 消息只作为范数权重 | 展示样本可保存两条真实 post-W_O 消息，以及选定节点各层完整残差/MLP 和所有 head code、带符号读出 |
| 正常/幻觉比较 | 只有三个距离量的粗位置匹配 | 加入逐 head 事件率、载体 FAI 与趋势比较；标签在全部采集、候选及审计保存之后连接 |
| 开销与复用 | 无 tqdm；缺少独立分析阶段；cache 覆盖未传给 scan 标签读取 | 各阶段 tqdm；按样本续跑；`--phase analyze` 不加载模型或 tokenizer；修正 cache 覆盖 |

## 完整一键运行

下面覆盖旧扫描中的 **train/test、QA/Summary/Data2txt、全部可用 response token**。以已有 `mechanism_all_v3`
读取相同输入身份；原始 attention 需要重新前向，旧四桶和 PCA trace 无法还原它。使用新目录，避免混用 v1。

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph && \
git pull --ff-only origin main && \
conda run --no-capture-output -n research \
  python -m experiments.reanchor_flow.attention_rhythm_run \
  --phase all --split all --task all \
  --samples-per-task 0 --max-response-tokens 0 \
  --scans experiments/reanchor_flow/outputs/mechanism_all_v3 \
  --model /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct \
  --cache /share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876 \
  --source-info /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/source_info.jsonl \
  --device cuda:0 --dtype bfloat16 --query-chunk 8 \
  --window 10 --future-lo 10 --future-hi 100 \
  --plots-per-task 3 --map-tokens 128 --relay-examples 2 \
  --evaluate --bootstrap 500 \
  --output experiments/reanchor_flow/outputs/attention_rhythm_v2
```

每样本一次无梯度前向采全部 head 曲线。每个 split/task 的 3 个无标签展示样本额外一次前向，**同一次**采原图
和最多 2 个两跳实例；不是对每条路由消融。`--map-tokens 128` 仅限制原图窗口，不截断统计或完整时间轴。
`--plots-per-task 0` 同时关闭原图和载体细节的第二次前向，仍保留所有样本的逐 head 结构统计。
如旧扫描已经截断 response，程序如实报告 `processed/full`，不会把缺失 token 补成已观测。

续跑使用同一命令/目录，只复用配置及 tokens/mask 一致的 NPZ。改变捕获配置用新目录。
已有这次原始 trace 后，重新比较和绘图只需：

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph && \
conda run --no-capture-output -n research \
  python -m experiments.reanchor_flow.attention_rhythm_run \
  --phase analyze --evaluate --bootstrap 500 \
  --output experiments/reanchor_flow/outputs/attention_rhythm_v2
```

分析阶段使用 `index.json` 已冻结的样本范围。首次标签连接需要原 cache，此后复用同名 `.labels.npz`。
也可只跑 `--split test --task QA --sample-id 12693`，但它不能代替总体结论。

## 精确测量与坐标

基础定义对应 [Attention Illuminates LLM Reasoning，§4.1，Eq.7/9/10](https://arxiv.org/html/2510.13554v2#S4.SS1)。
论文式 local/global 分组是跨度排名，本项目主分析保留 head，不能把排名当成事实功能鉴定。

- `distance[l,h,q] = sum_s A[l,h,q,s]*(q-s)`；`waad` 把距离截为 `min(q-s,W)`。因此 1→8→1 的局部变化
  即使完全处在同一个旧 bucket，也能被观察到。
- `fai[l,h,b]` 平均满足 `b+Hlo <= q <= b+Hhi` 的 response-query 对载体 b 的 attention；分母为实际可用行数，
  无未来行是 NaN。FAI 使用未来回答，是离线复用描述。
- `message_waad` 仅是 `A*||W_O V||` 行归一化后的幅值对照，不是事实贡献。**实例中的真实消息保留向量和符号，
  不通过模长决定其支持还是反对生成。**
- 行同时包含 `P-1..N-1`：`P-1..N-2` 预测 response tokens；`P..N-1` 是输入了 response token 的载体行。
  完整时间轴按 token b 对齐 `WAAD(b-1)` 与 `FAI(b)`；同一载体的层序统计则对齐 `WAAD(b)` 与 `FAI(b)`。
  前者是两种观察视角，不能把 teacher forcing 中的相邻 token 自动连成生成因果边。

H0 使用 response 归一化进度上的 OLS slope，保留 `(layer,head)`。原始 prompt/history 占比之外，同时报告
`prompt_mass - P/(q+1)` 的趋势，以均匀来源关注为可见位置数量的对照。它不等于排除了所有位置/词法混杂。
正常 token 子集与完全正常回答分开；少于 3 个可用位置或没有该类样本时为缺失，不补成零。

## 同一载体与两跳实例

先对每个 head 保留整条连续曲线，再用显式峰约定（严格邻居极大，WAAD 相邻突出度至少 0.5；FAI 至少为有效范围的 10%）
描述候选。平台、端点、空峰集不强造事件；此约定不声称复现论文未公开的峰检测源码。

主配对矩阵检验：head `(l,h)` 在位置 b 的读取峰，是否与更深的 head `(k,g)` 对**同一 b** 的后续复用峰对应，`k>l`。
它覆盖所有 head 对，使用完整未来窗口的 FAI 峰，保存各对的来源数、来源等权 lift 和正向来源比例。
null 是相同有效位置和峰数下的均匀占位期望，没有保留时间自相关，所以 lift 不能直接解释为因果性或显著性。

展示实例保留 `s → b → q` 的实际边，满足 `s<b<q`、`l<k`，且 q 有观测到的下一个 token。
第一条边是 b 的最强过去来源；第二条边是该 FAI 窗口中具有观测到的 q+1 的最强后续读取。
每个载体最多展示一个 head 对，再按两条 attention 系数的乘积取展示预算。**该乘积不用于幻觉评分或总体 head 筛选。**
没有合法实例就保存空集合；两跳图只是一条原生计算见证，不是完整 DAG，不把未展示的旁路判为不存在。

对展示实例的 b 与 q，第二次前向保存每层：

1. 完整 residual 输入、post-attention 状态、attention 写入、MLP 写入；
2. **全部 head** 的 pre-W_O 净写入 code（无 PCA/随机投影），以及两条边的完整 post-W_O 消息；
3. 每个节点自己的 observed-next-token − frozen native runner 方向下的逐 head/MLP/残差读出；
4. attention 求和、残差相加和最终读出的有限精度差，单列为 rounding，不归给某条消息。

正的读出值只表示支持该 observed token 相对 runner；observed 可能是幻觉，runner 不一定是真实答案。
某状态中仍有相似方向、某边读出为正，均不能证明某项事实保留或被正确使用。四种证据失败分类仍需语义配对及功能确认。

## 输出、模块与下一步判断

| 输出 | 读什么 |
|---|---|
| `summary.md`、`summary.json` | 完整覆盖数、正常/幻觉数量、逐 head 趋势和 H−N 差异、source bootstrap 区间及未检验事项 |
| `population_<split>_<task>.png` | 每层每个 head 的原始行为、趋势与标签差异；没有跨 head 平均 |
| `head_pairs_<split>_<task>.npz/.png` | 实际写入/读取 head 对，同载体、递增层序，及可用来源数 |
| `gallery.html` | 按 split/task/样本 ID 查阅产物 |
| `<split>/<task>/<id>.timeline.png` | 完整回答逐 head 曲线，事后标注正常/幻觉；读出与载体坐标明确区分 |
| 同名 `.png`、`.full_sources.png` | 近对角原图与全部 source 列，未经 top-k 或重归一化 |
| 同名 `.relay.png`、`.relays.json` | 两跳位置、文本、层/head，以及接收位置的全 head 与 MLP/残差读出 |
| 同名 `.npz`、`.audit.npz`、`.labels.npz` | 无标签原始测量/有限载体状态、派生审计、独立的事后标签；契约见 SCHEMA.md §13 |

`attention_rhythm.py` 负责流式测量；`attention_relay.py` 负责两跳选择与原生状态检查；`attention_rhythm_report.py`
负责统计；`attention_rhythm_plot.py` 负责绘图；`attention_rhythm_run.py` 只编排数据、阶段和续跑。
它们复用 `experiments/common/llama_message_intervention.py`，没有第二份 Llama 前向。

先判断原图中是否有可重复形态、H0 是否成立、同载体层序关系是否普遍，以及 H/N 差异是否超出粗位置对照。
有形态后再研究其中的内容与因果使用；本入口不强行输出一个新幻觉分数。

## 验证与成本边界

```bash
python -m pytest -q \
  experiments/reanchor_flow/tests/test_attention_rhythm.py \
  experiments/reanchor_flow/tests/test_attention_rhythm_entrypoint.py \
  experiments/reanchor_flow/tests/test_attention_relay.py \
  experiments/reanchor_flow/tests/test_attention_rhythm_pipeline.py
```

测试包括密集公式、chunk 不变性、同桶锯齿反例、FAI 缺失、两种 token 对齐、非法层序拒绝、可见 history 数量对照、
实际 head ID/source 平衡，以及真实 tiny Hugging Face Llama 的消息/读出和全任务全 split 流程。
本次验证为 **38 passed**，其中包含“attention 非零但 V 消息为零”的对照，以及缺失未来不污染其他有效位置的比较。
**本地是 CPU tiny-Llama 验证，没有运行服务器的 8B/RAGTruth，不能据此承诺机制成立、检测效果或任意长度下 24GB 不溢出。**

GPU 只保留当前层与当前 query chunk 的 attention；单样本曲线为 O(LHT)，示例状态约为 O(LKD + LHKd_head)，
K 为少量展示节点。每样本 head-pair 分析有 O((LH)^2) 的 CPU 输出；总体按 source 流式汇总，不常驻所有样本的配对矩阵。
