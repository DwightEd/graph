# 原生路由与多观测切换状态

## 统一模型候选

```bash
git pull --ff-only origin main
python -u main.py support --stage model \
  --output outputs/native_support_ragtruth4
```

复用 `route_filter_v3/features/` 的小缓存；仅在它尚不存在时提取原始 NPZ，不载入模型权重。
详细架构、公式、假设和文献见 [JOINT_STATE_DESIGN](../../iclr/JOINT_STATE_DESIGN.md)。

| 输入/状态 | 在同一个模型中的作用 |
|---|---|
| 功能路由 R | 提供已有实测支持的风险方向；输出为其潜在均值 |
| attention 路由 A | 与 R 联合判断读取状态是否仍可由此前状态解释 |
| log(1+熵) | 帮助判断统计状态改变，不直接作为正向风险或开关 |
| 可能的段长度及其后验 | 决定各段历史对当前状态估计的影响 |
| 完整 3×3 协方差 | 处理观测相关性，避免当作独立证据相乘 |

新段与延续分支按预测概率竞争，参数积分为 Student-t 预测分布。
`--window 16` 在模型里表示先验期望段长，同时是普通均值对照的窗口；不截断模型段长。
保留所有可能起点，首行强制新段仅是初始化。状态切换不代表语义 reanchor。

新输出 `joint_state_v4/w16/`，不会修改 v1/v2/v3 结果：

| 文件 | 内容 |
|---|---|
| `report.html` | AUROC/AP、同答增量、逐词风险、切换概率及新段/延续贡献 |
| `evaluation.json` / `comparisons.json` | 同 token 的完整评价和固定对照差值 |
| `scoring_protocol.json` | 评分前冻结的参考均值/协方差、参考 source/回答 ID 及模型假设 |
| `responses/0000/scores.npz` | 全部检测分数；`run_posterior[t,n-1]`、三维 `state_mean` 等 |
| `tokens.csv` / `onsets.csv` / `recovery.csv` | 逐词轨迹、起点及错误结束后的正常词排名 |

固定主候选 `joint_state`；主要比较基线 `route_mean` 已在四答达到 AUROC 0.779200。
`route_state` 用完全相同的概率结构但只观察 R，检验 A/熵的贡献。
`instant_state` 使用同一参考先验逐词独立更新，检验时间推断的增量。
原 R、A、熵均保留，没有根据测试标签调整融合权重或自动选择方法。

默认每个目标 source 的参考统计只用其他 source，各 source 等权，标签不进入统计。
这是对现有四答的 source-held-out 诊断，不是外部独立训练。只有一个 source 时会明确报错。
已有独立参考缓存可追加 `--reference-output outputs/native_support_reference`；参考的模型、
task、generator 与来源定义必须一致，同 source 仍排除。无需额外神经网络训练。
新模型输出为路由状态，不是幻觉概率；尚未得到自然数据 AUROC，不保证超过普通均值。

`--stage evaluate` 会优先评价对应窗口已完成的 v4 模型结果；缺少标注时保留旧有效评价。
`--stage optimize`/`score`/默认 `run` 继续使用 v3；`--stage compare` 保留 v2。

## v3 原生路由：已有比较

固定 `routing_imbalance` 基线，以近期相似逐头路由状态加权平均原分数，检验时间降噪。
唯一新候选为 `route_state_filter`；同窗口普通均值和先合并 head 的滤波是必要对照。
不消融、不反向传播、不训练关系分类器；自然标签只在全部分数保存后用于评价。
[v3 公式及依据](../../iclr/ROUTE_FILTER_DESIGN.md)。四答已报告逐头滤波 AUROC 0.777385，
普通均值 0.779200；未显示逐头加权的总体 AUROC 优势，不宣称机制成立。

## 已有四答结果直接重算

```bash
git pull --ff-only origin main
python -u main.py support --stage optimize \
  --output outputs/native_support_ragtruth4
```

首次读取旧 NPZ 中的 attention、value energy 及必要标量，提取紧凑的逐头分区表；
之后只读小缓存，不重算 collapse、Gram、跨来源校正或旧 signed support。
已存在 v2 的 source_regions 时直接复用，否则只载入 tokenizer 恢复官方来源区间。
不载入模型权重。输出在 `outputs/native_support_ragtruth4/route_filter_v3/w16/`。
原始 NPZ、v1 和 v2 文件保留。`score` 与 `optimize` 同义；`compare` 保留原 v2 比较。

四答已实测的基线为 AUROC 0.754983/AP 0.199373，旧支持传播为 0.572333/0.103939。
不能保证滤波提高它；这次输出会直接给同样本差值，阴性结果也保留。

## 新版方法和输出

每个物理 layer/head 保存 source、严格历史、response self、other 的消息范数，
以每层全部头的总范数归一为 head×region 联合分布；全零层单列 inactive。
比较逐层联合分布的 Hellinger 距离，给当前及此前最多 15 个分数加权。
距离尺度只从已观察到的邻接行估计；状态相同时严格等于普通因果均值。
状态变化大可降低旧分数权重，但不保证首错无延迟；分区相似也不代表来源地址或语义相同。
图没有新增风险递推或因果边，滤波权重只是检测器对已测路由分数的平均权重。

| 方法 | 用途 |
|---|---|
| `routing_imbalance` | 原始功能消息路由基线 |
| `route_mean` | 同窗口普通因果均值，检验单纯平滑 |
| `route_state_filter` | 唯一新候选；保留头身份的状态滤波 |
| `route_pooled_filter` | 合并头后的同一滤波，检验 head 身份是否有增量 |
| `attention_displacement` | 读取路由对照 |
| `entropy` | 当前不确定性对照；不作为检测开关 |

没有官方来源块时，上表两种原始路由改用明确命名的 prompt 版本，不冒称事实证据。

`route_filter_v3/w16/` 中：

| 文件 | 内容 |
|---|---|
| `report.html` | 各法 AUROC/AP、同答比较、增量及逐词轨迹 |
| `evaluation.json` | 全错误、首错、span 起点、延续、前后半段；pooled/来源等权/同答 |
| `comparisons.json` / `comparisons.csv` | 候选相对原路由、普通均值及合并头对照的差值 |
| `onsets.csv` / `high_risk_normals.csv` | 每个起点及高风险正常词的同答排名 |
| `recovery.csv` | 错误后第一个正常 token 的同答正常词排名，检查旧高分滞留 |
| `tokens.csv` | 逐词分数、当前权重、有效平均词数和距离尺度 |
| `responses/0000/scores.npz` | 完整分数和按 lag 保存的滤波权重；lag=0 为当前 |
| `scoring_protocol.json` | 标签读取前保存的固定规则及来源协议 |
| `summary.json` / `evaluation_status.json` | 本次是否完成评价，不自动宣布获胜方法 |

紧凑缓存位于 `route_filter_v3/features/0000.npz`，保存 head_profile 的 `[T,L,H,5]` 表和原路由。
默认窗口 16 来自历史代码默认值；不是在这四答上调出来的最优值。显式 `--window 8` 写 w8，
不会覆盖 w16，也不会重新读大缓存。不同窗口不是默认搜索网格，不得根据同一测试集选最优窗口。

优先看 `route_state_filter − routing_imbalance` 和 `route_state_filter − route_mean`。
若只有总体增益但首错/恢复恶化，应报告时序滞留；不能把延续分数改善称为机制发现。

## 历史 v2 比较的判读

`--stage compare` 仍计算原来的 13 方法，写 `route_comparison_v2/`。
[v2 公式与历史来源](../../iclr/NATIVE_ROUTE_REDESIGN.md)。它不再随默认 run/score 重算。

1. 先看 `report.html` 的同样本方法表。主方法是 `routing_imbalance`；没有官方来源区间时，
   明确改名为 `prompt_routing_imbalance`，整个 prompt 不能冒充已核验的事实证据。
2. 看 `evaluation.json` 的全错误、span 起点、每答首错、延续及前/后半段。
   每项同时给 pooled、`source_balanced` 和 `within_answer`；只有两类都有的回答参与同答 AUROC。
   `weighted_prevalence` 对应来源等权 AP 的基准。四答只有两个首错，不能据此认证普适能力。
3. 看 `onsets.csv` 中每个首错/起点相对同答正常词的排名，和 `high_risk_normals.csv` 中的排序反例。
   没有阈值，不把高分正常词直接称为已发生的误报。
4. 比较 `functional_collapse` 与 `attention_collapse` 才能讨论 value 范数带来的增量。
   `_offline` 两列恢复历史完整 prompt/最终回答长度协议，仅作离线对照；不能叫在线检测。
   因果版本只使用普通 prompt、已观察到的位置和前 3 行。所有结构校正按 source 分开拟合、
   校准和评价，不读标签；不足三个来源时缺测，四来源结果仍只用于诊断。
5. `focus_*_write` 将读取峰窗口与同一头的正负写入对齐；FFN/残差账本仍保留。
   它们用于解释，未按这四答标签拼权重或选择最佳头；支持实际输出不等于事实支持。

消息路由使用 `sqrt(edge_value_energy) = ||A W_O V||`，不能用平方能量直接替代。
结构先保留头身份再算来源支持、头 Gram 有效秩和历史锚点支持，衡量路由冗余而非证明因果协同。
所有方向固定为高分高风险。`support_graph` 与 `direct_prompt` 保留原公式作为 v1 对照。

## 一键运行

在 graph 根目录、现有 research 环境中：

```bash
git pull --ff-only origin main
python -u main.py support \
  --dataset /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset \
  --limit 4 --balanced --query-chunk-size 8 \
  --output outputs/native_support_ragtruth4 --resume
```

这条命令从官方 `response.jsonl` 与 `source_info.jsonl` 中选 4 个完整回答，
用原始字符区间与同一次 tokenization 的 offsets 自动生成 `annotations.json`，
完成原生前向、保存分数后自动输出 AUROC/AP。无需创建 `token_labels.json`。
默认筛选 QA/test/llama-2-7b-chat；observer 为既有 Llama-3.1-8B-Instruct，
模型路径可用 `--model` 改写。每个官方回答都是 teacher-forced replay，不重新生成。

`--balanced` 明确使用标签选择 2 个带错误标注、2 个无错误标注的回答。
标签不进入评分或选头，但这属于按标签分层的小样本诊断；其 AUROC/AP 不代表整个数据集，
尤其 AP 随阳性比例变化。去掉此选项则按官方 ID 顺序取前 4 个匹配回答，不看标签选样本；
若样本只有一类，AUROC 返回 null 并注明原因，不伪造 0.5。

仅检查输入/生成标签而不载入模型权重，可给同一命令增加 `--stage prepare`。
重复 run 命令会复用逐 token 缓存。必须使用新的官方回答输出目录，不覆盖旧前缀结果。

不带 `--dataset` 的旧命令仍保留：`python -u main.py support --resume`，
默认模型和 4 个自然前缀来自 `examples/prefixes.json`，共 322 个已观察回答 token。
这些前缀止于原决策词之前，没有加入 supported/unsupported 候选续写，不能评价首错与后续延续。
ID 中的 supported/unsupported 仅保留档案名称，不认证整段回答正确，也不参与计算。
缺标签时写出 `evaluation_status.json`，不把档案名字转成整段 token 标签。
以上均不是全量实验，本轮没有运行服务器 8B。

```bash
# 更换模型位置；输入 token IDs 必须来自这个 tokenizer。
python -u main.py support --resume --model /path/to/Meta-Llama-3.1-8B-Instruct

# 已采集后，仅 CPU 重算分数/报告，无需载入模型。
python main.py support --stage score --output outputs/native_support_v1
```

采集可断点续跑，缺失文件才前向；评分按全部已观察回答 token 重建因果窗口。
读旧 native_trace_audit 结果不会得到新分数：旧文件使用固定人工候选方向，需新的原生前向读出。
原审计文件不会修改。

## 采集速度与批量计算

默认一次前向计算 8 个连续缺失位置，原生 causal mask 限制每行只能读取此前 token。
这适用于已有回答的 teacher forcing；真正在线生成仍需等待前一个输出。
206 个连续回答位置需要 26 次采集前向，另有分块 prompt prefill；不是每词单独调用模型。
`--prefill-chunk-size` 管 prompt 预填充，`--query-chunk-size` 管回答位置的批量采集。
显存余量允许时可把后者调到 16；长 prompt 显存紧张则减到 4。

每个位置仍需自己的原生竞争词、逐头来源写入、FFN 和残差账本，不能用前一词的结果替代。
这些投影在 GPU 按块完成，只传回标量；不再构造/搬运随后丢弃的完整逐头残差写入，
也不采集 FFN 中间激活。每个来源的投影 value energy 只计算一次，随 KV cache 增量更新。
attention、energy、熵等字段保留；v1 对照 risk 的定义和历史递推没有改动。

逐 token 缓存默认使用未压缩 NPZ，减少 CPU 压缩时间，但会增加磁盘占用。
需要节省空间可加 `--compress-cache`；新旧 NPZ 都能直接读取，可在同一输出目录续跑。
调整 query chunk 或压缩选项不会废弃已有文件，文件仍逐 token 原子写入。
全部采集缓存已齐时，run 直接重算分数和评价，不再加载模型权重。

每答结束打印本次新增/复用 token 数，以及 capture/write 秒数，并写入
`responses/0000/capture_timing.json`。capture 含 Gram、prefill、原生前向、投影和 GPU→CPU；
write 是 NPZ 写入；wall 是本答此次采集总耗时，不含载入模型及评分。
每个新 token 文件记录实际 `query_chunk_size` 和块内均摊的 `capture_seconds`。
续跑时计时文件描述最新一次补采集，不能与首次完整采集混为一谈。

分块 GEMM 可能造成浮点末位差异，不承诺逐 bit 相同。软件验证覆盖 float32/bfloat16、
与原 teaching 逐步账本的核对、块内未来 token 不影响早期结果、缺口续跑和新旧压缩缓存混用。
未在服务器 8B 上测量加速倍数，不把前向调用次数减少等同于端到端提速倍数。

## 原始缓存与历史 v2 输出

以下 v2 文件只在 `--stage compare` 时计算；新版默认输出见上面的 `route_filter_v3/w16/`：

| 文件 | 内容 |
|---|---|
| `route_comparison_v2/report.html` | 方法比较、逐 token 路由曲线及读取/写入表格 |
| `route_comparison_v2/tokens.csv` | 13 种固定方法分数及逐词解释字段 |
| `route_comparison_v2/summary.json` | 主方法、来源协议、完成数量、账本误差及评价摘要 |
| `annotations.json` | 使用 --dataset 时自动生成的真实评价标签，含字符区间、token IDs 和有效位置 |
| `route_comparison_v2/evaluation.json` | 各法、各答、各阶段 AUROC/AP，pooled/来源等权/同答比较 |
| `route_comparison_v2/evaluation_status.json` | 本次评价状态；缺标签时不覆盖此前有效评价 |
| `route_comparison_v2/onsets.csv` / `high_risk_normals.csv` | 逐答逐法起点排名及高风险正常词 |
| `route_comparison_v2/calibration.json` | 按来源划分、无标签校正系数及 causal/offline 协议 |
| `route_comparison_v2/source_regions.json` | 官方来源区间与已核对的 prompt IDs；只读 prompt |
| `responses/0000/token_000000.npz` | 每层每头每来源 attention、signed write、value energy；残差账本 |
| `responses/0000/capture_timing.json` | 本答最新一次补采集的批量大小、复用数量、采集和写盘耗时 |
| `route_comparison_v2/responses/0000/scores.npz` | 新旧分数、每层结构量、头锚点、同窗口写入及 v1 历史边 |
| `settings.json` | 实际模型、token 序列和边界，便于重算及续跑 |

路由主分数越高表示消息范数更偏向历史而非来源；分数不是幻觉概率，0 不是分类阈值。
熵/惊讶度单独作为对照，不参与路由组合或挑选位置。FFN 正负写入不是正确/错误标签。

## 输入完整自然回答

`--input` 接受 JSON，沿用 teaching 的原始 token ID 回放：

```json
{
  "model": "/path/to/model",
  "responses": [
    {
      "id": "answer-1",
      "source_id": "source-1",
      "prompt_length": 3,
      "token_ids": [1, 2, 3, 4, 5],
      "token_text": ["<s>", "Question", "Answer:", " text", "."]
    }
  ]
}
```

例中 ID 仅展示结构，应使用真实的 prompt+response IDs。
`token_text[i]` 必须是原 tokenizer 的 `decode([token_ids[i]])`，不要求这些片段能拼成整句解码。
保存的回答必须是实际观察值，不从标签或候选表合成。不要提供未来正确答案作为候选。

## 已有分数直接补评

官方回答流程会自动生成标签并评价。若已保存分数，补评只需 CPU：

```bash
python main.py support --stage evaluate \
  --output outputs/native_support_ragtruth4
```

程序默认读取上述输出目录中的 `annotations.json`，不用手写不存在的文件名。
只有已有外部标注文件时才传 `--annotations`，它不是要求新建一个空文件。
外部标注格式为 `{"answer-1": {"token_ids": [4, 5], "labels": [0, 1]}}`。
ID 与回答 token 必须完全对应，1 表示幻觉；评分阶段不读取此文件。
已有 v3 时默认补评 w16；显式其他窗口需传相同 `--window`。没有 v3 时补评 v2，
只有旧 v1 时仍支持原两方法补评。缺标签不覆盖之前有效的 evaluation.json。
不翻转方向，不以当前测试标签选阈值。没有标签时不输出伪造的 AUROC 或告警准确率。

## 代码与验证

- `state_audit/native_forward.py`：因果分块采集、增量 KV、逐 token 兼容缓存。
- `state_audit/forward_ledger.py`：GPU 批量原生竞争词、消息投影和残差账本。
- `filtering.py`：逐头分区、状态距离、因果滤波及两种直接对照。
- `filter_features.py`：首次提取和重复使用紧凑缓存。
- `optimize.py`：冻结评分后评价；`filter_evaluation.py` / `filter_report.py`：增量和恢复报告。
- `routes.py`：历史消息范数/attention 路由、头来源结构及同窗口写入。
- `calibration.py`：按来源分离的结构校正，明确 causal 与历史 offline。
- `comparison.py` / `comparison_evaluation.py`：复用缓存、冻结评分、同口径比较。
- `source_regions.py`：复用 teaching offsets，从官方 prompt 恢复来源区间。
- `score.py`：保留 v1 预算、头模式及历史递推作为对照。
- `pipeline.py`：按 token 保存/续跑/评分；`report.py` 输出可读结果。
- `evaluate.py`：冻结分数后的独立标签评价。
- `ragtruth.py`：官方回答选择、复用 teaching tokenization 与字符标签对齐。

针对性验证：

```bash
python -m pytest -q tests/test_native_support.py tests/test_native_support_evaluation.py \
  tests/test_native_routes.py tests/test_route_filter.py
```

CPU 小型随机 Llama 的软件测试不能证明真实模型的检测有效性。
