# 原生路由比较：恢复历史信号

固定主基线恢复为历史 `routing_imbalance`，从同一原生缓存同时计算 attention 路由、
来源/头/时间结构和 v1 净支持对照。读取、消息范数、对实际输出的有符号写入分开保存。
不消融、不反向传播、不需要正确候选或角色表；标签只在分数冻结后用于评价。
[历史结果、公式及重新设计依据](../../iclr/NATIVE_ROUTE_REDESIGN.md)。

## 已有四答结果直接重算

```bash
git pull --ff-only origin main
python -u main.py support --stage compare \
  --output outputs/native_support_ragtruth4
```

只读已有 `settings.json`、逐词 NPZ 和官方来源文本，CPU 计算，不加载模型权重。
首次恢复官方来源区间需要原模型的 tokenizer，随后复用 `source_regions.json`。
新结果位于 `outputs/native_support_ragtruth4/route_comparison_v2/`；原始缓存及 v1 结果保留。
`--stage score` 与 `compare` 同义；缺标签时仍保存分数，AUROC 明确 unavailable。

原四答的 v1 AUROC 是 0.572333、AP 是 0.103939；历史路由基线约 0.7 来自其他总体和协议。
本轮恢复了历史公式并统一比较口径，没有提前宣称当前四答已恢复 0.7。

## 如何读结果

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

采集可断点续跑，缺失文件才前向；评分始终按全部已观察回答 token 重建历史递推。
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

## 输出

原始目录仍保存采集输入；所有新版派生结果位于其 `route_comparison_v2/` 子目录：

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
已有 v2 结果时评价所有固定方法并刷新报告；只有旧 v1 分数时仍支持原两方法补评。
不翻转方向，不以当前测试标签选阈值。没有标签时不输出伪造的 AUROC 或告警准确率。

## 代码与验证

- `state_audit/native_forward.py`：因果分块采集、增量 KV、逐 token 兼容缓存。
- `state_audit/forward_ledger.py`：GPU 批量原生竞争词、消息投影和残差账本。
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
  tests/test_native_routes.py teaching/state_audit/tests/test_native_trace.py
```

CPU 小型随机 Llama 的软件测试不能证明真实模型的检测有效性。
