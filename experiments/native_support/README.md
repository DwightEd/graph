# 原生来源支持检测

核心定义只有一个：`risk = direct_risk + inherited_risk`。
先用原生 W_O A V 写入衡量 prompt 对实际输出的支持，再沿正向历史写入与相似头模式继承此前分数。
不消融、不反向传播、不需要正确候选、角色表或幻觉标签。全程 teacher forcing。
[公式、分析顺序与边界](../../iclr/NATIVE_SUPPORT_DESIGN.md)。

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
attention、energy、熵等解释字段保留；核心 risk 的定义和历史递推没有改动。

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

默认目录 `outputs/native_support_v1/`：

| 文件 | 内容 |
|---|---|
| `report.html` | 自包含的逐 token 曲线与表格 |
| `tokens.csv` | 每个 token 的总分、直接项、历史项、熵、惊讶度、FFN 写入和读取增强 |
| `summary.json` | 完成数量、账本误差、运行边界 |
| `annotations.json` | 使用 --dataset 时自动生成的真实评价标签，含字符区间、token IDs 和有效位置 |
| `evaluation.json` | 有真实标签时输出 AUROC/AP；包含样本选择协议和阳性比例 |
| `evaluation_status.json` | 本次是否完成评价；缺标签时明确报告，不覆盖此前有效评价 |
| `responses/0000/token_000000.npz` | 每层每头每来源 attention、signed write、value energy；残差账本 |
| `responses/0000/capture_timing.json` | 本答最新一次补采集的批量大小、复用数量、采集和写盘耗时 |
| `responses/0000/scores.npz` | 逐 token 分数、完整头模式、下三角历史边权、逐头 prompt 关注峰 |
| `settings.json` | 实际模型、token 序列和边界，便于重算及续跑 |

R 越高代表来源支持越弱；R 可正可负，0 不是分类阈值，R 不是幻觉概率。
回看和熵不进入评分，不用它们挑选位置。FFN 正负写入不是正确/错误标签。

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
评价 `support_graph` 与 `direct_prompt` 的全错误、每答首错、span 起点和延续 AUROC/AP。
不翻转方向，不以当前测试标签选阈值。没有标签时不输出伪造的 AUROC 或告警准确率。

## 代码与验证

- `state_audit/native_forward.py`：因果分块采集、增量 KV、逐 token 兼容缓存。
- `state_audit/forward_ledger.py`：GPU 批量原生竞争词、消息投影和残差账本。
- `score.py`：预算、保留头身份的模式、因果历史递推。
- `pipeline.py`：按 token 保存/续跑/评分；`report.py` 输出可读结果。
- `evaluate.py`：冻结分数后的独立标签评价。
- `ragtruth.py`：官方回答选择、复用 teaching tokenization 与字符标签对齐。

针对性验证：

```bash
python -m pytest -q tests/test_native_support.py tests/test_native_support_evaluation.py
```

CPU 小型随机 Llama 的软件测试不能证明真实模型的检测有效性。
