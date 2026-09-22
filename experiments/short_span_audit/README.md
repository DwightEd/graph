# 短幻觉 span 检验

先检验 1–2、3–4、5–8 token 的错误能否在结束前发现。已有冻结分数评价和新贡献采集分开运行；没有训练新检测器，也不把标签挑选位置上的归因称为全流检测成绩。

## 1. 已有结果：CPU 一键检验

在 graph 根目录：

```bash
git pull --ff-only origin main
bash experiments/short_span_audit/run.sh
```

默认读取 `outputs/head_cross_terms_v1`，优先复用其中 `continuity/` 的冻结快照。没有 continuity 快照时直接读原 `predictions/`，复用现有标签对齐和正常匹配，不重新拟合分数。

也可在另一台电脑上解压旧 continuity review 包后运行，**不需要原始 attention、大模型、tokenizer 或 RAGTruth 文件**：

```bash
INPUT=/path/head_cross_terms_v1/continuity \
OUTPUT=outputs/short_span_audit_v1 \
bash experiments/short_span_audit/run.sh
```

可选参数：`--tasks QA Data2txt`、`--methods all__raw all__moment all__pair_state_smooth`、`--fpr .01 .03 .05`、`--bootstrap 500`。

若也有重锚归档，可加 `--reanchor outputs/reanchor_nodes_v2`，连接短段及正常对照起点前的确认事件。旧事件是 local→old，不自动等于回到适用证据；未确认与不存在严格区分。此连接不改变检测分数或阈值。

结果写入：

| 文件 | 内容 |
|---|---|
| `summary.md` | 分任务的短段核心结果 |
| `metrics.csv` | 长度分桶、onset、段内命中、缺失、实际正常 FPR、正常段误报 |
| `spans.csv` | 每段起止、首次观察到报警、完整漏检或缺失未决 |
| `paired_spans.csv` | 同长度正常对照，以及此前 15 步是否有错误 |
| `comparisons.csv` | 短段方法差的 source 配对 bootstrap |
| `answers.csv` | 每回答 TP/FN/FP/TN、可测和缺测 token 数、是否全正常 |
| `cohort.json` | 全部短错误及已有正常配对的精确目标位置、文本 |
| `reanchor.csv` | 可选的旧重锚事件连接 |

自动打包：`outputs/short_span_audit_v1_review.tar.gz`。

两套阈值口径必须分开：

- `frozen`：沿用原方法保存的报警；原混合 CAL 阈值不是正常 FPR=5%。
- `descriptive_test_normal_fpr_curve`：根据本组全部 TEST 正常 token 分数取阈值，保守处理并列分数，并报告实际 FPR。用于离线同误报预算比较，**不是无标签部署校准**。其 bootstrap 固定已报告阈值。

所有方法在共同可测位置比较。旧快照已按原方法集合取过共同覆盖，选择更少方法不能恢复当时被裁掉的行。缺测段报告下界、上界及未决数，不把下一个可测 token 冒充 onset。段结束后的报警不算成功。所有 span 等权，正常对照用相同长度及相同 max 聚合。

配对沿用相同回答、相同长度、粗词面类别、位置和重复度约束；并未完全控制实体或关系语义。正常对照前 15 步有错误的 `recovery` 独立报告，防止把平滑拖尾解释成正常稳定模式。

新增 deadline=0/1/3/7 的及时召回，用**全部 span**作分母，且始终在段尾停止；缺测提供上下界。长段另分 9–16、17+，保留旧 9+ 汇总。匹配段另外输出 TP/FN/FP/TN，以及各方法的召回差、正常段误报差、recall−FPR 差的 source bootstrap；后者只对双方完整可测的相同配对计算，不冒充总体检测收益。

错误结束后观察最多 15 个正常 token，遇到下一错误或回答结尾停止。连续报警尾长遇到缺测即截尾，不能用后面一次零值声称已经及时退出；报告尾长下界和截尾原因。另报“段内完整漏检、结束后才报警”的数量。全正常回答至少一次报警率独立报告，不能用正常 token FPR 替代。

## 2. 新测量：逐目标采集 attention、贡献、熵

此阶段复用 `teaching/state_audit` 的 `ModelAdapter`、原 token 对齐、存储接口和新 `attribution.py`。只回放，不重采样回答、不更新模型参数。

模型依赖未安装时：

```bash
pip install -e 'teaching/state_audit[model]'
```

先小量核对，再继续全部目标；两次命令复用逐目标文件：

```bash
MODEL=/path/to/Meta-Llama-3.1-8B-Instruct \
bash experiments/short_span_audit/run_capture.sh --limit-answers 2

MODEL=/path/to/Meta-Llama-3.1-8B-Instruct \
bash experiments/short_span_audit/run_capture.sh --resume
```

省略 `MODEL` 时，读取 CPU 审计保存的 `input_settings.json.tokenizer` 作为观察模型路径。因此在原服务器续跑只需 `bash experiments/short_span_audit/run_capture.sh --resume`；模型搬家后再显式设置 `MODEL`。

数据与 attention cache 默认使用原 `input_settings.json` 的目录，搬家后显式加 `--cache /path/to/test/attention --dataset /path/to/RAGTruth/dataset`。

默认层 `--layers 8:24`（8 到 23，全部 heads），默认 `--device cuda --dtype bfloat16`。其他模型须显式指定适当层范围；当前 adapter 支持 Llama、Mistral、Qwen2，不能直接承诺支持 Qwen3。模型 tokenizer 必须与保存的 token IDs 匹配。不会截断前缀或删除标注后半段来规避显存问题。

24GB 显存采用分块历史 KV 预填充，再对当前预测位置反传。`--prefill-chunk-size 256` 是默认计算块大小，不是历史窗口；所有历史 token 仍参与原生 attention。它只调整工作内存，不改变采样位置、层或评分公式，也不改变续跑身份。

现在同一回答的目标按时间顺序共用 KV：首次预填充历史，之后只推进尚未计算的 token。每次求导后 detach 缓存，不保留上一个目标的反传图。固定的每头 `W_O^T W_O` 也只计算一次。一个会话内前缀 token 不再为每个目标重复前向；续跑时只选缺少 NPZ 的目标，其间历史仍按原文推进。旧的已完成 NPZ 和 settings 可继续复用。

旧版遇到 CUDA OOM 后，更新代码并运行上面的 `--resume` 命令。已写完的逐目标 NPZ 会复用。若预填充仍吃紧，可加 `--prefill-chunk-size 64`；它不会减少 KV 本身的存储或模型权重。无需重跑 CPU 分数检验，也无需删除已完成的贡献结果。

对 target `t`：

1. 输入只有 `token_ids[:P+t]`，预测位置 `q=P+t−1`，目标 ID 是 `token_ids[P+t]`。先在无梯度下用 SDPA 分块计算 `[:q]` 的 KV，再用原生 eager attention 计算位置 `q`。
2. `F = logit(y_t) − logsumexp(other logits)`；对当前目标做一次反传。
3. 经实际 `o_proj` 输入求导，计算各边 `A × dF/dA`。GQA 显式展开 KV 对应关系；不合并多个目标的 loss。
4. 保留 `[layer, head, key]` attention、signed contribution、`||A V W_O||²`，以及预测熵、目标 log-prob 和 margin。
5. 按普通 prompt、近期历史、早期历史聚合，正负贡献分开。特殊 token 仅从统计中排除；不更改模型原生 attention、不重归一化、不平均 heads。

这里只有 prompt/history 分组，**没有自动识别适用证据或值—scope 绑定**。`value_energy` 是逐边平方范数之和，不能当作相加后消息的范数。贡献是一阶消息 gate 敏感度，不是完整删除效应，不跨层求“总因果贡献”。

从最早选层的当前位置 head readout 建立梯度。由于因果掩码，更早位置的状态不依赖当前位置的消息扰动；将历史 KV 固定并保留当前位置的后续 Q/K/V 路径，可以计算相同的当前消息 gate 导数。这样无需保存整个前缀的 attention 反传图。历史预填充和当前查询仍读取完整的、符合模型掩码的来源，不将历史梯度为零误解为历史没有贡献。

分块 SDPA 与整段 eager 使用不同数值核，低精度结果不保证逐位一致。软件验证比较原始整段回放的 logits、各头 attention、贡献与能量，并检查有限差分和无未来信息；这里没有实测用户 8B 检查点在 24GB GPU 上的峰值。

输出在 `outputs/short_span_audit_v1/contributions/`：

- `answers/<split>/<task>/<id>/<target>.npz`：原始边、逐来源聚合、目标读数。
- `paired_head_sources.csv`：每个 layer/head/source 的短错误减正常配对差，onset 与段内分开，并按正常对照此前 15 步是否有错误区分 clean_history/recovery。
- `paired_measurements.npz`、`paired_records.json`：保留逐配对数组及来源身份。
- `token_readouts.csv`、`paired_readouts.csv`：逐 token 及配对熵、log-prob、margin。
- `report.json`：已完成/预期目标数、observer/generator、测量范围。

新 NPZ 和 `token_readouts.csv` 还保存 `capture_seconds`、实际新增 `prefill_tokens`，CUDA 运行保存峰值 allocated/reserved 字节数。`report.json.cost` 汇总这些实测数据。耗时包含预填充与贡献计算，不含模型加载和磁盘保存；首次目标承担初始预填充，失败尝试的耗时不计入。旧文件没测过的成本保持缺失，CPU 没有 CUDA 显存数字。

自动打包 `outputs/short_span_audit_v1/contributions_review.tar.gz`，包含汇总、配对数组与回答身份；逐目标大边数组保留在本地，不塞进 review 包。

配对统计先在段内取平均，再在 source 内取平均；不把一个 span 的 token 或不同 heads 当独立样本。原始逐 token、逐 head 数组仍保留。中断后可 `--resume`；只重汇总无需加载模型：

```bash
PYTHONPATH=teaching/state_audit/src \
python -m experiments.short_span_audit.capture_reporting --audit outputs/short_span_audit_v1
```

这一步使用标签选择短段与对照，属于机制检验；没有无标签 FIT/CAL 全流采集，不能将这些配对数据直接拟合后声称新检测器有效。当前原回答生成器可能是 Llama-2，而观察模型是 Llama-3.1；输出中保留两者身份。

## 代码职责

`inputs.py` 读冻结数据、列出采集目标；`metrics.py` 计算评价及配对对照；`lifecycle.py` 计算及时发现、退出和回答误报；`reanchor.py` 连接确认事件；`reporting.py` 写报告；`capture.py` 调用 teaching 采集；`capture_reporting.py` 汇总逐头配对。模型细节只在 teaching 内实现。当前检测问题与本次实测结果见 [REFACTOR_REVIEW.md](REFACTOR_REVIEW.md)。

```bash
PYTHONPATH=teaching/state_audit/src python -m pytest -q \
  tests/test_short_span_metrics.py tests/test_short_span_pipeline.py \
  tests/test_short_span_capture.py tests/test_short_span_lifecycle.py \
  teaching/state_audit/tests/test_attribution.py \
  teaching/state_audit/tests/test_attribution_memory.py
```
