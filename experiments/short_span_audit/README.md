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
| `cohort.json` | 全部短错误及已有正常配对的精确目标位置、文本 |
| `reanchor.csv` | 可选的旧重锚事件连接 |

自动打包：`outputs/short_span_audit_v1_review.tar.gz`。

两套阈值口径必须分开：

- `frozen`：沿用原方法保存的报警；原混合 CAL 阈值不是正常 FPR=5%。
- `descriptive_test_normal_fpr_curve`：根据本组全部 TEST 正常 token 分数取阈值，保守处理并列分数，并报告实际 FPR。用于离线同误报预算比较，**不是无标签部署校准**。其 bootstrap 固定已报告阈值。

所有方法在共同可测位置比较。旧快照已按原方法集合取过共同覆盖，选择更少方法不能恢复当时被裁掉的行。缺测段报告下界、上界及未决数，不把下一个可测 token 冒充 onset。段结束后的报警不算成功。所有 span 等权，正常对照用相同长度及相同 max 聚合。

配对沿用相同回答、相同长度、粗词面类别、位置和重复度约束；并未完全控制实体或关系语义。正常对照前 15 步有错误的 `recovery` 独立报告，防止把平滑拖尾解释成正常稳定模式。

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

数据与 attention cache 默认使用原 `input_settings.json` 的目录，搬家后显式加 `--cache /path/to/test/attention --dataset /path/to/RAGTruth/dataset`。

默认层 `--layers 8:24`（8 到 23，全部 heads），默认 `--device cuda --dtype bfloat16`。其他模型须显式指定适当层范围；当前 adapter 支持 Llama、Mistral、Qwen2，不能直接承诺支持 Qwen3。模型 tokenizer 必须与保存的 token IDs 匹配。不会截断前缀或删除标注后半段来规避显存问题。

对 target `t`：

1. 输入只有 `token_ids[:P+t]`，预测位置 `q=P+t−1`，目标 ID 是 `token_ids[P+t]`。
2. `F = logit(y_t) − logsumexp(other logits)`；对当前目标做一次反传。
3. 经实际 `o_proj` 输入求导，计算各边 `A × dF/dA`。GQA 显式展开 KV 对应关系；不合并多个目标的 loss。
4. 保留 `[layer, head, key]` attention、signed contribution、`||A V W_O||²`，以及预测熵、目标 log-prob 和 margin。
5. 按普通 prompt、近期历史、早期历史聚合，正负贡献分开。特殊 token 仅从统计中排除；不更改模型原生 attention、不重归一化、不平均 heads。

这里只有 prompt/history 分组，**没有自动识别适用证据或值—scope 绑定**。`value_energy` 是逐边平方范数之和，不能当作相加后消息的范数。贡献是一阶消息 gate 敏感度，不是完整删除效应，不跨层求“总因果贡献”。

从最早选层的完整 head readout 建立梯度，减少之前层的反传激活。已用 tiny Llama GQA 检查子集/全层归因一致、原生输出一致及微扰一致；未实测 8B 在单卡 24GB 的所有长前缀显存。

输出在 `outputs/short_span_audit_v1/contributions/`：

- `answers/<split>/<task>/<id>/<target>.npz`：原始边、逐来源聚合、目标读数。
- `paired_head_sources.csv`：每个 layer/head/source 的短错误减正常配对差，onset 与段内分开，并按正常对照此前 15 步是否有错误区分 clean_history/recovery。
- `paired_measurements.npz`、`paired_records.json`：保留逐配对数组及来源身份。
- `token_readouts.csv`、`paired_readouts.csv`：逐 token 及配对熵、log-prob、margin。
- `report.json`：已完成/预期目标数、observer/generator、测量范围。

自动打包 `outputs/short_span_audit_v1/contributions_review.tar.gz`，包含汇总、配对数组与回答身份；逐目标大边数组保留在本地，不塞进 review 包。

配对统计先在段内取平均，再在 source 内取平均；不把一个 span 的 token 或不同 heads 当独立样本。原始逐 token、逐 head 数组仍保留。中断后可 `--resume`；只重汇总无需加载模型：

```bash
PYTHONPATH=teaching/state_audit/src \
python -m experiments.short_span_audit.capture_reporting --audit outputs/short_span_audit_v1
```

这一步使用标签选择短段与对照，属于机制检验；没有无标签 FIT/CAL 全流采集，不能将这些配对数据直接拟合后声称新检测器有效。当前原回答生成器可能是 Llama-2，而观察模型是 Llama-3.1；输出中保留两者身份。

## 代码职责

`inputs.py` 读冻结数据、列出采集目标；`metrics.py` 计算评价；`reanchor.py` 连接确认事件；`reporting.py` 写报告；`capture.py` 调用 teaching 采集；`capture_reporting.py` 汇总逐头配对。模型细节只在 teaching 内实现。

```bash
PYTHONPATH=teaching/state_audit/src python -m pytest -q \
  tests/test_short_span_metrics.py tests/test_short_span_pipeline.py \
  tests/test_short_span_capture.py teaching/state_audit/tests/test_attribution.py
```
