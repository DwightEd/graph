# S10 结果完整性审计

日期：2026-09-14
审计角色：fresh Codex reviewer，same-family、read-only、provisional
范围：冻结主结果的数值、标签范围、分割、哈希、预测复现、阈值误报与结论边界；未重拟合、未运行 GPU、未改实验代码。200 次来源 bootstrap 未重复运行（本任务明确为可选）。

## 裁决

**完整性：PASS；无合并/推送阻断。结论使用：WARN，必须保留限定。**

冻结预测和 `evaluation.json` 的主指标可由官方测试标签独立精确复算，最大绝对差为 **0.0**。全部冻结/完成清单哈希一致，989 个特征文件哈希一致，固定分割可独立精确再生，冻结模型参数可将 12 个保存分数流复现到最大误差 **8.33e-17**。

主方法支持“在本次固定来源留出上改善 token 排序”。它不支持“已经成为可用报警器”：错误目标在约 4.25% 负 token FPR 下只召回 12.57%，并在 98 条完全正常回答中误报 85 条；起点目标在约 4.42% 负 token FPR 下召回 32.14%，并在全部 98 条正常回答中至少误报一次。

## 独立复算

审计器先由 `prediction_index.json` 取 150 个 test ID，再逐行扫描官方 `response.jsonl`。全部 17,790 行均为 ID-first schema，但只对命中这 150 个 ID 的行执行 JSON 反序列化。150 个标签的 `id/source_id/split/response SHA256` 与冻结特征身份全部一致。

测试范围为 150 回答 / 150 来源 / 30,721 token；52 条回答有错误，98 条完全正常。官方标签包含 85 个 label 对象，得到 2,307 个错误 token 和 84 个唯一 onset token。少 1 个 onset 是因为回答 `15651` 有两个起止位置和文本完全相同的重复 label 对象（仅 `implicit_true` 字段不同），二者合理折叠到同一 token 起点，并非漏映射。

| 目标 | 分数 | AUROC | AP | 与记录差异 |
| --- | --- | ---: | ---: | ---: |
| error | raw entropy | 0.6000032601 | 0.1072510750 | 0 |
| error | raw remote-history-excl16 minus source | 0.6639191841 | 0.1168736553 | 0 |
| error | raw negative margin | 0.5819926922 | 0.0924476040 | 0 |
| error | instant | 0.7404971767 | 0.1622671931 | 0 |
| error | combined | **0.7492396695** | **0.1661344589** | 0 |
| onset | raw entropy | 0.7098796662 | 0.0085832639 | 0 |
| onset | raw remote-history-excl16 minus source | 0.6571153072 | 0.0094657882 | 0 |
| onset | raw negative margin | 0.6465493793 | 0.0041685687 | 0 |
| onset | instant | 0.7515169955 | 0.0157419081 | 0 |
| onset | combined | **0.7790782076** | **0.0158807904** | 0 |

四个学习 arm 的 log loss、Brier、threshold metrics 以及八个学习/raw 流的回答内 AUROC 也全部精确匹配。combined 相对 instant 的 error 增量为 AUROC +0.00874249、AP +0.00386727；记录的来源 bootstrap 只支持 AUROC 增量稳定为正，AP 区间跨 0。onset 的点估计增量为 AUROC +0.02756121、AP +0.00013888，两项记录的来源 bootstrap 区间均跨 0。

## 固定阈值与误报

| 目标/arm | 正例召回 | 负 token 误报 | 完全正常回答 any-alarm | 回答内 AUROC |
| --- | ---: | ---: | ---: | ---: |
| error / instant | 297/2307 = 12.87% | 1286/28414 = 4.53% | 84/98 = 85.71% | 0.663742 |
| error / combined | 290/2307 = **12.57%** | 1209/28414 = **4.25%** | 85/98 = **86.73%** | 0.680542 |
| onset / instant | 21/84 = 25.00% | 1372/30637 = 4.48% | 98/98 = 100% | 0.771608 |
| onset / combined | 27/84 = **32.14%** | 1355/30637 = **4.42%** | 98/98 = **100%** | 0.806699 |

`evaluation.json` 已正确报告这些数值；未发现把 token 级约 5% FPR 冒充回答级 5% FPR 的情况。高 any-alarm 率来自一条回答包含许多 token 后的累积效应，必须与排序改善一起呈现。

## 哈希、分割与来源链

- `complete.json` 与 `prediction_freeze.json` 共核验 15 个结果引用，另核验 3 个协议代码引用、feature manifest 和 records 引用：**20/20 匹配**。关键 SHA256：`evaluation.json` `58ada313...0678`，`predictions.npz` `5e7b1471...ac14`，`models.json` `cc6516ec...2259`，`thresholds.json` `214f95d2...3cc8`，`prediction_index.json` `d9287e82...f03`，`protocol_freeze.json` `29d1f6ec...90ff`。
- `executed_experiment.py` 与当前 `graph/structural_detector/experiment.py` 同为 `1cfe7a6a...55b6`；`executed_features.py` 与当前 `features.py` 同为 `289998ae...c13`。导出 manifest 的代码哈希与当前 `reanchor/src/decoding/structural_export.py` 同为 `fb3573c2...f65f`。
- feature manifest 为 complete，`planned=completed=989`、`labels_read=false`、`model_forwards=0`。`records.jsonl` 哈希 `4effde22...b905` 匹配；989/989 个 per-record NPZ 哈希、形状和有限值检查通过。
- 上游 observer 的 `COMPLETE` 为 `completed=17790 failed=0 total=17790`。feature manifest 引用的 parent settings/input 哈希与现存父缓存完全一致，989/989 个 parent manifest 均存在且哈希匹配。父 settings 记录的 RAGTruth `response.jsonl` SHA256 `e4c2e4ac...8073` 和 `source_info.jsonl` `0dffc26e...578b` 与当前官方文件一致。
- 上游无标签 roster 独立筛选得到恰好 989 条 QA + llama-2-7b-chat 记录，顺序及 source/split/response/token-count 身份与 feature records 全部一致。
- 839 个 official-train 来源和 150 个 official-test 来源无交集。按 `SHA256("20260914:" + source_id)` 独立重做 80/20 分割，得到 671 train、168 calibration、150 test，与 `protocol_freeze.json` 和连续、无重叠的 `prediction_index.json` 逐项相同；全索引共 222,741 token。
- 从 989 个冻结 NPZ 独立重算 11 维特征，四个 raw 分数流逐元素完全一致；再由保存的均值、标准差、逻辑回归系数和正斜率校准重算 8 个学习分数流，最大绝对差 8.33e-17。不存在 phantom result。

冻结保证属于**代码顺序 + 内容哈希**证据：执行快照在 test annotation join 前写入并哈希模型、阈值、全量预测、索引和协议；随后才读取限定的 150 个 test ID（`experiment.py:110-149`）。它没有外部 append-only 时间戳，因此本审计不把文件 mtime 当作独立证明。

## A–F 完整性检查

### A. Ground-truth provenance：PASS

标签来自官方 RAGTruth `response.jsonl`，并按 ID-first 方式只反序列化所需 ID；`targets` 检查 source、split、response hash 和 token-offset 映射（`experiment.py:28-53`）。没有从模型输出生成 ground truth。测试评价类型为 **real_gt**；预测特征则来自 Llama-3.1 对原 Llama-2 回答的 observer replay，不能解释为原生成器内部轨迹。

### B. Score normalization：PASS

模型只用 train 的加权均值/标准差做特征标准化，并用 calibration 拟合受限为正的 slope/intercept（`experiment.py:56-73`）。AUROC/AP 直接作用于保存分数；没有用模型自身 test 最大值、均值或结果统计作归一化。raw 分数同时报告。

### C. Result existence and numerical match：PASS

请求范围内文件全部存在，哈希链一致，`evaluation.json` 中主/raw 数值与独立 sklearn 复算完全一致。`S10_RESULTS_20260914.md` 的表格、增量解释和报警限制均与冻结结果一致。

### D. Dead code：PASS

`fit`、`predict`、`metrics`、`bootstrap` 及四个 arm 均由主执行路径调用并在模型、预测或 evaluation 中留下对应输出（`experiment.py:131-180`）。本范围未发现声称使用但未执行的指标函数。

### E. Scope and claim language：WARN

证据是单个固定 seed/分割、989 个自然回答、150 个官方测试来源；测试集曾被历史描述性工作使用。它足以支持该固定来源留出上的排序结果，不足以支持跨模型、跨数据集、完全未见确认集、稳健部署或普遍机制结论。combined 的回答级误报尤其排除“可靠报警器”表述。

### F. Evaluation type：real_gt + observational predictors

目标是真实数据集人工标注；特征是 observer replay 的观测量。结果是监督 token 检测关联证据，不是无监督结果、原生成器因果机制、外部事实核验或结构必要性证明。

## 结论影响

- **支持**：在这次固定的来源互斥测试上，combined 对单项 raw 特征有更好的 token 排序；error 的 combined 相对 instant AUROC 小幅提高。
- **需要限定**：回答内 AUROC 和 onset 点估计改善是描述性结果；onset 相对 instant 的 bootstrap 区间跨 0，error AP 增量区间也跨 0。
- **不支持**：可部署/可靠报警、回答级 5% FPR、跨数据/跨模型泛化、原生成器机制或因果知识正确性。

在 `S10_RESULTS_20260914.md` 当前限定措辞保持不变的前提下，本审计没有发现阻止合并或推送的结果完整性问题。
