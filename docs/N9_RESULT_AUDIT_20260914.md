# N9 冻结结果完整性审计

**日期**：2026-09-14
**审计者**：fresh Codex reviewer（same-family，read-only）
**审计效力**：`provisional`；不是 cross-family 独立验收
**总体裁决**：**WARN**
**完整性状态**：**warn**

本次只审计已经冻结的 N9 主结果。没有调用 GPU、没有重新拟合、没有改写输入、捕获、预测或评价产物。独立计算直接读取冻结的 `predictions.npz`、`prediction_index.json`、构造测试 gold，以及通过 ID-first helper 选出的 32 个 `DEV_IDS` 自然标注。没有遍历读取 800 个大型捕获数组的全部内容。

WARN 不表示发现伪造结果。冻结指标可以复现，ground truth 来源和 fit/evaluate 隔离均通过。WARN 来自四个需要显式保留的边界：真实 graph 没有胜过 nodes 或同容量 shuffled graph；受控 W1 窗口没有覆盖任何 core/onset；自然结果只是已反复使用的开发 observer 数据；追踪表仍错误地写作 RUNNING/未完成。此外，`predictions.npz` 的文件系统 mtime 晚于完成见证约 6.7 秒，虽然当前字节 SHA 与 freeze 和 complete 记录完全一致。

## 1. 独立复算

复算使用 `sklearn.metrics.roc_auc_score` 和 `average_precision_score`，没有导入项目的 `weighted_metrics`。自然标注只通过 `next_iteration.confirmation_assess_scoped.selected_annotations` 读取命中的 ID；该 helper 先从每行字节前缀取 ID，非目标行不做 JSON decode（`confirmation_assess_scoped.py:11-37`），并且目标集合与固定 `DEV_IDS` 完全相等（`development_assess_scoped.py:10-13`）。自然 response SHA 和 source ID 逐条匹配。

### 1.1 构造测试集

固定分母为 256 responses、32 held-out sources、8,128 tokens、208 core-positive tokens、128 onset tokens。下表的每一项均从冻结预测独立复算；与 `controlled_evaluation.json` 的最大绝对差不超过 `1.4e-14`。

| arm | micro AUROC | micro AP | equal-response AUROC | equal-response AP |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.976333041958 | 0.693384881660 | 0.976745238373 | 0.702081910386 |
| nodes | 1.000000000000 | 1.000000000000 | 1.000000000000 | 1.000000000000 |
| graph | 1.000000000000 | 1.000000000000 | 1.000000000000 | 1.000000000000 |
| shuffled_graph | 1.000000000000 | 1.000000000000 | 1.000000000000 | 1.000000000000 |

原结果位置见 `controlled_evaluation.json:5-16`、`:1161-1172`、`:2317-2328`、`:3473-3484`。项目自定义加权 AUROC 因浮点累加把两个 equal-response 值写成 `1.00000000000002...`；sklearn 返回严格的 `1.0`。这是约 `3e-14` 的表示误差，不改变排序，但后续报告应把指标限制在 `[0,1]` 或直接采用库实现。

完美分数不能解释为 graph 有效。nodes、graph、shuffled graph 的 AUROC/AP 全部相同；graph 相对 nodes 的来源 bootstrap log-loss 增益只有 `2.79e-08`，95% CI `[-9.37e-09, 7.23e-08]`，相对 shuffled graph 为 `5.39e-08`，95% CI `[-4.89e-09, 1.51e-07]`（`controlled_evaluation.json:9573-9587`）。两个区间都跨 0。全窗口 binding delta 相对 nodes/shuffled 的区间也跨 0（`:9599-9611`）。

### 1.2 自然开发集

固定分母为 32 responses、16 sources、5,170 tokens、281 annotated-error tokens、28 annotated-span onsets。RAGTruth 文件当前 SHA256 为 `e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073`，与 `natural_evaluation.json:437` 相同。复算与原 JSON 的最大绝对差小于 `9e-15`。

| arm | micro AUROC | micro AP | equal-response AUROC | equal-response AP | onset AUROC | onset AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.337609158187 | 0.040092982634 | 0.378558465095 | 0.034728785191 | 0.716737511808 | 0.010472161831 |
| nodes | 0.655260665784 | 0.125184574513 | 0.660427577521 | 0.106285021431 | 0.615262265933 | 0.008273539670 |
| graph | 0.627143220055 | 0.087414257844 | 0.632799034031 | 0.074503707477 | 0.574824970828 | 0.006762610587 |
| shuffled_graph | 0.647434978225 | 0.098574202274 | 0.670538107648 | 0.087564787939 | 0.693053008835 | 0.012516334690 |

Graph 在自然 micro AUROC/AP 上分别低于 nodes `0.02812/0.03777`，也低于 shuffled graph `0.02029/0.01116`。原 graph 数值和固定分母见 `natural_evaluation.json:206-231`；shuffled 数值见 `:308-333`。文件自身也正确标注了“known development observer data”、自然 entity-core/semantic-unit gold 不可用（`:436`）。这些结果不支持采用 graph，也不支持原生成器机制、因果必要性或独立自然泛化主张。

受控校准阈值在自然域没有保持预期工作点。Graph 的固定阈值为 `2.2788e-08`，自然正常回答 token-FPR 宏平均为 `0.12237`，自然 recall 为 `0.23132`；在固定 0.5 阈值下 recall 为 0（`natural_evaluation.json:223-228`）。这属于显著域偏移，不是自然校准成功。

## 2. W1 时序与覆盖

W1 的实现是因果的。捕获读取 `q=P+t-1` 的 attention 行（`internal_error_units_capture.py:77-86`）；`head_statistics` 明确拒绝未来 attention mass（`revisit_state.py:21-26`）；事件阈值只使用 `[t-16,t)` 历史（`:41-52`）。绑定 node/message 则是观察当前 token 后的 `q=P+t` 状态（`internal_error_units_capture.py:87-109`），不能写成提前预警。

从 288 个主结果 NPZ 中只解压小型 `event` member，独立重建向前的 W0/W4/W8 mask，结果与两个评价 JSON 完全一致：

| 数据 | W | token coverage | onset coverage | core coverage | events |
| --- | ---: | ---: | ---: | ---: | ---: |
| controlled test | 0 | 0.057210 | 0.000000 | 0.000000 | 465 |
| controlled test | 4 | 0.113558 | 0.000000 | 0.000000 | 465 |
| controlled test | 8 | 0.118356 | 0.000000 | 0.000000 | 465 |
| natural dev | 0 | 0.065571 | 0.071429 (2/28) | 0.046263 | 339 |
| natural dev | 4 | 0.297485 | 0.571429 (16/28) | 0.313167 | 339 |
| natural dev | 8 | 0.464217 | 0.821429 (23/28) | 0.587189 | 339 |

受控原结果见 `controlled_evaluation.json:34-36,58-60,82-84`；自然结果见 `natural_evaluation.json:233-238,259-261,283-285`。所有 128 个受控 binding pair 在 W0/W4/W8 都是 `neither_hit`，条件配对排序不可用。

这里还有一个预先存在的设计限制：事件条件为 `t > 16`（`revisit_state.py:50`），最早只能在 response token index 17 触发，而且窗口只向后扩展（`internal_error_units_evaluate.py:57-61`）。受控 128 个错误回答中，108 个 onset 位于 index 8-16，因而结构上不可能被 W1 命中；另 12 个在 index 17、8 个在 index 18，本次也全部漏掉。自然 W8 的 23/28 覆盖需要覆盖 46.42% 的全部 token，正常回答窗口覆盖同样为 46.50%（`natural_evaluation.json:384-406`）。因此 C2 的受控部分为负，自然部分至多是高预算开发集覆盖诊断。

## 3. A-F 完整性检查

### A. Ground-truth provenance：PASS

- 受控 gold 由固定模板和 source binding 规则生成，与输入分文件保存（`internal_error_units.py:34-71,150-163`）。这是明确标注的 constructed auxiliary supervision，不是从模型输出反推的“真值”。
- 捕获模块不打开 gold，并在 manifest 写入 `labels_read=false`（`internal_error_units_capture.py:223-232,250-255`）。
- 自然 gold 来自 RAGTruth `response.jsonl`。评价先要求目标恰好等于固定 `DEV_IDS`，再调用 ID-first helper；随后校验 response SHA 和 source ID（`internal_error_units_evaluate.py:331-345`）。没有解析非目标 ID 的 label payload。
- 输入 manifest 明确自然完整语义单元真值不可用（`outputs/n9_internal_error_units_inputs_20260914_v3/manifest.json:27`）。

评价类型：controlled 为 `constructed_gt / auxiliary synthetic task`；natural 为 `real_gt` 的 RAGTruth 字符 span，但只在 known development observer replay 上评价。两者都不是 human eval，也不是原生成器内部轨迹。

### B. Score normalization：PASS

没有指标除以模型自身输出的 max/min/mean，也没有按测试分数重缩放。AUROC/AP 的实现只按 score 排序和相同值分组（`grounding_contrast_evaluate.py:14-34`）。读出器的 feature mean/std 只在 train 上拟合，固定 `C=1`；温度/截距只在 calibration 上拟合且斜率约束为正（`internal_error_units_evaluate.py:76-100`）。这是训练预处理和预注册校准，不是结果归一化。原始 NLL、entropy、margin、attention mass 和 displacement 也单独报告（`controlled_evaluation.json:9538-9562`；`natural_evaluation.json:410-434`）。

### C. Result existence/provenance：WARN

通过项：

- 输入文件、完整 gold 和三个 split gold 的当前 SHA 全部与 input manifest 匹配；原始 R04 无标签 roster SHA 也匹配 manifest 的 `natural_roster_sha256`（input manifest `:17-25`）。
- full capture manifest 为 complete，800 planned/800 unique completed，ID 及顺序与 `inputs.jsonl` 完全一致；1600 个声明的 per-row 文件都存在，没有多余或缺失。manifest 记录 800 次实际 forward、4 个权重分片和执行版本（capture manifest `:2420-2454`）。
- 三个 full-run executed code snapshot 与当前被审代码逐字节一致，且 SHA 与 capture manifest `:2428-2431` 相同。评价 executed snapshot SHA 为 `627fad...`，与 freeze 和 complete 记录一致。
- `complete.json:2-16` 声明 controlled/natural complete、36 fits、0 CPU model forward；其中 8 个产物的当前 SHA 全部匹配。one-shot follow-up 记录 exit code 0；日志包含 36/36 无 warning 的 fits 和最终完成行。
- 本审计独立验证了全部 800 个小 JSON sidecar 的 SHA，以及按 8 template × train/calibration/test 加 4 个 natural endpoint 分层抽取的 28/800 个大型 NPZ SHA，均无不符。成功的 executed evaluator 自身在拟合前逐个验证全部 800 个 NPZ SHA（`internal_error_units_evaluate.py:239-254`）。这是代码路径证据；本审计没有再次对全部大型 NPZ 做完整字节哈希。

WARN 项：

- `EXPERIMENT_TRACKER.md:1-10` 仍写着 N9 RUNNING、GPU/CPU 未完成，和已经 complete/exit-0 的产物冲突。它没有虚构正结果，反而过时且保守，但必须改成实际完成和负面科学裁决，避免后续重复执行。
- 当前 `predictions.npz` SHA `86a301...` 同时匹配 `prediction_freeze.json:5` 与 `complete.json:10`，所以本次复算确实绑定冻结字节。然而其 mtime 为 11:29:29.493，晚于 `complete.json` 11:29:22.609 和 follow-up finished 11:29:22.822。可能原因包括共享文件系统 metadata/writeback，也可能是相同字节被重写；现有证据不能区分。因为 SHA 未变，这不改变任何数值，但文件时间线不能单独充当不可变性证明。

### D. Dead code / phantom metrics：PASS（有非结果性备注）

`features`、`fit_head`、`predict`、`summarize`、`paired_binding`、`localization` 和所有公开的主指标路径都在 `run` 中实际调用（`internal_error_units_evaluate.py:238-356`），对应输出 key 均存在。W1 的 `head_statistics`/`summarize_events` 由捕获器实际调用（`internal_error_units_capture.py:77-80,198-206`）。`grounding_contrast_evaluate.main` 和 `revisit_state.hold_state/main` 是被复用模块中的 P1/P3 入口，N9 只导入其 helper；没有 N9 主张依赖这些未调用入口。`summarize` 中的局部 `per_row` 列表被填充但不输出（`internal_error_units_evaluate.py:114-125`），属无害冗余，不是 phantom result。

### E. Scope assessment：WARN

- Controlled：8 固定模板、96 source groups、单一构造器、单一 tokenizer/model observer、单一固定 seed；384 train / 128 calibration / 256 test。独立核对为 48/16/32 个 source，任意 split 间 source 和 `pair_id` 重叠均为 0。九组模型（main + 8 LOTO）共 36 fits，无 convergence warning。
- 容量为 baseline 8、nodes 648、graph 1160、shuffled graph 1160 维。Graph 与 shuffled graph 同容量，但 graph 与 nodes 不同容量；计划已经披露该差异。Controlled 值词表跨 source split 复用，测试的是留 source，不是未见 value。
- Natural：只有 16 source / 32 已知 development responses；没有独立 validation/test，没有多 seed capture，也没有同一原生成模型的内部状态。Natural strict entity core、semantic unit 和 scope F1 不可用。
- 捕获是 BF16 执行后派生 float16 存储。800 个已哈希 sidecar 报告最大量化绝对误差 `0.0031109`、5,816,852 个非零到零下溢；按代码保存张量元素数估算约占 `0.105%`。attention row-mass 最大误差 `6.20e-06`，完整 O 重建最大相对误差 `0.001727`，分组 O 重建最大 `0.002401`，均低于代码的 0.02 gate（`internal_error_units_capture.py:59-65,94-129`）。量化没有破坏工程 gate，但属于科学解释边界。

不能使用“comprehensive”“robust generalization”“graph necessity”“causal mechanism”或“original-generator early warning”等措辞。

### F. Evaluation classification：PASS

| block | classification | 可支持的范围 |
| --- | --- | --- |
| N9 controlled | constructed_gt / auxiliary synthetic task | 固定模板、teacher-forced observer 内的 source-held-out 可读性诊断 |
| N9 natural | real_gt | RAGTruth 已知开发回答的 annotated-error-token 探索性迁移 |
| W1 | self-supervised event selector joined to the above GT | pre-token事件的覆盖诊断；不是错误标签、不是因果干预 |

## 4. Freeze / fit 隔离

评价器先验证完整 800 capture 和输入 ID（`internal_error_units_evaluate.py:218-230`），然后只打开 train 与 calibration gold（`:232-237`）。每个模型的标准化和逻辑回归只用 train indices，温度及阈值只用 calibration indices（`:257-276`）。它随后为全部行生成预测并写出 `predictions.npz`、index 和 freeze（`:278-287`），之后才打开 `gold_test.json`（`:288-303`），最后才打开自然 annotations（`:331-353`）。静态代码顺序与当前 artifact SHA 一致，实测 split/source/pair 隔离也通过。

这一检查排除了评价脚本中的 test-label fit 和 natural-label fit。它不能证明在本次审计范围之外没有人工根据 test/natural 结果选择整个 N9 方法；计划与历史明确将自然 development 视为已使用数据，因此任何自然结果只能保留为探索性。

## 5. Claim impact

- **C1（内部 graph message 相对基线增加绑定判别信息）**：**unsupported for graph-specific efficacy**。受控 graph 不优于 nodes 或 shuffled graph；自然 graph 低于二者。只能说构造任务对 node/readout 可分，且该可分性没有建立 graph message 的必要性。
- **C2（W1 定位事实选择，窗口内绑定优于同容量对照）**：**unsupported on controlled; exploratory coverage only on natural development**。受控 onset/core 覆盖全部为 0；自然 W8 的 23/28 覆盖伴随 46.42% token 预算和 46.50% 正常回答窗口占用。
- **工程主张（800 条捕获、冻结预测、CPU fit/evaluate 完成）**：**supported within bounded audit scope**。文件、哈希、快照、ID 对齐和独立复算均通过。
- **自然 entity core、semantic scope、原生成器机制、图因果必要性、独立泛化**：**unavailable / unsupported**。

## 6. 必须保留的后续动作

1. 将实验追踪从 RUNNING 改为完成，并明确记录 `COMPLETE_NEGATIVE / EXPLORATORY_TRANSFER_ONLY`；不要重复运行或覆盖冻结目录。
2. 不采用 graph 作为有效主线，不把 nodes/shuffled 同样完美的构造分数包装成 graph 增益。所有自然负迁移数值原样保留。
3. 在任何 W1 结论旁同时报告 16-token burn-in、受控 108/128 onset 结构不可达、实际窗口 token/正常回答覆盖和 0 core hit。
4. 后续若需要新的 W1 受控检验，应在看不到本测试结果的独立 roster 上让事实核心出现在 burn-in 之后，并预注册等预算位置/随机事件对照；不得修改本轮冻结结果或在本轮 test 上调 window/threshold。
5. 后续生成 JSON/表格时将 AUROC/AP 的微小浮点越界 clamp 到 `[0,1]`，同时保留本次原始文件不变。
6. 记录 `predictions.npz` 的 mtime 异常；后续冻结可使用只读权限或独立、带时间戳的 hash witness。当前结论只依赖匹配的冻结字节 SHA，不依赖 mtime。

## 7. 审计边界

这是同家族 fresh-agent 语义审计，按 skill 规则只能标为 provisional。为遵守 bounded main-results 范围，本审计没有重新哈希四个模型 shard，也没有重新读取/重算全部 800 个大型高维 NPZ；验证了 manifest/执行快照、全部小 sidecar、28 个分层大型样本、288 个主结果的 event member，以及 successful evaluator 对 800 个 NPZ 的逐文件 hash gate。没有重做 GPU forward、float16 投影或 36 个 fit。该范围足以验证报告数值、GT join、split isolation、freeze 顺序和常见完整性失败模式，不构成对底层模型运行的全量法证复演。
