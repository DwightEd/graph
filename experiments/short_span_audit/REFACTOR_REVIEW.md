# 当前检测的问题与重构实测

2026-09-22。基于 main `3bfc3d5`，纳入此前未推送的显存修复 `24bfcb9`。
本轮修改采集计算与评价，不通过 TEST 调分数、选头、换方向或增加新的异常评分器。

## 1. 检测仍有什么问题

1. **现有分数主要是 attention 构型的罕见度，不是事实支持度。** 当前这份实验只输入 self-attention 信号；raw 是当前逐头构型的近邻距离，moment 加入过去窗口的二阶矩，smooth 平滑标量。即使异常可分，也不能由此读出主体、时间、否定和范围的适用关系。
2. **短段与起点检出弱，平滑会削弱短脉冲。** Data2txt 的 144 个 1–8 token 错误段，raw 在段结束前命中 46 个，moment 41 个，smooth 11 个；严格起点分别只有 3、4、6 个。总体 AUROC 不能替代这个目标。
3. **正常 token FPR 小，不代表完整回答很少误报。** 例如 QA raw 的正常 token FPR 为 3.15%，但 97 条全正常回答中有 93 条至少一次报警。旧报告缺少这个重要口径。
4. **新贡献只是测量，尚未成为独立验证的无标签检测器。** 新采集用标签选择短段及正常对照；正负贡献是实际选择 token 的局部门控敏感度。prompt/history 分组没有独立证据适用性，正贡献不等于真，负贡献不等于假。观察模型重放也不能冒充原生成模型的机制。
5. **成本此前不适合逐目标运行。** 原版完整前缀反传保留大量 attention 激活；仅修为单步反传后，每个目标仍重复预填充整个历史、重复计算固定投影 Gram 矩阵，浪费计算。此前没有实际每目标耗时和显存记录。

## 2. 本轮实测：449 条冻结回答

读取原 continuity 快照中的冻结分数、报警和匹配，重算三个任务。
与最新上传的 `short_span_audit_v1_review.tar(1).gz` 逐项比较，原有各长度桶的 span 统计以及 frozen / TEST 诊断 FPR **完全一致**。
以下沿用每个方法自己的原冻结混合 CAL 阈值，不能当作相同正常 FPR 的算法优劣比较。

|任务|方法|短段结束前命中|短段起点命中|正常 token FPR|全正常回答至少一次报警|
|---|---|---:|---:|---:|---:|
|Data2txt|raw|46/144|3/144|4.47%|27/27|
|Data2txt|moment|41/144|4/144|4.71%|27/27|
|Data2txt|smooth|11/144|6/144|4.88%|17/27|
|Summary|raw|11/22|0/22|4.52%|97/99|
|Summary|moment|7/22|1/22|4.03%|74/99|
|Summary|smooth|4/22|2/22|4.01%|49/99|
|QA|raw|5/11|1/11|3.15%|93/97|
|QA|moment|3/11|1/11|2.52%|59/97|
|QA|smooth|2/11|1/11|2.29%|31/97|

Data2txt raw 有 36 个完整漏检的短段，在其后最多 15 个正常 token 中出现报警；moment 26 个，smooth 8 个。这些都仍算短段失败。这里只描述时间关联，不声称后续误报由前一错误导致。
观察区间遇到下一段错误或回答末尾停止；上表完整正常回答误报数不依赖未测 token 的推断。
TEST 已反复查看，仍是探索性分析；本轮没有新自然数据检测 AUROC，也没有声称检测精度提高。

## 3. 已落地的改动

- `state_audit/model/replay.py`：用 SDPA 分块扩展完整历史 KV，保持原生 causal/sliding-window mask。
- `state_audit/attribution.py`：当前 query 单独求导；同一回答按严格递增目标复用 KV，目标之间 detach，禁止携带上一目标的反传图。固定每头 `W_O,h^T W_O,h` 每回答只算一次；attention、正负贡献和 value energy 的定义不变。
- `capture.py`：去重目标、只计算尚未保存的 NPZ；缺口历史无梯度推进；异常和保存失败仍释放 hooks/参数标记/迭代器。旧已完成 NPZ 可直接续跑，不缩短历史、不平均 heads。
- `capture_reporting.py`：保存真实耗时、新增预填充 token 数、CUDA allocated/reserved 峰值及覆盖；旧文件未知成本不补零，CPU 不伪造显存。
- `lifecycle.py`：deadline 0/1/3/7 的全 span 分母及时召回与缺测上下界；段后连续报警长度、迟到报警、截尾原因；正常回答整体误报。
- `comparisons.py`：将来源重采样和配对比较从评价主流程拆开；匹配段同时比较召回与误报，增加完整配对 TP/FN/FP/TN 和 recall−FPR 差。
- `metrics.py` / `reporting.py`：保留旧字段，增加 9–16 / 17+ 长段桶、逐回答表及上述诊断。

采集的每个目标仍有一次独立 backward；不是将所有目标 loss 相加后冒充逐 token 贡献。
KV 复用不删除 target 之间的历史位置，也不消费尚未出现的未来词。
低精度 SDPA/eager 舍入可能不同；不承诺逐位相同。

## 4. 验证与尚未完成的研究

软件回归 **226 passed**。已运行小型 Llama、Mistral（含滑窗）、Qwen2 的完整前缀/缓存/顺序复用对照，核对 logits、attention、有符号贡献与能量；验证有限差分、无未来信息、跨目标梯度释放、中断续跑、缺测分母和报警截尾。顺序采集测试明确断言每个输入前缀 token 只前向一次。另重跑上面的 449 条真实冻结分数。

当前环境只有 CPU，**没有用户 8B 检查点和 24GB GPU 的实测峰值或端到端采集速度**。相关指标现已在真实运行时记录；降低 prefill chunk 也不能消除权重和完整 KV 自身的存储开销。

未完成：独立适用性/主体—值—scope 表征；新增贡献在独立来源 FIT/CAL/TEST 的无标签全流验证；联合即时/持续分支的独立校准；在正常复述、低熵错误等对照上的机制确认。本轮不添加来源错接训练、图网络或标签调参去掩盖这些缺口。

## 5. 运行

```bash
git pull --ff-only origin main
bash experiments/short_span_audit/run.sh
bash experiments/short_span_audit/run_capture.sh --resume
```

默认沿用审计 `input_settings.json.tokenizer` 的观察模型；模型搬家时显式设置 `MODEL`。已有 settings/部分结果用 `--resume`；首次采集去掉该参数也可以。若预填充仍吃紧，加 `--prefill-chunk-size 64`，不需要重跑 CPU 审计或删除旧 NPZ。
详细输出字段、安装和测试命令见 README。
