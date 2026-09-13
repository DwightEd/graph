# Ownership source/layer — fresh agent-follows-doc witness

日期：2026-09-12。身份：独立执行代理，未编写或修复本次 runner；仅原样执行一次，之后进行只读产物核验。实验属于看过前两轮结果后冻结的探索性消歧。科学审稿状态：`REVIEW_UNAVAILABLE`。

## 文档、环境与唯一执行

已读取 `/root/.agents/skills/run-experiment/SKILL.md`、其 `shared-references/compute-env-contract.md`、`graph/.aris/compute/local.md` 和 `graph/docs/OWNERSHIP_SOURCE_LAYER_PLAN_20260912.md`。研究根目录及所有可见祖先、graph/reanchor 根目录、相关 scripts/refine-logs 路径未发现适用的 `AGENTS.md` 或 `CLAUDE.md`。`rg` 在本环境不可用，文件检查用 Python / find 完成；这不影响 runner。

`graph/.aris/compute/env-spec.json` 按排序键、无空白 JSON 的规范形式计算 SHA-256，前八位为 `8d044d57`，与现有 ledger 一致。复用 research 环境，无安装、重建或环境规格修改。预检 GPU 0 为 NVIDIA GeForce RTX 4090，24564 MiB 总显存、1 MiB 已用，无计算进程；目标输出目录不存在。父代理确认其余 GPU 作业已结束；运行期间只见一个计算进程。结束后 GPU 再次为 1 MiB、无计算进程。

从 `reanchor` 根目录执行的实验命令完全按文档：

```bash
bash scripts/run_ownership_source_layer.sh --output outputs/ownership_source_layer_20260912
```

外层仅增加 `set -euo pipefail`、目标/日志不存在检查及 `2>&1 | tee /tmp/ownership_source_layer_witness_20260912.log`，不改变命令参数或环境。stdout/stderr 全部在 `/tmp`，未写入实验输出目录。实验启动一次，exit code 0；无重试、覆盖或代码修改。

产物：`reanchor/outputs/ownership_source_layer_20260912`。runner 内部 elapsed 为 **26.869655 s**（不包括 Python 启动/import 阶段），CUDA allocated peak 为 **16,365,352,448 bytes / 15.241 GiB**。设置记录 torch `2.8.0+cu126`、transformers `4.57.1`、BF16、eager attention、seed 20260912。此 witness 验证复用环境中的文档执行，不证明干净安装可复现，也不替代独立科学审稿。

## 矩阵、来源和数值核验

- 完整覆盖 **4 trace × 2³ = 32 source worlds** 和 **32 层 × 2 窗口 = 64 layer patches**，条件唯一、无缺项。每份保存 logits 为 `(2, 128256)` 且全部有限；source worlds 共 64 个端点记录。
- 新 manifest **139/139** 文件哈希及覆盖集合匹配，输出目录除 manifest 本身外无未登记文件。上轮 multilayer manifest **116/116**、首轮 factorial manifest **18/18** 也独立重算通过。settings 中 **7 个 code SHA-256**、**18 个 input SHA-256** 全部匹配；7 份 `executed_*` 内容与登记源码哈希一致。
- 32 个保存 token arrays 逐元素核验：A 只改 prompt 位置 76 的 `12→14`；B 只改位置 341 的 `14→12`；H 只改对应 trace 的既往 grill 上界 `14→12`，生成步依次 99/100/98/109。除这些指定位置外 tokens 完全不变。A=B 时 prompt 数值多重集合保持原样；A≠B 时改变。A-only/B-only 的两个来源值相同，不能用作竞争归属准确率样本。
- `samples.jsonl` 显示 00012/00013/00014/00015 的真实生成 seeds 分别为 **0/1/2/3**；`00012` 等是 trace 名称。四者 source_id 全为 **14375**，保存 prompt tokens 逐元素相同。四条 seed 是一个 source 的生成对照，**独立来源数仍为 1**。
- 所有 source 端点及每个 layer 被干预端点的 margin、原生 argmax 与 max tie count 均从保存的全词表数组独立重算匹配。候选 token IDs 为 14→975、12→717。
- 未来 H 对所在 grill 预测及已读取的更早位置，独立比较 **20 个端点配对的全词表 logits**，最大误差均严格为 **0**。00014 的 t61 控制也位于其 H 之前。其余没有保存的更早位置不计入此项独立检查。
- 00012 的 A=B∈{0,1}、H∈{0,1} 共四个世界，对首轮对应两个端点的全词表复现误差均为 **0**。
- 64 个 layer 记录均有非空 prefix：32 个各 **99** queries，另 32 个各 **131** queries，共 **7360** 个早期 query 检查；记录的最大逐 logit 误差全部 **0**。已检查 executed runner 的断言实际比较 `full_logits[:step]`，没有空切片。完整早期 logits 未保存，因此这一总 prefix 结论基于本次真实 runner 执行及记录，不能说是事后独立重算了所有早期 logits。另有 **32** 个 onion patch 保存的 grill t99 早期端点，已直接从保存数组独立核验与 baseline 完全相同。
- 旧单层 7/15/23/31 × 两窗口共 **8** 个候选 margin 精确复现：grill 为 7.0/4.25/4.125/7.0；onion 为 −10.375/−9.5/−8.875/−10.375。

## 数值目标窗口的 A/B/H 平均主效应

单位为 `logit(14)−logit(12)` 的平均变化；在另两个二值因素上等权平均。全部 32 原始条件和全词表结果保存在 `source_worlds.json` 及对应 `.npy`，不能将边际主效应解读为不存在交互。

| trace | window | A 主效应 | B 主效应 | H 主效应 |
|---|---|---:|---:|---:|
| 00012 | grill | 3.18750 | -16.31250 | 0.00000 |
| 00012 | onion | 17.71875 | 0.03125 | -1.65625 |
| 00013 | grill | 3.68750 | -13.31250 | 0.00000 |
| 00014 | grill | 3.81250 | -16.43750 | 0.00000 |
| 00015 | grill | 4.93750 | -15.06250 | 0.00000 |

本结果在同一 source 的四条生成上表现为：正确 grill 更强地响应 B，00012 的错误 onion 窗口更强地响应 A。grill 的 H=0 是未来 token 不能影响该预测的因果一致性检查；onion 的 H 平均效应非零，不能据此声称历史完全无关。新增正确 grill 窗口没有增加错误窗口数。

## 全部单层结果

原始 baseline margin：grill **7.125**，onion **−10.375**。32 层穷举中，grill margin 范围 **3.625–7.750**，onion 为 **−10.750–−7.250**。两个窗口均无单层候选翻转，也无原生 argmax 改变；全部 64 个被干预端点 max tie count=1。这仅说明当前定义的单层 X 干预未足够翻转这两个窗口，不推出一般单层不可能有效，也不把多层效应假定为单层效应之和。

| 零基层 | grill margin | grill argmax | onion margin | onion argmax |
|---:|---:|---|---:|---|
| 0 | 7.000 | 14 | -10.375 | 12 |
| 1 | 7.000 | 14 | -10.375 | 12 |
| 2 | 7.000 | 14 | -10.500 | 12 |
| 3 | 7.000 | 14 | -10.375 | 12 |
| 4 | 7.000 | 14 | -10.375 | 12 |
| 5 | 7.000 | 14 | -10.500 | 12 |
| 6 | 7.125 | 14 | -10.375 | 12 |
| 7 | 7.000 | 14 | -10.375 | 12 |
| 8 | 6.875 | 14 | -10.250 | 12 |
| 9 | 7.000 | 14 | -10.375 | 12 |
| 10 | 6.875 | 14 | -10.375 | 12 |
| 11 | 7.000 | 14 | -10.375 | 12 |
| 12 | 7.125 | 14 | -10.375 | 12 |
| 13 | 6.875 | 14 | -10.500 | 12 |
| 14 | 6.750 | 14 | -10.375 | 12 |
| 15 | 4.250 | 14 | -9.500 | 12 |
| 16 | 3.625 | 14 | -7.250 | 12 |
| 17 | 6.875 | 14 | -10.250 | 12 |
| 18 | 6.250 | 14 | -9.500 | 12 |
| 19 | 7.750 | 14 | -10.750 | 12 |
| 20 | 7.625 | 14 | -10.500 | 12 |
| 21 | 7.000 | 14 | -10.500 | 12 |
| 22 | 6.625 | 14 | -9.375 | 12 |
| 23 | 4.125 | 14 | -8.875 | 12 |
| 24 | 5.125 | 14 | -9.250 | 12 |
| 25 | 7.750 | 14 | -10.750 | 12 |
| 26 | 7.625 | 14 | -10.625 | 12 |
| 27 | 7.500 | 14 | -9.500 | 12 |
| 28 | 7.375 | 14 | -10.375 | 12 |
| 29 | 7.125 | 14 | -10.375 | 12 |
| 30 | 6.500 | 14 | -10.250 | 12 |
| 31 | 7.000 | 14 | -10.375 | 12 |

## 文档与实际产物的差异及范围

CLI 和声明矩阵正常完成，没有需要即席修复的执行差异。计划要求“对每个窗口报告 A/B/H 平均主效应”，但现有 `summary.json/source_main_effects` 只聚合 **5 个数值目标窗口**。00013 的 `correct_onion` t126（实际 token ` over`）、00014 的 `simmer_control` t61（实际 `20`）、00015 的 `alternative_grill` t142（实际 `10`）具备全部原始 full-vocabulary logits、原生 argmax/tie、逐世界相对基线 JS / 固定实际 token logp 变化，却没有聚合 A/B/H 的全词表效应报告。这属于**原始 run 摘要的报告范围缺口**，不是漏跑条件。未为 witness 改代码或重跑，也未将这些控制窗口的机械 14−12 候选差作科学解释。

父代理随后在独立 CPU 分析阶段补齐了所有窗口的因素分布对比，保存于 `reanchor/results/ownership_source_layer_analysis_v2_20260912/source_contrasts.csv` 与 `numbers.json.factor_distribution_effects`，没有改动原 run summary。witness 另行检查该分析 manifest **8/8** 哈希通过；CSV 共 **96** 逐对对比，恰为 8 窗口 × 3 因素 × 4 对比；24 组平均/最大 JS、argmax 变化计数均从 CSV 独立聚合匹配，每个 before→after 仅切换对应因素一位。这证明控制窗口的聚合报告现已补齐。本次补充核验没有重新调用模型，也不把分析脚本当作独立科学审稿。对于已证实全 logits 完全相同的未来 H 控制，约 1e−8 的 JS 数值残差来自分布计算的有限精度，不能另计为未来因果影响。

本实验没有拟合检测器、没有跨 source 验证，不能报告自动回看定位准确率、约束归属检测准确率、跨来源泛化或图模型必要性。独立执行核验与科学审稿分别陈述；`REVIEW_UNAVAILABLE` 保持不变。

只读核验脚本及完整机器可读结果分别在 `/tmp/ownership_source_layer_verify_20260912.py`、`/tmp/ownership_source_layer_verification_20260912.json`。上述脚本仅读取已保存结果，不调用模型或占用 GPU；未往实验输出目录添加任何文件。
