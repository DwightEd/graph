# 第三轮工程复审

日期：2026-09-12。范围仅限第二轮提出并刚修复的四项：机制置换 seed、机制错误有效集、冻结 AUROC 的 score manifest、报警配置与配对对照。没有扩展全项目，没有使用 GPU，也没有修改 v3 正在运行的代码或 v2 结果。

## 结论

**APPROVE（工程修复范围）**。四项修复均已落实，未发现仍会阻断 v3 写入新目录或改变既定负研究结论的问题。该结论只确认软件行为和审计边界，不表示 C1/C2 得到支持，也不是外部科研裁决。

## 1. 条件级稳定随机置换：通过

`reanchor/src/decoding/binding_validation.py:139-147` 使用模板、布局、对象、世界和末层编号生成 `reading_seed`，并显式传给 `source_context_readout`。`binding_validation.py:223-225` 用 SHA256 从固定 base seed `20260912` 和完整条件身份派生 64-bit seed；`binding_validation.py:191-192` 将 base 和派生方案写入 settings。运行脚本默认输出也已改为独立的 `binding_validation_v3_20260912`，不会覆盖 v2。

针对 96 个合成条件身份的纯 CPU 检查得到 96 个唯一 seed。3 端点产生全部 6 种排列，4 端点产生 23/24 种排列；各 endpoint 位置不再在所有条件中重复同一映射。单次随机排列仍可自然包含固定点，这是预声明随机 null 的正常结果，不再是每个同端点数条件系统复用同一固定点模式。

`route_graph/context.py:94-107` 同时保存 seed、完整排列、实际移动 endpoint 比例和跨候选身份移动比例。因此每个条件的 null 强度可审计，不能再把没有跨候选移动的抽样误写成充分破坏归属。

对应回归位于 `reanchor/tests/test_binding_validation.py:70-82`，覆盖 seed 唯一性、重跑稳定性和各位置映射变化。

## 2. 实际模型错误与各方法有效集：通过

`binding_validation.py:251-268` 的 `actual_errors` 现在对 layout 的全部 `selected` 条件计数，不受顶层共同有效集影响；每个方法分别报告 `valid`、`valid_errors` 和只在本方法有效错误中的 `error_correct`。

`binding_validation.py:269-289` 又为 context 相对 direct、shuffled、count 分别构造两方法交集，独立报告共同条件、共同实际错误和双方正确数。一个无关对照失效不再删除 context-vs-direct 的错误条件。`test_binding_validation.py:61-68` 覆盖了“存在真实模型错误、direct 有效、context 无效、四方法共同集为空”的回归场景。

当前 v2 的 128 条原生 top-1 全部正确，因此这些字段不会事后创造错误检测证据；v3 仍应按实际模型预测报告。

## 3. 冻结 AUROC score manifest 与输出审计：通过

`route_graph/archive_evaluation.py:108-112` 在读取标签和运行 `RouteEvaluator` 前，要求相邻 `complete.json` 标记 `labels_used=false`，且其中的 `scores_sha256` 与实际 score 文件一致。被修改或未完成的 score 文件不能进入标签评价。

`archive_evaluation.py:116-121` 在最终评价产物中写入 score、archive index 和 label 文件哈希，并使用原子 JSON 替换保存。这样可以从评价产物追溯冻结 score、恢复归档和实际标签输入。`tests/test_route_alarms.py:101-128` 验证 score 哈希不符时在接触不存在的 label 文件之前失败，证明检查位于标签接入之前。

## 4. 报警参考配置和配对非图对照：通过

校准器在 `route_graph/alarms.py:42-44` 固定 `neighbors=3`、`per_source=8`。评价入口在 `alarms.py:127-138` 除 archive 和 score 哈希外，还要求评分完成记录明确为相同的 3/8；同一 archive 上用其他参考配置生成的 scores 不能混入。`tests/test_route_alarms.py:94-98` 覆盖配置不符。

`alarms.py:284-316` 的来源配对 bootstrap 现在覆盖 entropy、negative margin、position，以及 observed/null/signal 的 direct 与 kNN 版本。纯数组调用产生预期的 9 个 baseline × 2 个报警指标，共 18 个配对结果。每项继续在同一 source 内先求方法差，再按 source 重采样。

全流 95% fit 分位仍不等于测试集等实际报警预算；“每回答至多一次”只固定容量，实际发出数仍可不同。现有 report 的 scope 保留了这一限制，因此本修复没有把当前报警结果升级为 C2 证据。

## 验证

- graph 定向回归：`tests/test_route_alarms.py tests/test_source_context.py`，8 passed。
- reanchor 定向回归：`tests/test_binding_validation.py`，3 passed。
- 额外纯数组检查：条件 seed 唯一且端点排列随条件变化；配对 bootstrap 输出全部 18 个预期键。
- 全程设置 `CUDA_VISIBLE_DEVICES=''`，未使用 GPU；没有重复运行全套测试。

第二轮报告保持原样，作为问题与修复过程的审计记录。v3 必须继续使用新输出目录，不能回写或混合 v2 数字。负研究结论保持不变。
