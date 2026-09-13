# Soft graph runner engineering review — 2026-09-13

范围限定为 `route_graph/soft_graph_runner.py`、`experiments/evaluate_soft_graph.py`、`experiments/interleave_soft_graph.py` 及新增 CPU pipeline tests。未运行 GPU，未把该工程检查当作机制或效果结论。

## 结论

**Critical：0**

**Required：2**

1. **冻结设置与执行代码快照不是一个可恢复的原子对象。** `soft_graph_runner.settings()` 在 [soft_graph_runner.py](../route_graph/soft_graph_runner.py#L58) 先发布不可覆盖的 `settings.json`，随后才逐文件复制 `executed_code/` 并发布其 manifest（L59–66）。进程在其中断后，下一次 `settings()` 因已有 settings 直接返回（L49–56），既不补建也不核验快照。wrapper 也只要求 settings 存在（[interleave_soft_graph.py](../experiments/interleave_soft_graph.py#L204)），会在停止 population 后运行这种没有完整、可核验执行快照的审计。应将快照建在临时目录、逐项核验后以完整 manifest 原子发布，并把 settings 与该 manifest 绑定；已有 output 需在任何 prepare/phase/wrapper 入口拒绝缺失或哈希不符的快照。添加“settings 发布后、snapshot 完成前中断”的 resume 回归。

2. **评价器没有强制使用或校验运行时冻结的代码快照。** evaluator 只读取 live import 的 `read_artifact`，而 result 中记录的 `evaluator_sha256` 只是事后描述（[evaluate_soft_graph.py](../experiments/evaluate_soft_graph.py#L16), [evaluate_soft_graph.py](../experiments/evaluate_soft_graph.py#L91)）。它没有要求 `executed_code/manifest.json` 存在，也没有将它同 `settings.code_sha256` 和实际 evaluation/runner 源关联。因此 run 完成后修改 live evaluator/runner，再评价同一 artifacts，仍会写出一个“冻结”结果，实际算法却已变。应在评价前验证完整 snapshot manifest 与 settings，且从 snapshot 执行评价器或拒绝 live evaluator/runner 哈希和 snapshot 不一致；为两种篡改都加无标签读取前的拒绝测试。

## 已确认

- A→B→C→D→merge 的每行完整 upstream roster、上游字节哈希、B feature 文件和 D donor artifact 都会递归验证；merge 之后才将 progress 标为 `complete`。
- evaluator 在 `complete` 前拒绝；随后才读取标签字节，并核对 evaluation manifest、input/label hash、annotation identity 与原 response span。每个非空白词必须同 merge words 一一对应，故全词分母不会被 selected/native 成功数缩小。
- interleave 保留单一 scheduler lock、同一 `.phase.lock` 文件描述符交接、pidfd 目标核验、population 输出锁和失败后的有界恢复；它与旧 native wrapper 共用全局 interleave lock，避免两者并发。
- 独立 CPU：`tests/test_soft_graph_pipeline.py` **5 passed**（14.58s）；Ruff 对三模块和该测试通过。

不将尚未接入的 structural/null controls 或没有 GPU 结果列为本次软件缺陷；native 输出仅被代码标为 target-dependence，未被提升为正误路由证明。

## 复审闭合（2026-09-13）

此前 **2 项 Required 均已关闭**；当前范围为 **Critical 0 / Required 0**。

- `settings()` 现在先逐项生成和验证 `executed_code` 及 manifest，最后才发布 `settings.json`。已有 settings 的每次 prepare/resume 也会验证 manifest、每个快照字节和 live 源哈希。因此复制中断不留下可被 wrapper 接受的 settings，损坏快照不能恢复。
- evaluator 在打开任何 labels 文件之前调用同一 `verify_executed_code()`；快照 manifest、快照文件和 live runner/evaluator 代码任一不一致都会拒绝。若 live 源变化，必须从保存的快照执行，而不能给旧 artifacts 写入由新代码计算的评价。

独立 CPU 验证：`tests/test_soft_graph_freeze.py tests/test_soft_graph_pipeline.py` **9 passed**（13.20s）；相关 Ruff 检查通过。
