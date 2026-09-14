## 2026-09-14 S10 当前产物

- 主实现：structural_detector/{features,experiment}.py；运行scripts/run_structural_fusion.sh。
- 无标签采集：../reanchor/src/decoding/structural_export.py；原生依赖迁入../reanchor/src/route_graph。
- 结果：docs/S10_RESULTS_20260914.md，results/s10_structural_fusion_20260914。
- 历史代码归档：archive/pre-structural-20260914；清理清单docs/STRUCTURAL_REFACTOR_20260914.json。未跟踪的notebook自动checkpoint移至outputs/structural_refactor_preserved_checkpoints_20260914保留。

# Research Output Manifest

原始逐数组哈希索引保留在各实验输出的 manifest.json；本索引记录代码、方案、报告与入口，不复制数组清单。

| Timestamp | Skill | File | Stage | Description |
|---|---|---|---|---|
| 2026-09-13T01:46:22+08:00 | monitor-experiment | ../reanchor/docs/PROJECT_HANDOFF_before_20260913_014622.md | implementation | 更新前备份 |
| 2026-09-13T01:46:22+08:00 | monitor-experiment | ../reanchor/docs/PROJECT_HANDOFF_20260913_014622.md | implementation | 本轮版本 |
| 2026-09-13T01:46:22+08:00 | monitor-experiment | ../reanchor/docs/PROJECT_HANDOFF.md | implementation | 当前入口 |
| 2026-09-13T01:46:22+08:00 | monitor-experiment | ../reanchor/docs/mechanism_experiment_before_20260913_014622.md | implementation | 更新前备份 |
| 2026-09-13T01:46:22+08:00 | monitor-experiment | ../reanchor/docs/mechanism_experiment_20260913_014622.md | implementation | 本轮版本 |
| 2026-09-13T01:46:22+08:00 | monitor-experiment | ../reanchor/docs/mechanism_experiment.md | implementation | 当前入口 |
| 2026-09-13T01:46:22+08:00 | monitor-experiment | docs/RESEARCH_STATUS_before_20260913_014622.md | implementation | 更新前备份 |
| 2026-09-13T01:46:22+08:00 | monitor-experiment | docs/RESEARCH_STATUS_20260913_014622.md | implementation | 本轮版本 |
| 2026-09-13T01:46:22+08:00 | monitor-experiment | docs/RESEARCH_STATUS.md | implementation | 当前入口 |
| 2026-09-13T01:46:22+08:00 | experiment-plan | refine-logs/EXPERIMENT_PLAN_before_20260913_014622.md | implementation | 更新前备份 |
| 2026-09-13T01:46:22+08:00 | experiment-plan | refine-logs/EXPERIMENT_PLAN_20260913_014622.md | implementation | 本轮版本 |
| 2026-09-13T01:46:22+08:00 | experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | 当前入口 |
| 2026-09-13T01:46:22+08:00 | research-refine | refine-logs/FINAL_PROPOSAL_before_20260913_014622.md | implementation | 更新前备份 |
| 2026-09-13T01:46:22+08:00 | research-refine | refine-logs/FINAL_PROPOSAL_20260913_014622.md | implementation | 本轮版本 |
| 2026-09-13T01:46:22+08:00 | research-refine | refine-logs/FINAL_PROPOSAL.md | implementation | 当前入口 |
| 2026-09-13T01:46:22+08:00 | experiment-plan | refine-logs/EXPERIMENT_TRACKER_before_20260913_014622.md | implementation | 更新前备份 |
| 2026-09-13T01:46:22+08:00 | experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260913_014622.md | implementation | 本轮版本 |
| 2026-09-13T01:46:22+08:00 | experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | 当前入口 |
| 2026-09-13T01:46:22+08:00 | analyze-results | docs/MECHANISM_ITERATION_20260913_before_20260913_014622.md | implementation | 更新前备份 |
| 2026-09-13T01:46:22+08:00 | analyze-results | docs/MECHANISM_ITERATION_20260913_20260913_014622.md | implementation | 本轮版本 |
| 2026-09-13T01:46:22+08:00 | analyze-results | docs/MECHANISM_ITERATION_20260913.md | implementation | 当前入口 |
| 2026-09-13T01:46:22+08:00 | monitor-experiment | docs/RAGTRUTH_POPULATION_RUN_STATUS_20260912_before_20260913_014622.md | implementation | 更新前备份 |
| 2026-09-13T01:46:22+08:00 | monitor-experiment | docs/RAGTRUTH_POPULATION_RUN_STATUS_20260912_20260913_014622.md | implementation | 本轮版本 |
| 2026-09-13T01:46:22+08:00 | monitor-experiment | docs/RAGTRUTH_POPULATION_RUN_STATUS_20260912.md | implementation | 当前入口 |
| 2026-09-13T01:46:22+08:00 | research-refine | refine-logs/REFINE_STATE_before_20260913_014622.json | implementation | 更新前备份 |
| 2026-09-13T01:46:22+08:00 | research-refine | refine-logs/REFINE_STATE_20260913_014622.json | implementation | 本轮版本 |
| 2026-09-13T01:46:22+08:00 | research-refine | refine-logs/REFINE_STATE.json | implementation | 当前入口 |
| 2026-09-13T01:46:22+08:00 | run-experiment | .aris/compute/local.md | implementation | 复用环境与执行恢复见证 |
| 2026-09-13T01:46:22+08:00 | research-refine | findings.md | implementation | 追加关系/定位负结果及范围 |
| 2026-09-13T01:46:22+08:00 | research-refine / run-experiment | docs/RELATION_MECHANISM_RESULTS_20260913.md | implementation | 本轮代码、协议或证据；原始数组见各run manifest |
| 2026-09-13T01:46:22+08:00 | research-refine / run-experiment | docs/MISSING_CONSTRAINT_PLAN_20260913.md | implementation | 本轮代码、协议或证据；原始数组见各run manifest |
| 2026-09-13T01:46:22+08:00 | research-refine / run-experiment | route_graph/relation_readout.py | implementation | 本轮代码、协议或证据；原始数组见各run manifest |
| 2026-09-13T01:46:22+08:00 | research-refine / run-experiment | tests/test_relation_readout.py | implementation | 本轮代码、协议或证据；原始数组见各run manifest |
| 2026-09-13T01:46:22+08:00 | research-refine / run-experiment | docs/RELATION_ONLY_PLAN_20260913.md | implementation | 本轮代码、协议或证据；原始数组见各run manifest |
| 2026-09-13T01:46:22+08:00 | research-refine / run-experiment | docs/RELATION_ROUTING_PLAN_20260913.md | implementation | 本轮代码、协议或证据；原始数组见各run manifest |
| 2026-09-13T01:46:22+08:00 | research-refine / run-experiment | ../reanchor/src/decoding/relation_inputs.py | implementation | 本轮代码、协议或证据；原始数组见各run manifest |
| 2026-09-13T01:46:22+08:00 | research-refine / run-experiment | ../reanchor/src/decoding/relation_only.py | implementation | 本轮代码、协议或证据；原始数组见各run manifest |
| 2026-09-13T01:46:22+08:00 | research-refine / run-experiment | ../reanchor/src/decoding/relation_routing.py | implementation | 本轮代码、协议或证据；原始数组见各run manifest |
| 2026-09-13T01:46:22+08:00 | research-refine / run-experiment | ../reanchor/scripts/interleave_relation_only.py | implementation | 本轮代码、协议或证据；原始数组见各run manifest |
| 2026-09-13T01:46:22+08:00 | research-refine / run-experiment | ../reanchor/scripts/run_relation_only.sh | implementation | 本轮代码、协议或证据；原始数组见各run manifest |
| 2026-09-13T01:46:22+08:00 | research-refine / run-experiment | ../reanchor/scripts/run_relation_routing.sh | implementation | 本轮代码、协议或证据；原始数组见各run manifest |
| 2026-09-13T01:46:22+08:00 | research-refine / run-experiment | ../reanchor/scripts/analyze_relation.py | implementation | 本轮代码、协议或证据；原始数组见各run manifest |
| 2026-09-13T01:46:22+08:00 | research-refine / run-experiment | ../reanchor/results/relation_analysis_20260913/manifest.json | implementation | 本轮代码、协议或证据；原始数组见各run manifest |
| 2026-09-13T01:46:22+08:00 | research-refine / run-experiment | ../reanchor/results/ragtruth_population_partial_20260913.json | implementation | 本轮代码、协议或证据；原始数组见各run manifest |
| 2026-09-13T01:46:22+08:00 | code-review-and-quality / run-experiment | refine-logs/relation_analysis_engineering_20260913.md | implementation | 独立工程/执行/分析报告 |
| 2026-09-13T01:46:22+08:00 | code-review-and-quality / run-experiment | refine-logs/relation_only_engineering_20260913.md | implementation | 独立工程/执行/分析报告 |
| 2026-09-13T01:46:22+08:00 | code-review-and-quality / run-experiment | refine-logs/relation_only_witness_20260913.md | implementation | 独立工程/执行/分析报告 |
| 2026-09-13T01:46:22+08:00 | code-review-and-quality / run-experiment | refine-logs/relation_routing_engineering_20260913.md | implementation | 独立工程/执行/分析报告 |
| 2026-09-13T01:46:22+08:00 | code-review-and-quality / run-experiment | refine-logs/relation_routing_witness_20260913.md | implementation | 独立工程/执行/分析报告 |
| 2026-09-13T01:46:22+08:00 | experiment-audit | .aris/relation_population_audit_artifacts_20260913_before_20260913_014622.json | implementation | 更新前备份 |
| 2026-09-13T01:46:22+08:00 | experiment-audit | .aris/relation_population_audit_artifacts_20260913_20260913_014622.json | implementation | 本轮版本 |
| 2026-09-13T01:46:22+08:00 | experiment-audit | .aris/relation_population_audit_artifacts_20260913.json | implementation | 当前入口 |
| 2026-09-13T01:47:51+08:00 | research-refine | refine-logs/FINAL_PROPOSAL_20260913_014751.md | implementation | 明确整样本与O4的Q/K捕获范围及V/O因子存储 |
| 2026-09-13T01:47:51+08:00 | research-refine | refine-logs/FINAL_PROPOSAL.md | implementation | 明确整样本与O4的Q/K捕获范围及V/O因子存储 |
| 2026-09-13T01:47:51+08:00 | run-experiment | results/relation_iteration_final_checks_20260913.json | implementation | 最终冻结代码、检查与实时运行快照 |

| 2026-09-13T02:36:21.408455+08:00 | research-refine | reanchor/docs/PROJECT_HANDOFF.md | architecture revision | 当前范围和未验证项；历史保留 |
| 2026-09-13T02:36:21.408455+08:00 | research-refine | reanchor/docs/mechanism_experiment.md | architecture revision | 当前范围和未验证项；历史保留 |
| 2026-09-13T02:36:21.408455+08:00 | research-refine | docs/RESEARCH_STATUS.md | architecture revision | 当前范围和未验证项；历史保留 |

## Native audit implementation 20260913_042134

新增route_graph/audit_*.py、causal_groups.py、frozen_reader.py、evidence_anchor.py及真实native后端。入口audit_runner，调度experiments/interleave_native_audit.py。自然36条清单outputs/native_audit_design_20260913；审查native_{validation,pipeline,scheduler}_engineering_20260913.md。此时未启动GPU，部署前复审尚未关闭。

Native audit 2026-09-13T04:47:53+08:00: docs/NATIVE_METHOD_MODEL_20260913.md, docs/NATIVE_AUDIT_RUN_20260913.md, outputs/native_audit_v1_20260913/settings.json, experiments/evaluate_native_audit.py, experiments/summarize_native_audit.py; full36 launch preflight, no empirical result claimed.

| 2026-09-14 | research-refine | refine-logs/token_flow_20260914/ENGINEERING_RESULTS.md | implementation | RoutingResidual方法/独立评审/工程证据；自然效果未验证 |
| 2026-09-14 | research-refine | refine-logs/token_flow_20260914/FINAL_PROPOSAL.md | implementation | RoutingResidual方法/独立评审/工程证据；自然效果未验证 |
| 2026-09-14 | research-refine | refine-logs/token_flow_20260914/REFINEMENT_REPORT.md | implementation | RoutingResidual方法/独立评审/工程证据；自然效果未验证 |
| 2026-09-14 | research-refine | refine-logs/token_flow_20260914/REVIEW_SUMMARY.md | implementation | RoutingResidual方法/独立评审/工程证据；自然效果未验证 |
| 2026-09-14 | research-refine | refine-logs/token_flow_20260914/round-0-initial-proposal.md | implementation | RoutingResidual方法/独立评审/工程证据；自然效果未验证 |
| 2026-09-14 | research-refine | refine-logs/token_flow_20260914/round-1-refinement.md | implementation | RoutingResidual方法/独立评审/工程证据；自然效果未验证 |
| 2026-09-14 | research-refine | refine-logs/token_flow_20260914/round-1-review.md | implementation | RoutingResidual方法/独立评审/工程证据；自然效果未验证 |
| 2026-09-14 | research-refine | refine-logs/token_flow_20260914/round-2-refinement.md | implementation | RoutingResidual方法/独立评审/工程证据；自然效果未验证 |
| 2026-09-14 | research-refine | refine-logs/token_flow_20260914/round-2-review.md | implementation | RoutingResidual方法/独立评审/工程证据；自然效果未验证 |
| 2026-09-14 | research-refine | refine-logs/token_flow_20260914/round-3-refinement.md | implementation | RoutingResidual方法/独立评审/工程证据；自然效果未验证 |
| 2026-09-14 | research-refine | refine-logs/token_flow_20260914/round-3-review.md | implementation | RoutingResidual方法/独立评审/工程证据；自然效果未验证 |
| 2026-09-14 | research-refine | refine-logs/token_flow_20260914/round-4-refinement.md | implementation | RoutingResidual方法/独立评审/工程证据；自然效果未验证 |
| 2026-09-14 | research-refine | refine-logs/token_flow_20260914/round-4-review.md | implementation | RoutingResidual方法/独立评审/工程证据；自然效果未验证 |
| 2026-09-14 | research-refine | docs/TOKEN_GRAPH_INFORMATION_FLOW.md | implementation | 当前路由残差方法、数学边界与执行接口 |
| 2026-09-14 | research-refine | refine-logs/token_flow_20260914/REFINE_STATE.json | implementation | RoutingResidual精炼与审查归档；科学未验证 |
| 2026-09-14 | research-refine | refine-logs/token_flow_20260914/REFINE_STATE_20260914_final.json | implementation | RoutingResidual精炼与审查归档；科学未验证 |
| 2026-09-14 | research-refine | refine-logs/token_flow_20260914/round-5-review.md | implementation | RoutingResidual精炼与审查归档；科学未验证 |
| 2026-09-14 | research-refine | refine-logs/token_flow_20260914/score-history.md | implementation | RoutingResidual精炼与审查归档；科学未验证 |
| 2026-09-14 | research-refine | refine-logs/token_flow_20260914/round-4-review.raw.json | implementation | 原始评审逐字保存；md仅规范化Markdown行末空格 |
