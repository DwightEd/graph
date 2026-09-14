# 当前实验审计：S10（2026-09-14）

[完整审计](docs/S10_RESULT_AUDIT_20260914.md)：fresh同家族只读复算，完整性PASS、主张使用WARN/provisional。主/原始指标独立复算最大差0；20/20元数据哈希、989特征哈希及671/168/150来源划分核验通过；正常回答anyalarm限制已报告。可选200次bootstrap未独立重跑，不称跨家族外部科学验收。没有合并或发布阻断。

## 下方为历史审计记录

# 实验审计状态

日期：2026-09-12。审计员：不可用。Overall verdict：NOT RUN；integrity_status：unavailable。

已按照 experiment-audit 收集代码、原始结果、配置、标签和主张路径，见 `.aris/audit_artifacts.json`。所需独立科研审稿后端没有可调用工具；没有执行外部完整性审计，也没有给出 PASS/WARN/FAIL 裁决。工程复审和本地数值存在性检查不替代它。

11项数字通过本地确定性的JSON路径核对，记录于 `.aris/evidence_precheck.json`；标准技能 evidence_check.py 未能解析，按Policy B跳过，使用明确标注的本地替代检查。该检查只证明数字存在。

当前研究主张均未获支持，判定保持 pending Codex review。完整实验范围和限制见 [本轮报告](docs/METHOD_ITERATION_20260912.md)。

最新O1–O3机制实验的代码/配置/原始输出/分析路径另在`.aris/ownership_audit_artifacts_20260912.json`，10项数字本地JSON路径核验通过，见`.aris/ownership_evidence_precheck_20260912.json`。这只是证据存在性检查；独立科学审稿/完整性裁决仍NOT RUN。实际机制观察及未支持的自动归属主张见docs/OWNERSHIP_MECHANISM_RESULTS_20260912.md。
