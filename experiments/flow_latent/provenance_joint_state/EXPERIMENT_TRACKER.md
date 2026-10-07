# Experiment Tracker

2026-10-08：实现、四轮探索实测及数值复核已完成，有效检测未成立。原始产物在 graph/outputs/provenance_joint_state_20261008_*，历史设计/结果不覆盖。

|Run ID|目的|状态|实际证据与边界|
|---|---|---|---|
|PJ001|来源操作/采纳/吞吐|DONE|8答×32位置 pilot；53整答106前向；4 full-native 梯度 canary；6科学测试。测量通过，不是检测通过|
|PJ002|signed 相对来源的增量|DONE，阴性|4个固定监督诊断 probe，signed AP 弱于 source/abs，weights不进入 detector；无词项完全匹配的语义因果识别|
|PJ003|无标签邻居联合状态|DONE，收益未成立|四轮，三seed；banded精确后验与交叉协方差 EM；v4全部15最终模型收敛；native .563585，来源单元 .671476|
|PJ004|真实边相对链/乱边/节点|统计对照DONE，物理有效性门未完成|ΔAUROC区间含0；null只有63/159过25%质量门，未BPE匹配；只重连统计描述符，B3有限端点物理replay未做|
|PJ005|新来源确认|NOT RUN|当前53来源已在历史2965来源范围暴露；未构建盲确认，不将旧17答称独立验证|

四轮产物：v1，v2，v3，v4_converged。v2补legacy前的评价在before_legacy_*保留；旧分数无变更。v4从冻结v3继续，objective停止不按AUROC。最终1/16起点、6/6正常答报警；真实边未胜乱边，停止继续堆维度或将negative adoption当标签。

数值复核：488个hash、131组AUROC/AP、228个后验重现（误差0）、MAP单调与冻结continuation衔接通过；当前可读性整理不改变已保存分数。检查脚本及路径见代码目录README。fresh experiment-audit状态待报告，不能提前记PASS。

详见RESULTS.md / RESULTS.json / METHOD_IMPLEMENTATION.md。没有有效图检测提升、没有图必要性、没有在线预警结论。

2026-10-08 审计后补记：fresh experiment-audit 已完成 WARN；A/B/C/D/F PASS，E为旧17答、优化seed与未跑盲确认的证据范围限制。上述“待报告”为审计前状态。此补记只更新进度，原审计输入在 .aris/traces/experiment-audit/2026-10-08_provenance_joint_state/audited_inputs/ 保存；算法、分数及结果未变。
