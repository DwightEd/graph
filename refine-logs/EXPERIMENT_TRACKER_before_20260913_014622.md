# 本轮实验跟踪

| 实验 | 状态 | 实际完成 | 证据 |
|---|---|---|---|
| R0 恢复 | DONE | 217完整回答/47,880 token；严格逐行复核通过；1部分、38缺失 | results/method_iteration_20260912/recovery |
| R1 自然评分 | DONE，负结果 | 107 fit来源，108适用test回答；首错AUROC residual 0.4361 / entropy 0.8442 | results/method_iteration_20260912/baseline |
| R1 校准/报警 | DONE，负结果 | 106留一校准来源；59首错，命中16/24；未完成等实际预算主比较 | baseline/alarms_v3.json、calibration/thresholds.json |
| R2 初始机制 | DONE | 96条件全模型正确，无错误检测证据 | reanchor/outputs/binding_validation_20260912 |
| R3 冲突前缀 | DONE | 32条件全模型正确 | reanchor/outputs/binding_misbound_20260912 |
| R2+R3 审查修正版 | DONE | 128条件全覆盖、全模型正确；G主层冲突前缀26/32，D 27/32 | reanchor/results/binding_validation_v3_20260912 |
| 自动回看定位 | NOT VALIDATED | 未用人工窗口声称自动定位 | 下一轮独立验证 |
| C1/C2 研究主张 | NOT SUPPORTED | 无方法有效性/图必要性主张 | docs/METHOD_ITERATION_20260912.md |

不再扩跑此读出的自然检测；下一轮先检验同对象不同动作和有符号信息贡献。研究审稿后端不可用，工程复审另行记录。

## 第二轮：消息采纳与错误后续影响

| 实验 | 状态 | 实际结果 | 证据 |
|---|---|---|---|
| B0 一查询smoke | DONE | 独立执行退出0、sham/前缀误差0、GPU峰值16.52GiB | refine-logs/adoption_smoke_20260912.md |
| B0 完整v1 | DONE | 1自然回答7查询；暴露全删除近似误差最高200.8% | reanchor/results/adoption_pilot_20260912 |
| B0 数值迭代v2 | DONE，局部读数适用 | 保留.1/1并增加.01；1%削弱误差0.8%–3.4%；sham/前缀精确不变 | docs/EVIDENCE_ADOPTION_RESULTS_20260912.md |
| B1 新来源无标签干预 | PLANNED | 未采集，不把旧测试集移作训练 | refine-logs/EXPERIMENT_PLAN.md |
| B2 自监督图/非图 | PLANNED，条件执行 | 尚未训练；图必要性未证明 | 同上 |
| B3 自动连续错误/恢复 | NOT VALIDATED | 没有自动影响终点和新检测性能 | 同上 |

当前已有测量仪器，不等于已有有效的连续错误检测器。

## 最新收敛：属性图表示

完整真实同源正误样本图捕获DONE：00012/00013，730/681节点，32层/32头/4096维，保留prompt内部拓扑，无标签、无投影/剪边。独立文档调用退出0，新增2项tiny模型测试通过。原始数组reanchor/outputs/attributed_samples_20260912（约6.1GiB）；协议docs/ATTRIBUTED_SAMPLE_GRAPH_20260912.md。下一步仅检验约束归属可辨性；不将成功捕获改写为已经区分正误。

## 最新O1–O3：真实窗口归属机制

| 实验 | 状态 | 实际结果 | 证据 |
|---|---|---|---|
| O1 来源S×历史H、单层X/E/XE/MLP | DONE | 4输入世界，64干预+4sham；来源效应强，单层无翻转 | outputs/ownership_factorial_20260912 |
| O2 跨层/逐query/端点交换 | DONE | 98条件；全层final-query X与端点交换两窗口均翻转；4sham、76非空前缀误差0 | outputs/ownership_multilayer_20260912 |
| O3 独立来源A/B/H、全单层 | DONE | 同源4seed共32世界；错误数值跟随A，正确grill跟随B；64单层X无翻转 | outputs/ownership_source_layer_20260912 |
| O1–O3 数字/工程核验 | DONE | 18+116+139 manifest文件核验；4核心测试、最终11项相关回归；4轮审查修复闭合 | docs/OWNERSHIP_MECHANISM_RESULTS_20260912.md |
| O4 同数值、仅动作/阶段归属变化 | NOT RUN | 用来区分数值身份与关系合法性；需冻结读出/同输入非图及端点对照 | 下一轮当前优先项 |
| 自动合法归属/100%回看/图检测增益 | NOT VALIDATED | 不用局部因果翻转宣称检测成功 | 同上 |

O1–O3输出均位于reanchor；两组独立分析在results/ownership_analysis_20260912与ownership_source_layer_analysis_v2_20260912。科学审稿仍不可用。所有GPU任务结束，既有未跟踪文件保留。

## 当前：RAGTruth全量通用机制

| 实验 | 状态 | 实际覆盖/位置 |
|---|---|---|
| P0 CPU算子/数据/工程审查 | DONE | 2核心tiny测试；2965源匹配；恢复/评价边界修复闭合 |
| P1 独立GPU及恢复见证 | DONE | 6回答×8条件；1491token；2616最大输入；误差检查0；两次resume无重算 |
| P2 全量RAGTruth | RUNNING | PID16928；目标17790回答，QA5934/Summary5658/Data2txt6198；输出reanchor/outputs/ragtruth_population_20260912 |
| P3 标签独立评价 | QUEUED | 全量测量保存后自动执行，任务/生成器/官方split分组 |

当前真实计数看输出progress.json，运行记录见docs/RAGTRUTH_POPULATION_RUN_STATUS_20260912.md。
不是新主线已收敛：自动约束归属读出、O4关系对照和图检测增益仍未验证。
