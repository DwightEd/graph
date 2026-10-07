# 项目文件与原始证据

设计与规划在METHOD_DESIGN.md、EXPERIMENT_PLAN.md；数学修正/保真度/同容量追加分别在MAP_CORRECTION.md、FIDELITY_CHECK.md、CAPACITY_CONTROL.md。RESULTS.md报告所有版本、漏例和限制；RESULTS.json保存全部配置、seed与补充对照，未选择最佳模型。

原始证据位于graph/outputs，完整逐文件索引为flow_latent_20261007_map_cycles/MANIFEST.md（超过2000项）。大数组和pickle留在这些原位置，不提交Git。

| 目录/文件 | 内容 |
|---|---|
| flow_latent_20261007_capture | 53答完整A/V/head、输入、捕获快照与运行记录 |
| flow_latent_20261007_v* | 初轮9组，非一致MAP；失败代码快照保留 |
| flow_latent_20261007_map_v* | 修正版9组，每组node/native/rewired |
| flow_latent_20261007_head16 | fit-only逐头坐标及53答派生消息 |
| flow_latent_20261007_head16_v3* | 追加第三版3组及各自模型/后验/评价 |
| flow_latent_20261007_head16_control_seed* | 最初同容量对照3模型，绑定记录缺口保留 |
| flow_latent_20261007_head16_control_bound_seed* | 绑定修正重现3模型，与前轮分数逐项完全一致 |
| 各评分目录/tokens.html | 全部17回归答逐token风险、报警与真实标签展示 |
| NUMERIC_AUDIT.json | 114组AUROC/AP、69拟合density及模型/cache/容量契约复算 |
| RAW_CACHE_MANIFEST.json、DERIVED_CACHE_MANIFEST.json | 全部缓存字节SHA256，原拟合之后建立，后续校验PASS |
| NODE_RETENTION.json、LAG_PAIR_WITNESS.json | fit/held-out方差及实际节点位置对齐证据 |
| EXPERIMENT_AUDIT_FRESH.md/.json | fresh同家族代理的独立读文件审计（provisional） |

最初53答只运行一次原生8B前向。21组三模型拟合加6次单对照（含3次绑定重现），共69模型；其中27初轮模型不称一致MAP。无全量新检测结果，未建立有效幻觉检测器。
