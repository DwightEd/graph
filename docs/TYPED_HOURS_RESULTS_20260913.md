# Typed-hours完整RAGTruth编译结果

2026-09-13。输出 `outputs/typed_hours_population_prepare_20260913`，独立文档命令仅运行一次、exit0，编译370.43秒；见 `refine-logs/typed_hours_prepare_doc_witness_20260913.md`。

| 完整分母/状态 | 总量 | train | test |
|---|---:|---:|---:|
| 全部回答 |17790|15090|2700|
| Data2txt回答 |6198|5298|900|
| QA/Summary（provider范围外） |11592|9792|1800|
| 来源时间表无法按本协议解释 |3978|3438|540|
| 可以核验来源时间表的回答 |2220|1860|360|
| 这些回答的base句段 |20614|17304|3310|
| 回答关系/范围不在语法内 |20604|17295|3309|
| typed支持、无需错误对照 |8|8|0|
| 合法单值B/A |2|1|1|

来源时间表去重370份；所有17790行与未知状态保留。2个原生输入均精确token对齐，没有alignment失败。模型forward0、reader0、标签读取0；这不是人工标注准确率评价。test只冻结预测，不检查其示例内容/标签或按其失败改语法。

**覆盖不足以充当主线**：6198个Data2txt回答只有2个合格B/A，回答语法覆盖仅10个base。当前结果只说明source-schema代数能产生少量可核验自然决策，不说明归属/检测模型有效。不能把两个候选当总体召回或只丢掉未覆盖样本后报高准确率。

settings_object_digest：`224973133762ed3cda588f606c53b875ea9b7088b222ab974b88c80fe53a6ade`。独立核验18163项manifest产物、50份代码与快照、4个tokenizer/config文件、原始input与settings/summary SHA全部一致，18215个输出文件无遗漏/额外文件。训练合格对照全部只有9273/source14637，因此native固定hash每来源1条的选择最终自然退化为1条；没有按模型偏好筛选。

下一步不继续堆窄hours模板；当前通用关系模型规格见 `CURRENT_METHOD_20260913.md` 和 `refine-logs/source_relation_encoder_design_refinement_20260913.md`。CPU规则事实不冒充原生回看/采纳证据。


2026-09-13 元数据追加更正：旧 `constraint_graph.owner` 的 root-owner ID 实为 business-name scalar anchor。真实记录根是 `literal_graph.root`；370个来源的对应更正见 `outputs/typed_hours_owner_metadata_correction_20260913.json`，独立影响审查 `refine-logs/typed_owner_metadata_correction_review_20260913.md`。仅 owner 图元数据命名有误，全部七天来源值、B/A原文、对齐和native数值保持；旧冻结代码/产物未改，不需要重跑GPU。
