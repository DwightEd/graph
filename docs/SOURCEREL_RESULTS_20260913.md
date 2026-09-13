# SourceRel-Mini：完整来源重建与自然迁移结果

2026-09-13。数据、38541个冻结高维视图、20epoch小头及完整旧36自然清单均已实际运行结束。工程完整性通过；当前不能作为可靠自然约束归属模型接入主线。

## 来源重建

仅official-train Data2txt的883个来源，按source hash分720训练/163验证；每来源8个query，共5760/1304。没有RAGTruth幻觉标签、response训练目标或另一LLM真假标签。监督为原source字段指针。候选含同值不同字段、同字段不同记录的难负例，masked-query完全相同则保留multi-positive。

38541个辅助视图共3954619 tokens，最大1200，不截断；Llama3.1-8B causal final-layer last-nonpad hidden，4096维。编码9636forward，实际编码循环474.05s。小头两个4096→128无bias投影、7类type缩放、cosine/0.07，共1049472参数，CPU训练20epochs，最低验证NCE选epoch20。逐query预测、checkpoint与实际执行代码都留存。

| 同一来源验证池 | TF-IDF | 冻结hidden cosine | SourceRel-Mini |
|---|---:|---:|---:|
| 1304 queries top1 | .80598 | .30215 | .95399 |
| 1304 queries recall@5 | .99003 | .83206 | 1.00000 |
| homograph 808 queries top1 | .88243 | .29579 | .96658 |
| same-field / other-record 242 queries top1 | 1.00000 | .32231 | .77273 |

小头整体60个top1错误中55个在same-field/other-record子集。不能以整体95.4%掩盖记录绑定失败。TF-IDF拟合仅使用实际保留的train-source视图，验证来源不进入词表拟合；与小头使用相同候选池。两来源sanity只是工程接口，不混入以上结果。

## 自然迁移

旧36个开发回答来自6个来源、三个任务各12回答、全部official train。两个Data2txt来源13717/14637都实际进入source_train，所以这不是自然source-heldout测试。原1619个surface slots全部保留；997个QA/Summary槽位标域外，622个Data2txt槽位给候选。687视图、172实际feature forwards；无超长/空池排除。599/622的top1相较同视图冻结cosine改变，只是候选变化，不是准确率。

可直接从原文与候选路径核对的失败：

- 6303的Santa Barbara/California槽位分别以.994/.979的ranking mass选到name；这些mass不是校准的正确概率。
- 6301的地址220、套间1被限制在number候选池，池内只有business/review stars；正确地址内部数值根本没有候选节点。
- 9271的overall business rating 4.0首选review_stars，ranking mass .857。
- 每天营业的time_range首选某个weekday，top5至多包含5天，不能表示完整七天约束集合。
- 来源13717的3条review_text有90/187/99 words，全部被40word短scalar候选政策排除；14637也缺一条44word评论。长文本保留作其它字段的上下文，不等于它本身有可选证据节点。
- unknown-valued attributes被从候选池移除；但未知值的WiFi字段仍可能是当前关系的适用来源，只能说值未知，不能把关系owner本身丢掉。

上述是透明的原文/图结构诊断，没有人工natural-owner金标准，不报owner accuracy。622项完整诊断保存在 `outputs/source_relation_transfer_diagnostics_v1_20260913/diagnostic.json`，没有新的模型调用或RAGTruth标签读取。

## 方法决策

来源字段重建不能替代自然归属训练目标；当前JSON视图到自然masked span有明显分布差异。停止当前单向量头的追加epoch/扩大批次，不以其95.4%作为主线已收敛。

下一设计必须同时修复：完整source原文及字符串内部组件的provenance、未知值关系owner、字段语义与记录身份分别匹配、量词/区间等集合约束、query训练视图与自然输入的一致性。候选表示修复在先；更深GNN、简单扩大topk或降低门限都不能恢复根本不存在的节点。原生回看、路由/聚合及连续span仍未闭合。

## 实际产物和验证

- 数据：`outputs/source_relation_data_v1_20260913`；独立document witness `refine-logs/source_relation_data_doc_witness_20260913.md`。
- 特征：`outputs/source_relation_features_v1_20260913`；独立document witness `refine-logs/source_relation_features_doc_witness_20260913.md`。
- 完整小头：`outputs/source_relation_train_v1_20260913/checkpoint.pt`，SHA256 `9c716c94ffffd7fc5d114f931f2973290f1dd798c227cef6d57dc2ddc1ed2ff6`；训练见证 `refine-logs/source_relation_train_doc_witness_20260913.md`。
- 自然候选：`outputs/source_relation_transfer_v1_20260913`；独立document witness `refine-logs/source_relation_transfer_doc_witness_20260913.md`。
- data/model、features、train、transfer独立工程Required均闭合。原始源字段监督、全部分母、实际模型/代码/输入/array/checkpoint哈希核验通过。它们不构成自然语义或因果机制通过证书。
