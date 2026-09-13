# Typed native v2：真实query/layer联合路径，未获强证书

2026-09-13。输出 `outputs/typed_native_v2_20260913`。完整CPU父编译的train合格集合只有1条，全部进入本批：9273/source14637，原回答6 PM、来源七天16:0、支持值4 PM；同句WiFi和非目标字符保留。原生成器Llama2-7b，内部测量模型是Llama3.1 observer。

完整事件F=logP(B)-logP(A)，Δ=F_base-F_intervened。各后缀11token，shared prefix711token。以下是有限干预原始结果，不是检测准确率或原模型生成原因：

| 测量 | F或Δ |
|---|---:|
| Baseline F | -9.011938 |
| 选中A-message content移除Δ | -12.862700 |
| 同位置content半强度Δ | -1.511797 |
| A原输入全移除仅经同A-V地址进入recipient，Δ | -15.254831 |
| 同origin半强度Δ | +0.232414 |
| 其余query、同A/同layers移除Δ | -0.117205 |
| 其余layers、同A/同queries移除Δ | -4.921040 |

程序自动选择query707–710、layers16–31，共4query×16层的联合组。完整source A为七天全部21个endpoint key，未挑某一天。搜索从122个共享query及全部32层开始，62实际forward、31测量组；全部为联合层组，单层0；测4个非邻接query union，保留60个pending及未枚举子集的边界。**没有搜索到唯一单点，也不能把未测单层叫失败。**

选中组content与origin的全移除都令原本正确A偏好转向B，主要变化位于目标slot：baseline slot/context分别-9.115641/+0.103703，content移除后+3.878763/-0.028002。这提供了该observer中局部来源消息影响当前目标的原始证据。其余query效应小，但其余层仍有显著贡献，因此不能称唯一必需层组。

**强证书0**。538项预冻结control census包含完整无关leaf及其不相交双leaf组合；baseline后、search前冻结position2/origin0个实际ID。选中范围仅复核这些ID，position0/origin0通过，未回填；此外origin半强度与全强度反向。sham和重复一致不足以抵消这些缺陷。状态为 `raw_native_measurements_uncontrolled_or_not_selective`，不称稳定采纳、幻觉修复、错owner路由或聚合失败。baseline本来A_preferred，也是必须报告的限制。

实际90forward：41个recipient记录×2，加8个donor前向；缓存返回不冒充新forward。原生高维节点32层×143token×4096维bf16，共18743296数值，0非有限值；从已计数baseline采集。6个唯一donor NPZ/manifest对，96个layer字节哈希一致。17个manifest产物和54份live+snapshot代码完整验证。节点特征与query/layer超边可供下一步研究，但组效应不能拆成单边真值。

独立文档wrapper/native均实际exit0。原population暂停于17180/17790、failed0；恢复PID167241，独立实际增长17180→17187。旧17180份manifest、4元数据、8份population代码保持。完整见证 `refine-logs/typed_native_doc_witness_20260913.md`，机器证据 `refine-logs/typed_native_integrity_evidence_20260913.json`。未读取RAGTruth标签或改变冻结语义/门限。

旧v1只是CPU预检失败，无GPU；tuple/list序列化比较错误修复后新建v2，旧目录保留。v2并非第二次重复GPU调参。

本批直接推进了真实原文对照→自动query/layer搜索→原生节点/消息落盘的实现，但没有解决通用语义覆盖、内部独立归属、错误B路由、稳定聚合或连续错误范围。下一步模型结构见 `CURRENT_METHOD_20260913.md`。


2026-09-13 元数据追加更正：旧 `constraint_graph.owner` 的 root-owner ID 实为 business-name scalar anchor。真实记录根是 `literal_graph.root`；370个来源的对应更正见 `outputs/typed_hours_owner_metadata_correction_20260913.json`，独立影响审查 `refine-logs/typed_owner_metadata_correction_review_20260913.md`。仅 owner 图元数据命名有误，全部七天来源值、B/A原文、对齐和native数值保持；旧冻结代码/产物未改，不需要重跑GPU。
