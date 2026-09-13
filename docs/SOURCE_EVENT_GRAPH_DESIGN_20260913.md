# 条件性的下一轮结构：先建立来源事件图，再核对回答归属

2026-09-13。此为 v3 运行期间的结构设计，不是当前冻结 v3 实现，也不是已成功的实验。主线保留不以幻觉标签训练；不会为了提高覆盖改成监督检测器。v3 必须完整执行和评价，才能确定主要失败环节。

## 已观察问题与待判定问题

v1/v2 共两版均 native0；v3首条回答已看到原句mask的自答通过，但来源输出仍会改写 condition_role 或编造 source_quote（尚是局部运行观察，不能当36条统计）。格式合法不代表事实核对可靠。当前单role方法还有定义上的限制：当事件多个角色都错误时，每次mask留下的其他错误角色都会使条件核对失败。这里不能把“非目标条件缺失”偷换为“目标值不存在”。

因此下一轮考虑改变模型接口与图的先后顺序：来源先于回答独立形成带精确坐标的事件图，后续模型只引用已存在的节点/关系ID。这样既减少逐问题重复抽取，也将编造引文从后端真假判断中排除。能否提高语义准确率仍需独立评价；合法ID并不证明事件归属正确。

## 模块1：来源图独立编译

输入仅 source_id、完整来源及带坐标文本单元，不含 response、风险分数、native效应或标注。同一source在六个生成器间复用一个图。第一步抽取事件/角色的原始文本片段；程序验证每个quote与所属原始单元，保留多处occurrence并用不同ID区别，不能把位置不唯一混成实体不明确。节点ID由来源SHA、角色、原始span生成。

事件节点保存谓词跨度、角色节点集合、时间/条件/否定/模态与未解析文本，角色—事件边保留原文证据。指代边仅指向更早原文。多句联合证据允许组成带成员ID的超边；推导关系由冻结reader估计，不能假设各片段单独相似就联合支持。任何解析失败与未覆盖单元仍在来源图清单中，禁止把没有抽到事实当作来源不存在事实。

输出接口：`compile_source_graph(source, source_only_predictions) -> {nodes, events, relations, coverage, failures, source_sha256}`。图没有天然truth标签，也不由生成器注意力决定语义归属。

## 模块2：事件对应与最小可检验范围

输入完整来源图、原文回答事件图；原始来源仍可见。先核对事件身份与条件，而不是先假定所有非目标条件都正确。回答值可以参与事件比较，但不能反向修改已冻结来源图、添加证据或来源值。后续引用仅可使用冻结node/event IDs。

为每个回答事件保留三层结果：候选来源事件集合、逐角色same/conflict/missing/unknown表、联合same-event/near-event/unknown判断。模型估计与程序引用完整性分开记录。候选搜索须保留全部已发现同实体/动作/阶段的竞争事件；不能挑最高相似的一项就认定为应采纳证据。没有候选时仍须区分来源图漏提取与全文事件缺失估计。

- 单槽可比：两个独立核对均认为同一事件、所有非目标角色等价，且目标的来源值有确切原文。生成现有 single-slot A/B。
- 多槽可比：两次核对均识别同一事件，多个角色有唯一来源替代。同时替换这些具体角色，完整事件比较；记录joint scope，不宣称单槽归因。
- 仅语义支持、需要推导或改写：保留derived_support状态，没有精确对比不能进入单槽证书。
- 事件未说明：必须对完整来源而非抽取图作两次独立缺失核验；可比目标为整个事件承诺与两种固定withholding，范围为event。不能称这些模板是来源提供的正确事件。
- 事件候选互相冲突、条件未决或抽取不全：明确弃权与覆盖不足，不用自由改写补一个可测答案。

所需新接口：`align_events(source_graph, response_frame, predictions) -> event_alignment`、`compile_event_contrast(alignment, response) -> exact_edits | withholding | unresolved`。

## 模块3：原生图继续定位真实信息选择

来源事件图只给规范证据候选；原生回看候选仍由observer对完整事件差的响应自动搜索。E/V、固定历史质量的成对路由、原始输入介导、MLP交互与无关匹配对照沿用已有审计逻辑。若目标是joint事件，只能发布event级证书；不能把多个改动中的效应随意归给某个数字。搜索结果继续允许一个query组或多个分离组，不预设唯一回看点。

高维节点内容与实际消息拓扑保留；不加入一个没有训练目标的GNN。可扩展的无标签全流候选可以以后读取配对状态/消息敏感性，但这种敏感性只能用于优先测哪些位置，不能自行作为事实真假分数。它与语义锚点独立评估，失败不能隐藏在弃权之外。

## 模块4：以关系边划定延续和恢复

单槽目标可与此前具体答案槽位对应；joint目标只能与此前事件对应。语义联系、先前原始输入介导和错误方向必须同时报告。将当前为支持且纠正此前错误者单独处理；无已测依赖不推出新主题，标点也不强制切断长边。事实span与条件影响span分开，未测中间词保持未知。

图可用于关系自适应分组，但不能靠把相邻分数平滑成一段来证明错误传播。未来若形成全流分数，须对照同分数的常数链平滑，以验证图关系本身对恢复/换题边界的作用。

## 执行与取舍

先完成v3并报告全部词、抽取失败、自答失败、条件表失败、有效引文失败、多mask非目标失败、对比可用率和实际native数。达到某个覆盖目标也不等于准确。若v3主要仍是字段格式问题，先压缩来源输出接口到冻结ID，不调低真假或机制阈值；若有效条件表显示多role错配主导，再增加joint事件对比。两种改动分别冻结，避免一次换掉所有环节而无法解释变化。

v3的六个来源已是开发集，不能再叫未见测试。任何主方法有效性声明须新来源、固定协议、与同一语义读者且无图的基线配对比较。监督probe仅可单列为信息可解码上界，不能用于不以幻觉标签训练的候选或证书。

参考方法细节与可用边界见 refine-logs/structured_alignment_followup_20260913.md（SRLScore/FENICE）和 native_graph_literature_2026_followup_20260913.md（SIRG/CORTEX/CausalGaze）。本文件的结构组合与保守范围是待验证设计推断。

## 审查后的具体接口修订（实施前）

以下明确替代前文仍允许自由quote的来源输出设想，并落实 source_event_graph_design_review_20260913.md 的七项要求。

1. **原始清单与抽取图分开保存。** 程序先按原文坐标建立完整raw inventory（包括结构化字段、标点、文本单元），再由reader选择其片段构成事件。raw inventory覆盖是字节/字符覆盖，事件图覆盖是抽取覆盖，二者都不是来源事实完备性。`graph_absent`只可进入unresolved；N必须另看完整原始来源核验，不能由图未找到触发。

2. **模型仅返回指针和有限枚举。** 首版接口为`events:[{unit_id, roles:[{role, start_token, end_token}], qualifiers:[...]}]`，token指的是程序生成的局部lexical token编号，end为exclusive。原始source_quote、source_span、节点ID全部由程序从raw inventory构造，不接受模型自由生成引文或新节点。每个role指针必须落在允许原文单元、保持顺序、非重叠；角色枚举受限。`subject + predicate`是最小完整事件，object可选，避免排除合法不及物陈述。缺主语的片段可以保留partial_event，不能补世界知识。Data2txt字段边优先由原JSON结构直接编译，不能硬套谓词句。

3. **回答图同样独立冻结。** reader构造response graph时没有source graph，也没有来源答案；两图摘要冻结后才对齐。来源输出不能回写或补全回答的角色。提取失败与未知角色保留。

4. **身份依据是对齐的必要输出。** `identity_basis`引用两个图中用于确定同一事件的entity/action/stage等稳定角色指针；必须说明哪些角色固定、哪些冲突。不允许把所有判定事件身份的角色都改掉之后仍叫same_event。只有two-pass identity_basis一致、未改角色联合支持、变化角色有唯一来源值，才进入joint contrast；否则near_event/unresolved。实际自然数据会有无法建立身份锚点的整事件幻觉，方法必须承认该覆盖边界。

5. **三种验证分别发布。** `reference_integrity`仅证明pointer/ID绑定原文；`relation_verification`保存冻结reader对角色、事件身份、联合条件的估计；`contrast_validity`验证按已批准角色指针构造的实际文本编辑。任何一层失败不得提升到后一层，合法ID不自动具有语义权威性。

6. **作用范围不可升级。** derived_support没有exact edit则不进slot certificate；joint原文多处编辑只发joint_event证书；whole-event withholding单列event_commitment，与single-role N不共用准确率/选择性门槛。所有native证书都是选定干预族的有限效应，不代表唯一或最小原因。首个source-graph实现只支持reference_integrity和对齐记录，joint/missing的native扩展须后续单独实现审查。

7. **历史边保持三重核验。** 语义角色/事件关系、前后错误/纠正方向、前一原始输入片段经native消息的介导效应分别通过。相同event ID、相邻句或正向history attention都不够。source graph可组织候选，不能代替这些证明。

计划实现的第一个小模块为纯CPU `source_event_graph.py`：建立raw pointer inventory，编译和验证来源/回答独立抽取，构造带provenance的event graph。它不调用GPU、不评分真假、不添加新监督，也不改正在运行的v3。随后再以完整自然失败分解决定是否集成到下一批次；单独编译器通过测试不能称归属机制成功。
