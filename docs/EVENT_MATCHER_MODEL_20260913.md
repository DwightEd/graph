# 事件候选图模型：已实现的接口、接入缺口与取舍

2026-09-13。此文件描述 v3 之后的候选层原型；它不在冻结 v3 中，不是已有检测成功结果。主线仍在 graph，原机制数据采集在 reanchor。v3 在本文撰写时 A 33/36，尚未加入评价标注。

## 问题锚点

对冻结LLM，在不以幻觉标签训练的条件下，测量来源和历史的信息实际传递、聚合与输出采纳；区分疑似错误被继续沿用、被反对/纠正以及新陈述开始。事实区间和影响范围分别评价，熵只提供候选起点。

候选图要解决的是开放来源问答供给失败：先保留可定位的来源事件/字段/原文位置，再提出对应关系。匹配质量不能替代支持、冲突、约束适用性或内部采纳判断。

## 当前已经编码的模型

1. `source_event_graph.py` 独立编译来源与回答的原文指针事件。事件包含有限角色及原始坐标，未分配文本和失败保留。Data2txt 用 AST 白名单读取字典/列表/有限标量，字段树是原文结构，None 是未知。不会执行来源代码，也不生成支持性真值。
2. `event_matcher.match_catalog` 把角色、事件和字段整理成候选目录，并给来源增加原文词元回退节点。相同值在不同位置/列表记录中保持独立 ID；128-token 原文单元只用于地址，不被当作语义事件。
3. `span_feature_capture.capture_post_block` 通过冻结模型的真实 block hook 流式读取高维残差。接口输出 `[node_or_event, selected_layer, hidden]` 的 float32 数组；预期部署到 Llama3.1-8B 时为32层、每层4096维。当前还没有该原型的真实GPU特征结果。
4. 每个 span 按**非空白原文字符**池化。一个字符被多个 byte-fallback token 覆盖时，质量在这些 token 间等分；模型仍运行完整输入。捕获记录绑定 inventory、完整 prompt/response、token IDs、offsets、每个节点的token成员、模型/分词器/层定义与数组字节。来源捕获不接收 response，来源与回答捕获必须使用同一 prompt/source-span。
5. `event_matcher.propose_event` 逐回答事件生成来源角色候选并做局部联合排序。所有特征来自同一冻结表示空间；不输入幻觉标注、reader事实标签、A/B差或native效应。

实际 unary cost 为：

```text
U(r,s) = 0.55 * mean_layer_cosine_distance
       + 0.25 * event_context_distance
       + 0.10 * surface_token_Jaccard_distance
       + 0.05 * role_type_mismatch
       + 0.05 * field_path_lexical_mismatch
```

缺少事件/类型/path桥接时用显式中性成本0.5。字段path词与回答事件原词只做兼容性候选特征，没有把字段名解释成自然谓词。当前**没有**实现数字区间逻辑、单位转换或bool自然化；这些不能被写成现有能力。

各成本通道分别保留 top4，再合并为候选池，加入 null/unknown。每个候选保留具体角色位置和事件/记录归属。`None` 保留 `null_literal`，不计作普通 matched。联合目标为：

```text
mean_role U + 0.25 * mean_role_pair provenance_penalty
```

同一事件/记录的不同候选为0；未知/异构关系为0.5；不同事件/记录或同一来源位置被不同角色复用为1。这些是候选排序中的原文归属惩罚，不是语义同一性真值。beam宽度32，保留8个assignment；不声称全局最优。来源没有column capacity，可跨多个回答事件反复使用。

每个assignment带剪枝前候选池中按固定哈希选出的同值其他事件、同字段其他记录控制候选；不同记录仍可能相关，须后续核验。所有assignment的 `identity_status=not_verified`、`downstream_allowed=relation_verification_only`。null、低相似度和竞争项都不产生E/C/N。

## 为什么没有直接做FGW或GNN

FGWEA的来源是已给定KG的无监督实体对齐，提供语义与关系联合排序的启发。srGW可以放松一侧质量约束，但本项目的字段树和自然语言角色图并不要求同构。首版采用逐角色候选加事件内关系惩罚，避免把来源复用与异构结构问题藏进全局OT超参。[FGWEA](https://arxiv.org/abs/2305.06574)，[srGW §3及附录7.3](https://arxiv.org/html/2110.02753)。这是本项目的设计取舍，不是两篇论文已经验证了本任务。

当前没有新增训练权重。是否需要一个仅由来源自身结构监督的所有者指针头，正在另行做必要性审查；尚未决定训练，不能宣称一个未经验证的自监督目标已经解决事实归属。

## 接入还缺什么

- 实际执行入口：精确原始tokenization预检、source-only/response-only事件抽取、模型/代码冻结、GPU特征捕获和产物缓存。当前只有纯CPU清单准备与capture函数。
- 关系核验：读取冻结候选ID和完整来源，判断同一事件、未编辑身份条件、逐角色支持/冲突以及未分配条件；不能让reader自由生成来源答案回填候选。
- 对照构造：字符串/数字/区间/None/bool分别处理；joint edit必须保留未编辑身份依据。尚未实现或验证这些新分支。
- 现有native衔接：完整A/B可测子集仍使用方向性干预、对照与原始输入介导。只有原始claim logP的依赖测量不能命名为路由错误或错误传播。

这些缺口直接决定下一轮是否仍会全弃权。不能把候选模块代码完成、更多native调用或解析率提高当成方法收敛。

## 已完成的验证与产物

44项CPU检查通过，独立工程审查Critical0/Required0；包括原文绑定、同值不同事件、剪枝分母、控制候选、特征捕获绑定、真实tokenizer重叠offset模式，以及随机tiny Llama上的hook池化对照。后者仅是软件检查，没有用于研究正误、训练或机制声明。

`outputs/source_event_inventory_20260913`包含36回答、6来源的原文清单；2个Data2txt字段图为42/51节点。47个清单文件哈希一致。该产物无模型调用、无标签、无对齐或检测结果。

审查记录：`refine-logs/event_matcher_minimal_spec_review_20260913.md`、`event_matcher_engineering_review_20260913.md`、`event_graph_end_to_end_failure_review_20260913.md`。问题锚点及既有v1/v2负结果保持不变。
