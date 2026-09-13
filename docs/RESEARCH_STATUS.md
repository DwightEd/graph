## A2-v1完整负结果与v2受控迭代（2026-09-13T15:46:56.948789+08:00）

v1训练/自然预测/评价/48来源实际擦除均完成。固定graph差分source-balanced AUROC全词0.505927、首错后0.512073，base NLL分别0.535477/0.516545；36答/6源/4733词是反复使用的开发集，6答来源进source-only预训练，30source-unseen同样无改善。结果/实现/完整分母见 graph/docs/GROUNDED_GRAPH_V1_RESULTS_20260913.md。

固定query擦来源X后坐标指针概率降0.399276，生成gain drop为−0.000657；原始gain本就−0.001492，因此不能把full擦除的正drop叫正收益被移除。source-copy任务中H_full已经含完整source，图生成分支可能被绕过。当前v2保持240来源、划分、9220锚点、参数、10epoch和loss，只换为真实all-source-erased H_empty查询及解码基底，原X保持；新特征/训练/预测/评价代码已工程闭合，fresh-doc见证正在启动真实capture，来源依赖诊断待审查。尚无v2数值。

当前方案与可执行指令：graph/docs/GROUNDED_GRAPH_RESTORATION_V2_20260913.md、GROUNDED_GRAPH_RESTORATION_V2_RUN_20260913.md。source48 anchor正增益+固定query真实X擦除降益为机制门控；全词和strictpostfirst两个差分分别报告，不翻方向。准确回看、自然适用约束、原模型路由/聚合、连续范围仍未解决。外部Codex MCP不可用，科学审计不称PASS；保留所有旧文件/未跟踪文件和分支。

以下均为保留的历史状态；当前以最新段落及实际manifest为准。

## 当前联合图模型与实际状态（2026-09-13T15:07:49.713783+08:00）

主线已写成单一 GroundedGraphAdapter：完整source图的4096维特征，两步128维消息传递，真实预测前h[t−1]查询，门控残差经冻结LMhead。固定自然检测score=logp_base−logp_adapter。当前方法/执行入口为 graph/docs/CURRENT_METHOD_20260913.md、GROUNDED_GRAPH_MODEL_20260913.md 和 GROUNDED_GRAPH_TRAIN_RUN_20260913.md。

Qwen来源模板实验因确证错误owner停止：6/240源，19/48机械可用但有段落归属错误，已落盘7298forward，SIGINT130；中断批额外forward未知。所有原始文件保留，不用于训练。替代的来源原文坐标重建已完成240源、9220锚点，192/48全文SHA隔离；训练特征135519token/240forward，自然36答5840token/4733词/36forward均exit0。自然开发6源中13717在source-only训练内，单独报告重合，不称独立test。

训练/自然预测/评价/实际payload擦除代码均已实现并完成工程检查。训练执行见证正在启动，尚无新自然检测数值；不称方法收敛。旧SourceRel95.4%仍只属弱字段检索；全量17790的post-first弱/负结果保留。准确回看、自然适用约束、原LLM路由vs聚合、连续影响范围仍未闭合。

以下为保留的历史检查点；当前状态以上述入口和真实manifest为准。

## 最新核验与方法修正（2026-09-13T14:00:27.286414+08:00）

完整来源库存已实际完成并独立审计：2965来源/17790引用，1241059组件，unknown3049、长文本字段4083、mapping failure0；CPU339.217秒，执行与独立审计均exit0。报告 graph/refine-logs/constraint_inventory_doc_witness_20260913.md。SourceRel与post-first的完整负结果保持，不能将字段重建95.4%当自然归属准确率。

用户指出反复局部实验的死胡同后，进一步明确：冻结Qwen推理图只作为外部语义参照，不足以回答内部信息是否能判定归属。新 docs/REASONED_GRAPH_METHOD_20260913.md 和 next_iteration/reasoned_graph{,_runner}.py 已写，工程审查尚未完成，未prepare/未GPU执行；其当前提示图仅field/record/context粗粒度，完整组件图仍保存在库存，不能混称。当前还在审查联合来源指针与grounded token重建的轻量内部图模型目标，未实现/训练，不宣称已选定有效结构。

当前没有GPU实验在运行。准确回看、适用归属、路由/聚合、连续范围四项仍未闭合。保留所有旧文件与分支，以下运行状态均为历史。

---

## 最新进展：SourceRel自然迁移失败已定位；修完整候选图（2026-09-13T13:26:02+08:00）

SourceRel-Mini完整运行结束：883来源、38541辅助4096维向量、20epochs。来源验证top1=95.40%，但same-field/other-record仅77.27%（TFIDF100%）；60错中55错在此组。自然旧36回答/1619槽位全部保留，622Data2txt给候选、997域外；两来源均已在source_train。实际出现地点→name高置信、地址数字候选只有评分、overall rating→review rating、每天营业→单个weekday；长评论和unknown-valued owner还被旧候选池排除。**完整源字段重建训练完成不等于自然约束归属成功，当前头停止推广/追加训练**。具体表格、反例与checkpoint见 docs/SOURCEREL_RESULTS_20260913.md。

全量post-first CPU分析也结束，无新模型调用：17790响应，严格首错后810750 tokens/148814 errors，旧all/first的504组指标对完全复核。post-first source-balanced组AUROC按token加权均值：entropy0.5480、history-smallJS0.5127、source-smallJS0.5533、history-minus-source-support0.5222；不是pooled指标，也不支持连续错误检测已解决。报告 refine-logs/population_postfirst_results_review_20260913.md。

当前实现 next_iteration/constraint_inventory.py：全部未知值字段/长文本、精确decoded-to-raw组件、field/context/record包含关系和七天key库存；QA/Summary也保留原文节点。工程复核中，完整2965来源CPU编译待执行。只有源拓扑完整性，不冒称语义图真值；已删除假定正确的q_relation/q_owner/q_condition桶、all-slot遮蔽和同句同事件硬绑定。自动query scorer仍在细化，不运行尚未设计好的新GPU模型。

主线graph、机制reanchor；回看/归属/路由与聚合/连续范围四项仍未闭合。全部旧代码和结果、两分支、未跟踪文件保留。以下“当前”与待跑项为历史。

---

## 当前进展（2026-09-13T12:54:03+08:00）

当前完整主线未实证收敛。方法结构与边界见 [CURRENT_METHOD_20260913.md](CURRENT_METHOD_20260913.md)。全量population17790/17790完成、failed0；typed full仅2对照，真实train native90forward仅raw影响、强证书0。SourceRel-Mini已实现数据/特征/训练CLI，883来源、7064query，完整长度预检通过；两来源sanity完成，全量编码/训练推进中。TFIDF source-validation top1=80.60%，不是自然归属准确率。自然迁移与路由/连续span仍待验证。

参考 [全量机制审查](../refine-logs/ragtruth_full_evaluation_review_20260913.md)、[当前实验计划](../refine-logs/SOURCEREL_EXPERIMENT_PLAN.md)。以下为历史状态。

---

## 当前状态（2026-09-13T11:27:21+08:00）

surface owner v1已完成但合格B/A=0、native=0；CPU读出诊断显示主要障碍是语义判断。当前实现fact-scope共享约束求解及Data2txt typed-hours真实正误对照，独立工程审查中，尚无typed native结果。主线未实证收敛；QA/Summary通用关系入口、自动回看、路由/聚合区分及连续错误范围仍待闭合。

最新方案：[事实范围与原生图修订](../refine-logs/fact_scope_graph_revision_20260913.md)；结果：[surface owner负结果](SURFACE_OWNER_V1_RESULTS_20260913.md)。以下旧“当前”按历史阅读。

---

## 最新运行：soft graph v1 完整36回答（2026-09-13T08:41:31.225750+08:00）

新的主线检测器已冻结并由独立文档见证启动，child PID 155856，wrapper session67861，输出 graph/outputs/soft_graph_v1_20260913。settings digest 9833b851b479f6ffe53d4a601a947d53fc13eb4d161946daf5f2e9e33fa32cd8，文件SHA fb13d3dabe6333c15f00ae722993148018cecf8c949010f8b01172788753bec4，47份代码冻结。72项相关CPU检查通过，独立runner/pipeline复核9通过，Critical0/Required0。当前没有新效果评价。

执行A结构图、B高维特征/候选、C有限关系判断、D每span原始输入介导/定位、merge全词四版本评分。完整结构见 graph/docs/SOFT_GRAPH_METHOD_20260913.md，运行指令见 graph/docs/SOFT_GRAPH_V1_RUN_20260913.md。原全量暂停于13669/17790、失败0，由同一独占wrapper完成后恢复。未修改任何旧冻结代码/结果，未跟踪文件均保留。

本版仅以观测回答span的完整logP测依赖，不把它冒充正误决策差。严格A/B层仍单独保留，本批不声称已区分错误路由和正确路由后聚合失效。标签仅在全部预测完成后评价。v3全部弃权/native0的完整负结果已归档；以下状态均按时间视为历史。

---

## 最新状态：v3 完整负结果与主线架构调整（2026-09-13T08:05:20.112574+08:00）

v3 的 A/B/C/D/merge 各36/36完成；4733词、260标注错误词，语义与机制覆盖0、错误召回0、实际native forwards 0。368问题中271 uncertain、97 invalid。独立完整性审计通过；这不支持自动回看、约束归属、路由/采纳或连续错误机制主张。结果见 graph/docs/NATIVE_AUDIT_V3_RESULTS_20260913.md。三个旧批次完整保留，停止仅修格式后重跑。

主线设计改为：全span固定能量图检测器 + 严格因果证据覆盖层。高维残差特征/角色与事件拓扑提出竞争来源；冻结Qwen有限选项估计完整来源及局部关系；正向且对照调整的native依赖调制图风险；仅明确前向reuse传播，纠正/引用/新话题阻断。Unknown向0.5收缩；局部未提及不推出全来源缺失。不训练幻觉二分类，不按RAGTruth标注调权重。四个同reader对照必须同批输出。审查规格见 graph/refine-logs/soft_graph_fixed_energy_review_20260913.md；当前正在编码，尚无新检测结果。

原全量已恢复并验证PID 153564，本次读取12802/17790、失败0。原wrapper/native均exit0；12574份旧manifest、4份冻结文件及8份代码哈希保持。实时状态以population progress.json为准；以下“当前”均为历史。

---

## 运行中快照：v3 与下一版结构工作（2026-09-13T07:09:44.864444+08:00）

v3 完整36条开发回答仍在执行，当前阶段 A，该阶段已完成 18/36，native PID 147669。最近已发布的18份 A 产物包含148个问题：110不确定、38无效，2327词中语义评分覆盖0，尚无有效风险对照供给。这里是无标注的运行中诊断，不能作为36条最终评价；尚未加入本版评价标签。v1/v2的完整负结果继续保留。

原 RAGTruth 全量机制任务暂停在12574/17790、失败0，由独占调度器在v3结束后恢复；不同时加载另一GPU作业。v3冻结代码、参数、阈值未改。运行指令与评价入口见 graph/docs/NATIVE_AUDIT_V3_RUN_20260913.md。

下一版结构工作已完成原文指针/字段图的CPU模块 route_graph/source_event_graph.py，26项检查通过，独立工程审查必修项已关闭。这只证明原文坐标、角色指针与字段结构完整性，不证明事实归属。outputs/source_event_inventory_20260913 已生成36回答/6来源清单，其中2个Data2txt来源字段图为42/51节点；47个清单文件哈希复核一致，无模型调用、无标签。

针对当前候选供给失败，已详细阅读 SRLScore、FENICE、SIRG、CORTEX、CausalGaze、HalluSpan、SiGHT、FGWEA；审查在 refine-logs/unsupervised_alignment_structural_review_20260913.md。保持不以幻觉标签训练的锚点，正在收窄高维角色候选与事件联合约束的匹配接口；不把相似度/null质量当作真假，不把监督或合成幻觉二分类训练换成主线。自动回看定位、适用约束、路由/采纳与连续错误范围仍未实证收敛。

以下旧“当前”和PID均为对应时刻历史。

---

## 当前更新：v2 完成负结果，v3 接口修订中（2026-09-13T06:30:50.935275+08:00）

36 条同源开发回答的 v2 已完整执行并评价：119 个问题中 93 个无效、26 个不确定，语义覆盖 0，错误召回 0，native forward 0。故自动回看、路由与采纳、连续错误边仍未获得本批次的实证验证。详情见 graph/docs/NATIVE_AUDIT_V2_RESULTS_20260913.md（在 graph 仓库中为 docs/ 下同名文件）。

v3 将纯角色 JSON 问题改为保留原句的单槽位 mask，未解析条件也保留为显式条件；重复文本按关系角色与原始坐标绑定。来源端不接收未遮蔽陈述或候选答案，解析仅接受完整 JSON 根并保留原始边界。所有请求的严格解析、边界恢复与失败进入冻结结果。47 项相关 CPU 检查通过；独立复审未结束，GPU 尚未启动。v1/v2 输出与负结果均保留，未改变因果阈值或使用标注挑样本。

原全量运行 PID 145947，12488/17790，失败 0；以下旧 PID/“当前运行”按历史记录解读，实时状态以 population progress.json 为准。研究任务仍在继续，软件可运行不等于方法有效。

---

## 最新状态：v1自然批次失败，v2原子槽位集成中（2026-09-13T05:36:30.868925+08:00）

36/36条v1已完成A/B/C/D/merge并独立评价：4733词、260标注错误词，语义覆盖287词（6.06%），错误召回0，native实际forward0。172问题中103无效、56不确定、13被reader判支持，无有效风险窗口。不能称回看/约束归属/路由检测已验证或方法收敛。详见 graph/docs/NATIVE_AUDIT_V1_RESULTS_20260913.md；原结果和executed_code快照保留。

主线graph正在集成v2：原子事件角色、精确跨度/指代绑定、确定性隐藏槽位提问、逐角色证据表核对；保持全上下文与弃权分母，重新跑同36条开发清单，无标签调阈值。真实自然效果待跑。旧全量reanchor任务恢复PID141861并继续产出；原wrapper因120秒验真超时exit1的事实保留，后续恢复已有独立证据。新的900秒等待修复已复审。

以下状态与PID均按时间保留为历史。

---

## 最新状态：完整自然批次启动预检（2026-09-13T04:47:53+08:00）

graph 的 A/B/C/D/merge 主线实现已完成部署前复核，native validation、pipeline、scheduler、evaluation 均无未关闭 Required。完整CPU回归最新独立结果87 passed，随后N position-only不得计入internal覆盖的专项回归通过；这些不代表方法有效性。

冻结执行输出：graph/outputs/native_audit_v1_20260913；36条输入；settings digest 9e5c7ae743f7157c08ffd5a1c2e93672e009c4e265408e99c99d5f217b80f85f。独立文档执行代理已被派发完整批次调用，当前是启动预检阶段，尚无新机制结果。运行文档：graph/docs/NATIVE_AUDIT_RUN_20260913.md；中文模型结构说明：graph/docs/NATIVE_METHOD_MODEL_20260913.md。

关键落实：native搜索不看reader证据候选；输入来源与key位置分开验证；连续错误绑定到先前具体答案槽位；同角色原始输入对照按答案长度冻结；正确恢复单独控制；N必须两个模板均通过位置与来源验证；所有无法判定与未覆盖词保留分母，标注只在预测全量冻结后加入。仍未验证：准确自动回看定位、当前约束归属可靠性、路由/MLP机制覆盖、连续错误范围与语义层之外的增益。

---

# 当前研究状态：2026-09-12

## 当前：原生消息图完整自然验证，部署前（20260913_042134）

以下旧“当前”、O6和旧PID均为历史。O6未运行，不再排队；不再证明已知的“关系影响决策”。

方法规格完成五轮真实GPT-5.5审查：6.22→6.90→7.12→7.79→8.30；规格READY_TO_IMPLEMENT，研究仍REVISE/实证NOT_READY。A/Qwen独立来源问答与逐条件引用、B/Llama候选盲的消息组搜索、C/不看native效应的固定候选核对、D/完整事件E/V/输入介导/跨角色/MLP交互、CPU全词覆盖与历史边已编码，入口route_graph.audit_runner。部署前工程复审尚在闭合，未启动新GPU任务。

新自然清单已冻结：outputs/native_audit_design_20260913，官方train每任务2来源×6生成器，共36回答，未读labels/quality，排除14375。36条原token对齐通过，最大输入1757。语义读者的预测不是评价真值；图教师是否提供增量仍需自然数据验证。

支持的纠正控制单独处理：当前E/此前C且问题全部条件等价，才以此前错误答案构造明确错误alternative；历史依赖正值不自动继承幻觉。N仅对预选content位置及其origin做第二withholding模板，N paired/MLP暂不测、不发完整证书。原文分组保留跨span长边。

执行上限320真实forward/Claim、1280/response，含B/A两分支、donor、逐组sham；最多4个不同Claim，有合格纠正控制时3风险+1控制。该保守cap不要求消耗完，失败不回填候选。它是机制教师，不是已验证的高速全量检测器。

原冻结population仍运行：10635/17790，失败0，PID20383。未修改其冻结代码、结果或未跟踪文件。新独占调度与恢复脚本正在复审。近期父进程23项相关CPU检查通过，独立审查员全graph tests 78 passed；正在修复的编排另须专项复验，不能将测试数写成方法效果。

未验证：来源问答准确率、自动位置/介导覆盖、连续错误vs纠正、图相对同读者非图增益、实际成本。RAGTruth没有唯一回看点或因果影响终点真值，不承诺100%定位。下一步是完整36条自然A–D验证，并按最大的实际失败环节继续迭代。


## 2026-09-13 方法优先修订（当前；以下旧“下一项O6”已失效）

核对时间 2026-09-13T02:36:21.408455+08:00。用户明确停止简单关系实验，要求先解决方法结构。O6单来源缺失约束行为实验已取消作为当前下一项，没有启动。现有全量任务继续：7219/17790，失败0，PID 20383；它仍是冻结observer机制测量，不能当新检测器结果。

本轮已按research-lit/research-refine精读QA事实核查、图事实核查、内部归因、无监督检测及分段方法，论文与固定代码笔记位于graph/refine-logs/*architecture_lit_20260913.md、grounding_qa_lit_20260913.md。两个完整草案接受真实GPT-5.5协作方法审查，6.22/6.90分，均REVISE；Codex MCP不可用，未冒充该后端或科学有效性审计。

正在修订的完整架构为 graph/refine-logs/architecture_round2_20260913.md：冻结来源语义锚点与真实计算图分开、完整陈述对比、同目标控制、自动联合query组搜索、区分事实区间与条件影响区间。**外部reader锚定并不证明内部图独立识别归属**；跨角色路由、聚合覆盖与自然证书覆盖仍是必须补齐的问题，未宣称收敛。

仅新增graph/route_graph/causal_contrast.py计算内核，7项CPU测试通过；它不是端到端新方法。正在进行结构复审及独立代码审查，没有新GPU实验、没有删除未跟踪文件或变更全量冻结代码。


## 2026-09-13 当前进展：O4/O5 已完成，全量继续

核查时间 2026-09-13T01:46:22+08:00：RAGTruth **5514/17790**，失败 **0**，状态 `running`，当前 PID **20383**。实时值以 `reanchor/outputs/ragtruth_population_20260912/progress.json` 为准。

本轮按 monitor-experiment、analyze-results、experiment-plan、research-refine、run-experiment 和 code-review-and-quality 工作流继续迭代；科学审稿后端不可用，未生成通过裁决。

O4 已完成 4 个关系世界、160 干预；数字/位置固定，仅交换 Before/After，模型 4/4 随关系选择。最终 query 的 X（来源内容）0/4 翻转，E（来源内连接）4/4；冻结 N/A/G 读出仅 3/4、2/4、2/4，未形成有效自动归属检测。
O5 据此追加并完成 167 条件：132 个单 query E 仅 t131 能翻转，32 个单层均不能；去掉 t131、此前全部 query 联合 E 仍翻转。因此没有唯一必需回看点的证据；来源读取正增量 top8 对该单点充分集合召回 0/1。

O4/O5 都已结束，当前 GPU 正在跑原冻结全量，不是继续重复局部案例。全量当前日志 `reanchor/runs/ragtruth_population_resume_relation_routing_20260913.log`。两次短调度均已安全恢复；O5 独立见证 5057→5101 新增 44 条，旧结果/settings/冻结代码未变。

4077 条已发布回答的独立 CPU 评价已完成，其中 4 个 QA 生成器组各 989 条完整。它是部分结果；没有全量 COMPLETE。Llama2-7B 官方 QA test 首错 entropy AUROC 0.8989、全部 token 0.5996；history JS 分别 0.3055/0.6058，评价总体不同，不能称为检测器提升。GPT4 仅 1 个首错。QA test 已用于诊断，后续改法须另设未查看的确认性评估。

尚未验证项按优先级：缺失适用约束/开始补具体时长的决策 → 独立来源归属读出 → 自动单点/联合回看候选 → 延续/纠正/换题的有符号影响 → 保留跨 span 边的自适应分组 → 图相对同输入非图增益。O6 目前仅设计，未执行。

主线仍在 graph，reanchor 负责机制执行。完整属性图和干预算子可运行；**新图→自动风险主线尚未收敛**。不把 O4 构造阶段规则的成功当成原自然错误修复；原来源没有给当前洋葱阶段唯一时长。

当前权威材料：graph/docs/RELATION_MECHANISM_RESULTS_20260913.md（数字/解释）、MECHANISM_ITERATION_20260913.md（待验证清单/部分全量）、MISSING_CONSTRAINT_PLAN_20260913.md（下一项）；graph/refine-logs/FINAL_PROPOSAL.md 和 EXPERIMENT_PLAN.md 已按负结果重写。


## 历史记录（截至 2026-09-12，旧 PID 和未运行状态不代表当前）

## 当前执行：RAGTruth全量机制入口验证

用户已授权启动全部17790回答。CPU验证、6条真实GPU小批次和两次恢复见证已通过；全量已后台启动，PID 16928，尚未全量完成。当前运行状态见[RAGTruth执行记录](RAGTRUTH_POPULATION_RUN_STATUS_20260912.md)。以下历史记录中的“GPU作业结束/未启动”仅对应此前轮次。新图→自动风险读出仍未完成，通用机制测量不等于检测器收敛。

可运行性澄清：最新属性图/约束归属方法尚未形成端到端检测器；图→自动风险读出是未完成的核心。用户已明确先用RAGTruth、优先收敛并实现主线。旧baseline的QA全989来源数据准备及输入覆盖预检通过，未启动全量GPU。见[数据与入口核对](LOCAL_DATASETS_AND_RUN_STATUS_20260912.md)。

最新O1–O3已执行：68条件单层factorial、98条件跨层/逐节点/拓扑对照，另32个独立来源A×B×H输入世界及64次完整单层扫描。错误洋葱数值跟随取出前阶段的来源约束；正确grill跟随烤制约束。两窗口全32层最终query X及动态端点交换能改变数值首选，任一单层X均不能。此为一个来源上的机制证据，合法归属自动读出、图检测增益与100%回看定位仍未验证。见 [最新完整结果](OWNERSHIP_MECHANISM_RESULTS_20260912.md)。下一项是数值不变、只换动作/阶段归属的关系对照，尚未运行。以下保留早期进展。

最新用户澄清后的进展：每个完整prompt–response以高维节点属性和关系拓扑建图。
已完成同source真实正误样本00012/00013的全层捕获：730/681节点、32层×32头、
4096维，保留prompt内部边；无候选投影/头平均/剪边。原始图约6.1GiB。
当前优先问题仅为真实窗口的约束归属可辨识性，尚未完成该有效性评价；见
[图表示契约及执行记录](ATTRIBUTED_SAMPLE_GRAPH_20260912.md) 和
[窗口约束核对](REAL_WINDOW_CONSTRAINT_INVENTORY_20260912.md)。

最新第二轮：已按用户扩展目标实现真实末层V/O有符号消息、包含MLP的后缀贡献与
固定后文访问干预。1条既有自然回答、7查询完成v1/v2；1%削弱下局部贡献估计误差
0.8%–3.4%，但整组删除有明显非线性，错误数值处也可低竞争。最大历史影响跨句
到“Note”元说明，尚不能确定错误终点。没有新增自然检测性能或图必要性证据。
见 [本轮结果](EVIDENCE_ADOPTION_RESULTS_20260912.md)、
[当前设计](EVIDENCE_ADOPTION_METHOD_20260912.md) 和
[执行前协议及迭代记录](EVIDENCE_ADOPTION_PLAN_20260912.md)。下文保留第一轮状态。

主线在 graph，reanchor 承载机制实验。当前默认实现是候选条件化有序路径 residual 与来源隔离参考评分。旧四边 factorial/AE 已废弃，其历史说明保存在 RESEARCH_STATUS_before_20260912.md，不能作为现行算法。

本轮新增完整回答归档恢复、严格因果首错报警、两层来源上下文诊断。217条完整旧捕获已恢复；108条可评价测试回答上 residual 首错 AUROC 0.4361，entropy 0.8442。冻结训练阈值的报警命中16/59与24/59，未支持图增益。

修订后的128个机制条件全部模型正确；主层冲突前缀context选对26/32、direct 27/32。没有错误识别证据，不将该诊断接入默认检测。自动回看定位、动作归属读出、等实际报警预算比较与完整新捕获均未验证。

详见 [本轮方法、结果、复现与下一轮门槛](METHOD_ITERATION_20260912.md)、[当前方案](../refine-logs/FINAL_PROPOSAL.md) 和 [实验跟踪](../refine-logs/EXPERIMENT_TRACKER.md)。研究主张未获支持，外部科研评审不可用；工程审查不代替有效性证明。
