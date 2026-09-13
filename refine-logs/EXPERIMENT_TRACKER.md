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

# 本轮实验跟踪

## 当前：原生消息图完整自然验证，部署前（20260913_042134）

以下旧“当前”、O6和旧PID均为历史。O6未运行，不再排队；不再证明已知的“关系影响决策”。

方法规格完成五轮真实GPT-5.5审查：6.22→6.90→7.12→7.79→8.30；规格READY_TO_IMPLEMENT，研究仍REVISE/实证NOT_READY。A/Qwen独立来源问答与逐条件引用、B/Llama候选盲的消息组搜索、C/不看native效应的固定候选核对、D/完整事件E/V/输入介导/跨角色/MLP交互、CPU全词覆盖与历史边已编码，入口route_graph.audit_runner。部署前工程复审尚在闭合，未启动新GPU任务。

新自然清单已冻结：outputs/native_audit_design_20260913，官方train每任务2来源×6生成器，共36回答，未读labels/quality，排除14375。36条原token对齐通过，最大输入1757。语义读者的预测不是评价真值；图教师是否提供增量仍需自然数据验证。

支持的纠正控制单独处理：当前E/此前C且问题全部条件等价，才以此前错误答案构造明确错误alternative；历史依赖正值不自动继承幻觉。N仅对预选content位置及其origin做第二withholding模板，N paired/MLP暂不测、不发完整证书。原文分组保留跨span长边。

执行上限320真实forward/Claim、1280/response，含B/A两分支、donor、逐组sham；最多4个不同Claim，有合格纠正控制时3风险+1控制。该保守cap不要求消耗完，失败不回填候选。它是机制教师，不是已验证的高速全量检测器。

原冻结population仍运行：10635/17790，失败0，PID20383。未修改其冻结代码、结果或未跟踪文件。新独占调度与恢复脚本正在复审。近期父进程23项相关CPU检查通过，独立审查员全graph tests 78 passed；正在修复的编排另须专项复验，不能将测试数写成方法效果。

未验证：来源问答准确率、自动位置/介导覆盖、连续错误vs纠正、图相对同读者非图增益、实际成本。RAGTruth没有唯一回看点或因果影响终点真值，不承诺100%定位。下一步是完整36条自然A–D验证，并按最大的实际失败环节继续迭代。


## 2026-09-13 更新（以下为当前状态）

核查时间 2026-09-13T01:46:22+08:00：RAGTruth **5514/17790**，失败 **0**，状态 `running`，当前 PID **20383**。实时值以 `reanchor/outputs/ragtruth_population_20260912/progress.json` 为准。

| 实验 | 状态 | 本轮实际证据/下一步 |
|---|---|---|
| P2 全量17790×8 | RUNNING | PID20383；同一冻结输出继续，当前日志后缀 resume_relation_routing_20260913 |
| P3 部分CPU评价 | DONE，部分 | 4077条；4个完整QA生成器组、另121条部分；不是全量评价 |
| O4 固定数字/阶段互换 | DONE | 4世界160干预，native4/4；最终X0/4，E4/4，N/A/G 3/4、2/4、2/4 |
| O5 E回看与层扫描 | DONE | 167条件；仅t131单query充分，32单层均不充分；without_final联合也充分 |
| O4/O5 独立工程/执行/CPU分析 | DONE | 190+175原始hash；完整logits/sham/前缀检查通过；分析原始重算无Required |
| 来源正增量top8 | FAILED，此干预定义 | 单query E充分集合覆盖0/1；不能宣称自动定位 |
| O6 缺失适用约束/承诺前决策 | DESIGN，NOT RUN | docs/MISSING_CONSTRAINT_PLAN_20260913.md；输入/判据/预算待冻结 |
| 跨来源归属、联合候选、连续影响、图分组/非图增益 | NOT VALIDATED | 按最新EXPERIMENT_PLAN依序推进，未排队GPU |

详细结果 docs/RELATION_MECHANISM_RESULTS_20260913.md；不要将下方旧“O4未运行”和旧PID当成当前状态。


## 历史记录（截至 2026-09-12，旧 PID 和未运行状态不代表当前）

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


## 2026-09-13T12:54:03+08:00 — population / typed results and SourceRel-Mini implementation

- Population COMPLETE17790/17790 failed0; `refine-logs/ragtruth_full_evaluation_review_20260913.{md,json}`: all36 groups, hashes verified; descriptive observer scope only.
- `docs/TYPED_HOURS_RESULTS_20260913.md`, `docs/TYPED_NATIVE_RESULTS_20260913.md`: full17790 compile2contrasts; soletrain native90forwards/raw/strong0. Freshdoc witnesses completed; oldpending entries historical.
- `outputs/typed_hours_owner_metadata_correction_20260913.json`, `next_iteration/typed_owner_metadata.py`: additive370 root/name-ID correction; independent impact review, no frozen rewrite or GPU rerun.
- `next_iteration/source_relation_{data,model,features,train,transfer}.py`; current plan `refine-logs/SOURCEREL_EXPERIMENT_PLAN.md`, method `docs/CURRENT_METHOD_20260913.md`.
- M0 fresh883sources/7064queries; fullCPUfeaturepreflight38541docs/3954619tokens/max1200; no truncation. Data/features/train independent reviews C0/R0 (scope engineering). TFIDF valtop1.8059816/top5.9900307. Sanity real21featureforward and2head epochs exit0; no effectiveness claim. Full features/20epochs/natural transfer pending at this entry.


## 2026-09-13T13:26:02+08:00 — SourceRel complete / post-first complete / inventory repair

- `docs/SOURCEREL_RESULTS_20260913.md`: actual full20epochs/source-val95.40%, samefield-otherrecord77.27% vsTFIDF100%; natural M2complete622proposals/1619slots, structural negative. All4 freshdata/features/train/transferwitnesses completed with hashes; no RAGgoldtrain/naturalGT.
- `next_iteration/population_postfirst.py`, `outputs/population_postfirst_v1_20260913`, `refine-logs/population_postfirst_results_review_20260913.md`: actual rootCPU session97043exit0,17790verified,504oldmetricpairs matched; post-firsthistoryJS.5127/entropy.5480; descriptiveonly.
- `refine-logs/source_relation_failure_refinement_20260913.md` superseded at relevant boundaries by `source_relation_failure_boundary_revision_20260913.md`; current `CONSTRAINT_INVENTORY_PLAN_20260913.md` / `next_iteration/constraint_inventory.py`: codewritten, engineeringreviewrunning/fullCPUcompilepending, no newrankerGPU.


## 最新核验与方法修正（2026-09-13T14:00:27.286414+08:00）

完整来源库存已实际完成并独立审计：2965来源/17790引用，1241059组件，unknown3049、长文本字段4083、mapping failure0；CPU339.217秒，执行与独立审计均exit0。报告 graph/refine-logs/constraint_inventory_doc_witness_20260913.md。SourceRel与post-first的完整负结果保持，不能将字段重建95.4%当自然归属准确率。

用户指出反复局部实验的死胡同后，进一步明确：冻结Qwen推理图只作为外部语义参照，不足以回答内部信息是否能判定归属。新 docs/REASONED_GRAPH_METHOD_20260913.md 和 next_iteration/reasoned_graph{,_runner}.py 已写，工程审查尚未完成，未prepare/未GPU执行；其当前提示图仅field/record/context粗粒度，完整组件图仍保存在库存，不能混称。当前还在审查联合来源指针与grounded token重建的轻量内部图模型目标，未实现/训练，不宣称已选定有效结构。

当前没有GPU实验在运行。准确回看、适用归属、路由/聚合、连续范围四项仍未闭合。保留所有旧文件与分支，以下运行状态均为历史。


## 当前联合图模型与实际状态（2026-09-13T15:07:49.713783+08:00）

主线已写成单一 GroundedGraphAdapter：完整source图的4096维特征，两步128维消息传递，真实预测前h[t−1]查询，门控残差经冻结LMhead。固定自然检测score=logp_base−logp_adapter。当前方法/执行入口为 graph/docs/CURRENT_METHOD_20260913.md、GROUNDED_GRAPH_MODEL_20260913.md 和 GROUNDED_GRAPH_TRAIN_RUN_20260913.md。

Qwen来源模板实验因确证错误owner停止：6/240源，19/48机械可用但有段落归属错误，已落盘7298forward，SIGINT130；中断批额外forward未知。所有原始文件保留，不用于训练。替代的来源原文坐标重建已完成240源、9220锚点，192/48全文SHA隔离；训练特征135519token/240forward，自然36答5840token/4733词/36forward均exit0。自然开发6源中13717在source-only训练内，单独报告重合，不称独立test。

训练/自然预测/评价/实际payload擦除代码均已实现并完成工程检查。训练执行见证正在启动，尚无新自然检测数值；不称方法收敛。旧SourceRel95.4%仍只属弱字段检索；全量17790的post-first弱/负结果保留。准确回看、自然适用约束、原LLM路由vs聚合、连续影响范围仍未闭合。

以下为保留的历史检查点；当前状态以上述入口和真实manifest为准。



## A2-v1完整负结果与v2受控迭代（2026-09-13T15:46:56.948789+08:00）

v1训练/自然预测/评价/48来源实际擦除均完成。固定graph差分source-balanced AUROC全词0.505927、首错后0.512073，base NLL分别0.535477/0.516545；36答/6源/4733词是反复使用的开发集，6答来源进source-only预训练，30source-unseen同样无改善。结果/实现/完整分母见 graph/docs/GROUNDED_GRAPH_V1_RESULTS_20260913.md。

固定query擦来源X后坐标指针概率降0.399276，生成gain drop为−0.000657；原始gain本就−0.001492，因此不能把full擦除的正drop叫正收益被移除。source-copy任务中H_full已经含完整source，图生成分支可能被绕过。当前v2保持240来源、划分、9220锚点、参数、10epoch和loss，只换为真实all-source-erased H_empty查询及解码基底，原X保持；新特征/训练/预测/评价代码已工程闭合，fresh-doc见证正在启动真实capture，来源依赖诊断待审查。尚无v2数值。

当前方案与可执行指令：graph/docs/GROUNDED_GRAPH_RESTORATION_V2_20260913.md、GROUNDED_GRAPH_RESTORATION_V2_RUN_20260913.md。source48 anchor正增益+固定query真实X擦除降益为机制门控；全词和strictpostfirst两个差分分别报告，不翻方向。准确回看、自然适用约束、原模型路由/聚合、连续范围仍未解决。外部Codex MCP不可用，科学审计不称PASS；保留所有旧文件/未跟踪文件和分支。

以下均为保留的历史状态；当前以最新段落及实际manifest为准。

