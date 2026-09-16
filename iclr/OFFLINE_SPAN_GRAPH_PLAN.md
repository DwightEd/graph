# Offline span graph：先接口、后实现的项目规划

## 当前交付

只有设计文档、数据字段、空函数/空方法。算法体统一为 `raise NotImplementedError`，
避免未实现函数悄悄返回None或假分数。这不是运行期防御检查。
没有参数训练、数据预处理或GPU前向；没有更换旧入口；不新增结果/缓存哈希框架。

实现位置：`experiments/unsupervised_token_graph/offline_span/`。
规范：`iclr/CODING_GUIDELINES.md`，沿用reanchor里用户确定的原文。
先阅读 `OFFLINE_SPAN_GRAPH_DESIGN.md`，再从data.py依次读模块。

## 文件和调用顺序

| 文件 | 空接口 | 一句话职责 |
|---|---|---|
| data.py | Sample / TokenGraph / SpanView / MatchedPair / DetectionResult | 数据结构只包含原始观察和输出，没有金标 |
| data.py | load_samples / load_observations | 复用现有样本、逐头cache和对应表征 |
| graph.py | build_token_graph / attach_source_context / index_later_readers | 原生运算边、材料上下文、后续读取索引分开 |
| regions.py | propose_spans / find_selector_candidates / build_span_view | 未知真假的完整区间视图，不硬依赖熵峰 |
| controls.py | match_evidence_controls / match_reuse_controls / build_contrastive_pairs | 只重接检测侧的关系配对 |
| model.py | EvidenceSpanScorer | 两侧关系编码与联合相容评分 |
| learning.py | contrastive_loss / train_scorer / save_scorer | 用原始与结构对照学习，不重构节点和边 |
| detection.py | score_spans / fit_unlabeled_reference / decode_spans | 整段评分、无标签长度校准、联合边界与token边际 |
| evaluation.py | evaluate_saved_predictions / compare_saved_ablation_runs | 最后才读标签，对同覆盖范围作比较 |
| run.py | prepare_graphs / fit_detector / score_answers | 未来流程入口，目前不提供可运行命令 |

拟定数据流：

    原样本 + 原生观测
           ↓
    TokenGraph（原生边 + 分开的分析连接）
           ↓
    SpanView（证据 / 选择 / 段内 / 后文）
           ↓
    MatchedPair → EvidenceSpanScorer
           ↓
    区间相容分数 → 离线区间推断 → token/span输出
           ↓ 先冻结
    独立金标评价

不要将整个数据集的原图或区间列表常驻显存；list类型表示一个样本或小批，不是要求
一次加载全量训练集。正式实现需要小批迭代时，在具体模块中直接写循环，不先造框架。

## 数据字段约定

- Sample：原source/response ID、任务/split、文本、原token_ids、prompt_length、回答offsets。
- TokenGraph.node_coordinates：[N,3]，绝对token位置、层、阶段编码；仅用于组图。
- native_edges：[2,E]，来源节点→写入节点；attributes包含kind、layer、head、attention等
  实际存在的观测。V索引或消息向量只在缓存确实提供时声明；分析连接不伪装成原生边。
- analysis_links：[2,A]，材料上下文等非因果关系；反向查询由index_later_readers返回。
- SpanView.start/end：回答token的左闭右开区间，不是字符下标；最终由原offsets映射字符。
- MatchedPair.control_kind：只指导采样/损失分组，不传入模型。绝对节点编号也不作为特征。
- DetectionResult.covered_tokens：[R]；所有原回答token保留，缺失分数不填0。

## 实现顺序和每步验收

### P0：本次只完成设计与骨架

核对语法、接口和文件路径。不得宣称算法测试通过，不生成随机分数演示成绩。
保留旧SourceFlow、FlowTracer、autoencoder与所有结果，尚不删除依赖模块。

### P1：只实现数据与图

先用各任务少量现有样本核对字段，确认缺什么；不为未知需求立即补采全数据。
确认query与预测位置、原生层序、逐head、后文索引和labels隔离。
attention-only与rich观测明确分版本；缺失表征不能悄悄补零。

### P2：只实现区间及错接

在同材料内展示一组实际配对：哪些内容没变，哪项对应被换掉。
核对节点数、局部质量、距离、词项重叠的匹配，不让“话题不同”就完成任务。
必须同时保留无回看、低熵入口、正常重复、短片段、可能纠正的后文。

### P3：实现关系评分和自监督训练

只训练一个EvidenceSpanScorer；固定原LLM。显式写出encode_evidence、encode_reuse、
forward和损失，避免万能模式分支。对比无边/原生关系/统计匹配错接。
参考集选择表示维度和资源预算；不用test效果选head、调正负方向。

### P4：实现片段推断

先保存原始区间分数，再加长度校准和区间推断。列出空解、单词、短段、长段、
两段之间正常过渡的手工小例；这是算法正确性检查，不是模拟自然效果。
明确长度预算、后文预算、边界惩罚和合并规则；同等资源比较前缀/离线模型。

### P5：冻结后自然评价

QA / Summary / Data2txt逐任务报告，样本按source拆分。
报告token与span，特别看严格首错后、结束后的误报和高置信错误。
固定无标签参数及阈值；结果负面时定位图、采样、表示或推断，不重新命名指标掩盖失败。

## 暂不实现

不训练新SAE、不提取通用自然命题、不做每token的JVP、不搜索全部因果电路。
不要求高熵才能启动，不强制首错后都为错，不用金标span构图或学习持续长度。
不把同句、同对象、attention连通或后文重复当作事实证明。

本次不新增CLI、不修改main.py/run_all.sh、不替用户启动训练。
阶段通过后再填下一模块；每次提交只说明实际新增的实现与实际运行的检查。
