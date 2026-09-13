# SourceRel-Mini：来源重建到自然归属候选的实施计划

2026-09-13。当前negative：surface owner36回答0合法B/A；typed full17790只有2合法对照；自然native1例90forward有条件影响、强证书0。保持无RAGTruth幻觉标签训练的锚点。该计划替代旧EXPERIMENT_PLAN的当前未执行段，所有旧运行记录保留。

## M0：输入和训练目标完整性（MUST）

使用冻结population inputs.jsonl中official train的全部Data2txt来源，按source ID去重。source-hash modulo5==0为内部source_validation，其余source_train；官方test不进入任何训练/调参视图。source literal graph提供每个field的实际occurrence监督。每来源最多8个query，按固定seed20260913哈希选择；保存未选分母。候选为同源、同broad type的完整短scalar（≤40words）；长文本仍保留作上下文，不被截断成错误值。

query是完整目标值遮蔽的局部parent标量字段集合、所属记录/业务context，无绝对ID或list序号；candidate保存field-path语义、owner上下文及值。图中真root用literal_graph.root，名称anchor单独区分。字面homograph保持独立；identical masked contexts保留全部可辨认正例，不能强迫任意索引。未建立跨字段same-fact关系时不假造closure。

监督只叫source-field pointer reconstruction。这个预训练任务的高准确率本身不能称自然约束归属成功；需与词面/冻结hidden基线比较，特别报告同值不同指针、同field不同record。

先做非首句/遮蔽/重复context/源划分/正负候选/梯度有限性的CPU完整性检查，然后一个固定小批特征编码和训练接口sanity；不把fixture当研究结果。

## M1：冻结特征与小模型训练（MUST）

既有Llama3.1-8B frozen encoder，bf16 SDPA、离线模型、最大输入4096，不截断。query/candidate canonical view重新编码是辅助表示，不冒称原生成轨迹。code/model/tokenizer/input/view哈希先冻结。完整节点词面与provenance保留；特征矩阵仅含hidden，不给source/record/occurrence绝对IDs。

SourceRelMini：4096→128 query/candidate无bias线性投影、6-7类broad type缩放、cosine/temperature0.07。第一版仅multi-positive owner InfoNCE；不训练visible same-record标签，不增加NULL/entropy/margin损失，不增加GNN。

seed20260913，AdamW lr1e-3、weight_decay1e-4、最多20epochs；source_train拟合，source_validation选ownerNCE最小checkpoint。maxepoch/lr等CLI可改，但每次run必须新冻结目录，不能用官方test标签选择。候选mask只限制同来源/同broad type，不引入真假标签。节点排序置换保持任务等价。

先比较冻结hidden cosine和词面TF-IDF（现有依赖可用时）基线，再比较小头；graph作为mask闭包、难负例与已知共享变量一致性，不把已给定结构的lookup包装成学习增益。估计feature成本需CPU长度预检后给出，不先宣称精确GPU时长；单4090独占，先等待原population自然完成。

## M2：自然parser-light候选迁移（MUST，M1后）

对旧36自然开发回答的完整surface slots，遮蔽目标值保留parent全文/此前必要语境，输出source owner top5和source-family mass。保留所有无法对齐/过长/未知。它不输出SCNI，也不直接发B/A证书。评价先报告candidate供给和与旧dense候选差异；RAGTruth人工span不是owner指针GT，不能拿它算owner准确率。

原typed可核验关系用源值比较；通用关系仍需独立verifier。若只是提高重建任务而自然候选仍失效，标迁移失败，不加深GNN掩盖。若另一个semantic-provider改善更能解决剩余缺口，应据实际自然错误选择，不锁死本head。

## M3：回到完整主线（条件MUST）

有足够自然适用关系与B/A后，运行原native query/layer、稳定且语义明确的输入→消息介导、sourceB路由和具体历史值复用。首版90forward的raw效果不作训练标签。连续span依赖真实值/关系复用，不靠同句同主题泛传播。用完整冻结预测与实际RAGTruth标签比较semantic-only/无图/图方法，含覆盖、PR、AUROC、首错后错误及source聚合。仍不得宣称唯一回看点或原生成器因果归因。

当前：M0代码实现/工程复核中；M1 model head已写但未训练，特征runner与trainer待写；M2/M3未跑。没有已训练checkpoint或主线有效性结论。
