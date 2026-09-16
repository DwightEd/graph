# CHARM 高表现来源审计（不是新检测器）

父模型：Attributed-Graph-Hallucination-Detection 的 train_charm_grid.py，
上游提交 13e907693aa954bf070e809d8afecdf26b3b88d8；代码来源固定到可读取的
DwightEd/Attributed-Graph-Hallucination-Detection 同一提交。只修改 graph/main。

## 问题与可证伪判断

1. 高 token AUC/AP 是否主要来自长错误片段内部？分别看回答首次错误、所有
   标注 span 起点、非起点错误、严格首错之后、正常 token。报告同一分数和
   同一正常负例下的 AUROC 分解，不能混用不同 scorer 或不同负例。
2. 固定阈值下，首错漏检却后续覆盖 >=80% 的回答/片段有多少？同时报告起点
   召回、延迟（漏检单独计数）、逐 span 等权覆盖、结束后正常 token 误报。
3. 模型是否只需要局部相邻？分别重新训练同一架构的真实入度图、局部化图、
   lag 内真实端点重连图、无消息基线。局部化保留 prompt 边、每个目标 RR
   入度和边属性集合，不声称保留 lag/outdegree；重连保持每目标的 RP/RR
   入度、权重向量和粗 lag，不保持 source 出度。记录实际改边比例。
4. 原出度归一化是否使用未来？保留原出度版本，再独立训练入度版本。
   同一个冻结模型执行 full/prefix 比较；这不是 token-before-emission 预警。
5. 固定模型的 RP/RR 删除、边通道置零、关系标记置零、节点特征置换、端点
   重连是依赖性诊断，不等同重新训练消融，不独自证明机制或因果真伪。
6. 保存每个 token 的 gold/score/预测/首错位置/结构属性，以及每个片段起点
   命中、内部覆盖和 spillover；HTML 显示 TP/FN/FP，避免只看整体 AUC。

## 保持的父模型定义

x[i,c] = A[c,i,i]；边 j→i 当任一通道 A[c,i,j]>tau，i 在 response，j<i。
edge_attr 为逐通道阈值截断的向量，edge_mark 区分 RP/RR。
节点(i)处理后预测其标签(i)，不偷换成 q→q+1。上游出度版本单独命名。
PyTorch index_add 实现与上游的 add/MLP 更新对应，state_dict 名称一致。
不用 GNN_train.py 的双向变体，不把 SourceFlow 的分数当成 CHARM 输出。

## 输入与学习协议

复用 unsupervised_token_graph 的 ResponseCache、CacheIndex、EvaluationBinding
和 source_info.jsonl 的 source_id；使用用户已有 train/test canonical NPZ。
构图只依赖 attention；标签单独关联，用于本次明确标识的监督基线及评价。
不需要重新跑 LLM，不改已有缓存或 SourceFlow 输出。
官方 test 保持隔离；train 内按 source 固定拆 fit/select/calibrate，所有变体
同划分同 seed；select AP 选 checkpoint，calibrate 正常 token 固定 FPR 选阈值。
外部旧 checkpoint 只能使用预先给定阈值（默认0.5），不能假定新划分是它
未见过的数据；与原高分是否同一 checkpoint/cohort 必须由实际文件确认。

## 实现顺序

先创建计划、graph/model/data/metrics/experiment/run 的接口骨架，再填充。
prepare 按样本落盘/续跑，fit 按模型落盘；audit 保存逐样本预测可续跑；
report 只用已存预测，不再次训练。有 CHECKPOINT 时不训练；没有时明确运行监督对照。
原路径：/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/{train,test}
标注：同 RAGTruth/dataset/response.jsonl；source_info.jsonl；原模型 tokenizer。

## 回归检验

CSR阈值并集与稠密参考一致（含float16）；标签不影响边/特征；父模型代数
与state_dict一致；入度版本prefix不变、出度版本可变；真实重连确实改端点；
首token纳入、长span但首错漏检、no-error、ties、边界spillover、val阈值不看test；
真实 tiny CPU 的prepare→train→audit→report、已有checkpoint→report流程。
测试通过只是实现验证，不是自然数据的机制结论。
