# A2-v1 完整结果：来源指针敏感，生成收益与自然检测均未成立

本轮训练、自然预测、评价、48来源实际擦除均已完成，所有输出和执行代码冻结。无官方test结论；36答是反复使用的开发集，6来源，4733词/260错误词，首错后742词/246错误词。全部词保留，unavailable0。

## 自然检测（source-balanced AUROC，固定高风险方向）

| 评分 | 全词4733 | 截至首错3991 | 严格首错后742 |
|---|---:|---:|---:|
| base_nll | 0.535477 | 0.552105 | 0.516545 |
| base_entropy | 0.523665 | 0.621364 | 0.486351 |
| graph_nll | 0.533458 | 0.537498 | 0.516961 |
| graph_difference | 0.505927 | 0.390467 | 0.512073 |
| no_edges_nll | 0.532638 | 0.529146 | 0.514793 |
| no_edges_difference | 0.491690 | 0.289787 | 0.502498 |
| permuted_graph_difference | 0.510390 | 0.406123 | 0.529989 |

图差分相对base NLL的AUROC差值：全词-0.029550，首错后-0.004472。差分方向、epoch未按RAGTruth标签调整。来源未进入预训练的30答单列亦无改善。仅6来源、单种seed、无置信区间，不能解释成越训练越差的显著趋势。

## 实现与训练

完整source库存的field/context/component/bundle节点，以Llama3.1-8B原prompt finalnorm状态池化为4096维X；9种节点、4种包含/成员及反向边、128维2步消息传递。每个目标token用真实预测前H_full[t−1]作为query，softmax分配来源后门控残差加回H_full，经冻结LM head预测。训练来源原文复制CE + 9220坐标锚点pointer NLL；只训练1,641,089参数。240来源按全文SHA分192train/48val，无RAGTruth真假训练标签。坐标指针不是自然回答应归属约束真值。

两臂同初始化/同样本顺序，各10epoch；graph选epoch7，no_edges选epoch4；selection仅source-val联合loss。276次原始observer前向（240训练源+36自然答），训练612adapter forward/480optimizer step，预测108adapter forward。

## 来源依赖诊断（48source-val，1861坐标锚点）

实际把owner payload prompt token替换成space220并重新编码48次；保存非零真实X_erased和Q_erased。图侧擦除固定原query；full擦除也换query。下面是每来源先平均，再跨48来源平均：

| 条件 | 臂 | adapter gain drop | 指针概率drop |
|---|---|---:|---:|
| graph_source_erased | graph | -0.000657202 | +0.399275607 |
| graph_source_erased | no_edges | -0.001100661 | +0.352580239 |
| full_source_erased | graph | +0.041304925 | +0.466800671 |
| full_source_erased | no_edges | +0.013513800 | +0.409332550 |

原始graph平均adapter gain本就−0.00149246。full擦除后的正drop表示原本负增益变得更负，不能说正收益被移除。固定query仅擦X时，指针概率降0.399276，生成增益却未下降（drop−0.000657）。因此指针敏感并未带来相应的生成使用。完整source隐藏状态使原文复制很容易、生成分支绕过图，是与结果一致的可检验解释；尚未证明它是唯一原因。

## 结论与后续

自然检测有效、正确约束自动归属、准确回看、路由/聚合区分、连续错误因果范围：均无支持。当前模型可运行，但未收敛有效方法。停止增加同一任务epoch；下一版仅改变source信息路径，以真实全source擦除H_empty作query/残差基底，同时保留原X，独立检验anchor正恢复收益和固定query来源依赖。详见GROUNDED_GRAPH_RESTORATION_V2_20260913.md。

工程证据：feature独立审计276packet/552arrays/model/code均通过；自然指标441cell独立重算max误差1.62e−14。该数值审计进程在最终JSON写入因numpy.int64序列化exit1，保留部分文件和错误，另恢复文档，未重跑/改指标；3条真实训练/预测/评价命令均exit0。48擦除独立审计96array/864artifact/165code检查exit0。报告grounded_graph_{feature,train,erasure}_doc_witness_20260913.md。

外部Codex MCP评审后端不可用；这是自身结果判定[pending Codex review]，不冒称跨模型科学审计PASS。旧EXPERIMENT_AUDIT.json unavailable状态保留。
