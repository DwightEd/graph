# CHARM 高分来源审计

这里不是新的幻觉检测器，也不是 SourceFlow。父模型是 `train_charm_grid.py`
（上游 commit `13e907693aa954bf070e809d8afecdf26b3b88d8`，可读取副本位于
`DwightEd/Attributed-Graph-Hallucination-Detection`）。目标是区分：
**首错识别、错误片段内部覆盖、局部连续性、真实端点/属性、未来出度**。
先读 `PLAN.md`；此次实现与任何自然数据成绩分开记录。

## 直接审计之前的高分模型（不重新训练）

在 graph 根目录，使用原实验实际保存的 checkpoint。路径不能从仓库源码猜出来：

```bash
CHECKPOINT="/实际高分实验/grid_best_model.pt" \
TASKS="Data2txt" OUTPUT="outputs/charm_checkpoint_audit" \
BOOTSTRAP=0 bash experiments/charm_structure_audit/run_all.sh
```

支持 `grid_best_model.pt` 的 `model_state/best_hp`，及 `original/train.py` 的
`checkpoint.pt` 的 `model_state/hyperparameters`，或纯 state_dict。不支持
`GNN_train.py` 的双向 HyperCHARM。字段不匹配严格报错，不静默加载一部分。
只准备指定 task 的 test 图；没有优化器或拟合过程。默认阈值0.5是预先给定值，
**不声称对应5% FPR**；可用 `THRESHOLD` 填原验证集已经选定的阈值，不能看测试
结果再调。AUROC/AP不依赖阈值。每个任务应使用对应任务训练出的旧模型。

模型和缓存必须是**相同 observer、通道顺序和构图阈值**。默认缓存是当前
Llama-3.1 observer 对 `llama-2-7b-chat` 回答的重放；两者不是同一个模型身份。
1024维相同不代表可以把原 Llama-2 observer checkpoint 接到 Llama-3.1 缓存。
旧 checkpoint 缺少这一元数据时，报告不能自动保证其历史实验兼容性。
旧源码的标签错位/实验选择若与这里不同，当前报告也不是历史指标的逐字复现。

## 重新训练公平消融（明确是新对照实验）

不设置 CHECKPOINT 时，脚本在相同数据/划分/初始化seed上分别训练：

| 变体 | 改动与用途 |
|---|---|
| `charm_out` | 父模型的原出度归一化；保留已发现的未来结构依赖以供对照 |
| `charm_in` | 仅将归一化改为入度；图仍来自真实 attention |
| `node_only` | 相同编码器/更新MLP/预测头，关闭所有消息；消息MLP不参与梯度 |
| `local_in` | prompt边不动；每个目标的k条历史边改为最近k个历史节点，保留边向量 |
| `rewire_in` | prompt边不动；同目标、同粗lag带内重选历史源，不是仅交换原边权 |

```bash
TASKS="QA Summary Data2txt" SEEDS="0" \
OUTPUT="outputs/charm_structure_audit" \
bash experiments/charm_structure_audit/run_all.sh
```

推荐先运行 `TASKS=Data2txt` 对齐原脚本默认任务，再扩到其余任务。
`LIMIT=40 EPOCHS=2 BOOTSTRAP=0` 是接口连通小试，不能报告为完整数据成绩；
用单独 OUTPUT。`VARIANTS="charm_out charm_in node_only"` 可先运行最重要对照。
`SEEDS="0 1 2"` 才是多次训练，不把一次seed的source-bootstrap当成训练方差。

官方 train 内按 source 固定拆成约80% fit、10% selection、10% calibration；
官方test保持隔离。selection AP只选模型，calibration正常token的固定5% FPR只选
阈值，测试不参与两者。它不同于原脚本的回答级80/20验证拆分，所以这是公平
归因实验，不冒充完全相同训练协议。训练有真实幻觉标签，不能称为无监督。
`node_only`另报固定beta=0.5的过去信息EWMA，阈值也在独立calibration上确定。

## 输入：使用已有文件，不制作新 metadata

默认已写入脚本：

- `/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/{train,test}`
- `/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/response.jsonl`
- 同目录 `source_info.jsonl`，来源键为 `source_id`
- `/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct`

复用 `unsupervised_token_graph/{data,cache_index,evaluation_data}.py` 的已修复
解析和token-ID校验。tokenizer只在原始offsets/文本身份缺失时加载，不加载LLM。
图按任意通道 `attention > 0.05` 的并集构造；x是全部通道的对角线，边属性是
逐通道超过阈值的权重。没有prompt→prompt边，没有平均head，没有再次归一化
attention。保留孤立节点。如果manifest的保存floor高于tau则拒绝所谓无损复现；
floor未知时记录为null，结果只对应已保存边。

节点i使用处理token i后的attention、对齐标签i，包含首个response token。
这是父模型的post-token检测，不是q→q+1的提前预测。

## 输出具体回答什么？

每个模型/seed/task有 `test/`，旧checkpoint模式有 `TASK/external_checkpoint/`：

- `report.json`：各任务整体、首错、span起点、continuation、strict_post_first的
  AUROC/AP、accuracy/precision/recall/FPR，统一分数/正常负例下的AUC分解。
- `spans.csv`：每个原标注span的起止、起点分数/命中、内部覆盖率、检测延迟、
  `onset_missed_later_80`、`onset_missed_later_all`。漏检的span不伪造延迟0。
- `tokens.csv`：每个token的文本、gold、概率、阈值预测、首错/起点标记、图属性。
- `gallery.html`：绿色TP、红色FN、黄色FP、黑框首错。优先显示漏起点却覆盖续写的
  例子（后验选例，不影响指标；不是随机样本）。重复offset可能重复显示子词文本。
- `samples/*.npz`：逐token分数、冻结扰动分数、最终embedding、真实结构统计。
- `comparison.json`：重新训练变体相对`charm_in`的配对source-bootstrap增量。

`boundaries.first_error_spans`给出最关键的计数：
`missed_onset_later_all_count`表示**首错token漏检、同一首次错误span的其余token
全部命中**。同时报告长度>1的分母、80%版本、条件于首错漏检的全覆盖比例。
不要把同一多token实体的后续子词误称为独立错误传播。

片段指标区分原标注span和标签并集的连续runs；IoU>=0.5采用明确的贪心一对一
匹配。另报错误结束后5个位置中的正常token误报率，以及全正常回答误报警率。
不采用“一点命中就把gold span全标对”的point-adjustment。

## 结构核查与解释边界

冻结模型另做 no_rp/no_rr、zero_edge、zero_mark、shuffle_nodes、rewire；这些
操作使用原divisor，避免把删边与重新归一化混淆。full/prefix例外，必须重新计算
前缀degree，才能暴露未来出度。prefix目标是均匀位置及标注起点；后者只用于
后验审计，不能作为线上候选事件规则。报告prefix覆盖而不声称全部token验证。

这些是固定模型的敏感性/OOD实验；**下降不等于重训后的结构收益**。真正的收益
归因以同划分、同seed的独立训练对照为准。local和rewire都保持每目标RR入度、
边向量、RP边；不保持source出度，local也不保持距离。逐样本记录实际改边比例；
如果改边太少，不把“分数差不多”解释为结构无用。

保存RR/时间链上HH、NN、混合标签邻居的input/embedding平均cosine及同标签比例。
高同质性/高相似性仍可能来自类别不均衡、词法、位置，而不是因果依赖。结构与
得分的Spearman只是描述性，最终以控制实验为准。

## 重用阶段与资源

`PHASE=prepare`只构图；`PHASE=fit`复用已建图跑训练和审计；有CHECKPOINT时
`PHASE=audit`只跑已有模型。必须保留相同 PREPARED/OUTPUT/任务配置。
已完成图和逐样本预测原子保存并可续跑；半写入`.partial`不算完成。完整训练的
checkpoint可复用；**中断的fit目前从该模型训练起点重跑，不是epoch断点恢复**。

只重算报告（不重新推理）：

```bash
python -m experiments.charm_structure_audit.run report \
  --predictions outputs/charm_checkpoint_audit/Data2txt/external_checkpoint \
  --output outputs/charm_checkpoint_audit/Data2txt/preview \
  --bootstrap 0 --completed-only
```

训练每次加载一张图，按父模型batch的response-token总数归一化累积梯度。
EDGE_CHUNK默认4096，训练checkpoint重算消息MLP，避免保留所有边的大输入在GPU。
CPU仍需一张[E,L*H]边属性图，不保证任意长样本内存充足；不截断样本冒充全量。
图/预测保存为自己的输出，不动之前SourceFlow结果。没有已训练模型路径时，
不能声称已经解释“那一次”高分；新训练也必须先检查父模型是否复现相近表现。

依赖为现有torch(>=2.4)、numpy、scipy、scikit-learn、tqdm、pytest；缺offsets才需
transformers。模型不依赖PyG；本轮验证的是原消息代数/参数键一致，不是实际
PyG运行对照。不要强制更新服务器的CUDA/PyTorch来运行本审计。

测试：`python -m pytest tests/test_charm_structure_audit.py -q`。
