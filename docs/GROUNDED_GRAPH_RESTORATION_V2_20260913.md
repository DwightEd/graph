# A2-v2：切断原文复制绕过图的路径

## 已观察到的失败

A2-v1自然开发36答/4733词/260错误词：固定graph差分AUROC全词0.505927、首错后0.512073，原LM NLL分别0.535477、0.516545。source未见过的30答同样未改善。没有官方test结论，不扩规模掩盖失败。

source-heldout48的固定query图payload擦除：graph坐标指针概率平均下降0.399276，而adapter增益下降为−0.000657（原始平均增益本就−0.001492）。因此目前的指针目标确实对来源内容敏感，但生成目标没有获得相应正向增量；full-erasure的正drop只是变得更负，不能称正收益被移除。文件：outputs/grounded_graph_evaluation_v1_20260913/evaluation.json、grounded_graph_erasure_v1_20260913/summary.json及独立见证。

**可检验解释**：完整source已经进入原LM预测前h，原文复制接近完成，残差图分支缺乏重建压力；joint loss主要学到pointer，而LM head继续依赖原h。域迁移和字段关系表征仍可能同时有问题，不能把这一解释称唯一原因。

## 只改变条件信息路径

保留同一240来源、全文SHA192/48划分、135519目标token、9220坐标锚点、相同source graph、1,641,089参数、2步128维GNN、10epoch与无边对照。

新增一次真实反事实observer前向：把prompt中**所有source token**替换为空格token，保持位置数量、其它prompt、回答前缀和目标不变。缓存 `H_empty[t−1]`。图节点X仍用原始完整source的已审计状态。新适配器以H_empty作为query和解码残差基底：

```
A_t = attention(W_query H_empty[t−1], GNN(X_original,G))
H_restore = H_empty[t−1] + gate * W_out Σ A_t V_node
p_restore = frozen_LM_head(H_restore)
```

这明确是**来源信息恢复模型**；query是实际source-erased observer状态，不冒称原生成状态。原始H_full仍保留用于基线与比较。当前token不进入query，来源图不含response。没有新真假伪标签、不改自然文本、不把擦除文本当自然GT。

预训练目标仍为source-derived token CE与坐标pointer NLL，但LM解码基底看不到prompt来源，图分支才有恢复缺失证据的压力。回答历史仍可能解决部分token，所以必须再做固定query、实际source-payload重编码X的对照，检验正向增益是否依赖来源。

## 预先固定评价

继续保留v1主差分 `logp_full(y) − logp_restore(y)`（高为风险），同时预先增加来源恢复差分 `logp_empty(y) − logp_restore(y)`（高为风险）。完整LM NLL/entropy、empty-source NLL、restore NLL、独立无边模型和破坏边对照同时报告；不翻转分数，不据开发标签改epoch。此处新增统计量属于显式开发迭代，不能把同一36答的改善当未见test成功。

首先按自然全词、严格首错后报告全部分母及source train-overlap。若自然分数与内容依赖均没有改善，拒绝“图已有效采用证据”，不继续加epoch。若出现增量，冻结具体统计量及模型后进入官方test独立评价。

## 残留问题

本改动不直接解决源文本到自然回答的域差异、长距离错误传播、100%回看定位或原模型的错误路由/聚合归因。它只针对v1中已经实测的“来源指针改变，但生成增益不随图内容改变”的结构缺口。

当前为v2实现与审查阶段，尚未运行；v1全部文件与结果冻结保留。

## 审查后固定的边界和门控

所有source_span正重叠token按整token擦除，包括字段key/格式符；token跨source边界时无法保留其外侧字符，receipt显式记录boundary_crossing_tokens，不声称字符级source外绝对不变。BOS、不重叠prompt token、全部response prefix保持原ID。

回答历史仍有复制旁路。因此保持原先checkpoint选择（source-val全词CE+pointer、epoch0也可选），但在所选模型上独立要求：48留出来源的source-mean坐标锚点生成增益相对empty LM为正，且固定同一H_empty、仅换用v1真实payload-erased X后增益平均下降。该门控同时报告按task和source正例数，不用自然标签选checkpoint。它只说明当前恢复模型的内容依赖，不等于自然owner真值。
