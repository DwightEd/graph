# CHARM audit：从 main 看整个实验

本版替换旧的 `deep_audit/learned_audit/cluster_audit` 多入口代码。
旧代码保存在 Git 提交 `96cc93f`；**原 checkpoint、prepared 图和输出文件不迁移、不删除**。
只重构 CHARM 子项目，其他研究项目不改。

风格参考 Ja1Zhou/LM_Negation @0ddfe7d 的
`src/zzj_mi/mi_module/attn_sink_pos_control_sweep_base_models_gen.py`：
参数 → 数据 → 原模型 → 明确的干预循环 → 输出。只参考代码组织，不移植其 attention sink 算法。

## 先读这九个文件

| 文件 | 只负责什么 |
|---|---|
| main.py | 选择实验，读数据，循环消融，保存结果 |
| data.py | 读取旧缓存、划分与分数；不生成额外模型特征 |
| model.py | 原 CHARM 消息和节点更新；原 checkpoint 参数名不变 |
| ablations.py | 明确改变一个模块或一种图输入 |
| train.py | 原来源划分上的监督训练、选模型和校准 |
| evaluate.py | token / 同回答 / 同配对比较 |
| positions.py | 标注 span 内的第几个 token、首中末、三等分位置 |
| matching.py | 同答等长正常对照；统计量只用于匹配 |
| __init__.py | 包声明，没有运行逻辑 |

不再在模块之间传递几十组特征、训练多个代理回归器、导出所有层embedding。
**没有图重构训练或无监督检测器。这里是已训练 CHARM 的审计。**

## 1. 已有结果直接分析（默认；不运行神经网络）

在 graph 根目录、research 环境：

```bash
python -m experiments.charm_structure_audit.main --mode report
```

默认 `--root outputs/charm_structure_audit_qa/QA/seed_0`，符合当前实验目录。
默认读取已完成的 charm_in/charm_out/node_only/local_in/rewire_in，以及已有node_only平滑分数。
每个模型使用自己的 threshold.json。输入CSV的报警必须与原阈值一致。
模型缺文件时明确打印未发现，不新训练；已有模型的token身份/顺序不同则停止比较。

默认复用 `charm_in/test/cluster_audit/pairs.json`，不重新配对。
这条命令在上传的自然QA导出包上实际运行过，六份原AUROC/AP均复现；具体统计见 ANALYSIS.md。

## 2. 冻结 charm_in，逐个删除模块

```bash
python -u -m experiments.charm_structure_audit.main --mode ablate \
  --ablations no_graph no_node no_edge no_source no_mark no_prompt no_history no_residual no_relay head_mean
```

只读取已经保存的 `charm_in/checkpoint.pt` 和原prepared图；**不训练，不运行8B模型**。
`--checkpoint`允许指定这个原模型的迁移路径；回放不一致直接停止，不通过拟合修复。
每份图的原分数复算必须与之前的 score 一致，然后进行模块删除。
逐回答显示进度，保存紧凑的 `scores/*.npz`，没有逐层表征导出。原结果不改写。
重复同配置命令复用已完成的分数文件；仍检查原模型回放。中断临时文件不计完成。

| 消融名字 | 删除/改变什么 | 保留什么 |
|---|---|---|
| no_graph | 全部邻居消息 | 节点投影、更新MLP、残差和预测头 |
| no_node | 输入节点属性清零 | 边、多头边数值、角色及后续网络 |
| no_edge | 消息中的多头边属性清零 | 邻居状态、连接、来源标记 |
| no_source | 消息中的源节点状态清零 | 目标自身状态、边属性和角色 |
| no_mark | prompt/history标记清零 | 两类实际连接、节点和边属性 |
| no_prompt / no_history | 对应来源的入边 | 另一类连接；冻结干预使用原分母 |
| no_residual | GNN更新中的跳连 | 更新MLP；不是删除输入节点属性 |
| no_relay | 源节点不传递从其他节点汇入的信息 | 源自身每层MLP更新；一层时与原模型一致 |
| head_mean | 每层头数值平均后复制回原维度 | 层身份、网络形状、连接 |
| one_layer | 只用第一层GNN更新 | 输入投影和原readout；冻结模式属强分布偏移 |
| local / rewire | 最近k历史邻居 / 原粗距离组内重选 | RP边、目标入度、对应边属性 |
| coupled_heads / independent_heads | 整向量共同换端点 / 各头独立换端点 | 每target/来源/距离组内的逐头权重列表 |
| head_names | 每份图内置换各层的head通道 | 节点和边联合置换；不是全数据固定可逆改名 |

改变 `main.py` 顶部 `ABLATIONS` 或 `--ablations` 就能控制实验。
不同运行要用不同 `--output`；每次原图归一化分母保持不变，避免删边与重新平均混在一起。
控制可能是不自然输入，性能下降是**该模型的敏感性**，不是该模块的泛化贡献。
`graph_changes.csv`分开记录边槽位变化和真正丢失/新增的RR邻接关系。

## 3. 同划分独立重训（只有显式 train 才训练）

```bash
python -u -m experiments.charm_structure_audit.main --mode train \
  --ablations no_graph no_edge no_source no_residual no_relay head_mean
```

复用原 `charm_in/training.json` 的 fit/select/calibration/test IDs，与原prepared index对齐。
BCE、正类权重、AdamW、学习率计划、selection AP选模型、独立calibration定阈值沿用原配方。
所有新模型写 `audit_train/`，不会覆盖已经跑出的模型。始终加入full作为本次重训参考。
删边模型在重训中按自身图计算degree，而不是借用冻结干预的分母。
head_mean清零/复制等是明确的新消融，不能冒充旧实验复现。只有自然数据重训之后才有成绩。
`--epochs`可显式覆盖原轮数；小试不是完整实验。已完成fit复用；中断fit目前从头重训，不是epoch断点。
同来源划分上的一个seed不能替代训练方差；用不同seed单独OUTPUT。

## 4. 需要新匹配时才运行 match

```bash
python -u -m experiments.charm_structure_audit.main --mode match
```

使用原prepared图，不重新提取attention。默认 `--prepared outputs/charm_structure_audit_qa/data`。
只是把原匹配协议收拢到一个文件：同回答、等长、位置差≤.25、重复差≤.15、
同首token表面类别、同“此前是否已有错误”；候选全部无错误标签。
cluster再匹配七项聚集描述符；cluster_heads再匹配逐head段均值。
结构caliper仍为 [.1,.1,.1,.1,.1,.5,.5]，逐头RMS≤.05、最大差≤.25；不根据test得分放宽。
只合并重叠标注，保留相邻标注；正常区间不重复用；候选少的错误段先配。
这不是人工正确事实边界，也不是模型自动发现的span。
匹配不读score和embedding；保留缺匹配。输出pairs.json、匹配状态及逐特征balance。
旧pairs.json不覆盖；以新的配对评价需显式传 `--pairs 新路径`。

## 输出怎么读

每个模型/消融目录内：
- `summary.json`：总体、首错/后续起点/续错、同回答排序和阈值。
- `positions.csv`、`offsets.csv`、`positions_by_length.csv`：有多少错误词、检出多少、在片段何处。
- `pair_scores.csv`、`matched_positions.csv`：同一对正确/错误窗口的分数差，且按相同相对区域比较。
- `tokens.csv.gz`：完整逐词ID、原文、标签、报警、span位置。顶层 `models.csv` 和 `paired_comparison.csv`横向比较。

相对位置按 `(offset+0.5)/length` 三等分；first/interior/last是另一种划分，不把两套计数相加。
短于3token可能没有中间三分之一区域，分母如实减少；按长度另报，避免长片段主导所有结论。
同一个错误span内没有正常标签，不能伪造真假AUC；配对AUC使用对应正常窗口。
“首token的配对AUC”是逐对胜率，既不是全数据首错AUC，也不等于阈值召回。
未做句子分割或语义实体识别，不能把span位置叫语法句子位置，不能把数字token叫独立事实决策。
来源重采样区间对应source均值，配对宏均值单列；两者不是同一个估计量。

旧的原始attention准备、特征探针、代理logit回归、大规模表征导出从当前主线移除，
并没有在新main下面再隐藏旧入口。需要查历史算法时读取96cc93f；旧报告仍保留。
本版直接使用已经构好的图；新缓存的原attention→图准备不是这次重构的运行入口。

## 验证

```bash
python -m pytest tests/test_charm_structure_audit.py -q
```

测试覆盖原消息代数/梯度、旧checkpoint格式、每个消融、因果前缀、禁中继、
标签/阈值/配对口径、四种模式的真实CLI和原文件不变。依赖numpy/pandas/scikit-learn/torch/tqdm/pytest；
不增加PyG、transformer-lens或新的LLM依赖，不要求升级服务器的torch/CUDA。


## 5. 连续标签 vs 当前token携带的上下文状态

这个审计专门回答：LDA/CHARM的单token效果是否只是因为幻觉span连续出现。

```bash
python -u -m experiments.charm_structure_audit.main \
  --mode state_context \
  --lda-window 10
```

它比较五个监督LDA输入：
- current：只看当前token的1024维layer×head self-attention diagonal；
- previous：只看前一token的同一表示；
- past_mean：只看过去窗口的平均表示；
- delta_previous：当前减前一token；
- current_residual_after_previous：先在FIT上用前一token线性预测当前token，再只用不可预测残差分类。

同时固定上一token标签：
- previous_gold_0：前一token都正常，比较新错误onset与正常token；
- previous_gold_1：前一token都错误，比较错误延续与恢复正常。

因此，若current或residual在这些固定历史标签条件下仍有明显AUROC，不能把监督性能解释成简单的“错误标签连续”。单个x_t虽然按token送入LDA，但它来自Transformer上下文化计算：q_t/k_t由前层contextual residual产生，softmax分母也包含全部可见历史key，所以x_t本身可以携带此前上下文状态。
