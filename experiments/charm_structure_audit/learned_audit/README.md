# 已训练 CHARM 究竟用了什么信息？

本目录只审计已有 `charm_out` 或 `charm_in`。直接导入上级 `model.py` 的 CHARM、
`load_checkpoint`；不重写或训练 CHARM，不重采 8B attention，也不改原始预测和阈值。
先读 PLAN.md：每项问题与代码文件一一对应。已有的 diagnose_saved 继续用于同答内部定位分析。

## 一条命令（QA，沿用已有目录）

```bash
bash experiments/charm_structure_audit/learned_audit/run.sh
```

默认输入：
`outputs/charm_structure_audit_qa/QA/seed_0/charm_out/{checkpoint.pt,training.json,test/}`
和 `outputs/charm_structure_audit_qa/data/graphs/{train,test}/`。
无需再填写 RAGTruth 标注、source_info 或 tokenizer；图和标签都沿用上一轮已准备的数据。
模型、原预测或图对不上时，实际推理与旧分数的检查会停止审计，不偷偷继续。

默认先复用旧诊断入口，将回答内/回答间及旧控制报告写到本目录的saved_scores/。
接着：计算每头特征及现有模型logit → 拟合小型解释模型 → 冻结模型做输入/消息对照 → 分首错与续写出报告。
训练的是解释 logit 的小回归器，不是新的幻觉检测器，不改 CHARM 权重。它只用原 fit/select；test只检查。
每份图单独读取，GPU按原 EDGE_CHUNK 分块，不把整套数据放进显存。
默认每个排列控制跑3个固定seed；状态替换在每个GNN层各跑一次。
结果逐回答落盘，中断后同一命令跳过完成部分；未完成回答重新开始。
没有epoch级续跑的问题，因为这里不训练神经检测器。

可分阶段运行：

```bash
STAGE=features bash experiments/charm_structure_audit/learned_audit/run.sh
STAGE=explain bash experiments/charm_structure_audit/learned_audit/run.sh
STAGE=intervene bash experiments/charm_structure_audit/learned_audit/run.sh
STAGE=report bash experiments/charm_structure_audit/learned_audit/run.sh --completed-only
```

`report`只读新审计分数，不进行模型前向；仍核对已保存的原配置。
`BOOTSTRAP=0`只关闭source均值logit差的区间，不关闭AUROC/AP。
更换种子、输入或head集合时使用新的OUTPUT；不要改旧settings绕过检查。
只查某几个预先选定的物理头可附加 `--channels 10:3 15:7`（例子，不是已发现重要头）。
它会额外清零这些头的节点/边通道，拓扑槽位保留。不能用test结果挑头后称独立验证。

## 三组关键对照

| 名字 | 改了什么 | 保留什么 |
|---|---|---|
| vector | 多头向量与邻居的配对 | 全向量、多头同看关系、每头熵 |
| layer | 不同LLM层在同一邻居上的配对 | 各层内部多头关系、每头熵 |
| head | 各head在同一邻居上的配对 | 每头自身的权重集合与熵 |
| positive_head | 每头正权重的相对大小配对 | 每头非零位置、熵 |
| head_names | 同层head编号；节点和边一起换 | 来源位置和熵的多重集，不保留物理头身份 |

所有排列只在同target、RP/RR、粗lag内进行；保留原边槽位、度数、分母。
**layer→head 的额外损失**更贴近“同一层多个头共同关注哪些邻居”。
**vector→layer 的额外损失**更贴近“不同层如何共同使用某个邻居”。
不能仅凭 head_names 换名后下降就说学到了多头合作，因为网络可能只是依赖某个单头。
独立打乱含零权重时会改变每个head的非零邻居，并可能制造全零消息槽位，诊断会明确计数。
这些控制不是重新生成的真实attention，冻结模型性能下降可能含分布偏移；不声称严格因果真伪。

## 两个head是否产生联合作用？

先在原fit集按每头熵与原logit的关联预选最多两个同层头对。这只是筛查，不是因果重要性排序。
对真实消息MLP计算：`联合项 = M(完整) - M(头A清零) - M(头B清零) + M(A/B都清零)`。
其他通道、来源状态全保持；从原消息中只减去联合项，后面各层照常运行。
`interaction_Lx_Ha_Hb_gy`报告该操作对首错/续写的影响。非加性存在但去掉后分数不变，不能说检测在用它。
零参考会制造非自然输入，必须结合上面的等熵重排；不能据此宣称两头在LLM里共同计算真假。
单独先跑intervene、没有fit筛查文件时，这项明确标未运行；之后增加筛查须新建输出，不能混入旧完成样本。

## 邻居状态替换，而非简单删边

通过实际消息MLP的钩子，只替换输入向量中“来源节点状态”部分；边属性、类型、目标状态不动。
`group_mean_g0`将同组来源状态替为组均值；g0/g1/g2表示GNN层，不是LLM层。
`same_label`和`other_label`只动同一批合格RR边，每条边都同时有同标签和异标签donor才参与。
来源限同答、同target前、同lag、表面类别及copy状态相同；粗熵、自注意力和log出度相近。
这是粗匹配，不代表事实或语义等价。匹配很少时，零变化不表示没用。
标签只在选择解释性干预对象时读取，不进入原模型前向。正常来源换错误来源也可能提高误报。
一次改变某层多条消息，后面各层照常计算，因此不是可加和的独立逐边贡献。

## 熵解释模型的边界

features保存全部物理layer/head，而不是先平均attention。
熵是在图里实际保留的非负权重（加回对角线）归一化后计算；没有恢复被压缩丢弃的边。
解释模型先保留9个位置/度数/质量等控制变量，再在fit集按与logit关系选择额外变量，默认合计最多64维。
原 fit/selection 用来拟合及选线性/小树回归器；不使用test标签选变量或阈值。
输出 context、加熵、加逐头质量/对角线、再加同层头重叠四组解释能力。
**小解释模型拟合差，不能证明不存在熵解释；残差仍有区分力，也不证明控制了所有逐头熵。**
解释器没有显式拟合全部高维交互。最后必须结合“熵不变但配对改变”的实际模型控制。

## 输出

写入原test目录下独立的 `learned_audit/`：

- `README_RESULTS.md`：首错/续写的控制差异表和通俗说明。
- `mechanisms.json`：各控制的原/改动后AUROC/AP、原阈值召回、source均值logit变化区间，及实际扰动量。
- `explanation_predictions.npz`：各解释器的逐token预测和残差，可复查是否只是整体AUROC相近。
- `score_explanation.json`：用熵等变量解释原分数的程度，首错的解释分数及残差排序。
- `token_effects.csv`：每个token的角色、原分数和每项控制对它的影响，不只挑“好看”例子。
- `samples/*.npz`：逐回答的原logit、控制logit、逐头特征和诊断；可复用。
- `features/*.npz`：用于解释模型的原分数和逐头测量。

区间针对source等权的logit变化，不是AUROC差的置信区间。不同seed分别报告。
这里重复使用了已看过的QA test，应称探索性归因，不是全新确认实验。
统计相关、冻结输入扰动和LLM真实事实计算是三种证据，不能混称同一个机制。

## 测试

```bash
python -m pytest tests/test_charm_learned_audit.py -q
```

测试包含已知“只使用熵”和“使用同一来源的双头共现”的小模型见证，检验对照确实区分这两类函数。
也检查原模型钩子与身份替换一致、分块映射、清理钩子、输入不变、分数不匹配时停止，及CPU完整流程。
自然数据和实际高分checkpoint未在本地运行；实际结论要看用户服务器生成的新报告。
