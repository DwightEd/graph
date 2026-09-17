# 模型内部的片段复用：不是把 HHI 改名为 self lock-in

## 先固定要检验的说法

本轮检验 **冻结 CHARM 的内部片段复用是否支持了错误高分**，不是证明生成 LLM 有吸引子。
因果图只有 j<q 的边，没有有向环；这里的“重复传递”指相邻 token 和多轮 GNN 更新中的继承。
GNN 的第 k 轮不等于原 LLM 的第 l 层。原节点/边的全部 L×H 通道原样送入训练后的 MLP。
不新增 attention 集中度特征，不训练新检测器，不重新配对，不挑 TP 后才定义主要分析。

研究单位是已经保存的错误/正常等长配对。两侧分别干预，绝不在同一次前向同时修改两个窗口。
这是使用金标范围的机制诊断，其 AUROC 是“固定配对诊断 AUROC”，不是可部署检测成绩。
原完整流的分数和阈值不改变。高低分、TP/FN只是冻结原分数后的分组描述。

## 三个可被否定的命题

H1 段内消息有功能作用：切断指向当前窗口的段内消息，最终 logit 下降。
H2 被传递的是继承信息：保持边、边属性和来源的外部上下文，替换掉来源此前从本段接收的状态，效应仍明显。
H3 有错误特异性：H1/H2的效应在错误侧强于结构匹配正常侧，且不是一般局部连接和词项复现就能解释。

只出现H1不是lock-in：正常连贯生成也依赖内部连接。只出现高分或大消息范数也不是。
多轮效应不增长、正常侧同样变化、或无可辨别随机对照时，不能宣布锁定。
通过上述测试最多支持“检测器使用了异常片段复用模式”，原LLM因果机制需要独立干预。

## 直接在训练后的 MLP 上做什么

对窗口 S 的入边，I={j→i: j和i都在S}；E={j→i: i在S而j在S前}。
E包括prompt和更早的response，**不等于正确证据**。所有删除置零的是msg_mlp的整个输出，
不是把attention改0后仍让MLP偏置产生消息。原节点属性、原union边和入度分母均不变。

1. full：原前向，必须复现原token分数。
2. cut_internal：每轮阻断I消息；cut_external：每轮阻断E消息；cut_all：两类均阻断。
   cut_all保留原完整模型的节点自身非线性/残差，不等于独立训练的node_only。
3. no_reuse：I上的来源状态替换为cut_internal同轮的来源状态；E保持当前前向。
   这样来源仍有自身特征和外部上下文，只不转传此前由本段传入的内容。第1轮应与full相同。
4. pulse_k：只在GNN第k轮阻断I，后续照常计算；衡量它对最终输出的作用，不把中层接同一readout当成真实分数。
5. random_r：按每个目标、粗距离组{1},{2},{3,4},...、是否同token-ID，随机阻断相同数量history消息。
   候选包含原段内边，不能假定它们都能换到外部；报告可交换数量和实际重叠。
   连续span中很多距离组全在段内，此时随机对照无法区分机制，不能把零差异当阴性证据。
6. last_swap_internal / external / own：仅在最后更新轮，把对应聚合消息或目标自身状态换成
   配对窗口同offset的真实激活，其他项固定；两侧独立替换。用于检查模型实际使用的组合。
   激活替换可能离分布，不能称作语义证据修复。

## 不用权重范数冒充贡献

输出精确分解最终logit：z=b+sum_u(w_u * ReLU(V h+b_V)_u)。
每个实验保存末层所有readout单元的有符号贡献（当前配置64个），并核对求和误差。
同时记录最后更新MLP的ReLU门相对full改变比例：门改变是机制线索，不是独立因果证明。
不比较不同模型的同编号神经元。

两类消息的2×2干预给出精确的有符号分配（默认零消息参照，有基线依赖）：
C_I = 0.5*((z_full-z_cutI)+(z_cutE-z_cutAll))
C_E = 0.5*((z_full-z_cutE)+(z_cutI-z_cutAll))
C_I+C_E = z_full-z_cutAll
interaction = z_full-z_cutI-z_cutE+z_cutAll

这能区分“自身已有高分”“段内消息支持高分”“外部消息抑制/支持高分”“两类组合交互”。
负贡献不取绝对值，也不把它归一成虚构的因果百分比。分解使用logit而非sigmoid概率。

## 汇总必须对齐

主要分析所有已锁定配对，逐词、首词、前后半、相对三等分，以及GNN脉冲轮。
每种干预先算 error_effect-normal_effect，再按pair平均、按source平均和bootstrap。
报告原/干预配对AUC、logit间隔，正常FP、错误TP和两侧同时成功数量；原概率分数保留在逐词表中。
只有窗口内才是干预输出，其余词不拼成一个“全体干预检测器”。
原分数定义每回答的高/低20%和TP/FN，原分组始终不随干预改变；不把“高分”当“误报”。
汇总表中的区间为探索性source区间；图展示均值，不作多重检验显著性声明。

## 计算和接口

```bash
python -u -m experiments.charm_structure_audit.main --mode lockin
```

默认root=outputs/charm_structure_audit_qa/QA/seed_0，prepared=outputs/charm_structure_audit_qa/data。
默认只读charm_in/checkpoint.pt、其原test预测和cluster配对。没有配对时停止，不重新选容易对照。
--lockin-random 3控制随机重复数；--lockin-pair-limit只用于显式小试，输出标记subset。
--lockin-stage report只读本次已有紧凑captures复画图；不运行任何checkpoint。

先每回答回放一次完整原模型，再只重算窗口内部受影响的计算；窗口之前的节点不可能受它影响，
因此复用这些节点的原激活等价于整图重算，而不是截断丢消息。该等价性在测试中验证。
每对保存1份npz，含各世界的逐词logit、末层贡献、激活门变化及控制覆盖，
不保存原边张量和所有逐层embedding。完成pair可续跑，配置/模型/图时间大小检查集中在入口。
所有新文件写audit_lockin，不修改旧checkpoint、分数、配对和阈值。

三个实现文件：lockin_forward.py（真实MLP与干预）；lockin.py（读数据/执行）；
lockin_report.py（配对评价/输出图）。main仅新增一个mode，不再接新特征管线。

## 文献的边界

Reasoning Fails Where Step Flow Breaks (https://arxiv.org/abs/2604.06695)研究attention-gradient步骤显著性，
不能把本实验叫它的复现。Activation patching的指标/替换方法会影响结论：
https://arxiv.org/abs/2309.16042 。本实验分别保存分数、排序、基线、扰动范围，不由一张图宣布机制。

## 先看哪些输出

- REPORT_zh.md / summary.csv：同一批配对的效应与来源区间，不重拟合阈值。
- detection_counts.csv：原/干预后的TP、FP、覆盖和两侧同时成功数；不是整段point-adjustment。
- component_summary.csv / pair_components.csv：自身基线、内部、外部、交互的有符号logit分解。
- tokens.csv.gz：每个高低分TP/FN/FP/TN的真实文本、位置、内部/外部贡献及no_reuse效应。
- random_adjusted.csv / random_summary.csv：仅在双方均有可交换来源的位置比较；保留零覆盖。
- readout_units.csv.gz：所有末层单元在各世界的贡献；求和精确重建logit差。
- controls/*.json：实际删边数量、原段内边重叠、粗距离和保留权重；候选缺乏变动不能证明无机制。
- figures/specific_effect、auc_delta、components、pulse_rounds、readout_units 的PNG/SVG。
- lockin_review.tar.gz：小型报告包，自动排除captures数组；不需要打包整个seed_0。

报告中的AUROC沿用原保存概率分数和干预后的sigmoid分数；logit只用于有符号贡献分解。
返回报告阶段用 `--lockin-stage report`，其余参数与原命令一致（例如用了pair-limit要保持）。

本轮另修正routes.depth_contrasts中零维数组应转float的问题；仅影响缺测汇总，不改变原路由定义。
训练/原始模型文件没有改动。
