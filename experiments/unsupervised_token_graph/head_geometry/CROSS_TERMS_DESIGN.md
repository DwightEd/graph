# 交叉项到底贡献了什么

本轮只检验一个问题：二阶矩检测的收益来自头间联合模式，还是各头幅度、持续状态、时间平滑？
这是无标签拟合、标签后置评估的表征消融，不是新的因果结论，也不预先保证提高 AUROC。

## 已有证据与理论的对应

有效的 head_interactions_v2 有 2 个来源、8 个有效 panel，恒等/恢复对照误差为零。
672 条 interaction 记录包含三个读出，不能当作 672 个独立实验；本次 sum 与
candidate_sum 读出重复。以下数值均为 supported 候选减 rival 候选的 log-probability margin。

|已有观察|理论能帮助理解什么|尚不能推出什么|
|---|---|---|
|headwear/unsupported：L23H6 在 L22H28 存在时贡献 +0.0981，删除后贡献 −0.1519|边际贡献依赖其他头；固定背景下二阶差分 I=+0.2500|不是一个跨背景稳定的“事实头”，也不是完整 Shapley 值|
|headwear：L31H14 的 history attention 质量约 0.9996/0.9985，删除效应却为 +1.125/−1.750|注意到内容与向输出写入正/负贡献不同|不能由 attention 强度判断支持、抑制或正确性|
|onion/supported：删 L21H11 损失 0.6020，恢复 L23H18 后只损失 0.1930|下游适应可以放大上游扰动|恢复的约 68% 是该读出的损失比例，不是信息流量比例|
|headwear/unsupported：删 L26H31 时 L27H8 自然适应缓冲损失约 0.1083|删除后的补偿，与 self-repair 研究的干预逻辑一致|不能说自然幻觉发生前已经出现这种补偿|

[Shapley / Adalina](https://arxiv.org/abs/2604.08438)研究跨 coalition 的归因及其高效近似；
适用于以后扩大机制审计，不能把一个背景的四条件差分叫 Shapley，也不直接给出检测器。
[Hydra effect](https://arxiv.org/abs/2307.15771)提示删除后其他组件会改变；
这正是本项目同时测自然下游响应和恢复基线消息的理由。
[Copy suppression](https://arxiv.org/abs/2310.04625)展示关注某 token 的头也可能抑制它，
但其 GPT-2 头的角色不能直接移植到本项目的模型。

两个例子的正常侧平均交互绝对值均高于错误侧；所以“交互越强/越混乱越幻觉”没有得到支持。
L31 的本次干预只改变候选首 token，onion 候选又有长度及属性不匹配，不能据此解释 span 持续。
当前 audit 还没有证明“正确证据路线被某个 history head 抑制”：输出竞争与路线抑制必须区分。

## 无监督结果目前支持到哪里

|任务|raw AUROC / AP|moment AUROC / AP|AUROC 差|
|---|---|---|---|
|QA|0.6447 / 0.1444|0.6900 / 0.1812|+0.0453|
|Data2txt|0.6466 / 0.1040|0.6732 / 0.1158|+0.0266|
|Summary|0.5560 / 0.0548|0.5568 / 0.0520|+0.0008|

这些是用户提供的旧运行结果。尚未取得该运行的完整 settings，不能只由 all__ 前缀判断用了哪些层。
moment 同时改变了表征、窗口、白化、维度和距离尺度。log_moment 与 log_diagonal 的对照
不能替代普通二阶矩的交叉项消融。条件能量略胜独立能量也不是充分证据：它还改变了残差标定。

raw 的近邻距离已经在完整多头向量中找联合状态，并非逐头独立打分。二阶矩可能改进的是
这种联合状态的表示和度量，而不是从零开始引入“协同”。单 token 的 zz^T 没有增加输入信息，
只是非线性特征映射；窗口 M 则额外使用过去 token。先区分这两个因素再讨论优化。

一个可解释的先验是保留有身份、有符号的“相对 FIT 基线一起偏高/偏低或反向变化”的模式。
这不是注意力越大越重要，也不是相关为负就代表抑制；z 的符号来自统计标准化，并非 OV 的贡献符号。
只看 zz^T 会把 z 与 −z 混为一谈，所以仍保留当前完整 z。
目前不需要为了这个问题引入黎曼距离；先检验普通矩空间中的可识别增量。

有监督 LDA 用标签决定哪些联合模式对应错误；无监督近邻只决定哪些模式偏离混合参考分布。
常见的错误模式可能不异常，罕见但正确的生成也可能异常。这是“可被有监督读出”和“可被异常检测排序”
之间的实质差距；仅换几何空间不保证消除。新对照检验表征是否有效，不预设这个差距已经解决。

## 要拆开的量

每层每个物理 head 的 self attention 使用 FIT 来源的中位数和 IQR 标准化为 z；保留零值，
沿用已去特殊 key、未重新归一化的缓存。稀疏压缩的零不等于原始注意力严格为零。
默认不沿 head 去均值、不做全协方差白化，不引入联合高斯。多 signal 时一个 head 是一个块。

在截止当前 token 的连续有效窗口 W 中计算：

\[
M_t=|W|^{-1}\sum_{s\in W}z_sz_s^T,
\quad \mu_t=|W|^{-1}\sum_{s\in W}z_s,
\quad C_t=M_t-\mu_t\mu_t^T.
\]

D(M) 为同一 head 内的块；X(M) 为不同 head 间的上三角项（乘 sqrt(2)）。
单个 self signal 时 D 就是对角线，X 就是不同头乘积。
因此 X(M)=X(C)+X(mu mu^T)：头间交叉项本身也包含持续状态，不能将它全部称为动态协同。
窗口均值只用于这个数学分解，不替代逐头表征，也不对 head 求平均。

X(mu mu^T) 等于窗口内各 head 独立随机重排后交叉矩的期望；它保留各头的平均状态，
去掉同步波动。这是解析对照，不用未来 token，也不需要随机打乱整条答案。
这些统计量不是因果交互 I=F11−F10−F01+F00，且窗口矩本身不编码时间先后顺序。

## 同场比较的读出

|名称（省略 all__）|表征/分数|所检验的问题|
|---|---|---|
|pair_state|当前逐头 z|无历史基线|
|pair_diagonal|z + D(M)|各头自身幅度与持续性是否足够|
|pair_cross|z + X(M)|不加对角块时，交叉项是否仍有用|
|pair_full（预设主候选）|z + D(M) + X(M)|完整分块二阶矩|
|pair_covariance|z + D(M) + X(C)|只保留同步波动的交叉项|
|pair_persistence|z + D(M) + X(mu mu^T)|只保留持续平均状态的交叉项|
|pair_full_w1|z + D(zz^T) + X(zz^T)|无需累积历史是否也能获得收益|
|pair_state_smooth|pair_state 原始异常分数的因果窗口均值|单纯平滑是否解释收益|
|raw / moment|原 v2 计算和距离|桥接旧结果，防止新坐标与距离掩盖退化|
|moment_diagonal / moment_cross|旧 moment 白化前只留同头块 / 跨头块|直接归因旧 moment 的增益|

各方法使用同一来源划分、同一参考 token、同一 5-NN 读出；不同表征下近邻身份可以变化。
pair_* 的 z/D/X 尺度仅从 pair_full 的 FIT 参考表征估计，其余方法复用对应列的尺度。
各块按 sqrt(块宽) 缩放，避免多达数千个交叉项仅靠数量淹没当前状态；不按 TEST 成绩调权。
z 和 D 保留原坐标；X 独立固定 sketch 到 256 维，绝不把 D/X 混投影后再做消融。
--dimensions 0 可精确保留所有项，代价是更大内存和距离计算量。
现有距离函数的最终 sqrt(总宽) 除法只给每种方法乘一个正常数，不改变排序或近邻身份。

pair_covariance / pair_persistence 固定相同 D(M)，只替换交叉块；不是整份矩阵的协方差对照。
moment_diagonal / moment_cross 在旧 contrast 坐标中拆分，再应用同一个 FIT 白化矩阵、同一个
sketch 和完整 moment 的参考尺度，保留相同当前 contrast。前者仍因层内去均值而含有其他头的
参照信息，故只用于归因旧方法；原始 raw 坐标的 pair_* 则检验不含该和为零约束的头间关系。
删除发生在白化之前，避免把旋转后的某条对角线误叫物理 head；它们不改变旧 moment 分数。
pair_full_w1 使用同一尺度但自己的瞬时参考表征；所有方法共同在完整窗口 token 上再比较一次，
避免窗口初始化长度不同。缺失 token 会截断历史，合法的 attention 零不会截断。
--coordinates contrast 只作为另开目录的敏感性实验：沿 head 去均值会人为引入和为零约束。

## 报告与判断

按 QA / Data2txt / Summary、生成器分别拟合并汇报，不把 ALL 当主结论。
标签仅用于冻结后评估：总体、首错、span onset、continuation、前后半段、previous_normal/error、
完整窗口与 warmup；对主候选减每个对照做 source 配对 bootstrap，同时看 AUROC 和 AP。
置信区间未做多重比较修正，这些已反复查看的测试数据只能给探索性结果。

1. full 优于 diagonal、cross 也优于 state：支持头间统计关系有额外检测价值。
   旧 moment 优于 moment_diagonal 则直接支持旧检测器依赖白化前的跨头项；二者需分别报告。
2. full 不能优于 persistence：不能宣称学到了同步变化；持续的联合状态可能已经足够。
3. full 不能优于 state_smooth：暂不需要用复杂关系解释收益。
4. full 优于 w1，但只在 span 后半段提升：更像累积确认器，不能宣称提前预警。
5. 旧 moment 明显优于分块 full：要检查原坐标、白化及度量，而不是宣布交叉项无效。
6. 只在某一任务有效：保留任务边界，不做通用幻觉机制结论。

不把 AUROC 差的比值叫“交叉项解释了多少百分比”，不同方法的排序变化不是可加因果分解。
当前 onset_vs_normal 是全部正常 token 对照，并非长度/位置/句法匹配的正常 span；后者仍需
独立评估设计。当前矩只描述同层不同头，未检测跨层有向传输、value/OV 载荷及语义适用性。

## 工程与运行

复用 head_geometry 的输入、源级抽样、近邻、校准和评估；cross_terms.py 只实现矩的分解与
共享尺度，reuse.py 只负责已有 observations 的复用。旧 geometry 默认行为保持兼容。
reuse 不重新读取大注意力缓存或加载模型；原 observations 应保持不变，设置记录其 manifest 摘要。
参考库仍从混合无标签 TRAIN 拟合（这属于无监督拟合，不是免训练/无需参考集）。

```bash
git pull origin main
bash experiments/unsupervised_token_graph/head_geometry/run_cross_terms.sh
```

默认从 outputs/head_geometry_middle 复用已准备的完整 observations，并继承实际 task、head、
数据集和 tokenizer 设置；可用 OBSERVATIONS 指向旧运行。窗口、参考预算、种子仍需与旧 settings
核对，想复现非默认旧参数请显式传入。不要与旧运行共用输出目录。

```bash
OBSERVATIONS=outputs/head_geometry_middle OUTPUT=outputs/head_cross_terms_v1 \
  bash experiments/unsupervised_token_graph/head_geometry/run_cross_terms.sh --bootstrap 1000
```

若参考运行目录不是 head_geometry_middle，只改 OBSERVATIONS 即可。精确交叉项和 contrast
敏感性运行分别使用新的 OUTPUT，并传入 --dimensions 0 或 --coordinates contrast；不依据 TEST
胜者反复挑种子。原始主实验先完成，再按剩余混杂决定是否需要这些敏感性运行。

检查 predictions/task_summary.md、predictions/comparisons.csv 和各任务 evaluation.json；
完整 all/evaluate 成功后自动生成 outputs/head_cross_terms_v1_review.tar.gz。
压缩包含设置、观测清单、参考来源清单及冻结分数/报告，不含大 attention 缓存和参考 bank。
实现验证只使用小型合成数据；真实数据 AUROC 需要服务器运行后再作结论。
