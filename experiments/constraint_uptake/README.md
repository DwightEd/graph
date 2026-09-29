# 候选条件下的来源消息传播与采纳

本模块把来源读取、传播、输出采纳分开测量。它是一套已运行的小样本测量与检测候选，不是已解决约束绑定的检测器。结果及具体失败见 `RESULTS_20260929.md`。

## 建模对象

来源位置j、当前预测query q、物理层l/头h的消息为

    m[l,h,q,j] = A[l,h,q,j] W_O[h] V[l,h,j]

V是原模型上下文化的原生value，可能包含关系信息，也包含语言、主题等内容。算法不预先标注prompt实体、证据或关系。key对应读取地址，不自动等于正确证据。

实际词y与固定候选c的输出竞争为f_c=z(y)−z(c)。每query选最终最强的3个非实际词，不人工指定正确答案。两个读出：

    local[l,h,q,j,c] = grad_immediate_lens(f_c) · m[l,h,q,j]
    native[l,h,q,j,c] = grad_native_downstream(f_c) · m[l,h,q,j]

local在attention写入后、该层MLP之前，用最终norm和输出矩阵作局部logit lens；native沿当前query后续attention/MLP真实运算反传。local不是校准后的中间层概率。两者变化说明同一消息的输出读出如何变化，不能仅凭符号断言事实真假。

复用 `decision_risk_flow.native`，past K/V保持原值，当前位置自己的K/V重新计算。query t预测回答token t，位置P−1+t；不同query独立，无跨batch未来泄漏。**这是当前query条件效应，未包含修改较早token并重算全部未来K/V的总效应。** FP32执行、冻结BF16权重，无训练或参数梯度。

## 不过度压缩信息

每个回答保存全部32×32物理头的attention，以及原生128维head value、native/local梯度经W_O的投影。它们精确重建逐key×候选的有符号效应：

    effect[j,c] = A[j] * (W_O^T grad(f_c))^T V[j]

这是模型原有head维度上的精确因子表示，不是PCA/SVD或随机sketch。当前self key单独保存；query=0时self仍属于prompt，后续self属于回答历史。没有只保留top8头或top-k来源地址；top8仅是第一轮失败读出，原数据仍完整可读。

## 三轮无标签候选

第一轮：每个head/candidate保存P+、P−、H+等正负作用，冲突比例

    conflict = 2 sqrt(P− H+) / (P_abs + H_abs)

另保留prompt反对比例、local正而native负的反转、正负抵消和prompt mass。第一轮按比例取最高8头、候选最大，暴露了弱头小分母饱和问题。

第二轮：全1024头按绝对输出作用量加权，再按3个竞争词的最终条件概率加权；主候选energy_conflict。energy_conflict_max分离候选聚合的影响。另用来源/历史净效应替代逐key正负量得到net_conflict，并保留energy_reversal等控制。没有依据标签训练、翻转方向或搜索权重。每层是不同扰动位点，跨层加权不声称是互斥的根来源归因。

各ratio固定>.5仅作探索工作点，不是错误概率或开发集校准FPR。margin控制用负margin>0，NLL控制用>log2；这些工作点不能视为公平的相同FPR比较。连续AUROC、逐案例排序和首错位置单独报告。第二轮受第一轮已知结果启发，属于开发迭代。

第三轮 `sink_center.py` 根据逐头平均prompt attention选择常驻参考b，用零和地址交换测量 `A_b A_j g^T W_O(V_j−V_b)`。它消除同一头各key共有的value偏移，并在±.25有限干预时保持attention总质量及非负性。参考地址不自动等于无意义sink，结果由独立有限重放验证。所有原始因子保持，32query分块仅减少临时内存。

## 单独、联合干预与hidden路径

`finite.py` 对无标签选择的位置，自动标点来源块最强/最弱响应及不同来源控制块，进行消息幅度0/±.05/±.25改变，保持其他消息和原历史。没有重新归一化attention，因此是message amplitude intervention，不是改变QK寻址。两个来源块的prompt没有第三块对照，明确缺测，不补零。

`interactions.py` 在既有正负配对声明、GSM首错步骤及正确同序号步骤的中间位置，作标签辅助机制审计。prompt按固定16token匿名块划分；正块、负块、联合、对照、正块+对照联合共5种处理。块名positive/negative表示最大/最小有符号响应，不保证数值正负。选头依作用量×冲突，使用标签只定位审计点，不拟合检测。

交互量Gamma=Δf(A+B)−Δf(A)−Δf(B)，与同类控制联合比较。每次保存32×4096的完整hidden差和跨层候选投影，观察同一消息扰动的层间传播；非加性、范数增长、符号变化均不自动等于关系执行正确。

`same_value.py`是标明用途的事后机制fixture：只检查GSM49的四个24位置，固定L30H29及24-vs-20候选，源key/历史key按实际效应选择，另保留最强非复现历史消息。它不进入检测分数；同处首错步骤不等于每个24均错误。保存的每条边是固定历史下当前query的条件效应，不把它们相乘冒充跨时间总效应。

## 样本与标签边界

- 两个RAGTruth来源、同prompt自然采样正负局部对照：14315头饰范围、14375洋葱阶段。31个局部已复核token，其他48个邻域位置unknown；不是官方全回答真假标签。正确局部一侧的其他内容可能有错误。
- 官方旧难例15604、9022，全文401token/34错误；分别属于QA与Data2txt，须看逐案例结果，不只看混合AUROC。
- ProcessBench GSM8K三组同题正负回答：43/240、49/243、123/313。标签只支持首错及此前步骤；44步中24已知（3首错、21正常），20后续unknown。第一轮每step均匀至多8query，后续对这六答全部2310token密集复验，公式不变。

全部为暴露的开发案例，无独立泛化或全测试成绩声明。观测为8B observer重放，即使模型名相同也不冒充原采样时的状态。不同自然前缀、同序号步骤不保证相同语义决策。

## 执行

使用现有research环境和本地Llama3.1-8B，无新安装。capture/finite/dense/interactions需要GPU，score/evaluate/reweight用CPU。已有历史缓存位置由data.py读取。

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python -m experiments.constraint_uptake.data --output outputs/uptake_new
python -m experiments.constraint_uptake.capture --output outputs/uptake_new
python -m experiments.constraint_uptake.score --output outputs/uptake_new
python -m experiments.constraint_uptake.evaluate --output outputs/uptake_new
python -m experiments.constraint_uptake.finite --output outputs/uptake_new
python -m experiments.constraint_uptake.reweight --source outputs/uptake_new --output outputs/uptake_weighted
python -m experiments.constraint_uptake.evaluate --output outputs/uptake_weighted
python -m experiments.constraint_uptake.dense --previous outputs/uptake_weighted --output outputs/uptake_dense
python -m experiments.constraint_uptake.evaluate --output outputs/uptake_dense/weighted
python -m experiments.constraint_uptake.interactions --dense outputs/uptake_dense --rag-factors outputs/uptake_new
python -m experiments.constraint_uptake.audit --source outputs/uptake_new --reweighted outputs/uptake_weighted --dense outputs/uptake_dense
python -m experiments.constraint_uptake.sink_center --dense outputs/uptake_dense --rag-factors outputs/uptake_new --output outputs/uptake_centered
python -m experiments.constraint_uptake.evaluate --output outputs/uptake_centered
python -m experiments.constraint_uptake.interactions --dense outputs/uptake_dense --rag-factors outputs/uptake_new --scored outputs/uptake_centered --output outputs/uptake_exchange
python -m experiments.constraint_uptake.same_value --dense outputs/uptake_dense --centered outputs/uptake_centered --output outputs/uptake_same_value
python -m experiments.constraint_uptake.audit --source outputs/uptake_new --reweighted outputs/uptake_weighted --dense outputs/uptake_dense --centered outputs/uptake_centered --exchange outputs/uptake_exchange
python -m pytest experiments/constraint_uptake/test_uptake.py -q
```

真实输出前缀 `outputs/constraint_uptake_20260929_{v1,v2,dense,v3,exchange,same_value}`；`units.csv`为全部已测token/步骤分数，`errors.csv`为各固定工作点的FN/FP，`interactions.json`含实际来源文本和候选，`paths/*.npz`含完整hidden传播差。所有原始大数据留本机，不提交Git。当前不替换原强路由基线。
