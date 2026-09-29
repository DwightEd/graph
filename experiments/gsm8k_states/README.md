# GSM8K 首错步骤、跨层状态与监督读出审计

本轮400答真实8B observer重放已完成，结果见 `RESULTS_20260929.md`。最终目标是首错步骤定位：第一越阈值步骤为预测；无报警输出−1。使用数据已提供、与原文核对过的步骤边界，不把任意空格或标点机械拆成步骤。首错之后标unknown，不编造后续步骤或token真假标签。

## 输入与状态对齐

沿用 `experiments.gsm8k_recurrence` 的400答、375问题及固定问题隔离分区：80fit、81dev、239evaluation。原数据、attention缓存位置见该模块README。评价239答包含127错误答、112全正确答；可确认909步，其中127首错、782正常，315后续未知。

`capture.py` 使用本地Llama-3.1-8B-Instruct、BF16、SDPA、完整原token。每个步骤保存32层4096维raw residual的入口、末端、均值，不做低秩投影。HF最后hidden是归一化结果，故由最终norm的pre-hook另取raw末层。步骤[a,b)、prompt长度P：入口P+a−1，末端P+b−1，均值P+a:P+b。实际词y_t预测状态为P+t−1。末端读出在观察完整本步后进行，适用于离线首错定位。

不是原采样时采集；即使生成器与observer同名也不升级为原生成过程。全32层状态原件保存；监督诊断固定取四层7/15/23/31全坐标拼接，16384维，不按评价挑层。

## 输出采纳的量化及限制

每token固定候选C_t：实际词及最终输出最高的31个非实际词。把每层raw residual经最终norm和lm_head读出为条件分布p_l(.|C_t)。这是受限候选logit lens，未经中间层概率校准，不是全部词表的层间真实分布。

固定竞争词c_t为最终最强非实际词：m_l,t=z_l,t(y_t)−z_l,t(c_t)。主候选是最后8层−log p_l(y_t|C_t)的层均值，再取步骤token均值。final_margin使用最终完整词表logits；final_nll与entropy同样来自完整词表。其他固定对照：相邻层JS、最后8层末端state路径/净位移比、31个跨层末端更新的Gram谱集中度1−exp(H(归一化特征值))/31。

margin正表示偏向原词，不表示原词为真；JS小既可能是稳定正确，也可能是稳定错误。谱对照没有复现论文的4096坐标相关图、原生Jacobian或完整因果传播，只检验跨层状态变化的简单预测力。BF16的selected einsum和full-vocabulary GEMM有数值差异：400答最终margin差异中位数/95分位为0，最大0.25；原始缓存保留，不能称两读出精确相等。

## 监督仅作诊断

`readout.py` 无监督分数不读真假标签；dev混合分布的问题等权95分位为报警阈值。标签不参与其参数或方向确定。

另列LDA end、end−entry、全层输出轨迹及位置/长度对照。仅fit标签拟合，dev正常步骤95分位定阈值，evaluation标签只在所有分数冻结后读取。标准化后使用固定0.5收缩协方差，w=Σ⁻¹(μ_error−μ_normal)。Woodbury求解保留所有输入坐标，单元测试与显式原空间求解相等。输出轨迹为158维；位置对照只有步骤相对序号、步骤长度、prompt长度和步骤起始token位置。

监督阈值与无监督阈值的校准群体不同，不能宣称相同固定FPR下比较。报告实际FPR、首错精确定位、正常回答无报警、全局及同答内排序。高维训练AUROC=1不作泛化证据。高AUROC不表示全部span边界可恢复。

## 第二轮聚合消融

第一轮结果暴露后固定第二轮，保持原输出另存。`aggregation.py` 比较最高20% token均值的NLL、负margin、最后8层conditional NLL及JS。主候选把attention_base和trajectory_top20分别按fit混合ECDF映射，再等权平均。均不使用真假标签拟合，但研究设计受已知结果影响，属于探索性复验；不按测试挑参数、不替换默认检测器。

## 运行

已有research环境，不需安装包；capture需要CUDA，后续CPU。目录必须新建。先已有 `gsm8k_recurrence` 的完成结果。

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python -m experiments.gsm8k_states.capture --output outputs/gsm8k_states_new
python -m experiments.gsm8k_states.readout --output outputs/gsm8k_states_new
python -m experiments.gsm8k_states.evaluate --output outputs/gsm8k_states_new
python -m experiments.gsm8k_states.verify --output outputs/gsm8k_states_new
python -m experiments.gsm8k_states.aggregation --source outputs/gsm8k_states_new --output outputs/gsm8k_states_aggregation_new
python -m experiments.gsm8k_states.evaluate --output outputs/gsm8k_states_aggregation_new
python -m pytest experiments/gsm8k_states/test_states.py -q
```

`readout`同时生成明确标记的监督诊断，不属于无监督主候选。`audit_previous`只读原历史QA LDA冻结分数/阈值和旧条件模型案例，输出每个FN/FP原节点位置；历史存档路径当前在本机Trash目录，未移动或恢复。新旧节点索引不可直接混用。

结果位置：`outputs/gsm8k_states_20260929_v1`（1.6GB新状态及全量读出），`outputs/gsm8k_states_20260929_v2`（聚合消融），`outputs/supervised_location_audit_20260929`（历史监督审计）。每轮 `errors.csv` 包含全部评价FN/FP步骤全文；`steps.csv` 包含未知后续步骤分数；原始大文件不进Git。
