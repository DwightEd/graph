# GSM8K步骤边界与多头状态重复

本模块是现有重复图方法的attention-only迁移试验。复用ProcessBench GSM8K的400份layer14×32头完整key attention，没有新模型前向，不冒称完整1024头JS/Jacobian方法。结果见`RESULTS_20260929.md`。

## 已找到的数据

从research目录起算：

- `demo/data/hf_datasets/ProcessBench/gsm8k.json`：400回答、375题、12生成器，193全正确、207存在首错。
- `demo/outputs/attention_traces/gsm8k_llama31_layer14/balanced/`：本轮使用的400份attention；完整key、token_ids、response_idx、step_ranges、生成器与observer provenance。
- `demo/outputs/attention_traces/gsm8k_llama31_layer14_nocap_hidden_hs15_matched_Llama-3.1-8B-Instruct/balanced/`：61份同名生成器、无长度截断，另存hidden layer15。不是额外61个独立回答。
- `demo/data/cct_traces/gsm8k_layer14/`：398份旧派生节点/超图特征，71维节点；不是完整attention，本轮未用。
- `demo/data/exact/processbench_observer_llama31_full`：失效软链接，目标`lys/data/ProcessBench/reasoning_error_detection/llama31_8b`当前不存在。
- `research/data/feature_extraction`近乎为空；上一级`lys/data/feature_extraction`主要是HaluEval/RAGTruth，实际可读GSM8K缓存在上述demo路径。

本轮400回答106872个response token的完整token IDs、原步骤文本、步骤token边界逐份复核一致。缓存声明上限2048，但本轮没有发现与完整原文不一致的截断。注意力是plain observer重放，不能当作原生成过程；生成器同名61答也不自动升级该证据等级。

[ProcessBench](https://arxiv.org/abs/2412.06559)标注首个错误步骤（0-based），−1表示全正确，不逐一标注首错之后各步；首错后的步骤在本实验为unknown，不能给它们伪造错误token标签。步骤边界预先给定，可用于分组而不读gold；边界不代表语义/因果独立，也不保证错误步骤内每个token都错。

## 固定方法

1. y_t预测query取response_idx−1+t；使用32个物理头的完整key分布，验证无未来attention、行质量近1后归一化。
2. source_only观测为负的平均head log(prompt attention mass)。source包含全部prompt，无实体标注。主base在fit问题等权线性回归中去除log prompt长度/log已生成长度趋势，再取fit混合分位；无标签拟合，不是事实概率。它只是attention可用的读取代理，不是原项目含OV等信号的强路由baseline。
3. 每head完整地址分布的Hellinger affinity，最后等头平均；保留逐head相似度。lag1..8分别用fit问题等权分位校准，边权低于.95置0；复用旧三跳max-min离线传播。
4. 主候选within_step仅允许相同步骤中的边。recurrence不限制步骤；distance_within使用相同lag/hops范围内的exp(−lag/32)；shuffled_within独立打乱各位置物理head身份，并重新用fit校准。
5. step风险为最高20% token风险均值；step_mean是无图base的全步均值对照。无标签dev问题等权step混合95分位、严格>。第一越阈值step为预测，无报警输出−1。不在评价集选参数。

375道题按problem文本SHA256排序，20% fit、20% dev、60% evaluation，同题全部回答不跨分区。旧demo已用过这些数据，新划分只保证本轮无拟合交叉，不能声称从未暴露的独立盲测。

## 运行

已有research环境、原缓存和本地Llama tokenizer；CPU即可。输出目录须新建，数据只读。

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python -m experiments.gsm8k_recurrence.measure --output outputs/gsm8k_recurrence_new
python -m experiments.gsm8k_recurrence.score --output outputs/gsm8k_recurrence_new
python -m experiments.gsm8k_recurrence.evaluate --output outputs/gsm8k_recurrence_new
python -m experiments.gsm8k_recurrence.verify --output outputs/gsm8k_recurrence_new
python -m pytest experiments/gsm8k_recurrence/test_contract.py -q
```

先冻结400回答所有分数再评价。AUROC/AP只包括首错步和可确认的正确步骤；首错定位准确率、全正确答无报警率及其调和均值另报。不是token AUROC。问题簇bootstrap300次的增量区间未做多重比较校正。`steps.csv`和`predictions.csv`保存全部evaluation步骤文本、首错与报警位置；`evaluation.json`含生成器分组和边模式统计。
