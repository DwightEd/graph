# 熵、reanchor 与内部写入的漏检审计

本轮优先回答“已知错误的原生内部信号到底有什么差异”，保留旧检测器，所有新候选独立输出。共享研究报告入口：`/share/home/tm902089733300000/a903202310/lys/codex/research/README.md`。

## 实现与实验单位

- `experiments/entropy_detection`：完整缓存的熵/惊讶度轨迹、轻量监督读出、中后层分布读出、自然同题样本迁移、拒绝信号融合。
- `experiments/internal_flow/audit_inputs.py`：8个原回答、9个官方错误片段、对应9个局部人工正确控制，以及两组自然同题局部配对。共24个claim/control视图，来自18个不同回答；不是24个独立问题。
- `audit_capture.py`：32层×32头；完整prompt保持。保存具体key attention、key对当前输出的局部投影、来源/具体证据/约束词/其他来源/指令/近16历史/远历史/标注段前缀的A·V·W_O写入，及残差、MLP读出。证据/约束mask是机制发现人工注释，不能当作自动检测输入。
- `audit_reanchors.py`：复用已有local-to-old-switch-v2的数学规则：当前cutoff、local window10、前三步基线、shift .1、排除特殊token、每头事件首点。测试逐项对照旧`measure_head`。完整attention与旧稀疏cache的覆盖不同，不声称历史节点逐个一致。
- `audit_decisions.py`：将官方错误段起点与“正确/错误候选首次分歧”分开，观察后者和后3token。零剂量、单头、整层多头、多层25%写入削弱及等范数随机对照。
- `audit_candidates.py`：7个明确候选的同前缀正确/错误读出，逐层穷举source/evidence/constraint/local_history/MLP的25%内部削弱；保存全部层结果。再在按首词效应选出的极值层测完整候选sum/mean logp，不把它解释为自由生成修复。
- `audit_prefix.py`：在语义分歧后前2token做精确前缀依赖干预，与前面2个token对照；没有后续claim token时明确跳过。
- `audit_precision.py`：四个主要漏检的冻结头选择，FP32 unembedding及0/.05/.1/.25剂量，检查微小正负效应的量化敏感性。

## 量的含义

对于位置t、层l、头h、来源集合S，写入为
`m(l,h,t,S) = W_O[h] sum_{j in S} A(l,h,t,j) V(l,h,j)`。
attention质量、写入范数、符号投影分别保存，不能互相代替。V已上下文化，来源key位置不等于纯来源事实。

局部投影为`m · grad_r log p_l(y_t | r)`，这里的梯度只经过该层最终归一化和输出矩阵，不经过后续层。原生效应为`log p(y_t) - log p_cut(y_t)`：正值只表示该写入支持实际词，不表示事实正确。多层同时削弱包含交互，不能相加解释为唯一因果路径。

同前缀候选margin为`log p(correct_first) - log p(wrong_first)`。WiFi的首词是but/and，因此额外保留完整候选评分；长度、措辞仍构成限制。MLP熵变化/残差可读出也不是模型已知真假的证明。

原回答是Llama3.1观察器对其他生成器文本的重放。自然两组配对才来自同一个Llama3.1生成器，且只审核了局部声明。完整prefill与原始逐token生成的BF16数值存在差异，原始生成熵与重放熵分列。

## 运行

研究环境Python：`/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python`。
从graph目录运行；所有输出必须指定新目录，不覆盖历史分数。

```bash
python main.py entropy-detect --packs outputs/probabilistic_detection_20260928_full/packs --output NEW_ENTROPY_RUN
python -m experiments.internal_flow.audit_run --help
python -m experiments.internal_flow.audit_decisions --help
python -m experiments.internal_flow.audit_reanchor_run --help
python -m experiments.internal_flow.audit_candidates --help
python -m experiments.internal_flow.audit_precision --help
python -m experiments.internal_flow.audit_report --help
```

CPU完整三任务预测先冻结再评价；已知8来源不参与本轮拟合和开发阈值。熵HGB及层读出是监督方法，融合还有dev标签选型。固定熵原始分数不训练，但开发正常阈值仍使用标签。

本轮结果不支持把熵、拒绝事件或reanchor存在性直接作为通用错误判据。熵轨迹只带来小幅匹配基线收益；层间pilot仍漏；拒绝融合补到部分错误但正常回答误报严重，全部保留为失败候选。新原生审计不是已经验证有效的最终检测器。
