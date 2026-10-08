# 来源程序读出：实际训练与自然迁移验证

2026-10-08。本轮完成18次GPU来源自监督拟合，另有2次四来源最小拟合诊断；复用已有内部状态，新增大模型前向为0。自然幻觉标签没有用于拟合、选模型或设定新模型阈值。训练来源671个，留出来源168个。

目前不能把“候选受到来源影响”或“字面来源程序兼容”直接当作自然幻觉判据。读入候选后的内部状态能区分来源程序的首次不兼容，但其自然迁移和误报控制仍未通过。

## 来源留出结果

| 实验 | best dev AUROC | 结论 |
|---|---:|---|
| 首次候选输入前，BCE，三seed | 0.518911 ± 0.013344 | 未通过 |
| 首次候选输入前，BCE＋pair，三seed | 0.520765 ± 0.011810 | 未通过 |
| 逐来源地址，dot，seed42 | 0.513366 | 末来源节点读取偏好；未通过 |
| 逐来源地址，cosine×8，seed42 | 0.527645 | 限制幅度仍未通过 |
| 匹配四条件，候选输入前，三seed | 0.519177 ± 0.015691 | 未通过 |
| 匹配四条件，候选输入后，三seed | 0.996755 ± 0.000368 | 程序门3/3通过；不是自然检测分数 |

三seed为42、123、2026，标准差采用样本标准差。每seed运行8epoch，按来源程序dev AUROC选checkpoint；保留best与last。地址两配置各有group_summed/address/payload_rewired三个分别训练的同参数控制。跨地址和首次读出不能宣称只改架构：模型容量和batch安排也不同。

dot地址模型所有dev地址峰值都在末来源token；cosine重新训练后仍有98.51%。这个节点可能已经整合上下文，因此不以此证明未读取语义。固定K、反转V的读出扰动对dot模型训练分数有作用，对dev没有稳定区分增量。它不是原生大模型因果干预。

代码中的地址是辅助读出的learned Q/K softmax，而不是按原生attention硬取最大历史token，也不是FlashTrace的`A V W_O`消息及L1 proximity。候选向量只在最后cosine读出参与，不参与地址权重；这不妨碍统计分类，但不能把该权重解释成指定输出的贡献。旧route已经用`A ||W_O V||`，仍丢失有符号输出方向。项目已有`token_backtrace.messages.sparse_edge_vjp`及`finite_edge_effect`，本轮地址读出未使用这两个原生输出作用算子。

候选输入后的seed42，同一候选在来源交换后的风险方向准确率为334/336；未见候选ID的AUROC为0.998316。候选自回音不能单独解释这个程序结果。配对差分置零后程序AUROC仍为0.996129，原生通道仍含来源，不能把它称为删除来源。

## 完整自然QA结果

900回答、150来源、124817有效token、6751错误token；740正常回答。主分数固定为三个source-selected模型在同一token上的平均，没有标点分组或幻觉span平均。

| 方法 | AUROC | AP |
|---|---:|---:|
| 旧来源＋路由＋native Huber | 0.779725 | 0.172794 |
| 新无标签条件路由＋来源＋同Huber | 0.731257 | 0.128977 |
| 新首次读出，候选输入前，三seed平均 | 0.529653 | 0.056327 |
| 新首次读出，候选输入后，三seed平均 | 0.677372 | 0.081851 |
| 候选输入后，seed42单模型（辅助） | 0.693878 | 0.086860 |
| 三seed候选输入后，去掉显式local消息（诊断） | 0.678225 | 0.082137 |

新post主分数减旧方法的来源bootstrap AUROC增量95%区间为[-0.127646,-0.069702]。因此不替换旧方法，也不从单seed辅助分数中重挑主方法。

来源fit的2684个first接收点仅32个有local邻边，而自然test有117617/124817（94.23%）接收点含8条local边。针对这一输入差异，冻结全部模型、只把两条件的local_attention归零，重新评分全部900答，AUROC仅提高0.000853。因此显式local图分支不是当前迁移失败的主要原因；该控制保留了已经包含历史信息的节点和边界，并未删除历史或证明完整传播机制。它是诊断，未被选作新主方法。

新post主分数零阈值下正常token误报率40.04%，740个正常回答全部报警。12219检出11/13错误token但误报120个正常token；12297检出8/49但误报27个。不能把较高召回解释为已解决这些样例。程序正确样本95%阈值迁移后误报更高，没有进行自然标签阈值调整。

逐token例子：12219仍漏位置227的` folding`、228的` a`；正常token的高误报包括位置14 `Pass`、141 `4`、13换行标题分隔。12297漏掉` Burn`、` flag`、` proper`、` ceremony`等被标注的引用内容，正常开头`Based`和编号`1`却报警。12471仍漏位置29/80的` iPhone`、87的` iTunes`。详细风险与官方标签在`natural_test/case_<id>.csv`，这些例子不能用统一升高来源读取量或调整零阈值来证明解决。

严格clean-prefix首错AUROC为0.749167；零阈值检出122/160首错，但此前没有正常token误报的仅3/160。候选已进入观察模型，不能称提前预防。旧R/A/H的query为P+t−1，也是在预测当前token时测量；新reader只使用L15，旧信号覆盖32层，不能把二者差距全部解释成时序。

## 重跑入口

在graph目录设置`PYTHONPATH=.:teaching/state_audit/src`、`OMP_NUM_THREADS=4`、`OPENBLAS_NUM_THREADS=4`，使用当前research Python环境。下面入口需要既有缓存，输出默认不覆盖完成目录；重跑请指定新的`--output`，或对已有输出只执行评价。

```bash
python -m experiments.flow_latent.local_graph_transport.run_first_choice fit --seeds 42 123 2026
python -m experiments.flow_latent.local_graph_transport.run_first_address fit --address-energy cosine --address-temperature 8 --run-name bounded_seed42
python -m experiments.flow_latent.local_graph_transport.run_first_timing fit --seed 42
python -m experiments.flow_latent.local_graph_transport.run_first_transfer evaluate
python -m experiments.flow_latent.local_graph_transport.run_conditional_source evaluate
python -m experiments.flow_latent.local_graph_transport.run_first_graph_control evaluate
```

新fit前需用对应`prepare`入口生成compact缓存。首次选择`--output`含cache，时序自然迁移`--fits`指向三seed checkpoint目录。完整自然评分是`run_first_transfer score`，评分/阈值先冻结，再`evaluate`读取官方标注；`all`顺序执行这两步。

原始产物：`outputs/first_choice_validation_20261008`、`outputs/first_address_validation_20261008`、`outputs/first_timing_validation_20261008/{natural_test,cache,seed_*}`、`outputs/conditional_route_validation_20261008`。逐token旧例CSV保存在两个自然评分目录。

相关CPU科学/软件检查115项通过。来源与自然冻结分数、首错时序、逐token报警计数及旧例由其他代理独立复算；这不构成跨模型家族的外审或完整复现保证。

Fresh experiment-audit为same-family provisional WARN：原始分数、标签时序、数字及冻结哈希复核通过；没有自然检测成功结论。条件CDF的既有FREEZE中“post-token”只描述local attention，复用的旧route/attention标量是原始预测位置观测，因此整体是混合时序。训练pack包含标签数组，但拟合、选型和阈值路径未使用。32-query native比较已从原始roster、native分数与地址分数独立重建，原combined JSON的后补合并脚本未归档。历史产物不追改。

这是历史测试已暴露条件下的探索性复验，内部状态来自Llama观察模型对六个生成器回答的重放。本轮只验证QA，没有新增GSM8K、Summary或Data2txt结果。来源程序读出是source-self-supervised，不称严格无监督；条件CDF分支不使用标签拟合或选型。没有宣称完整因果图、自然事实支持证明或整体方法已有效。
