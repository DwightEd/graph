# 来源读出诊断：完整文档顺序挑战（2026-10-09）

**目标未达：完整QA DEV1008回答/147981个token的无监督预定主方法AUROC .773897757。** 本轮新增的来源顺序挑战只在四来源八历史回答/1139个token验证，primary .568359136，方向资格失败；没有进入默认检测、融合或图，也未扩大三任务测试。它提供具体反证和逐词诊断，不是一套已有效的新检测器。

## 当前主线为什么误报和漏检

主线仍为逐token来源作用、路由读出→固定fit来源等权参考CDF→unary融合→历史邻接图。不是attention最大历史词当作证据。旧来源是完整prompt passages模板区域，输出作用是该来源对实际词生成的帮助，路由读出是消息幅度；它们均不直接验证主体、关系、适用范围。

完整QA DEV旧阈值的9987漏检中95.90%在图之前unary已低；5414误报中99.96%在unary已高。4806个错误词两来源作用均≤0，来源实际助推错误词且全部漏检。旧图移除2470正常误报也损409检出，其中375在长错误span。图可以降噪/传递，不能用不存在的事实方向造出错误起点。

本轮此前的固定16bin likelihood条件参考，不用自然标签fit，把来源作用条件于同视图native actual logp，保持route/融合/图不改：完整DEV AUROC .772523119→.773897757，AP .218084651→.234860398；AUC增量来源CI跨0。相同5414FP检出2136→2421，但各自fit95阈值首错46→13。full-only .781478564只是预设对照，未事后提升为主方法。精确lag8状态图/完整32head640坐标gate在同八答.414080低于旧.430271，状态持续推断也没有修好事实方向。

## 新代码怎样测量

`experiments/flow_latent/local_graph_transport/` 新增四模块：

- `source_order.py`：`document_blocks/compile_orders`完整移动三个原passage块，包括原ID/header/正文/尾双换行。两个循环分别2/3/1、3/1/2；问题、指令与chat包装不改。token映射双射、native完整retokenize一致、总P及prefix/suffix不变，否则拒绝。没有人工实体标注或attention排名选择。
- `source_order.py:complete_readout/logit_readout`：每条件原生完整32层fresh prefill、causal teacher-forcing，读P+t−1预测行；固定原回答前缀和实际y，full-vocabulary log_softmax。native最高非实际rival只在原条件选一次，循环复用，不把竞争词当作正确修复。
- `run_source_order.py`：固定全部八答，先做所有native/identity重复和旧输出再现，全部通过才执行两循环。保存全部target、源码/模型字节、输入身份、原始量及完整forward ledger，冻结后结束；不导入自然标签评价或拟合风险，fresh目录拒绝覆盖。
- `evaluate_source_order.py`：先验证冻结/依赖/identity，再从官方RAGTruth字符span对齐逐token标签。报告全部固定字段的AUC/AP、答内/首错、正常答报警、完整CSV和八答图；预算匹配报警明确为label-assisted oracle，不能当部署阈值。
- `test_source_order.py`：10个科学检查包括合并BPE分隔符、文档ID/bijection、固定竞争词、完整softmax、因果预测行及小Llama无未来读入。

每词保留三个条件的actual_logp及margin，两个循环各自有符号差。预冻结primary是

`margin_range = max(m_native,m_fwd,m_rev) - min(m_native,m_fwd,m_rev)`。

secondary为actual_logp_range、两循环均值减native的signed margin/logp。范围只作为顺序敏感性待验证，不转换事实概率、不翻方向、不平均span、不加入熵错误unary。新通道是输出响应；本轮没有采全多头新张量，不能称已经确定了具体信息路由或原生成器因果图。

## 实际结果与反例

| 同四来源八答 | token AUROC | AP | 同57正常token FP检出/77 | 正常答报警/5 |
|---|---:|---:|---:|---:|
| 旧native | .430271235 | .079948040 | 7 | 3 |
| primary margin_range | .568359136 | .084267032 | 5 | 4 |
| secondary actual_logp_range | .510883655 | .080650504 | 5 | 5 |

这是32次完整模型forward、296投影、0 backward的真实新采集，覆盖全部1139targets，官方77错误。所有原输出复现最大误差小于1e−4，identity重复误差为0，目标/rival/预测位置逐元素相同。真实GPU进程62.305秒，峰值16.947GiB，10CPU检查事前通过；CPU评价和绘图均exit0。

固定误报预算下，错误引用`12297:106/2`从漏检变检出，`15604:96/private`首错也上升；但`12219:223/not`由检出变漏检，养老金续词`15604:103/normal`仍漏。总体新增4TP损失6TP，56新FP替换56旧FP，净TP下降2。 29个native NLL<.01的错误续写全部漏检，57FP中也有30个同桶高置信正常词。误报35个content/subword、10个功能词代理、8个标点、4个含数字词，均为事后描述性分层，不用于删词或调分。没有attention来源选择，不能把这些误报归因于最大attention落到特殊token。新首错1/3但无先前误报的首错0/3。secondary首错AUC .8166只来自三个首错，不能称全token达到.8。

**机制限制**：重排同时改变RoPE位置和来源先前hidden/KV；正常跨材料整合也会敏感，错误local续写也能稳定。生成帮助不等于事实适用性，顺序稳定也不等于正确。不同generator的正负配对和固定不同前缀有混杂，Llama3.1为事后observer。需要新的事实关系方向资格证据；当前不应把这个不合格量塞入图以增加模块。

[Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/)提示正常位置敏感反例；[Stable-RAG](https://aclanthology.org/2026.acl-long.1188/)已有多顺序隐藏状态聚类/解码。本轮非其缓解复现，不把重排宣称新颖性。完整机制计划、独立设计审查和实际执行见共享`codex/research/refine-logs/source_order_equivariance_20261009/`。

## 运行与查看

从graph目录，用已有research Python；无需安装或改模型/环境。采集只接受新的输出目录，完成目录不要重跑覆盖：

```bash
PYTHONPATH=.:teaching/state_audit/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.flow_latent.local_graph_transport.run_source_order --stage capture --output outputs/source_order_equivariance_fresh
```

评价器需要capture FREEZE及独立EVALUATOR_AUX/ATTACHMENT绑定；当前唯一已验证实例是本次目录，不假称任意fresh目录可自动一键科学评价。本次已完成结果可直接查看：

- `outputs/source_order_equivariance_20261009/ORDER_DIAGNOSTICS.json`：全队列及每答指标。
- `ORDER_TOKENS.csv`：全部1139token的官方标签、native/两循环原量和预算诊断。
- `ORDER_KEY_TOKENS.json`：预留错词和正常fold/corner等反例。
- `ORDER_TOKEN_PANELS.png` / `.pdf`：全部八答位置曲线/官方错误span。

事后仅修评价器守卫接口，原AUX与脚本保留，四科学函数AST不变、原capture FREEZE不改；显式附件说明采集后修复，不追认预采集执行认证。完整逐词诊断、fresh独立审计与scope限制在共享目录；最新审计状态由EXPERIMENT_AUDIT.md记录。历史DEV/pilot已暴露，本轮不是盲验证或顶会有效性通过。

独立审计已DONE/WARN：全部9112派生数值、1291报告数值、1139CSV行/官方标签/报警复算一致；历史1139完整词表向量独立证实native rival及归一化。限制为历史小pilot、observer/generator混杂、cycle完整向量未留、evaluator事后守卫绑定和图事后回读；same-family/provisional，不能称顶会方案评估通过。
