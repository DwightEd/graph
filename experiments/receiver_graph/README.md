# 实际消息边、关键节点与连续风险聚合

本轮完成48答/6463token全1024头缓存测量、三轮小样本评分及12答288个真实8B有限干预测量点。没有新全测试，没有更换默认方法。三轮均未胜旧双起点的63TP/70FP。见[详细结果](RESULTS_20260929.md)。

## 图、节点与量化对象

预测回答token y_t的query绝对位置为P−1+t，历史y_j的key为P+j，j<t。这个key已经处理y_j，不能误当预测y_j前的状态。逐层逐头消息为m=A W_O V；缓存g为当前原词相对固定最强竞争词margin对消息乘法门的导数，包含当前query的原生下游响应，固定过去KV。

每头读用联合强度B=sqrt(A*abs(g)/sum_key abs(g))。同receiver/head下按lag {1},{2},{3,4},…与是否复制当前输出token分组，算端点条件均值B0。主图为mean_head max(B−B0,0)。attention原图、B0匹配图为对照。全部原始物理头/key的A/g仍在原缓存，新增head_readouts保留32×32×T×6，不作低秩投影；最终检测图仍有等头归约，不能称检测器完全没有压缩。lag1/2的单例组超额为零，这是此对照不可辨别的范围，不是消息无用。

四类候选独立于风险阈值：当前query的来源读取、来源abs门作用、来源地址分布相邻JS，以及历史key被未来query使用的累计图权重。JS在此为地址更新测量，未直接混进检测分数。carrier是历史key状态，其他三类描述预测当前token的query；均不是已验证的语义reanchor。source_use的分母只含source+answer-history，排除instruction；top8头均值只用于易读的审计表，完整头轴另存。

旧重复图的路径是“状态相似边”；新图的边是“实际被读取且对当前margin有局部作用的历史消息”。即使一组边j→t、t→u都存在，也未验证前一扰动通过后一个key影响u。严格跨token路径要在早期节点干预并重建后续KV再测量；本轮没有把局部边相乘冒称完整因果路径。

## 检测器

两通道为旧固定来源风险logit与−margin。按每任务4个无标签fit来源等权标准化，拟合相邻相关rho（截到[0,.99]），C_ij=rho^abs(i−j)。固定离线W=.5I+.5row_normalize(G+Gᵀ)，孤立点退回I；分别算Z=(Wz)/sqrt(diag(W C Wᵀ))，再取双通道max。不要求任何token先越过风险阈值。

C是AR1近似，W也是数据依赖的，因此这个统计量没有证明服从正态，更不是幻觉概率。对照rho=0用于观察重复证据被当独立的影响。无标签fit混有错误，dev95也不保证正常FPR=5%。原完整参考基线的阈值保持，其余方法用4个dev来源等权混合95分位、严格>。

v2分别用fit混合ECDF校准两通道再max，出现QA阈值=1导致零报警。v3只修复有限参考尾部饱和：使用等来源logistic核混合的−log survival，带宽固定fit标准差、logsumexp计算，不搜索带宽。三轮预测先冻结再评价，设计已参考暴露案例，不能声称盲测。没有自然标签拟合、人工prompt实体标注或按错例选头。

## 代码和复现

- `measure.py`：实际消息图、逐头地址JS/有符号作用、无标签选边。
- `score.py`：连续风险、相关校正、图对照和阈值。
- `balance.py`：v2/v3通道尾部校准，不重跑模型。
- `finite.py`：同头选中消息、匹配消息、二者联合，0/.01/.25/1剂量删边；不重新归一化A。
- `audit.py`：节点候选、所有FN/FP、全部旧71个漏检、物理轴和匹配质量核验。

需要现有完整缓存和本地8B模型，输出目录必须为新目录；不能重复覆盖已完成结果。

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python -m experiments.receiver_graph.measure --output outputs/receiver_graph_new_v1
python -m experiments.receiver_graph.score --output outputs/receiver_graph_new_v1
python -m experiments.receiver_graph.finite --output outputs/receiver_graph_new_v1
python -m experiments.message_js.evaluate --output outputs/receiver_graph_new_v1
python -m experiments.context_response.verify_restore --output outputs/receiver_graph_new_v1
python -m experiments.head_state_readout.audit --output outputs/receiver_graph_new_v1
python -m experiments.receiver_graph.audit --output outputs/receiver_graph_new_v1
python -m experiments.receiver_graph.balance --first outputs/receiver_graph_new_v1 --output outputs/receiver_graph_new_v2
python -m experiments.receiver_graph.balance --first outputs/receiver_graph_new_v2 --output outputs/receiver_graph_new_v3 --smooth
# 对v2/v3分别运行上述evaluate、verify_restore、head_state_readout.audit、receiver_graph.audit。
python -m pytest experiments/receiver_graph/test_graph.py -q
```

本次实际原始结果：`outputs/receiver_graph_20260929_v1`–`v3`；每版`TOKEN_AUDIT.html`和`all_errors.csv`列出逐token状态。v1另有`finite_effects.json`、`finite_summary.json`、`node_candidates.json`、`PREVIOUS_MISSES.md`。自然轨迹缺旧来源分数，保持融合不可用；仅rejection分支可评价31个局部已标注token，其余自然token真值未知。

文献定位：attention图需要区分读取与影响，参见[Quantifying Attention Flow in Transformers](https://aclanthology.org/2020.acl-main.385/)；一阶归因需有限干预核验，参见[AtP*](https://arxiv.org/abs/2403.00745)。本轮是相关思想下的自定义测量，不宣称复现这两篇方法。旧Dynamics启发的几何算子继续作为测量工具，不能给本轮token图提供现成的事实传播定理。
