# Pre-choice state and source-compatibility research

当前工作是出错前内部状态的机制诊断和来源程序监督pilot。旧4维图/EM方法仍是失败基线，原始RESULTS.md不覆盖；最新实测见[SOURCE_TRANSFER_RESULTS.md](SOURCE_TRANSFER_RESULTS.md)。自然token检测、错误类型及在线干预尚未验证。

## 观察时间与信息流

回答word t由位置P+t−1预测。prechoice_states_pilot只输入到该位置，保存r_before、attention_write、MLP_write完整4096向量和全prefix Q/K/V；r_next=r_before+attention_write+MLP_write。干预修改该预测位置的MLP写入，再由原生模型重算上层，读取真正的lm_head候选margin。不存在固定窗口风险广播。

native_prefix_pilot在历史t−2的head消息上加来源/随机方向，原生重算后继KV。该程序保存post-word marker的局部观测，candidate choice读取前一位置；不能将两个时点混称预警。物理响应是finite difference，尚非exact8B JVP/VJP。response_decomposition实现标准链规则的direct/transport/endpoint分解，exact JVP/VJP仅toy验证。

matrix_audit对原始8×1024self-head矩阵做完整Gram投影，只读当前行。它是监督方向保留率检验，不能解释生成机制或判断高rank就是错误。

## 唯一实验读出

prechoice_readout使用完整32×3×4096状态。每坐标μ/σ仅在程序fit来源计算，candidate是冻结lm_head行单位向量，不输入candidate做前向。g=b+Σα[l,s]〈d⊙z[l,s],e(c)〉/√4096，共4193可训练参数；另存786432 normalization统计。正g表示程序候选兼容性，尚非自然风险概率。

source_prechoice_pilot只读取已有role_controls的结构数据，交换相异bool路径，original/swapped×left/right共124prefix，候选True/False在同prefix互为正确/错误。23fit/8dev，source等权BCE+pair logistic+ridge，seed42、400fixedsteps，不选dev参数。它属于source-self-supervised/程序监督，不是严格纯无监督。自然标签、教师预测不进入训练。

程序满分和native排序同时满分；冻结迁移到confident-wrong旧12219仍排错。因此当前没有证明读出发现了decoder未采用的正确信息。α大小、探针梯度和raw lens符号不能代替因果定位。

## 可运行入口与依赖

在graph根目录，使用已有research环境（无需安装包或下载模型）：

```bash
export PYTHONPATH=.:teaching/state_audit/src
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python
$PY -m unittest experiments.flow_latent.provenance_joint_state.test_matrix_audit experiments.flow_latent.provenance_joint_state.test_response_decomposition experiments.flow_latent.provenance_joint_state.test_prechoice_contracts experiments.flow_latent.provenance_joint_state.test_prechoice_readout -v
$PY -m experiments.flow_latent.provenance_joint_state.prechoice_case_audit
$PY -m experiments.flow_latent.provenance_joint_state.source_prechoice_audit
```

两audit为CPU缓存复核，不启动LLM。13测试已实际通过；native margin独立复核误差≤1.91e−6，程序score独立复核≤1.754e−6。审查agent的默认Python缺依赖，不等于本research环境测试未跑。

采集/拟合入口：matrix_audit、native_prefix_pilot、native_branch_pilot、prechoice_marker_audit、prechoice_states_pilot、source_prechoice_pilot；冻结迁移入口prechoice_transfer_pilot。用同样`$PY -m experiments.flow_latent.provenance_joint_state.<module>`运行。原始输出已经存在，GPU/训练脚本会拒绝覆盖；复验应复制并记录新OUTPUT版本，原文件及protocol hash保留。source_prechoice_pilot需要outputs/full_head_message_design_20261007/role_controls.json；native3入口需要outputs/repetition_passage3_20261007/native_all_layers/inputs.json。矩阵/marker依赖旧prepared.pt及两seed自然监督self teacher，明确仅诊断。

完整最新设计与5轮审查：/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/source_transfer_20261008/FINAL_PROPOSAL.md。共享入口README.md、EXPERIMENT_TRACKER.md保留所有失败。下一实验先做native-discordant来源信息增量门，再验证同一读出的物理transport与fixed-budget pause；不重跑相同lookup题，不按旧2case挑层或标准化。

## 一手方法参考

[Tuned Lens](https://arxiv.org/html/2303.08112) §3对各层做affine译码和final-logits KL蒸馏以校正层基底；它预测模型分布，不提供事实真值。本项目raw lens尚未校正。

[LLM Internal States Reveal Hallucination Risk](https://arxiv.org/html/2407.03282) §3.2–3.4为query末token状态训练监督MLP，自动reference labels是response级；不能当无监督token预警。

[Pre-Generation Soft-Target Probing](https://arxiv.org/html/2606.21917) §3区分prompt采样风险和单个输出标签，使用多样本错误率监督。这里主接口为提出但未提交candidate的veto，分布hazard另报。

[Before the First Token](https://arxiv.org/pdf/2604.13068) §3–4报告其单层单位置均值差steering不能纠错；这说明probe可读性与该干预效果须分开，不能扩张成所有干预无效。
