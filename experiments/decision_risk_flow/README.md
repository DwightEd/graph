# 原生 Fisher 消息响应拓扑：已知样本回归

实测见 [2026-09-28 结果](RESULTS_20260928.md)：9 段中 6 段有任意报警，
仅检出 19/134 个错误 token；加入拓扑后六个含错回答的 AUROC 均下降。
这版未通过检测门槛，保留失败结果，不替换原检测器。

当前实现直接计算 Llama-3.1-8B 在原始 prompt 和完整回答上的内部消息响应。
不使用人工 prompt 实体、证据或约束区间；不删除来源，不改写回答。
这是 **observer 重放 + 轻量监督读出**。RAGTruth 回答来自多个生成器，不能将
观察模型状态冒称原生成器状态。固定过去 K/V，仅重算当前 query 的下游传播。

## 实现

1. `native.py`：保存原始过去 K/V，独立重算每个预测位置；当前 self K/V 重算，
   禁止一个 query 读取批中其他 query 的重算状态。捕获全部 32 层 attention/MLP 写入。
2. `kernel.py`：在 post-softmax 消息上定义乘法门控，门控不重新归一化。
   用自动选择的实际 token 与最强非实际 token 的 logit 差量化有符号响应。
   另计算 `K = D.T (diag(p) - p p.T) D` 的 8 维随机投影，`D = dz/dg`。
   每个节点聚合相同 16-token 地址窗口、结构组、深度带的门控；头/层已池化，
   不是完整逐头图。组为 prompt、最近 16 位置、较远历史、特殊 token、MLP；
   两个深度带各 16 层。MLP 也进入响应核。
3. `geometry.py`：量化响应能量、相干能量、近似有效秩、组间方向及功能相似图。
   图边由响应向量余弦 > 0.5 定义，每组最多取 16 个最高能量节点。
   图无向；不是原计算图的有向路径。随机方向对照保持节点能量，不保持度分布。
4. `train.py`：36 个 fit 来源训练 5 折 rank-8 双线性 probe 与等参数量 MLP，
   24 个 dev 来源选择 epoch。每折分别约 14.75 万参数；两种 probe 合计
   1,475,020 个参数。回归样本及其同来源其他回答全部排除。
5. `evaluate.py`：用来源交叉拟合的 probe 分数和机制特征拟合 logistic 读出。
   仅 dev BCE 选择 C；各任务 dev 正常 token 的 95% 分位数固定报警阈值，
   同时报正常整答最大值阈值。全部预测冻结后才评价 8 答、9 个官方错误片段。

主候选 `new_method` 有 99 个输入：8 个置信度/位置量、10 个注意力量、
30 个有符号选择响应量、40 个 Fisher 核量、9 个图量、2 个 probe 分数。
同时保留 confidence、attention、native、kernel、topology、shuffled、probe、
probe_kernel 八个消融/对照。风险 probe 的消息梯度、历史形成路径与显式状态转移
**尚未实现**，不能将本 pilot 描述成完整风险双响应方案。

## 运行

在仓库根目录使用已有 `research` 环境。一键从新目录完整运行：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m experiments.decision_risk_flow.run \
  --phase all --output outputs/transport_topology_cases_reproduction
```

默认 12 fit 来源/任务、8 dev 来源/任务、8 个固定回归回答，总计 68 答。
不是全量测试入口。原模型及 source-first 原文缓存路径沿用当前环境；可用
`--model` 与 `--packs` 指定位置。准备阶段拒绝覆盖已存在目录。

可按阶段运行 `prepare → static → fit → responses → evaluate`；采集按完整回答
NPZ 续跑，评价拒绝覆盖已完成结果。不要在相同目录混用不同模型、精度或投影秩。
本次实际输出为 `outputs/transport_topology_cases_20260928`。

## 数值与解释边界

`precision.py` 保持冻结 BF16 权重不变，逐层临时转换权重做 FP32 前向和输入梯度，
适配 24 GB GPU。普通 probe 的状态缓存使用 FP16；消息响应计算保持 FP32。
原生完整前向与 query 重算必须满足 max-logit 误差 ≤ 0.005、状态相对误差 ≤ 0.0005。
`validation.py` 在无标签自动选择的消息上做 2% / 5% 双向有限扰动，核验导数。
初次 BF16 失败及其测量保留于输出的 `archive_bf16/`，不混入有效结果。

8 维投影秩不是模型真实秩，单边能量存在抽样误差；不据此宣布完整谱机制。
响应符号、attention 大小和图连接都不是事实真值。主读出使用自然幻觉标签监督，
且每任务仅 8 个 dev 来源，尾部误报校准只能作小样本探索。
这些样本历史上已暴露，来源排除并不能让它们成为独立最终测试。

`cases/index.html` 展示逐 token 报警与邻近正常误报；`evaluation.json` 保存实际指标，
`native_gate_validation.json` 与每答 `replay_errors` 保存数值证据。

一键运行最后调用 `verify.py`，直接从官方字符区间和冻结数组重新计算 72 组
逐例指标，复核来源隔离、token 覆盖和两种开发阈值；这是同一执行者的数值复核。
独立调用：`python -m experiments.decision_risk_flow.verify --output RUN`。
