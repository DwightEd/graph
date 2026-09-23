# 来源分解与最终选择：基于成熟归因研究的实现

2026-09-23。本轮按用户澄清，依据其他成熟研究重构，不把之前的标量混合、预算平滑包装成创新。

## 采用什么，未采用什么

| 研究 | 采用的具体思想 | 本实现的边界 |
|---|---|---|
| DecompX，ACL 2023 | 保留来源分解直到输出；点态线性化使非线性模块进入传播 | 原文是 encoder/分类模型；这里显式适配 bias-free SiLU decoder，不宣称逐字复现 |
| ALTI-Logit，ACL 2023 | 用目标输出及对比输出读出作用方向 | 采用实际 token 对每个原生替代候选的 logit 差，不把向量范数叫支持 |
| AttnLRP，ICML 2024 | 乘法两侧分配相关性，区分归因规则与普通梯度 | SwiGLU 的乘法两侧各分一半；没有宣称实现完整 AttnLRP 的 Q/K 相关性规则 |
| Information Flow Reveals When to Trust Language Models，ICML 2026 | 跨层路径整体决定来源贡献；读取地址与原始来源不同 | 保留全部值路径，不只乘一个标量注意力 rollout；不复现其相关性标注/校准器或照搬论文 AUROC |
| ReDeEP，ICLR 2025 | 外部上下文与 FFN 作用应分别观察 | 保存 FFN 的选择作用；不把所有 FFN 写入认定为参数知识或反证据，不引入其监督回归 |

研究假设：prompt 信息可以经历史位置中继，而直接历史占比不能区分这种中继与回答自身的延续。把来源追溯到输入根，并保留对选择的支持/抑制，可能改善这类混淆。**这是待验证的检测假设，不是已证实的新机制或新性能。**

## 同一原生前向下的明确分解

给定已完成回答，teacher forcing 的 query 为 `P+t-1`。一次完整因果前向取得原生量；目标 y 与竞争候选 c 的差为 `logit(y)-logit(c)`。各目标分别反向，不对整段 loss 求和后冒充逐 token 归因。

归因固定原生 attention pattern、RMS 分母，并采用以下 SwiGLU 规则。令 `g=W_gate x`、`u=W_up x`、`theta=sigmoid(g)`，则原生写入为：

```
FFN(x) = W_down [(theta * g) * u]
```

在该原生点采用的线性算子为：

```
T_FFN = 1/2 W_down [diag(theta * u) W_gate + diag(theta * g) W_up]
T_FFN x = FFN(x)                       # 无 bias 的精确算术下
```

注意力和 RMS 同样构成固定点下的线性算子，残差包含恒等路径。这些算子保持 native forward 数值，但不等于原函数的 Jacobian：未归因 Q/K 变化与 RMS 尺度变化，乘法使用指定的相关性分配。

实现使用反向伴随避免显式保存“来源数 × 所有节点 × hidden”的向量场：

```
w_c^T T_L ... T_1 e_j = e_j^T T_1^T ... T_L^T w_c
```

这是同一线性分解的输出读出，没有训练压缩、SVD 或随机投影。历史位置的 value 路径保持连接，不 detach 历史 KV；输入根的位置、来源块和候选身份均保留。

对 bias-free 支持布局，逐输入 token 的作用之和应还原原生 logit 差。`ledger_error` 保存误差，尤其要看 bfloat16 下的数值误差。本地 float32 小 Llama 的该账本已核验；真实 8B 的精度、速度和显存尚未实测。检查账本不要求删除消息或运行反事实。

teacher forcing 的离散 token 仍是条件输入；输入归因不能穿过采样动作，因此不能把历史 token 的根贡献等同于原生成时的全部语义来源。

## 不在传播中压成少数标量

底层传播在原始 hidden 维度完成。写盘时保留：

| 数组 | 轴 | 含义 |
|---|---|---|
| `root_token_choice` | 可见输入位置 × 候选 | 输入根对每个选择差的有符号作用 |
| `root_positive/negative` | 来源块及 history/other/special × 候选 | 先在 token 上分正负，再分组；块内抵消不会丢掉 |
| `head_positive/negative` | 层 × 头 × 来源组 × 候选 | 当前 query 各实际消息端点的作用；层是不同观察切面，不能跨层累加成总贡献 |
| `ffn_choice` | 层 × 候选 | 当前 query FFN 写入对选择的作用；负值不是“反事实证据”标签 |
| `attention/edge_value_energy` | 层 × 头 × 可见 key | 与历史原始路由公式对照 |
| `candidate_ids/logits/tail_mass` | 原生候选 | 保留候选身份与未覆盖的词表质量 |

root 组是原始输入来源；head 组是该层实际读取的位置来源，两者不同。不能混用，也不能把多个层的终端作用相加。FFN 作用已包含在根贡献的传播中，不再作为额外风险加一次。

自动来源块只是分组，不是人工证据类型，也不自动知道实体、范围或否定是否适用。

## 固定检测候选与对照

对每个候选 c，P/N 是原始输入 token 正/负作用分组后的量，S 是来源，H 是历史。定义：

```
r_c = [(P_H-N_H) - sum_S(P_S-N_S)] / sum_all_groups(P+N)
root_route = sum_c softmax(native_alternative_logits)_c * r_c
```

候选权重只使用当前 query 原生替代候选的分布，不根据标注选方向。零分母用 EPS 保证数值定义，同时输出 `root_mass`；它不代表已确认正常。

该分数是固定的无标签检测候选，不是语义真假后验；强力支持一个错误选择仍可能低风险。这一局限不能靠改名或额外平滑解决。

同样本独立评价：

- `root_route`：跨层、跨位置值路径的原始来源读出。
- `direct_choice_route`：各层当前 query 直接消息的有符号读出，对照“读到了什么”和“来源最终起什么作用”。
- `raw_route` / `raw_attention` / `entropy`：历史 R/A/H，公式不改。
- `root_route_mean` / `route_offline_mean`：同一固定 16-token 离线窗口，允许使用未来；仅作为时间保持对照，不冒称隐状态真假模型。

来源份额用概率向量保存；相邻原始来源分布用 Hellinger 距离观察变化。它只提示来源作用变化，不自动宣称 reanchor。首错与延续在评价后分组，不能给评分提供金标边界。

## 运行、成本和输出

在已有 `settings.json` 的小样本目录中运行，不准备新数据，不扩到全量 RAGTruth：

```bash
git pull --ff-only origin main
python -u main.py transport --stage run --output outputs/native_support_ragtruth4 --resume
```

新结果位于 `value_transport/`，旧 `source_transport/`、`state_dynamics/` 和原始结果均保留。
旧 detached-KV 缓存没有输入根归因，不能伪造转换。首次需要新的采集；之后只改读出时：

```bash
python -u main.py transport --stage score --output outputs/native_support_ragtruth4
python -u main.py transport --stage evaluate --output outputs/native_support_ragtruth4
```

每答一次完整前向，多个独立候选反向；SDPA 与 FFN checkpoint 避免保存各层完整注意力和所有宽 FFN 中间量。默认 `--gradient-batch 1` 控制反向峰值；可明确增大来批处理，但不保证更快。当前没有 24GB 实测保证，`capture/*/timing.json` 记录时间和 CUDA 峰值。

`summary.json` / `evaluation.json` / `comparisons.csv` 包含 AUROC/AP 及首错、延续、前后半对照；`tokens.csv` 包含全部 token 诊断；`source_choices.csv` 包含来源对具体候选的支持/抑制；`high_risk_normals.csv` 和 `onsets.csv` 用于排名审计。

旧预算输出的只读审计命令保持不变：`main.py transport --stage audit --output ...`，明确只审核旧 `source_transport/`，不会重新采集或改分数。

## 清理范围与验证

删除 v3–v6 的标量滤波/切换/融合实现及其入口，删除旧共享预算图评分实现；共用特征读取和指标比较改为直接命名。保留源隔离的数据准备、原始 R/A/H、旧结果审计及独立 dynamics 实验。旧实现可在 Git `9d4ba17` 查阅，未删除用户模型、缓存或研究结果。

软件核验针对 native forward、归因账本、未来位置掩码、候选 ID、分组前正负分解、VJP batch/续跑、资源恢复，以及标签只在评价读取。没有自然数据新 AUROC，没有因软件测试就宣称检测有效。

## 原始文献

- DecompX: https://aclanthology.org/2023.acl-long.149/ （尤其 §3.3 的点态分解）
- ALTI-Logit: https://aclanthology.org/2023.acl-long.301/
- AttnLRP: https://proceedings.mlr.press/v235/achtibat24a.html
- Information Flow / 作者实现: https://github.com/rxu0112/RAG-information-flow
- ReDeEP / 作者实现: https://github.com/Jeryi-Sun/ReDEeP-ICLR
