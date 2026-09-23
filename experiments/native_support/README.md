# 逐 token 原生检测

默认流程：原生前向或已有缓存 → 全部token的固定R/A/H观测 → 保存分数 → 一次评价。
不把消融、机制审计、状态分段或参考集拟合作为检测前置。
设计与公式见 [TOKEN_DETECTION](../../iclr/TOKEN_DETECTION.md)。

新的离线状态动力学设计见 [NATIVE_STATE_DYNAMICS](../../iclr/NATIVE_STATE_DYNAMICS.md)。
该设计允许后续 token，并处理来源不确定性和 FFN 方向传递。
`main.py dynamics` 已实现原生敏感度采集、无标签状态拟合、离线评分、AUROC/AP 及层头报告；尚无新的自然 AUROC。

已有独立 reference/test 输入时：

```bash
python -u main.py dynamics --stage run \
  --reference-output outputs/native_support_validation32/reference \
  --output outputs/native_support_validation32/test --resume
```

这次需要补采原缓存没有的多方向导数，会加载 LLM；每个 query 一次前向，默认 12 个读出方向，
按 `--gradient-batch 4` 分批反传。**不是**只改旧分数的免费计算，也不是全量数据重跑。
24GB GPU 的显存/时间尚未实测，实际峰值写入每答 `timing.json`；方向 batch 可减至 1。
采集完成后 `--stage score` 只用缓存和小状态模型，不再加载 LLM。
输出位于 test 的 `state_dynamics/`：`summary.json`、`evaluation.json`、`comparisons.csv`、
`tokens.csv`、`report.html`，逐头原始观测在 `capture/`，训练模型在 reference 的 `state_dynamics/model/`。

从官方 RAGTruth 准备独立新来源并一键运行（最后两项是 train/test 答案数）：

```bash
bash experiments/native_support/run_dynamics.sh \
  /path/to/RAGTruth/dataset /path/to/Meta-Llama-3.1-8B-Instruct \
  outputs/native_dynamics_v1 outputs/native_support_validation32/test 64 32
```

采样按 source ID 和 seed 排序，不用标签平衡；排除给定已审阅输出的来源。
下面的 R/A/H 表和 support 命令继续作为原始基线。

| 指标 | 测量 | 角色 |
|---|---|---|
| routing_imbalance | 投影消息范数的history-source预算差 | 固定主分数 |
| attention_displacement | 注意力读取的history-source质量差 | 独立读取指标 |
| entropy | 下一token输出分布熵 | 独立不确定性指标 |

缺少官方source区间时，前两项明确改名为prompt_*。
消息范数不是对正确答案的有符号支持；三项独立报告，不事后选择赢家或强制融合。
R使用原公式，32答已报告AUROC=0.711589；本轮简化不意味着已有新的性能增益。

## 已有32答：一条命令评分与评价

```bash
git pull --ff-only origin main
python -u main.py support --stage score \
  --output outputs/native_support_validation32/test
```

直接复用route_filter_v3/features中的标量缓存，不加载模型、不需要16答参考集。
若尚无标量缓存，只从已有原生NPZ提取一次，保存到新的token_detection/features。
新提取不计算head_profile，不运行滤波或图传播；原始缓存不改写。

## 结果

均在输出目录的 `token_detection/`：

| 文件 | 内容 |
|---|---|
| summary.json | 固定主方法、范围与AUROC/AP摘要 |
| evaluation.json | 全错误、首错、段起点、延续、前后半段；来源等权、同答与逐答结果 |
| tokens.csv | 每个token的三个观测、主risk、回答/source ID与预测位置 |
| responses/0000/scores.npz | 对齐的标量观测；risk与原R逐值相等 |
| report.html | 指标表与完整回答风险轨迹 |
| onsets.csv / high_risk_normals.csv | 可选查看的起点及高分正常token |

标签只在全部评分落盘后读取，默认使用已有annotations.json。
无标签也能打分，但没有AUROC/AP。无阈值时不将高分正常词称为已判定误报。
起点、首错、延续只是评价分组，检测时不访问它们。

## 首次采集与补评

```bash
python -u main.py support --stage run \
  --dataset /path/to/RAGTruth/dataset --model /path/to/observer \
  --task QA --split test --generator llama-2-7b-chat \
  --limit 32 --query-chunk-size 8 --output outputs/native_qa --resume

python -u main.py support --stage evaluate \
  --output outputs/native_support_validation32/test
```

run使用已有teaching原生因果分块前向，然后执行同一个逐token检测器。
--resume复用已采集token；默认不按标签平衡抽样，样本不是全数据集。
evaluate优先读取已完成的token_detection，不因目录中残留旧融合输出而改换方法。

## 历史复现入口

以下为显式实验，不参与默认run/score；历史输出与原缓存保留。

| stage | 历史方法与说明 |
|---|---|
| optimize | v3因果滤波，[ROUTE_FILTER_DESIGN](../../iclr/ROUTE_FILTER_DESIGN.md) |
| model | v4联合状态，[JOINT_STATE_DESIGN](../../iclr/JOINT_STATE_DESIGN.md) |
| readout / validate | v5去收缩与历史扩展协议，[STATE_READOUT_VALIDATION](../../iclr/STATE_READOUT_VALIDATION.md) |
| fuse | v6分位最大值候选，[DIRECT_RISK_FUSION](../../iclr/DIRECT_RISK_FUSION.md)，尚无自然增益证据 |
| compare | v2历史路由比较，[NATIVE_ROUTE_REDESIGN](../../iclr/NATIVE_ROUTE_REDESIGN.md) |

## 核心代码

- token_detection.py：读取缓存、固定读出、保存、调用统一评价。
- routes.py：沿用原路由公式；filter_features.py：原生缓存到标量观测。
- comparison_evaluation.py / evaluate.py：仅评分后读取自然标签。
- run.py：统一入口；teaching/state_audit：模型采集和存储。

本轮只做针对性软件验证，不要求重跑全量模型，也不将软件测试当作真实检测成绩。
