# 原生路由检测与机制审计

当前检测入口：[Native Route Comparison](experiments/native_support/README.md)。
恢复历史约 0.7 的功能消息路由与 attention 路由基线，同缓存比较来源、头、时间的结构；
保留 v1 的 signed support/history 分数作为失败对照，不再默认它是更好的检测方法。
读取量、消息范数、候选方向写入分开保存；不消融、不反向传播、不用标签选择分数或方向。
[重新设计依据及历史结果](iclr/NATIVE_ROUTE_REDESIGN.md)。

```bash
python -u main.py support --stage compare --output outputs/native_support_ragtruth4
```

以上命令只读取已有采集缓存，在 `route_comparison_v2/` 写入报告及 AUROC/AP；不运行模型。
同时报告 pooled、来源等权、同一回答内的比较，避免不同口径互相替代。
历史 routing imbalance 在 971 个来源上的 pooled AUROC 为 0.721235；
用户四回答结果中 v1 support graph 为 0.572333。不同样本和模板不能直接作性能回退定量比较，
但此前没有保留强基线就更换主方法不合理。本轮代码恢复比较条件，没有宣称新自然成绩。

需要真实 AUROC/AP 时，使用少量官方完整回答，自动准备标注并在评分后评价：

```bash
python -u main.py support \
  --dataset /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset \
  --limit 4 --balanced --output outputs/native_support_ragtruth4 --resume
```

此处按标签分层选 2 个错误回答和 2 个无错误标注回答，仅为小样本诊断，标签不参与评分。
输出目录自动生成 `annotations.json`；评价位于 `route_comparison_v2/evaluation.json`，
不需要 `token_labels.json` 占位文件。`--stage score` 与 `compare` 都可从旧缓存重算。
不带 dataset/input 的 run 仍使用旧 322-token 自然前缀；其标签未知，不能用于首错成绩。

教学与独立复用入口：[State Audit](teaching/state_audit/README.md)。
短错误检验：[短 span 审计](experiments/short_span_audit/README.md)，分别提供已有冻结分数的 CPU 评价和复用 teaching 的逐目标贡献采集。
可单独安装，包含生成回答、准确 token 回放、逐层状态保存、离线审计和四世界消息干预；
支持 Llama、Mistral、Qwen2，并提供无需下载模型的示例。

两个项目的既有发现见[证据台账](docs/DETECTION_CONVERGENCE_20260919.md)。
既有审计方案见[正负配对机制审计](docs/PAIRED_MECHANISM_20260920.md)：明确适用性、
多头条件作用、下游补偿和时间持续性，再在同题重采样中逐项检验。
文献包括前轮 [18 项核查](docs/LITERATURE_TRANSPORT_20260919.md) 及本轮 GoS、Saliency、RAUQ 方法复核。
内部逐头模式可被监督读出；尚无已验证的通用无监督绑定检测器。

历史配对机制审计入口（包含干预，不属于上面的检测流程）：

```bash
python -u main.py flow
```

默认写入 `outputs/paired_head_transport_v1`，重复命令复用逐 world 缓存。
读取 reanchor 既有样本，以两侧 onset 的完整候选目标选出共同物理 head，
在子句前、起点、后半段、子句后测量单删、四世界条件作用、下游恢复和起点影响传递。
self 与更早历史分开；原值写回使用各候选分支自己的基线；另存严格只读前缀的置信度对照。
目前只核验了两个局部事实对，属于标签辅助机制审计，尚无新协议 8B 结果或检测成绩。
上传 v3 结果的实际重算见 [60 次单删与 24 组双删](results/paired_audit_20260920/README.md)。

`--stage inventory` 仅核对配对，`--stage screen` 只读原始 NPZ，`--stage report` 只重建报告。
原目标消息边实验改用 `python main.py flow-edges`，新运行写入 `outputs/target_transport_v4`；
只查看旧 v3 结果用 `flow-edges --stage report --output <原v3目录>`。
原 v2 局部 lens 实验用 `python -m experiments.path_conflict.flow` 显式运行。
`python main.py`显示入口；历史无标签基线需显式使用`unsupervised`。
现有无标签HMM保留为竞争对照：`python main.py regime --phase all --resume`。
本轮未取得其自然成绩，不把较少占用的状态直接称为已发现的幻觉状态。

旧grounding当前步回归存在互补质量恒等式，已改为过去路由预测基线。
`python -u main.py population --phase grounding --resume`写新目录，
不会覆盖旧成绩，也不再随population all自动运行。

## 历史基线：entropy seeds → local token reuse

显式 `main.py unsupervised`：无标签参考熵 → 疑似入口 → 实际局部 attention 多跳复用。
不训练入口/延续分类器，不加载 S10/S11 监督权重，不用正常样本标签校准。
标签只在全部预测冻结之后用于评价。**尚无新自然 RAGTruth 检测成绩。**

## 一键运行

在 graph 根目录及 research 环境：

```bash
python -m pytest tests/test_unsupervised_reuse.py -q
python main.py unsupervised \
  --population ../reanchor/outputs/ragtruth_population_20260912 \
  --output outputs/unsupervised_local_reuse_v1 \
  --tasks all --generators llama-2-7b-chat \
  --device cuda:0 --query-chunk 16 --window 16 --resume
```

首轮每答一次 teacher-forced backbone 前向，按层分块重建 response-query attention，
保留全部物理 layer/head 的局部端点。模型与数据路径从 population/settings.json 读取。
之后复用逐样本 NPZ。完整上下文不裁剪；局部窗口仅限制被保存的历史边，不重归一化。
这是 observer 回放与 attention 依赖代理，不是原生成器的 WV/WO 因果消息追踪。

阶段：`--phase capture` 仅采集；`--phase score` 无标签评分；`--phase evaluate` 补评；
默认 all。无监督允许最后用金标检验效果，不允许金标参与种子、传播、选头或阈值。
旧 S11 的 `outputs/s11_local_attention_all` 与本版缓存结构不同，保留但不混用。

| 文件 | 用途 |
|---|---|
| reuse_detector/core.py | 无标签参考、种子、端点继承及同质量/距离对照 |
| reuse_detector/capture.py | SDPA 正常前向 + 只读 Q/K/RoPE 重放及数值核验 |
| reuse_detector/run.py | 全任务 roster、缓存续跑、混合校准来源异常预算、冻结 |
| reuse_detector/evaluation.py | 首错、span 起点、延续、停止评价，完整来源固定权重 |

[完整方法与边界](docs/UNSUPERVISED_LOCAL_REUSE.md)。评价重点比较 reuse 与 seed_only、
single_hop、mass_matched_uniform、lag_group_permuted；特别看 continuation_vs_normal、
strict_post_first 和错误结束后正常 token 的误报。高 attention 可能是纠正而非沿用，
该方法不具备语义否定/绑定判定。高置信首错没有熵种子时也会漏检。

## 旧监督基线保留，但必须显式选择

- `python main.py supervised-s11 ...`：入口+延续的监督读出，原参数不变。
- `python main.py supervised-s10 ...` 或 `python -m structural_detector.experiment ...`：S10。
- `python -m structural_detector.audit ...`：对已经冻结的 S10 分数补评，不重训。

旧命令 `main.py transport ...` 会停止并说明它是监督 S11，避免继续误运行。
[S10 结果](docs/S10_RESULTS_20260914.md) 的 .7492 是全错误、.7791 是**所有 span 起点**，
不是每答第一次错误；这两项都不是新版无监督成绩。

## 信息论相关的另一个原型

`binding_detector/projection.py` 实现显式关系下的 `-log Q(合法绑定)`，需要可靠来源匹配
和关系数据包；它不参与本默认流程，不能从五列统计中自动恢复事实图。
旧复用基线使用熵和经验尾部，不估计真假两类密度、条件互信息或贝叶斯幻觉后验。
率失真定理解释高置信碰撞的可能性，不为旧attention传播提供有效性保证。

历史结果与归档均不删除。原 S11 说明见 docs/S11_LOCAL_PROPAGATION.md；
其中旧 transport 命令需改成 supervised-s11 才会运行。
