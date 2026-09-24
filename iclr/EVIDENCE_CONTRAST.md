# 原文来源／历史四条件检测

实现日期：2026-09-24。入口：`main.py transport-contrast`。
这是研究方案第一阶段的独立采集/评分原型；没有真假分类器，没有新8B检测收益声明。

## 目标与公式

固定原回答 token 序列。对当前文本单元起点 a 及其中 token t，设 Q 为非来源提示，
D 为已保存来源 token，H=y[:a] 为较早回答，U=y[a:t] 为单元内前缀。

| 保存字段 | 条件 |
|---|---|
| `logp_source_history` | log p(y_t \| Q,D,H,U) |
| `logp_history` | log p(y_t \| Q,H,U) |
| `logp_source_local` | log p(y_t \| Q,D,U) |
| `logp_local` | log p(y_t \| Q,U) |

四种条件均在预测 y_t 的 query 位置读取概率，不是读入 y_t 后的分类输出。
完整条件的 query=P+t-1；局部条件的 query=P_condition+t-a-1。
实际对应位置逐 token 保存，回答 ID、source ID、原始 token ID 均保留。

```
G_full  = logp_source_history - logp_history
G_local = logp_source_local   - logp_local

source_full      = -G_full
source_local     = -G_local
source_pair      = -(G_full + G_local) / 2       # 预先冻结的主候选
source_pair_span = mean(source_pair within the saved unit)
surprisal_full   = -logp_source_history         # 原文可预测性对照
```

1/2 是来源与远历史两个因素的对称分配，不是标签拟合权重。所有差值的单位是 nats/token。
`source_history_interaction=G_full-G_local` 只保存为诊断，不额外叠加为风险。
低来源作用不是事实错误的充分条件，高原文概率也不是正确证据。该方案不声称恢复了全部事实语义。

依据：[PMI-FAITH](https://aclanthology.org/2023.emnlp-main.639/) 的有/无来源条件概率，
以及 [CORTEX](https://arxiv.org/html/2606.31033v1) 对历史承接的研究。
CORTEX 使用真实 token 标签训练；本实现没有移植其监督分类器或把其成绩冒称为本方法效果。
本实现的四条件与对称分配是待验证候选，不是上述论文已证明的无监督 token 检测结论。

## 输入与局部条件

输入为现有 `transport-observable` 目录或它的轻量/完整 ZIP。需要：

- `settings.json`：观察者模型路径、完整原始 token 与 prompt 长度；
- `responses/*/sources.json`：原始来源组；
- `responses/*/scores.npz`：已对齐的 raw_route、observable_route、raw_attention、entropy；
- `annotations.json` 可选，只在所有新分数保存后读取。

只删除 prompt 中 `group_id < source_count` 的来源 token，保留问题、指令和其余模板 token。
不重新分词、不生成新回答、不选真假候选。来源组表示可用来源，不是已核验的适用证据。
删除后按新序列使用原生位置编码；来源长度和位置也随之变化，因此不是隔离了位置的因果干预。

句号/问号/感叹号/换行定义确定性文本单元；识别列表编号并避免在数字小数点中间切分。
这不是语义断言边界或模型发现的 reanchor。默认超过64 token的单元按token顺序拆块。
每个拆块单独重置远历史；标题、短项、标点、重复文本和尾部均参与评分。
单元规划使用完整文本，片段分数和固定窗口均值使用未来token，整个入口明确按离线检测报告。
在已经固定单元起点时，当前四条件似然只使用该目标以前的输入。

## 一键运行与恢复

在项目根目录、原 research 环境运行：

```bash
git pull --ff-only origin main
bash experiments/native_support/run_contrast.sh
```

脚本默认读取 `outputs/native_support_ragtruth4/observable_transport_v1`，输出到
`outputs/native_support_ragtruth4/evidence_contrast_v1`，自动恢复已完成条件。
这不扩展四答数据，也不复用旧方法的失败校正分数。

更换输入或输出可直接覆盖脚本默认参数：

```bash
bash experiments/native_support/run_contrast.sh \
  --input /path/to/observable_transport_v1_review_light.zip \
  --output outputs/evidence_contrast_new_sources
```

模型使用 settings 中的原路径及现有 teaching 适配器。本版已用随机初始化的小型
Llama/Mistral/Qwen2 验证；其他布局须先显式适配，不宣称已经支持 Qwen3。

采集使用原生 SDPA、无梯度、默认 BF16。预填充分块256 token，输出概率分块16 token，
不保存 dense attention 或词表全序列 logits，也不新增逐 token VJP。
一次只保留一个来源条件的 KV；每个局部单元开始前裁回固定 prompt 前缀，避免分支污染。
来源 prompt 的预填充在同条件各局部单元间复用。

如需降低工作缓冲，可以运行：

```bash
bash experiments/native_support/run_contrast.sh \
  --prefill-chunk-size 128 --query-chunk-size 4
```

分块参数可在恢复时调整，每个已完成条件保留自己的执行参数。模型、dtype、单元上限、
基线窗口与原数据变更需要新输出目录。单个条件中断会重算该条件；已完成条件跳过。
这里只测过小模型软件正确性，尚未测真实8B的24GB显存峰值或速度。

分阶段命令：

```bash
# 只采集，尚不读取评估标签
bash experiments/native_support/run_contrast.sh --stage capture

# 已采集：CPU重算固定分数、评价、打包，不加载模型
python main.py transport-contrast --stage score \
  --output outputs/native_support_ragtruth4/evidence_contrast_v1

# 只重做评价；--annotations 可提供另一个真实标签文件
python main.py transport-contrast --stage evaluate \
  --output outputs/native_support_ragtruth4/evidence_contrast_v1

# 只打包已有结果
python main.py transport-contrast --stage pack \
  --output outputs/native_support_ragtruth4/evidence_contrast_v1
```

score/evaluate 使用保存协议中的窗口，不从新CLI默认值悄悄改变公式。
输入仍可访问时，评价重新复制其中的标签；输入已移动时，可以使用已保存标签或明确传入 `--annotations`。
缺少标签仍输出完整分数，evaluation 写明不可用，不伪造 AUROC/AP。

## 输出与比较

每答保留 `views.json`（四条件输入、删除位置、单元边界）、`sources.json`、
`baselines.npz`、`with_source.npz`、`without_source.npz`、`scores.npz`、运行时间/显存记录。
真实标签存在时增加 `trajectory.png`，分开画 nats 分数与 route 分数，避免混淆量纲。

四个原始基线原样复制。raw/observable 各自再算固定16 token离线均值作强基线；
`risk` 仍保存 raw_route，新候选不会因当次评价较高就自动替代它。
本入口不压缩已有多头缓存；它利用完整冻结模型完成新的条件读出。
没有训练多头时序学生、语义教师或合成真假模型；这些步骤须以新候选实际收益为前提。

聚合输出包括：

- `coverage.json`：完整 token 覆盖、原来源删除量、单元数；
- `evaluation.json`、`metrics.csv`、`summary.png`：AUROC/AP，来源等权AP、within-answer、前后半段；
- `onsets.csv`、`span_budget.csv`、`normal_run_budget.csv`、`budget_by_answer.csv`：首错、延续和正常段；
- `detection_budget.json`：固定10%告警预算下的首错/片段起点召回、正常token误报和全正常回答误报；
- `comparisons.csv`：主候选相对原始/同窗口强基线的差值；
- `source_bootstrap.json`：同来源所有回答共同重采样，默认200次，可用 `--bootstrap 0` 关闭；
- `summary.json`：协议、评价、累计采集时间与显存峰值。

自动生成旁边的 `evidence_contrast_v1_review.zip`，包含上述结果与四列观测，可以CPU重新评分。
预算排名的平分按原回答/token顺序处理；不是用真值寻找最佳阈值。Bootstrap剔除单一类别重采样，
记录有效次数；来源太少时不能据区间声称可泛化。四答及已看过的32答仍属于探索数据。
不同任务/生成器应使用各自缓存目录运行，保留原cohort信息，不把多任务结果合成一个结论。

## 本次验证与边界

针对性测试覆盖：分块KV与独立完整原生前向相等；改变局部计算次序不污染结果；
固定单元时的未来回答不影响此前似然；首单元的full/local一致；空来源的作用为零；
来源删除和query索引；完整文本覆盖；标签置乱不改变分数；恢复无需再次加载模型；
ZIP输入、自动打包、原基线完整复制和全层能量加权重建恒等式。

没有真实8B前向、新自然数据AUROC/AP或速度提升结果。训练学生与大范围重构尚未启动。
先运行此独立原型检验检测收益；若不胜过强基线，就保留失败结果，不继续堆叠校正或状态结构。
