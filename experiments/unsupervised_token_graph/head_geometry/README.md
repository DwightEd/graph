# 无标签 head 几何检测

当前逐头向量完整保留；不添加历史均值特征，不对风险分数做移动平均。
新表示检验的是：以固定训练参照为原点，head 的带符号交叉乘积及矩阵对数，能否让无标签近邻检测读出更多差异。
实现与研究边界见 [DESIGN.md](DESIGN.md)。它是检测验证，不冒充已证明的信息流因果机制。

## 一键运行 QA

在 graph 根目录、已有 research 环境：

```bash
git pull --ff-only origin main &&
bash experiments/unsupervised_token_graph/head_geometry/run_all.sh
```

直接 Python 入口等价：

```bash
python -u -m experiments.unsupervised_token_graph.head_geometry \
  --phase all --tasks QA --output outputs/head_geometry_v1 --resume
```

复用原 train/test attention 缓存，不加载 Llama 权重。CPU 四线程、逐阶段进度条、NPZ 保存、逐答续跑。
默认路径：`/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/{train,test}`；
标注/身份目录为同 RAGTruth 下的 `dataset`。

特殊 token 由原 observer tokenizer 的 `all_special_ids` 取得，默认 tokenizer 为：
`/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct`。
路径不同用 `--tokenizer /实际目录`，仅本地加载。`--special-token-ids` 仅供已核对完整 ID 集合的显式输入；不能只填 BOS。
除 NumPy/SciPy/sklearn/tqdm/threadpoolctl 外，自动读取 tokenizer 需要现有 transformers；不需要 GPU。

先检查真实缓存接口：

```bash
python -u -m experiments.unsupervised_token_graph.head_geometry --phase inspect
```

`--train-cache`、`--test-cache`、`--dataset`、`--index`、`--source-info` 沿用已有接口。
支持 canonical CSR、dense attention/data、逐 layer/head 导出。裸缓存使用原身份元数据，不使用其幻觉标签拟合。
不要用 `--limit` 的小样本输出冒充全量成绩；小样本仍需独立来源的参考/校准组。

## 一次输出的对照

| method | 表示 |
|---|---|
| raw | 原 head 向量，FIT 中位数/IQR 标准化 |
| contrast | 去层共同分量的当前逐头向量，完整保留坐标 |
| moment | contrast + 参照白化后的窗口二阶矩减单位阵 |
| log_moment（主方法） | contrast + 参照白化后的窗口二阶矩的矩阵对数 |
| log_diagonal | contrast + 只保留各头边际能量的对数比例，移除 head 交叉项 |

所有方法共用同一参考 token 抽样和 k=5 近邻规则；高分表示混合 TRAIN 中少见的构型。
TRAIN 按 source 分为参考和独立校准来源；同 task/generator 分别拟合。
参考来源最多各取 16 个 token，库上限 4096。正常与幻觉回答均不按标签筛除。
校准为无标签混合分数的 95% 分位数，不保证 5% 正常 FPR；实测 FPR 在评价表报告。

默认 self 输入，以对应已有 self attention 监督审计；同时缓存 prompt 质量。
`--signals self prompt` 在同层联合保留两类 head 坐标，使用新 output 单独验证。
当前 raw/contrast 向量完全不投影；只对二阶展开部分使用固定双哈希投影到 256 维。
矩阵投影是有损的，固定 seed，不按标签选 seed。`--dimensions 0` 保存和使用所有二阶坐标，成本明显增加。
拟合后的特征 IQR 标准化会改变几何度量，所以这是“切空间描述子＋无标签近邻”，不是精确流形测地线近邻。

默认同时报告全层及四个等宽层段。32 层对应 L0–7、L8–15、L16–23、L24–31。
不只报告看起来最好的中层。各层段单独建立参照，使用相同 token；这是层段独立检测，不是冻结全层评分的贡献分解。
已有测试集被多次探索，本实验标记为探索性验证；没有使用其标签定拟合参数、方向、阈值或窗口。

## 结果在哪看

```text
outputs/head_geometry_v1/
  settings.json
  observations/manifest.json, inputs.json, *.npz
  reference/settings.json, complete.json, 0/geometry.npz, 0/bank.npz, ...
  predictions/freeze.json, *.npz, metrics.csv, evaluation.json
```

先看 `predictions/metrics.csv` 的 `group=ALL, scope=all, view=all_error`，比较全部五行。
再看 `first_error_until_first`、`span_onset_vs_normal`、`continuation_vs_normal`、`previous_error`。
`previous_error` 在前一词已标错的位置，区分继续错误与恢复正常；金标只用于评价分组。
`full_window/warmup` 检查短历史造成的窗口秩/长度混杂；缺测和首词缺行在 coverage 中单独报告。

`evaluation.json` 包含来源重采样的主方法减对照 AUROC/AP 区间、来源平衡指标，以及片段延迟和覆盖统计。
`position` 是只用当前绝对位置的独立控制，不依赖回答最终长度。
所有对照有相同可观测位置；不因一条方法得分不理想翻转方向。
分数先保存并冻结，随后才读取标注；更换标注可只重跑评价：

```bash
python -u -m experiments.unsupervised_token_graph.head_geometry \
  --phase evaluate --output outputs/head_geometry_v1
```

`--save-embeddings` 可另存逐答表示，行号在 `embedding_positions`；默认保存原逐头观测、拟合参数、参考库及全部分数。
改 window/signals/layers/dimensions/ridge 等计算参数要换 output，不混用断点文件。

## 怎么判断新表示有用

- contrast > raw：去掉层共同变化有收益。
- moment > contrast：带符号二阶关系有增量。
- log_moment > moment：矩阵对数变换有增量。
- log_moment > log_diagonal：head 间交叉项在边际能量之外有增量。
- 只有 continuation 提升：收益主要集中在持续阶段，应同时检查恢复正常和 warmup。

这些比较只检验检测增量。低分说明这条表示和当前无标签读出未成功，不自动否定监督可读信号；高分也不直接证明原 LLM 消息冲突或证据适用性。

## 已完成的软件验证

```bash
python -m pytest experiments/unsupervised_token_graph/head_geometry/tests \
  experiments/unsupervised_token_graph/fixed_graph/tests \
  experiments/unsupervised_token_graph/latent_regime/tests -q
```

覆盖实际缓存 reader 格式等价、特殊 token、q→q+1 对齐、无未来、缺测断点、SPD 已知解析解、
常量构型不被消掉、均值相同而头关系不同、符号保留、标签修改不影响分数、TEST 不影响拟合、完整 CLI 与续跑。
本轮 17 项新测试和 28 项既有回归测试，共 45 项通过。
本地用合成 fixture 运行上述端到端测试；另完成 32 层×32 头尺寸的有限值和耗时预检。
没有当前服务器的完整自然 TRAIN/TEST attention，尚无新自然数据 AUROC。
