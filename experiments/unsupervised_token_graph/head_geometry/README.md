# 无标签 head 检测 v2

修复零质量误删，并检验当前 token 的 head 条件依赖能否提供增量。不使用自然幻觉标签拟合。
公式、依据、实验边界及后续先验见 [OPTIMIZATION_V2.md](OPTIMIZATION_V2.md)。[DESIGN.md](DESIGN.md) 保留 v1 设计记录。

## 一键运行 QA

```bash
git pull --ff-only origin main &&
bash experiments/unsupervised_token_graph/head_geometry/run_all.sh
```

默认输出 `outputs/head_geometry_v2`。复用原 TRAIN/TEST attention，CPU 四线程，不加载 Llama 权重。
旧 v1 结果不能续跑成 v2：观测定义和拟合不同，必须重新 prepare/fit/score；默认新目录保留旧结果。
`--resume` 仅接受相同的 v2 设置。依赖 numpy/scipy/sklearn/tqdm/threadpoolctl；读取 tokenizer 需要 transformers。

## 仅使用中间层，分别运行三个任务

```bash
git pull --ff-only origin main
bash experiments/unsupervised_token_graph/head_geometry/run_middle.sh
```

这是 Llama-3.1-8B 的显式配置：仅 L8–23（从零编号）的 head，共 512 个；不按 TEST 成绩选头。
QA、Summary、Data2txt 分别建立参考/校准组，并分别汇报。输出到新目录
`outputs/head_geometry_middle`，不复用全层拟合结果；旧结果保留。
默认不再把中间层切成四段，`scope=all` 表示全部已选的中间层。
这次完整 L8–23 重拟合还没有自然数据成绩，不能直接引用旧 L16–23 子范围的 AUROC。

仅指定某些 head 时，`--heads` 使用原模型的 head 编号，并作用于每个已选层：

```bash
OUTPUT=outputs/head_geometry_middle_h4_h6 \
bash experiments/unsupervised_token_graph/head_geometry/run_middle.sh --heads 4 6
```

这个 head 列表只是语法例子，不代表已发现 4、6 是功能头。不同模型用 `--layers` 明确覆盖层列表，
不把 Llama 的层号直接套到其他模型。选择不同 head 或窗口时使用不同 OUTPUT。

任意已有完整分数可只重生成报告，不需要重新拟合或运行大模型：

```bash
python -u -m experiments.unsupervised_token_graph.head_geometry \
  --phase evaluate --output outputs/head_geometry_v2
```

`predictions/task_summary.md` 汇报实际层/head、三个任务的完成情况及各方法成绩。
`predictions/tasks/QA/`、`Summary/`、`Data2txt/` 分别保存 `metrics.csv` 和 `evaluation.json`；
未运行的任务显示“未评估”，不补零、不复用 QA 数值。总表增加 dataset/task/generator 列，
控制台只打印 task×generator，避免只有 QA 时重复打印同一份 ALL。
这里解析的是 RAGTruth；三个名称是其任务类别，并未新增任意其他数据集的解析器。

多头协同的具体定义与实验见 [COOPERATION_DESIGN.md](COOPERATION_DESIGN.md)。

默认缓存：`/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/{train,test}`；
默认 tokenizer：`/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct`。
支持 `--train-cache --test-cache --dataset --tokenizer --index --source-info`。
特殊 key 合并 tokenizer 的命名特殊 ID 和 `AddedToken.special` 标记，包括未列作 BOS/EOS 的聊天控制符；
只有已核对完整集合才显式传 `--special-token-ids`。实际集合保存在 settings/manifest。

## 零值与缺测

删除特殊 key 后保留未重新归一化的普通 key 子质量。self=0 或整行普通质量=0 都合法。
只有预测行未保存才记 NaN；不补首词行、不恢复稀疏缓存尾部。query `q=P+t-1` 预测回答 token t。
同时保存原行质量、普通质量、`head_observed` 与 `conditional_defined`。
`observations/manifest.json.coverage_summary` 逐答记录缺行、观测零质量和覆盖。
仍要求所选头的 query 行实际存在；修复没有把真缺行补成零。

## 冻结对照

| 方法 | 表示和评分 |
|---|---|
| raw | 当前逐头原子质量；参考库 5-NN |
| contrast | 去层共同分量但保留头身份；5-NN |
| moment | contrast + 窗口二阶矩；5-NN |
| log_moment | contrast + 窗口二阶矩的矩阵对数；5-NN |
| log_diagonal | contrast + 窗口边际能量；5-NN |
| independent_energy | 每个 head 的边际标准化残差平方，不使用其他 heads |
| **conditional_energy（v2 主候选）** | 留出整个 head，由同层其他 heads 条件化后的标准化残差平方 |

两个 energy 方法只使用当前 token，不平滑、不使用之前错误标签、不依赖 window。
按 head 计算后才聚合为 scalar，逐 head 原始能量另存。二者共用原坐标、参考抽样及 ridge，
区别是是否考虑头间依赖。能量直接评分，不再经 IQR+kNN，以免对角缩放被近邻标准化抵消。

默认只输入 self；新目录加 `--signals self prompt` 检验多信号，整个 head 的所有信号一起留出。
条件拟合使用 raw，不能使用 contrast 的头间和恒为零这一机械约束预测留出的 head。

TRAIN 按 source 分参考/校准；参考库最多 4096 行、每来源最多 16 个 token，不按标签筛选。
混合校准集 95% 分位数报警，不保证正常 FPR=5%。高分只表示偏离参考关系，不保证就是事实错误。
原五个近邻对照和两个 energy 方法使用相同来源及取样，各自无标签校准。

同时报告全层和四个等宽层段，不事后选最佳中层。默认 window=16，ridge=.1，关系 sketch=256。
当前逐头坐标不投影；`--dimensions 0` 保留全部二阶项但成本更高。
窗口二阶矩不是滚动协方差，也不记录窗口内事件顺序。

## 查看结果

`outputs/head_geometry_v2/predictions/metrics.csv`：`group=ALL, scope=all` 比较七种方法和 position 基线。
主比较 conditional_energy 减 independent_energy，同时比较 raw/contrast/log_moment。
一起看 all_error、span_onset_vs_normal、first_error_until_first、continuation、previous_error、front/back half。

`evaluation.json` 有 source bootstrap 区间、来源加权评价、报警及 span 匹配。
新增 `groups.ALL.availability` 给出有效 token、配置窗口、真实连续窗口长度直方图和完整窗口数。

预测 NPZ 中 `independent_energy__per_head` 和 `conditional_energy__per_head` 的形状为
`[有效位置, 选定层, 选定head]`，回答位置在 `embedding_positions`；这些能量尚未校准。
`--save-embeddings` 才额外保存所有表示。观测、拟合矩阵和参考库分别位于 observations/reference。

```bash
python -u -m experiments.unsupervised_token_graph.head_geometry \
  --phase evaluate --output outputs/head_geometry_v2
```

只评价可读取已有 frozen 输出，但不会将旧实验变为 v2。另跑时间对照时：

```bash
OUTPUT=outputs/head_geometry_v2_w1 bash experiments/unsupervised_token_graph/head_geometry/run_all.sh --window 1
```

两个 energy 结果应保持不变；窗口方法才会改变。

## 验证边界

软件验证使用合成缓存及解析条件高斯对照，覆盖零/极小质量、缺行、特殊 key、前缀因果性、
完整窗口、留出整个 head 的多信号回归、同边际不同配合、标签/TEST 不影响拟合，以及 CLI 续跑。
尚未在完整自然缓存上运行 v2，不能报告修复后的 AUROC。

```bash
python -m pytest -q experiments/unsupervised_token_graph/head_geometry/tests \
  experiments/unsupervised_token_graph/head_roles/tests
```
## 交叉项审计

新增可选 `cross_terms` 实验，保留旧 `geometry` 默认行为。方案与判读规则见
[CROSS_TERMS_DESIGN.md](CROSS_TERMS_DESIGN.md)。

```bash
OBSERVATIONS=outputs/head_geometry_middle \
  bash experiments/unsupervised_token_graph/head_geometry/run_cross_terms.sh
```

复用已有 observations，继承其任务/层/头选择，不重新加载模型或扫描大 attention 缓存。
默认输出 `outputs/head_cross_terms_v1`；成功后自动生成同名 `_review.tar.gz`。
主要看 `predictions/task_summary.md`、`predictions/comparisons.csv` 和各任务 `evaluation.json`。
`pair_full` 是预先指定的主候选，其他 `pair_*` 分别控制交叉项、持续状态与因果时间平滑。
这些是待验证的消融，不代表已经获得新的真实数据检测成绩。
