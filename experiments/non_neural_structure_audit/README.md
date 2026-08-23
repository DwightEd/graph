# 非神经结构审计

这个子项目先用无标签、可解释的统计审计判断哪些 attention-routing 关系值得建模，再决定后续神经模型是否有理由使用 exact response graph、head、layer order、多跳传播或非线性联合形式。

当前对象是 **prompt-connected 与 response-base attention-routing proxy**。它不是 evidence grounding，也不是 Transformer 真实计算贡献或因果 ancestry。

## 一键运行 30 条 smoke

PowerShell：

```powershell
Set-Location 'D:\projects\python_projects\research\graph_latest'
$env:PYTHON = 'D:/projects/python_projects/.audit_envs/llm_state_lab_py311/Scripts/python.exe'
& 'C:\Program Files\Git\bin\bash.exe' './experiments/non_neural_structure_audit/run.sh'
```

`Set-Location` 只是把当前目录切到仓库根目录，使 Python 包导入和相对输出路径一致；它不会下载、复制或切换代码版本。

Git Bash：

```bash
cd /d/projects/python_projects/research/graph_latest
./experiments/non_neural_structure_audit/run.sh
```

脚本默认使用：

- 数据：`D:/projects/python_projects/research/data/RAGTruth/llama31_8b`；
- 前 100 条 train QA 拟合无标签 reference；
- 前 30 条 test QA 做 smoke；`TASK_TYPE=all` 可显式改为全部任务；
- 2 次 endpoint null、5 次非恒等 layer shuffle；
- 50 次 grouped bootstrap、49 次 label circular shift。

这只验证工程路径。A1–A10 全部写成 `NOT_EVALUATED_SMOKE`。A0a 的 artifact/source/endpoint-null invariant binding 会执行，但 raw RAGTruth span/tokenizer gold alignment（A0b）和完整 pipeline label-permutation sanity（A0c）尚未实现，因此总 A0 固定为 `INCONCLUSIVE_A0_CONTROLS_MISSING`。不要从 30 条 smoke 宣布某种图结构有效。

常用覆盖参数：

```bash
TRAIN_LIMIT=30 TEST_LIMIT=30 \
NULL_REPLICATES=10 LAYER_SHUFFLE_REPLICATES=10 \
SWAP_ROUNDS=10 REFERENCE_CAPACITY=1024 \
OUTPUT_DIR=experiments/non_neural_structure_audit/outputs/preflight_30 \
./experiments/non_neural_structure_audit/run.sh
```

## 当前可做的 exploratory discovery

当前代码是目标协议的工程骨架，不是已经完成的论文级审计。它可以生成 exploratory metrics 来改进 null 和 control，但 A0 未完整通过时，discovery/confirmation 的 A1–A10 decision 全部是 `BLOCKED_BY_A0`。不要现在花费 499 replicates 追求正式结论。

### 1. 以低 replicate 运行全 QA exploratory（仍可能耗时数小时）

```bash
TRAIN_LIMIT='' TEST_LIMIT='' \
NULL_REPLICATES=10 LAYER_SHUFFLE_REPLICATES=10 \
BOOTSTRAP_REPLICATES=200 PERMUTATION_REPLICATES=99 \
SCOPE=discovery \
TOKENIZER=D:/path/to/local/Meta-Llama-3.1-8B-Instruct \
OUTPUT_DIR=experiments/non_neural_structure_audit/outputs/exploratory_full \
./experiments/non_neural_structure_audit/run.sh
```

流程默认只选 `task_type=QA`，使用完整 canonical train QA 作为 reference；test QA source groups 在读标签前确定性分成 50% discovery、50% confirmation。`split_plan.json` 绑定 dataset、reference、score manifest、source/sample IDs 与 audit source-code dependency digest；它不替代 Python/PyTorch/NumPy/scikit-learn 环境记录。

`499` 个严格 endpoint null 很昂贵。应先用 30 条、10–20 replicates 做 coverage/profile。最新两条 QA engineering smoke 的合法交换比例分别约为 `0.139`、`0.162`，不是总体 coverage 估计，并低于计划阈值 `0.70`。实际 decision 总体仍是 `BLOCKED_BY_A0`；若暂时忽略 A0、只检查 A2 pilot 质量，则是 `INCONCLUSIVE_NULL_INVALID`。在修好或替换 null 之前，不应启动昂贵的全量 A2。

### 2. 目标 confirmation 协议（完成 A0b/A0c 后才运行）

下面两条命令保留为冻结协议的可执行接口，不代表当前版本已经有资格产生正式 gate。只有补齐 gold trace/span-token 对齐、pipeline label permutation，以及各 audit 所需 control 后，才应冻结 confirmation。

当前 discovery 报告的 A0 不是 `PASS`，所以 `freeze-confirmation` 会直接拒绝生成计划；这是有意的科学停止规则。

未来补齐 A0 后，应按目标 499/2,000 参数重新生成 `outputs/formal_full`，不能把本节的 `outputs/exploratory_full` 改名后复用。

```bash
PYTHON=D:/projects/python_projects/.audit_envs/llm_state_lab_py311/Scripts/python.exe

"$PYTHON" -m experiments.non_neural_structure_audit.main freeze-confirmation \
  --split-plan experiments/non_neural_structure_audit/outputs/formal_full/split_plan.json \
  --discovery-evaluation experiments/non_neural_structure_audit/outputs/formal_full/evaluation_discovery/evaluation.json \
  --tokenizer D:/path/to/local/Meta-Llama-3.1-8B-Instruct \
  --output experiments/non_neural_structure_audit/outputs/formal_full/confirmation_plan.json
```

该命令不读 confirmation 标签。它冻结 discovery 结果摘要、audit 本包及共享执行依赖摘要、tokenizer 文件摘要、evaluation config 和 confirmation sample/source IDs。

### 3. 一次性评估 confirmation

```bash
"$PYTHON" -m experiments.non_neural_structure_audit.main evaluate \
  --split-root D:/projects/python_projects/research/data/RAGTruth/llama31_8b/test \
  --score-dir experiments/non_neural_structure_audit/outputs/formal_full/scores \
  --output-dir experiments/non_neural_structure_audit/outputs/formal_full/evaluation_confirmation \
  --scope confirmation \
  --confirmation-plan experiments/non_neural_structure_audit/outputs/formal_full/confirmation_plan.json \
  --tokenizer D:/path/to/local/Meta-Llama-3.1-8B-Instruct
```

若 score、dataset、source split、代码、tokenizer 或 evaluation config 与冻结计划不同，程序会在读取目标标签前拒绝运行；所有选中 NPZ 的 schema、列名、形状和 token IDs 也会在打开标签前做一次 label-free 预校验。`run.sh` 故意拒绝 `SCOPE=confirmation`，避免无意间重跑并覆盖已冻结 score。

本地 SHA-256 协议防止意外混用，但不能阻止操作者主动覆盖 plan。论文级运行应在打开 confirmation 前把 `split_plan.json`、`confirmation_plan.json`、代码 commit 和命令提交到 Git/只追加实验日志中。

## token 与标签对齐

- 所有 token（含标点和功能词）继续参与 routing/lineage，因为它们可能是中继节点。
- smoke 可对全部 token 评估。
- discovery/confirmation 强制提供本地 tokenizer，primary outcome 只保留 decode 后含字母、数字或中日韩文字的目标 token。
- cache alignment 是 `post_token_query_at_same_position`。
- 评估严格使用 `query t → response token t+1`，即 `score[:-1]` 对 `labels[1:]`；错误 token 自身的 query 不会泄漏进 pre-onset score。

## 数据流

```text
fit (labels closed)
  train attention -> bounded robust task/position reference

score (labels closed)
  one sample -> sparse routing -> conserved lineage
             -> response-endpoint null + non-identity layer shuffle
             -> hashed compact NPZ

plan (labels closed)
  frozen source groups -> discovery IDs + confirmation IDs

evaluate discovery
  label-free validation -> selected labels -> exploratory metrics
  incomplete A0 -> A1-A10 BLOCKED_BY_A0

freeze-confirmation (confirmation labels closed)
  discovery report + code/config/tokenizer digests -> confirmation plan

evaluate confirmation
  exact frozen plan -> conservative A0-A10 decision table
```

唯一入口是 `main.py`：

```bash
python -m experiments.non_neural_structure_audit.main fit --help
python -m experiments.non_neural_structure_audit.main score --help
python -m experiments.non_neural_structure_audit.main plan --help
python -m experiments.non_neural_structure_audit.main evaluate --help
python -m experiments.non_neural_structure_audit.main freeze-confirmation --help
```

## 模块职责

| 文件 | 单一职责 |
|---|---|
| `main.py` | 参数入口与五个阶段的直接调用 |
| `experiment.py` | `StructureAudit.fit()` / `score()` 无标签路径 |
| `lineage.py` | 六类守恒 routing lineage |
| `features.py` | 可解释坐标与预注册 relation scores |
| `nulls.py` | 可复用 `EndpointSwapPlan` 与严格 endpoint null |
| `reference.py` | 有界、train-only robust reference |
| `protocol.py` | 数据/source/code/tokenizer 的冻结协议 |
| `evaluation_data.py` | 唯一标签边界与单样本折叠 |
| `bounded_ensemble.py` | 磁盘缓冲的 exact pooled-AUPRC null；RAM 有界 |
| `bounded_samples.py` | 可复用的 compact sample memmap；避免跨样本矩阵堆在 heap |
| `relation_audit.py` | association、endpoint-null、layer-order 统计 |
| `temporal_audit.py` | pre-onset、pseudo-onset、lock-in 探索量 |
| `joint_form.py` | discovery-only grouped-CV 联合形式比较 |
| `decisions.py` | 保守 gate；缺 control 时不授权模型模块 |
| `attention_lifecycle.py` | 仓库级、可复用的单样本 attention 生命周期 |

## 输出与解读

```text
outputs/<run>/
|-- reference.npz
|-- scores/
|   |-- manifest.json
|   `-- samples/<sample_id>.npz
|-- split_plan.json                    # discovery only
|-- evaluation_discovery/              # or evaluation_smoke
|   |-- evaluation.json
|   |-- relation_metrics.csv
|   |-- temporal_audit.csv
|   |-- joint_form_cv.csv
|   `-- decision_table.csv
`-- confirmation_plan.json             # after discovery freeze
```

先看 `decision_table.csv`：

- `PASS` 只在完整 A0 或未来完整 gate 的全部 control 通过时使用；
- `CANDIDATE_*` 只允许进入下一轮审计，不授权神经模块；
- `INCONCLUSIVE_*` 表示 null、功效或必要 control 不足；
- `NOT_IMPLEMENTED_*` 表示代码没有假装回答该问题；
- `BLOCKED_BY_A0` 表示完整 gold alignment/leakage sanity 尚未通过；其他数值只能描述性探索，不能授权模块。

当前可运行代码覆盖 A0a artifact/source/endpoint-null binding、A1 association、A2 coarse-lag endpoint-null pilot、A4 layer permutation、A7 temporal exploration 和 A9 discovery CV。A0b/A0c、A3、A5、A6、A8 的正式 control 以及 A10 base-LLM intervention 尚未实现，所以当前版本只能筛查实现问题和候选关系，不能授权 exact graph、GRU/SSM、GNN 或联合神经模型，也不能支持 evidence-grounding、完整 graph ancestry 或因果机制主张。每个 metrics CSV 与 `evaluation.json` 都写入 `scientific_status`，必须与 `decision_table.csv` 一起解释。

公式见 [METHOD.md](METHOD.md)，完整审计矩阵与顶会方法依据见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)，内存边界见 [MEMORY.md](MEMORY.md)。
