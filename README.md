当前进展（2026-09-13T15:46:56.948789+08:00）：A2-v1完成且自然检测无改善；A2-v2实际来源擦除恢复版已实现，正在启动特征采集。见[完整结果](docs/GROUNDED_GRAPH_V1_RESULTS_20260913.md)、[当前方法](docs/CURRENT_METHOD_20260913.md)、[新版命令](docs/GROUNDED_GRAPH_RESTORATION_V2_RUN_20260913.md)。以下旧状态保留为历史。

当前进展（2026-09-13T15:07:49.713783+08:00）：完整source图与真实pretoken特征已跑完；联合图适配器训练/自然评价正在启动，尚无本版效果。当前实现见 [CURRENT_METHOD](docs/CURRENT_METHOD_20260913.md)，不要把旧SourceRel95.4%字段成绩当自然检测结果。

## 当前方法与运行入口（2026-09-13T12:54:03+08:00）

最新结构及已验证边界见 [当前方法](docs/CURRENT_METHOD_20260913.md)。当前在实现和验证 SourceRel-Mini 约束归属候选模块；[数据运行](docs/SOURCEREL_DATA_RUN_20260913.md)、[冻结特征](docs/SOURCEREL_FEATURE_RUN_20260913.md)、[小头训练](docs/SOURCEREL_TRAIN_RUN_20260913.md)均有具体命令。它尚不是已验证有效的完整检测器；下方soft_graph入口是已早停的历史版本。

RAGTruth全量机制已完成17790/17790；typed native90forward未通过强证书。不要将辅助source-pointer准确率、raw干预效应或旧全量运行等同于回看/路由问题已解决。

---

## 当前主线：证据约束的消息依赖图（2026-09-13）

新入口 `python -m route_graph.soft_graph_runner` 已接通结构、高维特征、有限关系判断、原始输入介导干预及全词风险。软件仍在自然验证前检查，尚无本版有效性结论。模型结构见 [SOFT_GRAPH_METHOD_20260913.md](docs/SOFT_GRAPH_METHOD_20260913.md)。

v1/v2/v3严格审计均未产生native干预，v3全部4733词弃权；旧负结果完整保留。`route_graph.audit_runner`是单独的严格A/B证据层，默认`main.py`仍是历史基线。以下旧“当前”和PID属于历史记录。

---

# Candidate-conditioned path residuals

**当前主线在迭代，尚未验证有效性。** 默认 main.py 仍运行旧路径残差基线；新增完整离线方法入口是 `python -m route_graph.audit_runner`。
原生图 v1 的36条自然开发样本已全部执行，语义覆盖6.06%、错误召回0、实际native干预0；问题构造失败使它尚不能检验内部归属。
原子事件槽位 v2 同一清单也已全部完成，语义覆盖0、native干预0。v3 保留原句的槽位 mask 已实现、复审关闭，正在同一清单上运行完整 A–D 验证（PID147669）。
[当前模型结构](docs/NATIVE_METHOD_MODEL_20260913.md)、[v1真实结果](docs/NATIVE_AUDIT_V1_RESULTS_20260913.md)、[v2运行与评价指令](docs/NATIVE_AUDIT_V2_RUN_20260913.md)给出实现与限制。
[本地数据盘点](docs/LOCAL_DATASETS_AND_RUN_STATUS_20260912.md)保留RAGTruth/HaluEval/BoolQ/CBUD的支持范围。

本仓库承载主线检测方法。相关机制和候选算子在
[reanchor 项目](https://github.com/DwightEd/reanchor) 中做具体样本验证；
主线采用的算法在本仓库集成到提取、图计算、评分和评价流程。
reanchor 新增的回看定位、多跳来源与表征关系残差目前是实验原型，尚未接入下面的默认流程。

当前方法用固定图算子读取**候选状态与真实 attention 端点的条件对齐**，用于无监督幻觉检测。不训练新神经网络，也不需要先进行模型消融。

先固定每行 attention 在角色、来源段落、距离区间中的质量以及 self edge，再比较真实路径与组内端点置换期望对候选状态的作用。保留路径、候选排名、层和最终 head 通道。二步 source-through-history residual 包含一阶偏离，不解释成纯二阶交互或真实因果贡献。

**状态：软件流程已实现，本轮自然旧捕获与机制验证均未支持检测有效性。** [2026-09-12迭代报告](docs/METHOD_ITERATION_20260912.md)记录数据恢复、冻结报警阈值和128条件机制复跑：residual首错AUROC 0.4361，entropy 0.8442；新上下文读出保留为诊断，未接入默认流程。 CHARM、TOHA、graph scattering、attention rollout 都是邻近已有工作，“固定图算子替代 AE”本身不是充分创新。[完整研究方案](refine-logs/FINAL_PROPOSAL.md)给出一手文献、精确公式和可证伪条件。[历史正负结果](docs/EXPERIMENT_HISTORY.md)继续保留。

## 默认执行路径

远端完整流程可用 `bash scripts/run_route_evaluation.sh`，自动运行准备、提取、评分和评价，默认沿用 reanchor 的远端数据／模型路径。参数和目前尚缺的 reanchor 机制评估见 [路由分析及远端运行](docs/REANCHOR_ROUTING_ANALYSIS.md)。

```text
RAGTruth 原始 prompt / response
  -> RagtruthPreparer.run(): 按 task/generator 选择，按 source 拆分
  -> FrozenGraphCapture.run(): 本地冻结 Llama 的因果 edges + candidate states
  -> PathEncoder.encode(): observed / conditional-null / residual / signal
  -> RouteDetector.run(): 条件参考 -> 冻结全流分数
  -> RouteEvaluator.run(): 此时才连接幻觉标签
```

入口是 `main.py`。从仓库根目录运行：

```bash
python -m pip install -r requirements-model.txt

python main.py prepare \
  --dataset /path/to/RAGTruth/dataset --task QA \
  --generator llama-2-7b-chat --max-sources 256 \
  --output outputs/run/input.jsonl

python main.py extract \
  --input outputs/run/input.jsonl --model /path/to/local/llama \
  --device cuda --dtype bfloat16 --output outputs/run/features

python main.py detect \
  --features outputs/run/features --neighbors 3 --per-source 8 \
  --output outputs/run/detection

python main.py evaluate \
  --scores outputs/run/detection/scores.jsonl \
  --labels /path/to/RAGTruth/dataset/response.jsonl \
  --output outputs/run/evaluation.json --bootstrap 1000
```

`--generator` 必须与数据中的 `model` 字符串相同。probe Llama 可以与原生成器不同，但此时属于 replay，不能声称是原生成器的内部轨迹。保留原始 prompt，采用 BOS + 单独分词后追加 response 的协议，不自动还原 chat template。

默认最终层、深度 `1 2`、候选数 `2`、全部终点 heads。用 `--end-layers` 指定从零开始的终点层，`--depths`、`--candidates` 指定预声明比较。终点必须有足够的前序层，参数写入 manifest。

`--max-tokens` 默认 2048；`--max-attention-mb` 默认 1024，只估计全部 attention 的存储，不包含权重、logits、激活和暂存。仍使用稠密 attention，长上下文可能昂贵。超过限制直接失败，不静默截断或漏样本；完成 manifest 在整个提取成功后写入。不下载模型。

## 无监督分数和对照

参考与测试 source_id 完全隔离。参考条件为 task、generator、prompt 长度区间、已生成长度区间，不使用回答总长度。每个来源等量取样。四种表征使用共享标准差和有效坐标，分别居中后计算 `mean(z²)`；`*_knn` 是补充读出。

恒定坐标不参与距离；整组无有效坐标时输出零分和 `reference_active_features=0`，表示参考没有辨别力，不表示正常。参考组不足所需来源数时失败，不跨组回退。

输出包含 `residual`、`observed`、`null`、`signal`、对应 kNN、entropy、negative-margin 和 position。评价保留全部响应 token，包含首 token；报告全流、span onset、response first error、continuation 的来源平衡 AUROC/AP，以及配对来源 bootstrap AUROC 差。生成 token 的候选覆盖率不等于正确答案覆盖率，后者当前未知。默认 evaluator 不选择报警阈值；本轮新增独立 `route_graph.alarms` 入口，用无标签fit来源留一评分校准阈值，并在首错处截断评价。它不保证测试中的相同实际报警预算。

## 文件职责

| 文件 | 职责 |
|---|---|
| `main.py` | 参数解析、构造对象、调用 `run()`、输出 |
| `route_graph/data.py` | 数据准备、无标签输入契约 |
| `route_graph/capture.py` | 冻结模型、token 对齐、原生提取 |
| `route_graph/operator.py` | 强条件零模型与有序路径读出 |
| `route_graph/detector.py` | 参考拟合、同条件评分与对照 |
| `route_graph/evaluation.py` | 标签连接、完整覆盖与统计 |
| `route_graph/metrics.py` | 来源平衡二分类指标与 bootstrap |
| `route_graph/archive.py`、`archive_evaluation.py` | 不完整旧捕获的严格恢复与探索性评分 |
| `route_graph/alarms.py` | 来源留一校准、首错截断报警与配对来源评价 |
| `route_graph/context.py` | 候选来源上下文诊断；未证有效，不进入默认检测 |
| `onset_analysis/analysis.py` | 错误起点和正常位置匹配、回看与选择的联合观测 |
| `onset_analysis/traces.py` | 读取已有 attention-audit v3 与独立标签 |
| `onset_analysis/statistics.py` | 首错／后续起点分组、事件相关性与不确定性 |

已删除旧 `control_graph`、四边 factorial 和分摊 whole-head margin 的评分代码。旧会话中有用的 onset 联合分析保留为 `python main.py onset-audit`，不再依赖旧图。已有 attention-audit v3 的主 NPZ 可用于这项低成本统计；它不能替代新路径算子所需的完整 token 连接。

```bash
bash scripts/run_onset_choice_audit.sh \
  experiments/reanchor_flow/outputs/attention_audit_v3 \
  "outputs/onsets_$(date +%Y%m%d_%H%M%S)"
```

这是利用已有标签做事件匹配的事后分析，不是无标签在线检测。输出 `events.jsonl` 和 `summary.json`，包括真正的首次标注错误、后续 span 起点、覆盖率、span 续写比例，以及事件级的选择兼容性—回看相关性。只跳过尚未采集的 trace，并报告数量；已有 trace 缺标签或格式损坏时失败。`word/number/alphanumeric` 只用于粗粒度匹配，不是实体识别。

新提取流程有样本／token 进度条，显示前向阶段、token 数；评分和 bootstrap 评价也显示进度。原始单次模型前向期间不会伪造层进度。[恢复的研究主线与旧结果](docs/ONSET_RESEARCH.md)说明图聚合、首错分析和当前测量边界。

```bash
python -m pytest -q
python -m ruff check main.py route_graph onset_analysis tests
```

安装 `requirements-model.txt` 后执行真实 tiny Llama CPU 集成测试。它使用随机权重，只验证软件。只安装基础依赖时模型测试跳过，不能据此声称模型提取已验证。
