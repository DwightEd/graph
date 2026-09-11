# Candidate-conditioned path residuals

本仓库承载主线检测方法。相关机制和候选算子在
[reanchor 项目](https://github.com/DwightEd/reanchor) 中做具体样本验证；
主线采用的算法在本仓库集成到提取、图计算、评分和评价流程。
reanchor 新增的回看定位、多跳来源与表征关系残差目前是实验原型，尚未接入下面的默认流程。

当前方法用固定图算子读取**候选状态与真实 attention 端点的条件对齐**，用于无监督幻觉检测。不训练新神经网络，也不需要先进行模型消融。

先固定每行 attention 在角色、来源段落、距离区间中的质量以及 self edge，再比较真实路径与组内端点置换期望对候选状态的作用。保留路径、候选排名、层和最终 head 通道。二步 source-through-history residual 包含一阶偏离，不解释成纯二阶交互或真实因果贡献。

**状态：已实现并验证软件流程；尚无新方法在自然数据上的检测有效性结果。** CHARM、TOHA、graph scattering、attention rollout 都是邻近已有工作，“固定图算子替代 AE”本身不是充分创新。[完整研究方案](refine-logs/FINAL_PROPOSAL.md)给出一手文献、精确公式和可证伪条件。[历史正负结果](docs/EXPERIMENT_HISTORY.md)继续保留。

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

输出包含 `residual`、`observed`、`null`、`signal`、对应 kNN、entropy、negative-margin 和 position。评价保留全部响应 token，包含首 token；报告全流、span onset、response first error、continuation 的来源平衡 AUROC/AP，以及配对来源 bootstrap AUROC 差。生成 token 的候选覆盖率不等于正确答案覆盖率，后者当前未知。未实施报警阈值选择。

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
