# Graph

面向完整回答的逐 token 无监督幻觉检测研究。代码保留完整 prompt、来源与生成历史，测量原生内部消息对后续生成的作用，再研究来源条件下的图联合检测。

**当前状态：架构重构和核心软件验证。** 原生消息/VJP、精确图割、逐 token min-marginal 已实现；自动语义提案、独立校准与真实 8B 检测验证仍待集成。不能把 CPU 图割通过或小模型导数通过当成新方法有效。

架构、公式推导、接口和待完成工作见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。研究主记录在共享的 `/share/home/tm902089733300000/a903202310/lys/codex/research/README.md`；历史结果保留在原来的 outputs/results/runs 与共享 tracker 中。

## 安装与验证

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python main.py --help
```

项目内直接复用 `teaching/state_audit` 原生模型适配器；没有另建算法项目。新核心在原有 `experiments/token_backtrace` 中：

| 文件 | 职责 |
|---|---|
| `trace.py` / `readout.py` | 全回答根归因、三角谱系、原 token 对比及有物理 head 身份的作用统计 |
| `messages.py` | pre-WO 整头消息、原生方向 VJP、对齐与有限干预 |
| `global_graph.py` | 来源/表达范围门控、精确图割、逐 token 分数、缺测台账 |
| `pipeline.py` | 明确的缓存输入/输出；不读取自然真假标注 |
| `span_metrics.py` | 冻结评分后按字符并集评价 RAGTruth span |

## 运行入口

所有阶段显式启动；先用 `GROUP STAGE --help` 查看参数。

```bash
# 原有有效无监督基线：从标量缓存重算，核对所有任务/分区的分数及阈值
python main.py baseline fixed --report outputs/my_verification/fixed.json

# 三任务完整test：重算无标签参考、冻结全部分数，再评价官方标注
# 运行旧fixed与独立token对照；不是尚未接通的新消息图
python main.py token baseline --stage run-test --output outputs/my_token_test

# 已有逐 token 对比基线，拟合和评价分开
python main.py token baseline --stage fit --output outputs/my_token_run
python main.py token baseline --stage pilot --output outputs/my_token_run
python main.py token evaluate --output outputs/my_token_run

# 新图核心：输入已经冻结并校准的 measured_token_graph_v1 事件
python main.py graph score --input measured_events.json --output scored_events.json

# 原完整回答根梯度缓存的解释谱系
python main.py graph lineage --trace trace.npz --prompt-length 100 --output lineage.json
```

另外保留 `evidence prepare/capture/score/evaluate`、`baseline source-capture/js/mmd/score/evaluate` 和 `history capture/verify/evaluate`。`evidence` 是既有自动来源实验；`history` 含已暴露样本的机制诊断。它们不冒充新方案的全自动语义提案器。旧监督、图重构、固定局部邻域训练入口已删除；对应的原始结果不删除。

旧 fixed 基线保留历史 unit/window 聚合，仅作强对照。新核心没有把固定窗口风险平均作为逐 token 检测，也没有把 attention 重复、高秩或 hidden 重构误差当作真假。

2026-09-30 全测试重算完成：三个任务各900答，共424,408有效token。fixed 的 QA/Summary/Data2txt AUROC 为 .890665/.752177/.758040，独立token的 odds_full 为 .799441/.689958/.620336；后者完整span覆盖为0/235、0/244、6/1054。分数与历史逐值一致，无新模型前向、无test调参；不是新图结果。详见 [验证记录](docs/REFACTOR_VALIDATION.md)。

## 研究约束

评分与校准不接收自然真假标签；RAGTruth 标签按字符 span 评价，重叠 token offset 按字符并集计数。GSM 只评价有定义的首错/前步标签，不制造首错之后的真值。预测端口 `P+t−1` 与生成后载体 `P+t` 分开。所有原 token 保留；无事件、未对齐、参考池不足和未完成测量均显式记录。归因影响不等于事实支持，经验尾秩不等于幻觉概率。
