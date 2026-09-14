# S10 审查与无标签绑定投影实验

## 先区分当前已经做了什么

读取的 graph 基线为 `7a16396801128c834b9f711f20f26d0915c9a9b4`，reanchor 测量接口为
`6eece0406b1efe87548ebb686cbc8e730e65bea8`。本改动保留 S10 模型、特征、默认 main.py
和既有结果，不重新训练、不替换为未经验证的检测器。

当前活跃入口：`main.py -> structural_detector.experiment.run`。
特征在 `structural_detector/features.py`，学习、标签及评估在 `experiment.py`；
原始测量的导出在 reanchor 的 `src/decoding/structural_export.py`。
旧 P1–P7/图适配器入口属于 `archive/pre-structural-20260914`，不再是 main 的实现。

### S10 实际方法

`D[t] = remote_history_excl16[t] - source[t]`。
`M[t] = max(H[j] exp(-(t-j)/4)), max(0,t-8) <= j < t`，首 token 的 M 为 0。
完整 11 维：H、negative margin、source mass、remote mass、跨头来源质量标准差、D、
H 差分、D 差分、M、H*D、M*D。来源质量先在层/头上平均；这不是逐头因果消息图。

分别对 `error` 和 `onset` 训练 L2 逻辑回归（C=1），再在自然标注校准集上拟合
正斜率缩放及截距。没有根据预测 onset 切换到 continuation 模型；也没有独立的
continuation 训练目标。所谓 causal 只表示特征不读取未来，不等于因果干预已经验证。

**旧文档中的“首错 0.7791”实际是所有标注 span 起点，不是每答第一次错误。**
`targets()` 对每个标注 span 的首个重叠 token 都赋 onset=1。测试 150 答，98 答完全
正常，因此每答第一次错误只有 52 个；84 个唯一 onset token 对应 85 个标注 span。

来源留出 QA / 原 Llama2-7B 回答 / Llama3.1 observer 的已提交结果：

| 方法 | 全错误 AUROC / AP | span 起点 AUROC / AP |
|---|---|---|
| 原始熵 | .6000 / .1073 | .7099 / .0086 |
| 远历史位移（排除最近 16 token） | .6639 / .1169 | .6571 / .0095 |
| 瞬时监督组合 | .7405 / .1623 | .7515 / .0157 |
| 时序监督组合 | .7492 / .1661 | .7791 / .0159 |

时序相对瞬时的全错误 AUROC 增量 CI 为 [.003440,.017046]，AP 增量 CI 跨 0；
span 起点 AUROC 增量 CI [-.008985,.056687] 也跨 0。
冻结负 token 5%FPR 阈值后，正常回答至少一次误报为 error 85/98、onset 98/98。
这是有效的排序基线，不是无监督方法，也不是可靠的回答级报警器。
测试来源曾被历史研究使用；不要在这批测试标签上再选模型、方向或阈值。
详见原 `docs/S10_RESULTS_20260914.md` 和 `results/s10_structural_fusion_20260914/evaluation.json`。

## 1. 补齐现有分数的独立评估（不再跑 LLM）

在 graph 根目录、research 环境下：

```bash
python -m structural_detector.audit \
  --features ../reanchor/outputs/s10_structural_features_20260914_v1 \
  --predictions outputs/s10_structural_fusion_20260914_v1 \
  --annotations /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/response.jsonl \
  --output outputs/s10_scope_audit_binding_v1 --bootstrap 200
```

输出 `evaluation.json` 和 `complete.json`。目标分别是：

- `all_error`：全部错误 token 对正常 token。
- `span_onset_full_stream`：每个标注 span 起点对所有非起点，延续错误也在负类。
- `span_onset_vs_normal`：每个 span 起点对正常 token，排除延续错误。
- `first_error_full_stream`：每答第一次错误对完整流中的其他 token。
- `first_error_until_first`：每答仅保留首错及以前；正常回答全保留。
- `continuation_vs_normal`：所有 span 内非起点错误对正常 token，排除所有 span 起点。
- `strict_post_first`：第一次错误之后，错误 token 对恢复正常的 token。

同一组冻结预测在不同 gold 评价视图中复用。首错位置只用于评价 mask，不反馈到特征。
保留两种头的所有指标；预声明 onset 类比较用 onset__combined，其他用 error__combined。
报告 pooled / source-balanced AUROC、AP、正例率、覆盖率及来源配对 bootstrap。
不训练新头，不选择最高分头，不重新调阈值，不把 onset 概率称为已校准的首错概率。

## 2. 新设想：不混合节点，只约束联合对应

变量 z_i 的候选是**来源出现位置 ID**。输入每个变量的 log_weights；默认显式假设
`Q(z)=prod_i softmax(log_weights_i)[z_i]`。它不是 LLM 的真实贝叶斯后验。
有相关性模型时可提供 `joint_log_weights`，轴顺序必须与 variables 一致。

完整、明确的证据关系定义合法集合 Omega。核心为：

```
Z = sum_{z in Omega} Q(z)
D_bind = -log Z = min_{supp(P) subset Omega} KL(P || Q)
P*(z) = Q(z) 1[z in Omega] / Z
```

所有计算使用自然对数（nats）、logsumexp；零概率保留，不使用 epsilon 伪造支持。
这是一条精确的信息投影恒等式，不是率失真论文保证检测有效的推论。

不完整关系分为 allowed、forbidden 和 unknown；未列出的 tuple 默认 unknown。
仅当 `closed_world=true` 且用户确实提供完整关系时，未列项才是 forbidden。
联合 forbidden=任一因子禁止，supported=所有因子明确允许，其余 unknown。

```
lower = -log Q(possible)       # possible = 非显式禁止
upper = -log Q(supported)
```

`binding_exact_nats` 只在闭合关系下输出；不完整关系仅输出 lower/upper，exact 为 NaN。
无关系、不一致关系不伪装成正常/错误；不一致关系是输入模型失败。
NaN 表示不可用，inf 表示真实的零支持概率。NPZ 保留二者；事件 JSON 中无限量为 null
并附 lower_infinite/upper_infinite。缺包、缺关系的原 token 都保留在分母。

### 输入从哪里来，以及本次没有完成的部分

**本次实现的是完整可执行的绑定评分/对照/评价核，不是自动自然语言事实抽取器。**
现有 S10 只有五列标量，无法恢复具体实体—字段—数值—条件的关系。
必须另有 `binding-packet-v1`，给出实际来源候选、节点匹配和有来源依据的关系表。
不能把同句、同段、共有实体或邻接矩阵非零直接当成事实支持。

本次没有自动将全部自然 QA/Summary/Data2txt 编译成可靠关系表；不能把 demo 当成这个
步骤已解决，也不能把指定材料的人工关系版本混入“全自动无监督”成绩。
接口支持全部任务及完整的 reanchor population roster，但语义抽取覆盖率需另行验证。
这也是“核验关系有效、自动关系无效”与“关系投影本身无效”的区分位置。

### 立即验证核心（仅构造例，不是新检测成绩）

```bash
python -m pytest tests/test_binding_projection.py -q
python -m binding_detector.run --demo --output outputs/binding_projection_demo_v1
```

两个例子的节点熵/置信度完全相同；正确对应 D≈.0998203，错绑 D≈2.3538784。
完整示例在 `examples/binding_projection.jsonl`，可复制为自然来源的数据包。

每条数据包包含 schema/id/source_id/official_split/task、原 response 或其 sha256、
**完整原 token offsets**，以及非标签定义的 events。单个 event 示例：

```json
{
  "target_span": [6,11],
  "available_after_char": 11,
  "variables": [
    {"id":"entity","candidates":["recordA","recordB"],"log_weights":[-0.051293,-2.995732]},
    {"id":"value","candidates":["recordA","recordB"],"log_weights":[-2.995732,-0.051293]}
  ],
  "factors": [{
    "variables":["entity","value"],
    "allowed":[["recordA","recordA"],["recordB","recordB"]],
    "closed_world":true,
    "provenance":"source field/record relation; interpretation separately verified"
  }]
}
```

允许用已保存的原生向量代替 log_weights：event 填 representations=相对NPZ路径和
representation_space=observer/layer/head/state说明；变量填 query_array/candidate_array。
前者必须为 [D]，后者 [候选数,D]，同一空间。逐物理 head 独立提供；代码不平均 head。
可选 cosine_logits 仅为未校准的匹配基线，temperature 默认 .1，不是消息因果贡献。

单个事件按其 target_span 归属分数，不向相邻词传播；同一 token 的事件重叠直接拒绝，
应提交一个联合约束问题而非随意 max/mean。`available_after_char` 明示该证据何时可用。
若用到后续文字，这就是离线事实定位，不能作为事前首错预警成绩。

### 自然数据包就绪后的完整评价

```bash
python -m binding_detector.run \
  --cases /path/to/binding_packets.jsonl \
  --roster ../reanchor/outputs/ragtruth_population_20260912/inputs.jsonl \
  --annotations /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/response.jsonl \
  --split test --output outputs/binding_projection_natural_v1 --bootstrap 200
```

`--roster` 固定所有回答分母，无 packet 的样本/token 也保留；不填补伪造零分。
没有 roster 时，报告仅覆盖 packets 提供的队列，不称全数据结果。
输出每答 NPZ（全部 token）、events.json、records.json、prediction_freeze.json、evaluation.json。
标签文件仅在 prediction_freeze 已写入后打开；记录 original spans/语义来源，不训练任何模型。

### 与 S10 相同 QA 测试队列比较

```bash
python -m binding_detector.run \
  --cases /path/to/qa_test_binding_packets.jsonl \
  --s10-predictions outputs/s10_structural_fusion_20260914_v1 \
  --s10-features ../reanchor/outputs/s10_structural_features_20260914_v1 \
  --annotations /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/response.jsonl \
  --split test --output outputs/binding_vs_s10_v1 --bootstrap 200
```

S10 模式自动采用其完整 150 答测试名单，校验来源/response SHA/offsets，不重新训练。
先看相同可用 token 的配对比较，再看覆盖率，不能拿部分绑定可用集对完整 S10 成绩。
不将绑定分数与 S10 概率任意加权，也不声称得到了校准后的幻觉后验。

## 3. 对照与验收

保持 Q 不变，只对候选身份做组内置换，同一变量在所有因子中使用同一映射。
因子拓扑、表格基数保持；这不是随机“幻觉图”或可校准的 p-value。
同时报告 node_uncertainty、node_entropy、无约束常数、均匀候选的约束成本。
这些是新匹配分布的节点基线，不冒充用户旧表中 AP=.1252 的已训练节点读出。

先检验核验关系/自动关系是否分别超过无关系、置换关系、S10；条件匹配的增量是目标，
不是默认成立。参数与方向不能按旧 test 调整。没有自然增量前不加GNN/JVP/逐边追踪。

本地只执行数学、输入、冻结次序、覆盖及合成端到端软件测试；未跑新的自然 RAGTruth
或 8B GPU 实验。旧数据、新 demo 和未来自然实验三者的结果不得混用。
