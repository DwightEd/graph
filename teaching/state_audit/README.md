# State Audit：生成、状态保存与审计

一个可单独复制、安装、运行的教学项目。只依赖本目录中的代码，不导入历史 `experiments`。

主线只有四步：**统一样本 → 生成回答 → 原样回放并存状态 → 离线审计**。
有现成回答时，从回放开始。原生干预是独立实验，通过重新运行模型检验消息对输出的影响。

## 先跑通

Python 3.10+。在本目录中安装：

```bash
python -m pip install -e '.[model,test]'
state-audit demo --output runs/demo
```

无需下载模型：demo 创建随机初始化的两层 Llama 和两条人工示例，执行真实 forward、状态保存和离线审计。
换成 `--family mistral` 或 `--family qwen2` 可演示另两种结构。随机模型不用于事实能力或幻觉检测评价。

按顺序打开：

1. `runs/demo/samples/000001/answer.json`：问题、回答、准确 token IDs、字符标注。
2. `runs/demo/audit/000001/tokens.csv`：每个回答词的预测位置、logp、熵和标签。
3. `runs/demo/audit/000001/heads_000.csv`：第一层逐 head 的读取质量和消息量。
4. `runs/demo/audit/000001/nodes.csv`：来源回看跃升候选；随机示例可能没有候选。
5. `runs/demo/audit/000001/roles_000.csv`：固定当前层输入的 key 内容交换实验。
6. `runs/demo/audit/000001/span_links.csv`：候选与标注 span 的位置关系及匹配正常对照。

`runs/demo/audit/summary.json` 汇总样本数、标注覆盖率及后续幻觉 token 占比。
它不是 AUROC 报告。完整字段见 [FORMATS.md](docs/FORMATS.md)，课堂顺序见 [LESSON.md](docs/LESSON.md)。

## 职责与阅读顺序

| 文件 | 只负责什么 |
|---|---|
| `pipeline.py` | 串起生成、采集、审计；建议先读 |
| `datasets.py` | 外部数据 → 统一 `Example` |
| `tokenization.py` | 模板、证据位置、生成 token 与字符位置的对齐 |
| `models.py` | 模型加载、原生层接口、最终读出 |
| `generation.py` | 新生成或录入已有回答；保存精确 token 序列 |
| `capture.py` | 安装 hook、回放、逐层落盘、移除 hook |
| `storage.py` | JSON / NPZ / CSV 文件读写与续跑契约 |
| `measurements.py` | attention、来源消息、候选节点的纯数组计算 |
| `roles.py` | 固定 Q 与位置，交换 RoPE 之前的 K |
| `annotations.py` | 观测完成后接入标签、匹配正常片段 |
| `audit.py` | 调用计算并写审计表 |
| `intervention.py` | 四世界消息删减与真实下游重跑 |
| `cli.py` | 参数与命令分发 |
| `demo.py` | 可离线运行的教学材料 |

没有训练器、注册器、模型基类或隐藏实验分支。纯计算不读文件，数据集适配不修改模型。
离线审计只需要 `pip install -e .`，不加载 PyTorch、Transformers 或模型权重。

## 使用真实数据和模型

### 1. 统一输入

每行一个样本。`prompt` 为完整的用户内容；`evidence` 是其中互不重叠的证据字符区间：

```json
{"id":"a1","source_id":"s1","prompt":"Mira wears blue. What color?","evidence":[{"id":"fact","start":0,"end":16}],"response":"Blue.","labels":[],"metadata":{"split":"test"}}
```

所有区间都是 Python 字符串的 `[start, end)`，不是字节。`response` 可省略；`labels: null` 表示未标注，`labels: []` 表示已审核且无幻觉。每条标签至少有 `start`、`end`，可保留 `text` 和其他原始属性。

RAGTruth 使用独立转换命令，保留官方 prompt、回答、标签、split、原生成器等信息：

```bash
state-audit convert-ragtruth --data /path/to/RAGTruth \
  --split test --limit 20 --output runs/ragtruth.jsonl
```

目录需包含 `source_info.jsonl` 和 `response.jsonl`。支持 QA、Summary、Data2txt；QA 按 passage 分来源，其余两类默认一个证据来源。证据必须能精确定位；无法匹配时明确报错，需提供统一格式中的显式区间，不猜测位置。格式依据 [RAGTruth 官方说明](https://github.com/ParticleMedia/RAGTruth)。

### 2. 一条命令运行

```bash
state-audit run --data runs/ragtruth.jsonl --model /path/to/checkpoint \
  --mode replay --output runs/audit --device cuda --dtype bfloat16 \
  --max-length 4096 --layers 18 20 22 --roles
```

支持 Hugging Face 的 `llama`、`mistral`、`qwen2` 三种 model_type；固定 Transformers 4.57.1 原生 eager 接口，单设备、单样本回放。默认采集所有层，head 始终保留原编号。
随序列长度变化的 dynamic/LongRoPE 会被明确拒绝：这种配置需要另外实现逐步采集，不能假设整段回放等价。

- `--mode replay`：在当前观测模型下读取已有回答；使用对应字符标签。这不声称恢复原生成器的内部状态。
- `--mode generate`：先生成新回答，再回放准确的生成 IDs；原回答标签清空为 `null`。新回答需重新标注。
- `--template chat`：使用当前 tokenizer 的 chat template；`raw` 原样编码输入，不自动追加 BOS 或其他包装。
- `--temperature 0` 为贪心生成；大于零时使用 temperature 和 top-p。样本实际种子为 `seed + 样本序号`，保存在回答记录中。
- `--max-length` 为显式资源界限，不做静默截断。长度应在所用 checkpoint 的支持范围内。

不同随机种子使用不同输出目录，以 `source_id` 关联同题回答；不能按文件位置假设正负关系。

### 3. 分步运行，复用状态

```bash
state-audit generate --data runs/ragtruth.jsonl --model /path/to/checkpoint \
  --mode generate --temperature 0.8 --seed 1 --max-length 4096 \
  --output runs/new_answers --device cuda --dtype bfloat16

state-audit capture --run runs/new_answers --layers 18 20 22

state-audit audit --run runs/new_answers --roles
```

生成和采集可加 `--resume`，配置及输入必须相同。成功样本的完成标记最后写入，中断的样本重新采集。
审计允许读取同一缓存反复运行；更换阈值或 `--roles` 时用新的 `--output`，避免混放不同实验。
状态缓存是完整数据，旧 attention-only 缓存不能补零伪装成包含 Q/K/V 的新缓存。

## 审计到底计算什么

设 prompt 长度为 P，回答第 t 个 token 的预测 query 为 **q = P + t − 1**。所有索引从 0 开始。
q 的 self key 是当前已输入的最后一个词，不能因为叫 self attention 就删掉它。

| 审计 | 计算 | 解释边界 |
|---|---|---|
| 来源读取 | 删除 tokenizer 特殊 key 后重归一化，统计每个 head 的 evidence / history / other mass | 最大来源不等于正确证据 |
| 消息载荷 | 用原始 attention 计算 `Σ_j a[h,q,j] V[h,j] W_O[h]` | 不对消息质量重归一化；来源是 key 所在区间，V 已包含上下文混合 |
| 消息方向 | 同 head 的 evidence/history 写入余弦；同层 evidence 写入的几何抵消比 | 相反方向不等于事实冲突，不等于下游抑制 |
| 回看候选 | 同来源 mass 超过阈值，且比过去 W 个 query 的均值增加到指定幅度 | 是来源回看跃升候选，不直接命名为已证实的重锚机制 |
| head 路由 | 保持 Q、位置、mask 不变，交换两组原始 K，再施加目的位置 RoPE | symbolic 只表示跟随内容，不能推出内容正确 |
| 标注关联 | 已冻结候选与 span 首词、之前窗口、段内位置关联；同答等长正常对照 | 标签辅助审计，不是无监督检测成绩 |

候选默认 `W=4, minimum_mass=0.25, minimum_rise=0.15`，只是公开的实验参数。
代码不使用未来 query、不强制每个幻觉前都有候选，也不把相邻满足条件的行自动合并为一个机制事件。
`nodes.csv` 同时给出 query token、预测 target token、head、来源 ID、该来源内最强 key token。

位置/内容交换借鉴 [Decoupling Positional and Symbolic Attention Behavior in Transformers](https://arxiv.org/html/2511.11579v1)。
教学版输出逐交换的两个 cosine，不做论文中的跨置换加权总分或频率因果实验。
每对来源最多取前 8 个普通 token，最多测 8 对；保存准确的两组 key 索引。
均匀注意力可令两项 cosine 同时很高，因此同时记录质量、对比度和 `identifiable`。
这只是当前输入上的探针，不是训练得到的通用 head 分类器。

## 从观察进入干预

```bash
state-audit intervene --run runs/demo --sample 0 --layer 0 --target 3 \
  --head-a 0 --head-b 1 --source evidence
```

同一层、同一 query，分别运行完整模型、删 A、删 B、同时删 A/B；另做零剂量 sham。
删减发生在 `o_proj` 前，只减所选 head 来自指定来源的原生 value 消息，不重新归一化 attention。
所有世界使用固定回答 prefix，继续经过原生下游层。

读出 F 是实际 target 的 **原始 log probability（nats）**：

```text
A 在 B 存在时的作用 = F(full) − F(without_A)
A 在 B 缺失时的作用 = F(without_B) − F(without_both)
interaction = 两个条件作用之差
```

结果在 `interventions/*.json`。有 interaction 说明本次干预的作用存在条件依赖；
它既不是“正确减错误”的事实 margin，也不证明这种协作是幻觉特有。需要独立设计正确/错误 claim 对和语义对照。

## 存储、验证和扩展

每个样本逐层保存 NPZ，JSON/CSV 可直接阅读；禁止 pickle。完整 attention 不做 top-k，未保存的层明确缺席。
模型计算中的特殊 token 保留，观测统计才排除它们；因此可以重建原生消息和执行可靠的删减。
采集检查 `A·V = head_readout`、`W_O·concat(head_readout)+bias = attention_write`；角色审计再检查原始 QK 重建。
`logit_entropy` 必定随读出保存，来自完整词表的原始模型分布，不是 temperature/top-p 后的采样熵。

全量 attention 每层存储量约为 `4 × H × T × S` 字节，S 是回放输入长度、T 是回答长度；
eager forward 还有 `S²` 的瞬时显存开销。W_O 每个采集层只保存一次。
这是一套便于检查的教学实现，长文本和大模型先选少量样本与层，不是多卡高吞吐框架。

```bash
python -m pytest -q
ruff check src tests examples
ruff format --check src tests examples
```

测试覆盖三种真实 Transformers 结构的小模型：生成/回放、GQA、QK/消息重建、无未来信息、标签隔离、续跑、hook 清理、sham、四世界效果，以及三类 RAGTruth 格式。
本次 CPU 验收：PyTorch 2.6.0、Transformers 4.57.1，37 项测试通过，Ruff 检查通过。
另外完成 wheel 独立安装，实际运行生成→采集→审计、已有回答回放、命令行消息干预和示例读取。
自然数据集上的机制效果与 AUROC 需另做实验，不能从这些软件测试推断。

扩展数据集只需转换为 `Example`；扩展模型先修改 `models.py` 与必要的采集接口，再加入原生重建测试。
含 Q/K 归一化、不同 RoPE、融合 QKV、交叉注意力的模型不能仅向支持列表添加名字。
以后添加 LDA 或无监督读出，应读取本项目的缓存，单独划分 reference/train/test，并按 `source_id` 防止泄漏。
