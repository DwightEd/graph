# 数据与索引契约

## 回答记录

`run.json` 保存模型地址/版本、配置、软件版本、生成参数、输入文件摘要及样本索引。
`samples/000000/answer.json` 是每个样本唯一的 token 对齐依据：

| 字段 | 内容 |
|---|---|
| `id` / `source_id` | 回答 ID / 同题来源 ID |
| `prompt_text` | 实际模板后的文本 |
| `prompt_ids` / `prompt_length` | 实际输入及长度 P |
| `response_ids` | 实际生成 IDs，或已有回答在观测 tokenizer 下的 IDs |
| `token_ids` | prompt_ids + response_ids，含生成的 EOS |
| `response` | 可读回答文本 |
| `response_offsets` | 回答内字符区间；无法严格验证时为 null |
| `token_strings` | 全序列的 tokenizer token 表示，不等于整词 |
| `key_sources` | 每个 prompt token 对应证据编号；−1 表示其他 prompt 内容 |
| `evidence` | 原始 prompt 内容内的来源字符区间和 ID |
| `special_token_ids` | tokenizer 定义的所有特殊 token |
| `labels` | 已有回答的原始标注；新生成回答为 null |
| `seed` / `mode` / `metadata` | 实际随机种子、生成/回放模式、数据集信息 |

生成 IDs 绝不会被重编码的结果替换。重编码只验证字符 offset 是否与原 IDs 一致；不一致就记缺失，停止字符标签关联。EOS 等特殊 token 的回答 offset 为 `[0,0)`，不算正常或幻觉词。

`--template chat` 在真实模板内定位未变形的 prompt 内容，再映射来源区间。落在区间边界的 token 按字符重叠归属；若同时覆盖两段证据则拒绝含糊映射。

## 每层状态

对 T 个回答词，只需输入前 S=P+T−1 个 token。保存的 query 是 `[P−1, …, P+T−2]`。
H 为 query head 数，K 为 KV head 数，d 为 head 宽度，D 为 residual 宽度。

`samples/000000/trace/layer_000.npz`：

| 数组 | 形状 | 含义 |
|---|---|---|
| `queries` | T | 预测 query 在完整输入中的位置 |
| `attention` | H × T × S | 原始 attention，包含特殊 key；未来位置为零 |
| `query` | H × T × d | RoPE 之前的 Q |
| `key` / `value` | K × S × d | RoPE 之前的 K / 原生 V，保留 GQA 的 KV 头 |
| `cosine` / `sine` | S × d | checkpoint 在该次 forward 实际使用的 RoPE 张量 |
| `attention_bias` | T × S | 实际因果/滑窗 mask 的加性 bias |
| `scale` | 标量 | 原生 QK 缩放 |
| `head_readout` | T × H × d | `o_proj` 输入重排后的逐 head 读取结果 |
| `attention_write` | T × D | attention 输出投影后的写入，含投影 bias |
| `residual_before` / `residual_after` | T × D | 整个 decoder block 前/后的 residual |

Q/K/V 是上下文化状态，不是独立词义。所有层号与 head 号保持物理编号；KV 头展开为 query 头的映射为 `kv_head = head // (H // K)`。

`trace/readout.npz` 保存 `queries`、最终归一化后的 `final_hidden[T,D]`、`target_logp[T]`、`logit_entropy[T]`。完整 vocabulary logits 按 32 个 query 一批计算，避免全序列 logits 常驻；不保存整个词表向量。

`weights/layer_000.npz` 保存 `output_projection[D,H*d]` 和 `bias[D]`。同一 run 每层一份。
NPZ 中的浮点状态转为 float32；这不把原先 bfloat16 计算变成 float32 推理。

`trace/complete.json` 最后写入，包含采集层和消息重建误差。失败时不会写成功标记；没有标记的样本可以用 `--resume` 重采。输入或采集配置变化必须换目录。模型地址对应的本地 checkpoint 也应保持不变。

## 离线结果

| 文件 | 用途 |
|---|---|
| `audit/settings.json` | 公开的窗口、阈值、角色探针开关 |
| `audit/000000/observations_000.npz` | 逐 token/逐 head 浮点观测；来源质量额外有来源轴 |
| `heads_000.csv` | 可读的逐 head 观测表 |
| `nodes.csv` | 每个满足阈值的 token/head/source 候选；不按标签筛选 |
| `roles_000.csv` | 每次 key 交换的 before/after、cosine、质量、对比度、原生重建误差 |
| `tokens.csv` | token 级读出和最后接入的标签 |
| `span_links.csv` | 金标 span 与同答正常对照的候选关联 |
| `summary.json` | 覆盖、数量和连续标注比例；没有伪造的检测分数 |

`evidence_mass/history_mass/other_mass` 是删除特殊 key 后的归一化概率；`ordinary_mass` 记录删除前保留下来的总量。
`evidence_write_norm/history_write_norm` 来自未归一化的原生消息，不混用两个量纲。
零向量余弦和零普通质量观测记为 NaN，CSV 中为 `nan`；未知标签为留空/null，不能当作负例。

`evidence_head_cancellation = 1 − ||Σ_h m_h|| / Σ_h ||m_h||` 存在 observations NPZ 中，形状 T。
它只是同层写入的几何抵消摘要，所有原 head 观测仍保留。

`nodes.csv` 中 `target=t`，`query=P+t−1`；`key` 是被读取的输入位置。
`span_links.csv` 的 `nodes_before` 严格使用 target 小于 span 起点的候选，首词同位置单列为 `node_at_onset`。
因此不能把同一首词上的关联计为提前检测。候选从 t=W 才能计算，首词需到 t≥2W 才有完整的前置 W 个可测 query；此前 `pre_onset_observable=False`，同时保存实际可测 query 数。没有证据来源时也不声称可测。

RAGTruth 字符标注投影后，重叠 token 区间合并，相邻而不重叠的区间仍分开。
特殊 token 不计入标注统计；若它把区间隔开，则分别保存普通 token 的连续区间。
后续标注比例为 `(幻觉 token 数 − token 映射 span 数) / 幻觉 token 数`，分母为零时 null。
这是标注结构统计，不代表有监督成绩中相同比例已被连续性解释。

正常对照在同回答内匹配长度、首 token 表面类别、相对位置差 ≤ 0.25、token 重复率差 ≤ 0.15、此前是否出现错误；观察邻域不含错误，正常区间不复用。
两侧必须都有完整前置计算窗口；早期 span 不强配给已经完成窗口预热的正常区间。
没有匹配时记为未匹配。当前教学版不匹配语法、语义或熵；结果不能解释为完全控制后的幻觉特异性。
