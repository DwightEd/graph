# RAGTruth全量通用机制：执行前协议

2026-09-12。用户明确授权开始全量机制实验。已有进展和未完成主线记录不变。

范围：RAGTruth全部17790已有回答（2965来源×6生成器），QA优先，随后Summary、
Data2txt；每回答全部token，包括首token。使用原始prompt与回答，BOS+分别分词；
本地Llama3.1-8B重放，因此是observer机制探测，不是这些原生成器的真实内部轨迹。
原生成器、task、原split、source_id、回答hash保留。数据适配只按这些字段与原文，
不得用质量标记、幻觉标签、是否有错误或位置来筛选。标签在独立评价阶段读取。

## 自动通用干预（各32层，对所有response预测query因果生效）

| 条件 | 操作 |
|---|---|
| base | 原始计算，保存逐层逐head来源/历史读取质量与分支范数 |
| sham | 同输入的零消息改动，完整O投影重放，检验数值不变 |
| source_01 | source的真实A×V消息削弱10%，不重归一化其他边 |
| source_1 | source真实消息全部减去；不等于删除全部来源信息 |
| history_01 | 距当前query至少16 token的response历史消息削弱10% |
| history_1 | 同组历史消息全部减去 |
| source_permute | 固定seed按response_id派生，在source内置换当前A的key端点，V保持；保留每head/query总来源质量和权重多重集合 |
| mlp_01 | response预测query的MLP实际分支输出削弱10% |

这是整段固定前缀逐query干预，会包含此前query中继变化与当步作用；不冒称单一
query局部作用。其他token仍保留，各query只可读其因果过去；history组排除self。
来源置换保持endpoint数量与source总质量；它不是语义合法/非法关系的真值标签。
baseline/topology同一计算协议，baseline原生top2在全部条件固定；记录完整词表JS、
原始回答token logp变化、原生argmax与并列数、top2候选差变化及熵，不把归一化差
当准确率。source_01/1与history_01/1可比较幅度非线性，但不代替导数验证。

baseline缓存逐层逐头source/remote-history attention质量、V/O源消息范数、MLP
更新范数；不是全高维属性图缓存。完整图仍使用sample_graph单独捕获。所有干预
真实重跑，不以候选隐藏投影或attention质量替代输出效应。高维V/O计算不平均head。

## 实现和数值验证

主算子graph/route_graph/population_mechanism.py；通用数据适配/断点调度与评价
在reanchor/src/decoding/ragtruth_population.py及相关模块。无新依赖/模型下载。
模型前向只取LlamaModel隐藏状态，逐块调用原LM head，避免为全部prompt存大词表
logits。不保留跨层完整attention矩阵，仅当前层用于实际消息运算。

每个回答进行sham与prompt侧不变性检查；未来response token不参与较早query。
先做tiny Llama精确sham/因果mask/实际干预测试，再真实GPU小批次和恢复运行见证；
随后后台全量。若OOM，记录该条失败并继续其余，不能截断或换条件后混为同一协议。
全部输入预分词，保存真实最大长度/超限清单；任一失败则总状态不能叫全部完成。

## 运行与保存

每response独立目录：原文与token/offset/source mask身份、逐token指标、baseline
profiles、设置与哈希manifest；临时目录完成后原子发布。已有完成样本按hash核验
复用，失败保留记录/临时结果，显式retry-failed才重试。拒绝在代码、模型stat、输入
或科学参数变化时继续旧目录。单进程文件锁防双写，不清理已有工作区/其他任务。

设置与进度在根目录，日志在根目录外，后台PID另存。每次成功/失败更新真实计数；
每任务/生成器结束可单独评价，最终全量评价按task/generator/official split分列。
数据集标签只用于这一步；按source聚合错误/正常效应及描述性排序指标，明确重放
协议与覆盖。不能由干预敏感性推断合法归属、幻觉概率或图检测已有效。

现有research@8d044d57环境，独立小批次（每任务2条，总6条）：

```bash
bash scripts/run_ragtruth_population.sh --output outputs/ragtruth_population_smoke_20260912 --smoke
```

完整任务使用新目录：

```bash
bash scripts/run_ragtruth_population.sh --output outputs/ragtruth_population_20260912
```

同设置恢复加`--resume`。独立科研审稿后端不可用，REVIEW_UNAVAILABLE，不伪造通过。
