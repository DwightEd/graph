# Constraint inventory v1：独立文档执行见证

结论：文档要求的一次完整 CPU 构造实际完成，独立坐标与产物完整性核验通过。该结论只覆盖来源表示和执行；没有运行模型、读取幻觉标签或获得语义证书。

## 实际执行

依据 `docs/CONSTRAINT_INVENTORY_RUN_20260913.md` 原样执行一次，复用 `research@03909e02` 及本地 Python 环境；无安装、代码修补、GPU 调用、覆盖或重试。

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.constraint_inventory --inputs ../reanchor/outputs/ragtruth_population_20260912/inputs.jsonl --output outputs/constraint_inventory_v1_20260913
```

工具 session **72797**、CPU PID **172144**，最终 exit **0**。2026-09-13 13:26:51 China 启动，最终 manifest mtime 为 13:32:30.747。按 `/proc` 进程启动时间和该 mtime 计算约 **339.217 秒**；这是实际构造时长估计，非预估算力预算。独立只读审计 session **99882**，exit **0**，审计本身 **119.582 秒**。

## 全量分母

| 项目 | 实际数量 |
|---|---:|
| response refs | 17790 |
| 去重 sources | 2965 |
| QA sources / refs | 989 / 5934 |
| Summary sources / refs | 943 / 5658 |
| Data2txt sources / refs | 1033 / 6198 |
| train / test refs | 15090 / 2700 |
| fields | 41729 |
| contexts | 78158 |
| lexical components | 1241059 |
| provenance bundles | 90781 |
| containment / membership edges | 2722375 |
| unknown value owners | 3049 |
| 超过 40 词的完整字段 | 4083 |
| weekday source-key inventories | 621 |
| component mapping failures | 0 |

六个原生成器各有 2965 refs。未按任务、split、模型或源内容跳过样本。`model_forwards=0`、`semantic_certificate_count=0`、`labels_read=false`。

## 独立核验范围

独立审计脚本只使用 Python 标准库，不调用 inventory 实现来复算其自身结论。

- 17790 response ID 无重复，与冻结 settings 顺序一致；逐行核对 response hash、官方 split、source ID、source span 边界及其在完整 prompt 中取出的源文本。同一 source ID 的六个来源视图一致。冻结输入仅包含原始允许字段，没有 gold 标签字段。
- 2965 source 封装的 settings object digest、source 内容摘要、任务和原文均一致。逐字核验 **1360946** 个 field/context/component raw span；另核验 **48609** 个 literal AST 节点的路径、原始拼写和解析值。
- 对 **17896** 个字符串字段，用独立 lexical escape 分词和 Python decoder 重建 decoded-character → raw-source 映射；核验 **1319217** 个 context/component、**13775443** 个已记录字符映射。所有 **1079618** 个既定正则词坐标保留为组件。这是词坐标保留，不是事实单元完整性。
- 所有 scalar AST 节点都有 field owner；None 值仍标为 unknown。所有 bundle/member/containment 引用有效；所有 bundle 的 `query_requirement_verified=false` 且 `semantic_joint_fact=false`。
- 621 个 weekday inventory 各自对应实际来源中七个具名字段，成员与 literal-field 边一致。它们不证明回答里存在或适用 all-days 量词。
- **49** 个 live code 文件及 **49** 个 executed-code snapshots SHA 一致；settings、输入、**2967** 个 manifest artifacts 全部逐文件核验，**3018** 个输出文件全部有归属，无额外 partial 文件。审计前后输入及 live/snapshot SHA 再次一致。见证 agent 未修改任何实现或旧冻结产物。

## 已要求的具体来源核验

source **13717**：保留 34 fields，包括 6 个 unknown owners；三条 review_text 分别 **90 / 187 / 99** 词，全文均保留。address 的 `220` 与套间 `1` 分别保留为 address 子组件，raw span 为 **[40,43)**、**[63,64)**，decoded span 为 **[0,3)**、**[23,24)**。

source **14637**：保留 42 fields，包括 1 个 unknown owner，以及 **44** 词 review_text。两例各自的 weekday bundle 都有七个来源成员，query requirement 仍未验证。

此次全量实际 `mapping_failures=0`，因此不能声称真实数据验证了“unsupported literal 拼写”分支；该分支的保留行为属于工程测试范围。本次检查确认所有实际出现的精确映射及其完整父字段。

## 关键摘要与边界

- Document SHA: `4472aea012b7189445fefa9c7a2af3cc57f5adf6bf9896c400d30388571037f8`
- Input SHA: `be388df51e83f60beafbcb4075bde85da11a40f0941279e715f0f334e28d67bb`
- Settings object digest: `3ff9424d7ce4c67d666bdd67d7b774eebb1a5ec47cfea7c385eb41c4594b115c`
- Settings file SHA: `b7a5579a668d78a2dfc29bf17016fafb4a17f07185422238479dbce596a1e958`
- Manifest SHA: `b4766185732788d9c96452b185e12b952a70f45dd4e95a36378b7c5a56573b09`
- Summary SHA: `ef5224b171bac5352627fcc00f4cf7cb747d60e8847b410259001d0426cecded`
- Evidence SHA: `c8f2410830b4c0269942ac35e524b88e984c96870b2aad30a3acf65b85f7e16b`

机器证据：`refine-logs/constraint_inventory_doc_witness_evidence_20260913.json`。只读审计脚本：`/tmp/constraint_inventory_independent_audit_20260913.py`，其 SHA 保存在机器证据中。

未发现文档与本次实际调用/产物范围的偏差。该库存修复了候选表示的丢失问题；它没有给出自动语义 role、query constraint applicability、owner 正确性、回看节点定位、信息采纳或幻觉检测性能结论。不得把 raw span、词组件或七日 key 的覆盖率写成机制成功率。
