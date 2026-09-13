# SourceRel-Mini M0 独立文档执行见证 — 2026-09-13

Fresh agent `/root/sourcerel_data_doc_witness` 先阅读 `docs/SOURCEREL_DATA_RUN_20260913.md`，再严格按其中唯一命令执行一次。执行 session **7762**，最终 **exit 0**；输出 `outputs/source_relation_data_v1_20260913` 的 manifest 与 summary 均为 `complete`。没有重试、覆盖、安装、GPU 模型加载或实现文件修改。

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.source_relation_data --inputs ../reanchor/outputs/ragtruth_population_20260912/inputs.jsonl --output outputs/source_relation_data_v1_20260913
```

随后用独立标准库校验脚本进行只读验证，session **12338**，exit **0**；核验时间为 **2026-09-13 04:43:06 UTC**。

| 项目 | 实际数量 |
|---|---:|
| 全部 annotation-free 输入行 | 17,790 |
| official train / Data2txt 输入行 | 5,298 |
| 去重来源 / 成功编译 | 883 / 883 |
| 内部 source_train 来源 / queries / candidates | 720 / 5,760 / 26,203 |
| 内部 source_validation 来源 / queries / candidates | 163 / 1,304 / 5,964 |
| 合计 queries / candidates | 7,064 / 32,167 |
| 有独立同值负例的 queries | 4,309 |
| 有另一 record 同字段负例的 queries | 1,429 |
| 每来源选定 query 数 | 全部为 8 |
| query + candidate 文本数 / 精确去重文本数 | 39,231 / 38,541 |
| model forwards / reader calls | 0 / 0 |

Query 类型计数：time_range 1,132，date 665，number 888，text 1,533，boolean 2,846。最长 query / candidate 分别为 4,213 / 5,183 **字符**；这些不是 tokenizer token 长度，完整 tokenizer 长度预检尚未执行，编码前仍必须完成。

核验结果：

- 逐行确认冻结输入恰好为 14 字段 allowlist；sample ID 无重复，全部 response SHA256 正确。只从 official train / Data2txt 选来源，没有读取 RAGTruth 标注文件。
- 独立提取的 883 个 source ID 与 settings 完全一致；每个 ID 在全量输入中仅属于 official train。按固定 seed / source ID SHA256 modulo 5 重新计算的内部划分完全一致；内部训练与验证的 source ID、完整 source 文本均互斥。
- 全部 883 个 source 数据 seal、source graph seal 和真实 dictionary root 节点通过；source 文本摘要与原输入一致。所有 query 候选池、非空正例、非空负例、同类型限制、候选 ID、8-query cap、未选 query 分母检查通过。
- 独立重算的完整 summary counts 完全一致。
- 原输入文件 SHA256、49 个 live code 文件、49 个 executed_code 快照、884 个 manifest artifacts 全部通过；935 个输出文件均被 settings / manifest / artifacts / snapshots 覆盖，没有未计入文件。
- settings、summary 的 `labels_read=false`、`model_forwards=0` 与 settings 的 `reader_calls=0` 一致；各来源的 `factuality_labels_used=false`、`native_effects_used=false` 均通过。

| 摘要对象 | SHA256 / canonical object digest |
|---|---|
| input file | `be388df51e83f60beafbcb4075bde85da11a40f0941279e715f0f334e28d67bb` |
| settings object | `ae68860d95ca3e4aa66ac730af2d01826268556ed2dc86a5d5104e0f7167d6bf` |
| settings file | `7777c6c4dff422c7d60278594ec0e598ab43eae61d7ad8ae4f95972e27975220` |
| summary file | `dd5647a0b2a2fc4966f300c18ec8dd2e3317861e9a2f942134903246a7aab00b` |
| manifest file | `821ce51544c44462bc2521e85fe808809c0c6b7fa2d0282e4fb6248027d40ec6` |

机器可读证据：`refine-logs/source_relation_data_integrity_evidence_20260913.json`，SHA256 `852853d73a6ccc8d9ff808cf290c3b1c7bc9a4acb40a38837d913cc9a7aeeee3`。

本次只证明原来源字段坐标构造的 **source pointer reconstruction 数据**已完整生成并通过完整性核验。没有训练模型，没有测得自然回答证据归属准确率、幻觉检测指标、自动回看节点准确率或任何 native 因果认证。后续需先完成不截断的 tokenizer 长度预检，再实施计划中的 frozen feature 编码及带同视图基线的训练/验证。
