# 本地数据与可运行范围

2026-09-12实际读取核对。父目录可直接访问：
`/share/home/tm902089733300000/a903202310/lys/data`。

新研究主线（高维属性图、跨层消息与合法约束归属）还没有通用的批量检测器。
sample_graph.py完成表示，ownership.py完成反事实干预；自动窗口/关系合法性评分
及跨数据集评价适配尚未完成。不能将旧main.py命令当成新方法的全量评测入口。

## 数据盘点

| 数据 | 本地内容 | 默认graph入口支持情况 |
|---|---|---|
| RAGTruth | 2965来源、17790回答、6生成器；QA989、Summary943、Data2txt1033来源 | prepare支持QA/Summary；evaluate使用原回答字符跨度标签；Data2txt未适配 |
| HaluEval | qa_data.json为JSONL，10000条；knowledge/question/right_answer/hallucinated_answer | 尚无适配；20000条正误答案应按问题/知识来源分组并用答案级评价，不能伪造token跨度标签 |
| BoolQ | dev.jsonl，3270条；question/passage/布尔answer | 尚无适配；没有原生成回答及幻觉跨度，不能把answer=true当幻觉标签 |
| CBUD | finite_field_predictive_alias/.../pilot_200中的200组有限域约束、状态与派生结果 | 构造机制样本，不是默认RAGTruth评测格式 |
| feature_extraction | 旧方法特征/结果缓存 | 必须按schema、模型、输入身份核对，不能直接冒充本轮高维图 |

当前本地模型还有Qwen3-8B，但旧capture只实现Llama候选投影，不能仅换MODEL_PATH
就宣称支持Qwen。现有默认probe为Meta-Llama-3.1-8B-Instruct，研究Python路径见下方。

## 可运行旧基线：RAGTruth QA的全部989来源，单一生成器

以下命令使用既有路径残差基线，不包含本轮自动合法归属读出。
数据为llama-2-7b-chat的989条QA回答；494来源用于无标签参考，495来源用于测试，
按source哈希分组（seed20260911），不是RAGTruth官方train/test划分。
Llama3.1重放Llama2回答，属于observer replay，不是原生成器内部轨迹。

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph

PYTHON_BIN=/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
TASK=QA GENERATOR=llama-2-7b-chat MAX_SOURCES=989 SEED=20260911 \
DEVICE=cuda:0 DTYPE=bfloat16 MAX_TOKENS=2048 MAX_ATTENTION_MB=8192 \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
bash scripts/run_route_evaluation.sh
```

脚本自动生成新输出目录与日志，成功后有COMPLETE、evaluation.json。
默认最终层、2候选、深度1/2；保留entropy/margin等对照。此前自然旧捕获中该基线
首错AUROC0.4361，entropy0.8442，因此这些命令用于基线复核，不代表方法已有增益。

本次只完成CPU数据准备与运行前检查，没有启动该全量GPU作业。
实际BOS+分开分词协议下，989条最大输入1033token，2048上限无超长；所有测试
长度分层均有至少3个参考来源。见results/local_dataset_readiness_20260912.json。
这是输入覆盖检查，不是对全量耗时/显存/最终运行成功的保证；旧extract逐token图
算子较慢且没有逐样本断点续跑，不提供未经实测的耗时承诺。

不要将MAX_SOURCES随手改成32/256当作同协议smoke：这两个子集分别有2条测试
回答缺足够参考层，detect会失败。也不要仅把TASK改成Summary后启动全量：
943条中21条超过默认2048token（最大2515），7条测试回答参考层不足。
这些问题没有被静默截断、丢样本或改阈值掩盖。

因此当前没有“一条命令测试新方法的所有数据集”。需要先完成自动关系读出，
再适配各数据集输入/评价粒度及可续跑的批量提取；不能用新机制结果包装旧命令。
