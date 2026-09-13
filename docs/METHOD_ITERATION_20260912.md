# 2026-09-12 方法迭代：来源上下文兼容性与首错报警

本轮完成了文献核查、最小算子设计、实现、两轮机制实验、审查修正后的复跑，以及旧自然数据的恢复和报警评价。结果不支持当前算子的检测有效性或图必要性。保留实现作为可复现的诊断，不把它接入默认检测流程。

## 研究问题与文献边界

在不提供正确实体—属性关系、不使用幻觉标签拟合检测器的条件下，能否利用真实计算连接识别“取回了相关内容，但归属约束没有保持”？回看定位、关系读出、错误判别是三项不同的待验证能力。

- [Feng & Steinhardt, ICLR 2024](https://arxiv.org/abs/2310.17191) 已通过激活替换研究 binding ID；绑定敏感性和 patching 本身不是创新。
- [Representational Analysis of Binding, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.967/) 提醒绑定表征可能混入出现顺序。本轮增加顺序颠倒，保持事实归属。
- [CHARM](https://arxiv.org/abs/2509.24770) 已将 attention 与激活组成图并训练检测器；[TOHA](https://arxiv.org/html/2504.10063) 包括无需幻觉标签的 copying-head 选择。图、attention 拓扑或免训练不能单独承担新颖性。
- [Attention Deficits in Language Models](https://arxiv.org/abs/2602.19239) 在受控程序任务中区分绑定与输出读出失败。不能把已知候选、oracle checkpoint 或监督 probe 的能力迁移成自然生成的自动检测结论。

本轮只测试一个最小假设：候选在 source 中的过去上下文，与当前回答经历史中继读取的 source 上下文是否相容。它没有预先获得实体位置或正确关系。

## 实现与可否定的含义

实现位于 [route_graph/context.py](../route_graph/context.py)；机制驱动位于 reanchor 的 `src/decoding/binding_validation.py`。主层对固定为从零计数的 (15,16)，(30,31) 只作敏感性；窗口16、原生 top-4 候选、端点置换base seed=20260912，各条件按模板/布局/对象/世界/层的稳定SHA256派生seed。

令 q 为预测下一 token 的位置，A 为 head 平均后的 attention，S 为 source token，H 为最近16个已输入回答 token。候选 c 的 source 出现位置为 J_c，只按 token ID 精确匹配。

```text
K = source 中排除所有候选 token 身份后的 key
B_j(k) = A_(l-1)[j,k]，k ∈ K 且 k < j
T(k) = Σ_(u∈H) A_l[q,u] A_(l-1)[u,k]
G_c = Σ_(j∈J_c) A_l[q,j] cosine(B_j,T)
D_c = Σ_(j∈J_c) A_l[q,j]
risk_G = 1 - G_top1 / Σ_c G_c
```

G 是对直接读取 D 的上下文重加权，不是正确概率。它只使用非负 attention，缺少 V/O、MLP 和有符号贡献。端点置换对照随机重新配对 B_j，保持候选直接读取质量；这是读出的置换诊断，不是一次模型因果干预。词频对照使用 |J_c|，并列最大不计选对。

必须覆盖原生 top-1 且至少两种候选。任何已匹配候选的出现位置缺少过去上下文，G 都不可判定；无历史来源质量也不可判定。D、词频分别保留独立有效集，避免图方法的缺失掩盖对照能力。逐条件同时记录共同覆盖和候选级覆盖。

归档模块按完整回答保存 NPZ，校验输入身份、完整 token 流、候选、字符区间、字段白名单、四种表征的共同宽度及 residual=observed-null。字段白名单禁止把标签混进评分输入。报警模块绑定分数和校准所用归档哈希，只在分数冻结后接入标签；到每条回答首错当步为止计算报警。

## 自然数据：恢复结果与边界

旧文件 `outputs/routes_20260911_110359_256/features/features.jsonl` 保留原状。其体积为 8,525,150,267 字节；缺原始完成 manifest，无法确认层/路径布局和 checkpoint 哈希。每个 view 实际1920维、候选4，不能用维度反推配置。

计划256条中恢复217条完整回答、47,880个 token；1条只有109行，38条缺失。严格复核全部恢复行通过。恢复不等于完成原始捕获；下列结果均为探索性旧捕获基线，不能充当当前默认配置的正式结果。

按原 source 拆分：107个训练来源、23,223个 token；110条完整测试回答中，108条满足冻结的 k=3/cap=8 分层参考要求，共23,910个 token。另2条在长前缀分层中缺足够参考来源，整条排除并保存清单，没有按标签修改 k。首错回答59条。

| 分数 | 全 token AUROC | 首错子集 AUROC |
|---|---:|---:|
| 路径 residual | 0.3775 | 0.4361 |
| entropy | 0.5792 | 0.8442 |
| negative margin | 0.5697 | 0.7550 |
| position | 0.6380 | 0.3839 |
| conditional null | 0.5057 | 0.3215 |
| signal | 0.4814 | 0.4253 |

这些是来源等权 AUROC。首错子集沿用现有 evaluator：保留所有正常 token（包括首错后的正常 token）和每条回答的首个错误 token，因此不是在线报警指标。主残差方向预先固定，看到低 AUROC 后没有翻转符号。完整 AP、来源 bootstrap 区间和各子集见 [evaluation.json](../results/method_iteration_20260912/baseline/evaluation.json)。

## 严格截断到首错的报警

阈值来自训练来源留一评分的95%来源等权分位，不读幻觉标签；106个来源、22,951个 token 可校准，1个来源因同样的分层覆盖问题排除。冷却8 token，命中窗口为首错前8步至首错当步，实际观测15,664个 token。

| 分数 | 命中首错 | 召回 | 报警总数 | 未命中报警数 | 每回答至多一次报警的命中数 |
|---|---:|---:|---:|---:|---:|
| residual | 16/59 | 27.12% | 501 | 485 | 4/59 |
| entropy | 24/59 | 40.68% | 499 | 475 | 2/59 |
| negative margin | 21/59 | 35.59% | 529 | 508 | 2/59 |
| position | 2/59 | 3.39% | 53 | 51 | 0/59 |

Residual 相对 entropy 的配对来源召回差为 −13.56 个百分点，200次 bootstrap 的95%区间为 [−32.25, 3.39]；不能声称存在显著优势或显著劣势。单次容量诊断的差为 +3.39 个百分点，区间 [−5.13, 11.91]，同样不支持优势。两者实际发出单次报警的回答数分别93和105。

相同训练分位不保证测试中相同报警数；“至多一次”也只匹配容量，没有匹配实际报警数。后者是在工程审查后补充的诊断，不能替代预声明的等预算主比较。首错后 token 完全不参与以上报警统计；过早报警和无错回答上的报警均保留。见 [alarms.json](../results/method_iteration_20260912/baseline/alarms_v3.json) 和 [thresholds.json](../results/method_iteration_20260912/calibration/thresholds.json)。

## 机制实验与迭代

使用本地 Llama-3.1-8B-Instruct，bf16/eager，8模板×2对象×2 source 世界。两个世界交换数值归属，但输入 token 多重集、prompt长度、回答前缀不变。正确关系只用于构造和评价，读出入口没有正确候选参数。

第一轮96条件包括原始、source顺序颠倒、无关数值干扰。模型全部答对，主层 G/D/置换也全部选对，不能隔离连接必要性。因此追加32条件：将目标对象与某个固定值绑定的回答句重复32次，再预测数值；同一前缀在一个 source 世界一致、另一个冲突。不会因为强制前缀有错就给预测步贴错标签。

工程审查发现有效覆盖耦合、缺失候选上下文被记零和固定循环置换等缺口。第一轮修正在v2目录复跑。第二轮工程复审又发现每条件重置同一个seed，使同端点数恒用同一个弱置换；最终改为条件标识派生seed，在v3新目录完整重跑128条件。所有旧输出及执行代码保留，下表仅使用v3。256个读出中，3端点出现全部6种排列、4端点出现全部24种排列，平均跨候选归属移动比例为0.55534；不是强制每次移动。逐条件核对确认原生logits、G/D/词频支持与v2完全相同，只有置换及附加统计改变。

最终结果：**模型128/128正确，各方法128/128满足算子的token覆盖条件；没有实际错误预测条件。** 下表数值是读出支持最大的候选是否等于构造正确值，均以32条件为分母；不是幻觉检测准确率。统计独立单位只有8模板，每个模板的4个配对条件单独保留。

| 层对 | 布局 | G 上下文 | D 直接读取 | 随机置换 |
|---|---|---:|---:|---:|
| (15,16)，主比较 | 原始 | 32 | 32 | 32 |
| (15,16)，主比较 | 逆序 | 32 | 32 | 32 |
| (15,16)，主比较 | 无关干扰 | 32 | 32 | 32 |
| (15,16)，主比较 | 冲突前缀 | 26 | 27 | 26 |
| (30,31)，敏感性 | 原始 | 32 | 32 | 27 |
| (30,31)，敏感性 | 逆序 | 30 | 29 | 24 |
| (30,31)，敏感性 | 无关干扰 | 24 | 19 | 18 |
| (30,31)，敏感性 | 冲突前缀 | 23 | 19 | 17 |

词频对照每组0/32，可能选中高频空格等非数值候选；并列最大也不计正确。末层部分条件 G 的数值好于 D，但主层不成立、模型全正确，不能据此事后挑层或宣称错误检测/图必要性。

最终候选核对发现更强限制：**只有62/128条件同时把两种构造数值纳入原生top4**（原始5/32、逆序6/32、干扰28/32、冲突23/32）。例如top4可以是 `[15, 空格, x, 左括号]`，其两个source匹配项不代表两个竞争数值。因而原始/逆序的32/32大多不能说明数值归属识别。`scripts/inspect_binding_coverage.py` 用构造事实在冻结预测后核验此覆盖，未改变检测输入或分数；原始数字全部保留，附加结果为 `constructed_value_coverage.json`。完整条件、模板配对、输入和源码见 reanchor 的 `results/binding_validation_v3_20260912/`。

## 结论与下一轮约束

C1“无需关系输入的上下文路径读出更稳健且能识别错误”未获支持：主层无优势，冲突前缀出现读出失配，但模型实际选值正确。C2“图结构提高等预算首错召回”未获支持：旧捕获来源不完整，主分数没有优势，等实际预算比较仍未完成。可靠的自动回看事件定位本轮也没有验证。

下一轮应先检验观测是否足以表达关系，再设计分数：

1. 先保证候选覆盖有竞争的正确值与错误值，且候选产生不依赖正确关系标签；token数量不能替代这个检验。再新建、冻结未检查的模板和词汇集合，加入“同一对象、不同动作、不同数值”的配对；保持候选内容边际不变。先验证读出能区分动作归属，不能只认对象或数字位置。
候选产生的下一版可预先固定为：原生top-2，加上source中非特殊、解码去空白后含字母或数字的token身份里，按同一原生logit排序的前2个未入选项。只使用token/source元数据和logits，不输入正确关系；多token值和词形变体仍显式记为未覆盖。此方案仅是下一轮待测的候选生成协议，不已实现，也不能保证语义候选齐全。参数须在新模板运行前冻结，不能在当前128条上反复调到覆盖为止。

2. 用模型原生预测构成正确/错误评价；若仍没有实际错误，报告样本不具识别力，停止该错误检测实验。不能用强制错误前缀充当当前预测错误。
3. attention-only 算子先保持冻结。新的机制假设是：应保留候选相关的有符号 V/O 消息及当前对象/动作状态，才能区分“注意到”与“使用了约束”。先在固定前缀下做路径干预、direct/no-edge 对照和 sham，证明信息存在，再讨论自动读出。这是待验证设计，不是本轮实现或结论。
4. 通过机制门槛后，在新采集、完整 manifest、source 隔离的数据上测试。冻结预算策略、层、窗口与阈值；首错、分句边界、正常回看和错误续写分别报告。当前已查看的测试结果只能用于发现问题，不能用于选择下一版并继续当作盲测。

不继续在这128条上调层、扩特征、训练 AE/GNN，或把残差改号来制造正结果。工程审查与软件测试不能替代研究结论。外部研究审稿后端不可用，研究裁决标记为 `[pending Codex review] / REVIEW_UNAVAILABLE`，未宣称审稿通过。

## 复现

使用既有研究环境，不安装或替换模型。所有新运行使用新输出目录，原始结果保留。

```bash
# graph 仓库；路径按本机设置
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
py=/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python
"$py" -m route_graph.archive \
  --features outputs/routes_20260911_110359_256/features/features.jsonl \
  --input outputs/routes_20260911_110359_256/input.jsonl \
  --output outputs/recovered_reproduction
"$py" -m route_graph.archive_evaluation score \
  --archive outputs/recovered_reproduction --output outputs/baseline_reproduction
"$py" -m route_graph.alarms calibrate \
  --archive outputs/recovered_reproduction --output outputs/calibration_reproduction
"$py" -m route_graph.archive_evaluation evaluate \
  --scores outputs/baseline_reproduction/scores.jsonl \
  --labels /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/response.jsonl \
  --output outputs/baseline_reproduction/evaluation.json --bootstrap 200
"$py" -m route_graph.alarms evaluate \
  --scores outputs/baseline_reproduction/scores.jsonl \
  --calibration outputs/calibration_reproduction/thresholds.json \
  --labels /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/response.jsonl \
  --output outputs/baseline_reproduction/alarms.json

# reanchor 仓库，调用 graph 中的读出；单模板加 LIMIT_TEMPLATES=1
OUTPUT_DIR=outputs/binding_reproduction bash scripts/run_binding_validation.sh
```

大体积原始特征、压缩归档和逐 token 分数保留在服务器 outputs；小型结果包、逐条件机制结果及执行源码保存在两个仓库的 results 新目录。文件清单包含 SHA256；模型 settings 记录本地模型文件大小/mtime、包版本和执行代码哈希，不冒称重新校验了全部权重内容。

## 软件验证记录

graph完整测试39项通过，reanchor完整测试55项通过；新增代码ruff检查和格式检查通过，脚本bash语法检查通过。独立执行最终脚本的单模板smoke完成16条件。两个小型结果包的全部文件哈希、128条件的预声明组合及执行源码/案例哈希均核对通过；11项关键数字通过明确JSON路径的确定性核对。工程复审第三轮 `refine-logs/engineering_review_round3.md` 对已指出修复给出APPROVE，仅适用于工程范围；外部完整性审计状态见 `EXPERIMENT_AUDIT.md`，当前为未执行。

已有AUROC结果的输入/输出哈希由独立事后核对文件 `results/method_iteration_20260912/baseline/evaluation_identity.json` 记录，未伪装为原评价时生成的元数据。新评价入口会校验分数完成标记并写入分数、归档及标签哈希。报警v3补齐全部预先已有非图对照的配对来源区间，并核对评分k=3/cap=8；原方法点统计与旧报警报告完全相同。
