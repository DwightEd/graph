# 从路由竞争到确定：reanchor 分析边界与运行方式

## 当前判断

当前 graph 是候选条件的路径异常基线，还没有实现完整的 reanchor 事件检测与竞争收敛模型。一键运行产生自然数据检测评估，不能直接叫作 reanchor 机制验证。

从“筛选路由并确定”切入，核心应是观察：**候选状态已经可读后，哪些来源路径进入竞争、哪些被排除、支持是否稳定地集中到一个候选，以及确定过程是否失效。** 高 attention 峰提示某种读取行为，峰高本身不说明读到了正确候选或竞争已解决。

这与[事实召回的富集—提取阶段](https://aclanthology.org/2023.emnlp-main.751/)及[非事实幻觉的机制分析](https://aclanthology.org/2024.findings-emnlp.466/)相容，把其迁移为当前 reanchor 检测器仍是待验证假设。

## 三个必须分开的观测对象

| 对象 | 应观察什么 | 对 reanchor 的意义 |
|---|---|---|
| 候选可读性 | 同一候选在 source/history 状态中的支持代理 | 竞争前有没有可读的候选信息 |
| 路由分配及对齐 | source/passage/history 的读取质量，加上端点与候选的有符号对齐 | 是否重新分配读取、连到了支持哪个候选的端点 |
| 确定过程 | 固定候选坐标下随层推进的领先差、跨 head 冲突、赢家切换和持续性 | 竞争是否解决，是否不稳定或错误确定 |

错误确定可以很稳定、很低熵；稳定性或谱熵下降不能单独作为正确性判据。状态可读性也不等于真实事实已被完整编码，应保持 proxy 边界。

在同一预测位置 q 内，可固定候选集合，沿层分析筛选。若跨生成位置分析，应追踪相同来源／anchor 身份，不能直接把相邻位置不同词汇候选的排名变化解释成事实竞争。完整分析还需要早、中、晚层候选的并集，避免只看最终 top-K 的幸存者偏差。这些扩展尚未实现。

## 目前代码做了什么

1. 本地冻结 Llama 重放原始 prompt/response。预测 y_t 时只读取 q=prompt_tokens+t-1 及其之前的信息，包含首 token。
2. C_q 为 q 的最终 logits top-K。冻结 final norm 和输出 embedding 的 cosine 差构造 top-1 相对其它候选的节点信号 X_l。
3. 对 source、history、source-through-history 路径读取 P=(I+A)/2 的作用。前序层 head 等权平均，末步保留 head，沿真实层顺序传播。
4. N 在 role × source unit × log-lag 内均匀重分配原质量，self 不动。Q=(I+N)/2，保留 observed、null、residual=observed-null、无边的角色信号均值。
5. 来源隔离的参考按任务／生成器／已知长度匹配。共享尺度和有效坐标后，mean(z²) 为主分数，kNN 为补充；不使用幻觉标签定向评分。

强零模型检验“同组内连到了哪个候选端点”。**它固定 passage/source/history 之间的质量，而 reanchor 可能正发生在组间重新分配。** 因此不能单凭 residual 声称抓住 reanchor；observed/null 仍有价值，还需显式保留路由质量和 anchor 身份。

默认最终层；`END_LAYERS="7 15 23 31" CANDIDATES=4` 保留 Llama-3.1-8B 的多个终点层和更多最终候选，仍不会自动产生事件、候选淘汰轨迹或确定性指标。

## 评估是否完善

| 项目 | 当前状态 |
|---|---|
| source-disjoint 参考、标签仅评价时连接 | 已实现 |
| 全 token、span onset、response first error、continuation | 已实现，含 token 0 |
| 来源平衡 AUROC/AP、bootstrap、配对 AUROC 增量 | 已实现 |
| observed/null/signal、margin、entropy、position、kNN 对照 | 已实现；entropy 是输出分布熵，不是谱熵 |
| 参考无有效变化的覆盖报告 | 已实现；零分不表示可信正常 |
| 独立阈值校准、误报警率、首报警提前量／延迟 | 尚未实现 |
| 无标签 reanchor 事件定位与事件前后对齐分析 | 尚未实现 |
| 固定 anchor 身份、候选淘汰与竞争收敛统计 | 尚未实现 |
| 正确答案候选覆盖、两类失效的独立验证 | 尚未实现；生成 token 覆盖不能替代正确答案覆盖 |
| 多模型／任务／独立拆分、TOHA／谱基线完整复现 | 尚未完成 |

现有评估是分数排序基线，还不是完善的 reanchor 机制或在线报警评估。first_error 子集 AUROC 不等于“固定误报率下抓到首错”。

后续应先用无标签的路由重分配规则发现事件，再比较正常 reanchor、错误附近 reanchor、仅有高峰但无重分配的读取、无 reanchor 的错误。事件定义不能先要求冲突消退，再验证冲突消退。未来窗口只用于事后机制分析，不能进入当时的检测分数。

## 远端完整运行

新增 `scripts/run_route_evaluation.sh`：任意工作目录启动，自动进入仓库，执行 prepare → extract → detect → evaluate，保存完整日志，失败停止且保留现场，成功才写 COMPLETE。

代码位于 `origin/agent/graph-structure-audit`。在远端激活已有 GPU Python 环境，进入 graph 仓库后运行以下命令；需要切换到该分支才能取得本次重构：

```bash
git fetch origin &&
git switch agent/graph-structure-audit &&
git pull --ff-only origin agent/graph-structure-audit &&
python -m pip install -r requirements-model.txt &&
DEVICE=cuda:0 DTYPE=bfloat16 \
END_LAYERS="7 15 23 31" CANDIDATES=4 \
MAX_SOURCES=256 BOOTSTRAP=200 \
bash scripts/run_route_evaluation.sh
```

默认沿用 reanchor 仓库记录的路径：

- MODEL_PATH=/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct
- RAGTRUTH_DIR=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset
- TASK=QA，GENERATOR=llama-2-7b-chat；这是 Llama-3.1 对已有 Llama-2 响应的 observer replay。
- DTYPE=bfloat16，MAX_TOKENS=2048，MAX_ATTENTION_MB=16384。

2048 token、32 层、32 heads 的 float32 attention 存储估计约 16 GiB，不是总显存估计。实际 bf16 attention、权重、激活、logits 和暂存仍需空间。脚本显示设备空闲显存；尚未在远端验证显存是否足够。超长样本或参考条件不足明确失败，不截断或按标签筛掉样本。MAX_SOURCES=32 可检查小规模运行，但参考稀疏风险更高。

MODEL_PATH、RAGTRUTH_DIR、PYTHON_BIN、OUTPUT_DIR、DEVICE、DTYPE、MAX_TOKENS、MAX_ATTENTION_MB、NEIGHBORS、PER_SOURCE、DEPTHS、SEED 等均可覆盖；`--help` 给出全部默认值。

结果在 outputs/routes_<时间>_<pid>/evaluation.json；特征在 features，分数在 detection/scores.jsonl，日志在输出目录旁的 .log 文件。没有 COMPLETE 不能视为全部阶段完成；有 COMPLETE 也不代表方法有效。
