# 首次事实选择、回看与竞争解决

## 恢复来源与证据

本说明恢复自本机另一会话「pull graph最新版代码，里面的reanchor_flow/message_dag写得有点乱…」，会话 ID `01a083e6-49fc-7922-8b05-00efe68f2293`。原实现可在 Git 提交 `68fafdc`、`244491c`、`a2a40fd` 中追溯。本次仅保留它的 onset 分析与必要读取代码，删除旧 control_graph 和已失败的 factorial／whole-head 分摊评分。

原会话用户提交的 688 来源汇总已原样保存在 [ATTENTION_AUDIT_688.json](ATTENTION_AUDIT_688.json)。这是历史结果，不是本次新路径算子的结果：

| 评价位置 | 正例数 | attention displacement AUROC | negative observed margin AUROC |
|---|---:|---:|---:|
| 全部幻觉 token | 20,123 | 0.7122 | 0.5861 |
| span 内续写 | 19,453 | 0.7151 | 0.5808 |
| 严格 N→H 起点 | 667 | 0.6331 | 0.7078 |

续写占全部幻觉 token 的 96.7%。因此全 token 指标很容易被续写主导。旧起点计数不是每个回答的首次错误；另有 3 个正例不在旧 onset/continuation 两组中，不能强行使计数闭合。汇总数据不足以重建具体实体、逐事件关联或三者的因果顺序。

旧 constraint displacement 的 AUROC 约 0.55；把整个 head 的 margin 按无符号来源质量分配，会混淆同一 head 内不同来源的作用方向。这条评分捷径不再保留。

## 应继续的研究问题

核心问题是：在同一个事实选择事件上，候选竞争是否与远处信息重读同时发生，重读后竞争是否得到解决？

1. 将回答首次标注错误与后续 span 起点分开；正常事实选择采用同回答、相近位置、相同粗 token 类型的对照。
2. 在预测目标前的窗口中记录 evidence、far history、local+self 的质量变化，以及集中／分散变化；保持目标位置身份。
3. 对同一位置联合分析候选竞争与回看。不能从两项独立 AUROC 推导二者相关，也不能先取 source 均值再声称解释了单个节点。
4. 固定候选实体／答案身份，比较读取前后候选优势的变化；多 token 实体最终需要序列候选概率，不能用相邻生成位置不同词汇的 top-1 排名替代。
5. 进一步区分：没有回看、回看后候选优势恢复、回看后仍冲突、稳定地选择错误候选。这些是待检验模式，不能预先作为真值。

潜在 insight 是“重读到的信息能否进入并稳定事实选择”，而不是“出现高 attention 峰就正常”或“低 margin 就幻觉”。现有实现仍没有正确答案候选、实体绑定、固定 anchor 轨迹或读取后的正确候选恢复量，因此不能声称已识别知识富集失败／答案提取失败。

## 当前图怎样使用结构

当前 `route_graph` 的节点是 layer × token 状态，候选对比是节点信号通道。设 A[i,j] 为接收位置 i 读取来源 j 的权重：

```text
z[start,i,c] = role_mask[i] * candidate_signal[start,i,c]
message[l,h,j→i,c] = A[l,h,i,j] * z[j,c]
z_next[i,c] = (z[i,c] + sum_j message[l,h,j→i,c]) / 2
```

前序层平均 heads，末步保留每个 head，按相邻层顺序传播。两步 source-via-history 在中间施加 history 掩码且排除 query 自环，最后读出目标 q。连接到不同候选支持节点、经不同中继或改变层序会影响响应；测试包含同边权／同特征值的重连和层序倒置。

这是固定、线性的消息传递，没有新训练的变换矩阵。P=(I+A)/2 是分析约定，不是原模型 AV/W_O/MLP 的精确前向重建。结构测试证明表征依赖连接，不保证平方距离标量在每次重连后都不同，更不证明真实检测收益。

零模型在 role × source unit × log-lag 内均分原质量，保留 self，提供 observed/null/residual 对照。它保留组间质量，可能消除 reanchor 的组间重分配信号，故 residual 不能单独承担事件检测。当前 readout 汇总了来源端点；下一步需要保留高贡献端点及中转位置，而不是添加更多抽象类别。

## 已恢复并完善的低成本实现

入口：`main.py onset-audit` → `OnsetChoiceAudit(config).run()`。

- `onset_analysis/traces.py`：读取已有主 NPZ 的 attention 组质量、普通 token 质量、熵、top1 集中度、observed margin、预测熵；标签单独读取。
- `onset_analysis/analysis.py`：合并 H span，取第一个字母／数字 token，寻找正常对照，测量预测窗口内最大 remote gain 对应的观察。正常窗口包括计算首个差分所需的前一预测位置。
- `onset_analysis/statistics.py`：分别给出 first_error/later_onsets；报告原始 span 结构和未匹配覆盖；保留每个事件，以每个 source 等质量的加权秩相关衡量选择与回看关联，并按 source 重采样。一个来源只能提供描述性相关，不提供来源 bootstrap 区间。

首 token 缺少上一预测位置时，不虚构回看差分；计入未覆盖首错，后续错误不替代它。`focused_far_history` 只表示远历史读取代理，不声称该 token 已被证明是 relay。峰值窗口长度会影响最大值，需要预先固定并做窗口敏感性检查；结果仍是关联而非因果效应。事件级相关也未消除所有任务、难度和来源混杂。

`instability = -observed_margin` 中 observed_margin 是数据中 token 相对最强替代项的优势。若轨迹来自别的模型，它测量观察模型对该 token 的兼容性，不能当作原生成模型的信心。新图主流程的 negative_margin 则是负 top-1/top-2 gap，两种指标不是同一个量。

## 运行

在 graph 仓库根目录、已安装 requirements.txt 的 Python 环境中：

```bash
bash scripts/run_onset_choice_audit.sh \
  experiments/reanchor_flow/outputs/attention_audit_v3 \
  "outputs/onsets_$(date +%Y%m%d_%H%M%S)" 3 64 200
```

无需 GPU 或重新前向；audit 路径可替换为现有 index.json 所在目录。`events.jsonl` 保留匹配双方的 token 位置／文本、span_start、是否回答首错、margin 代理、回看模式及变化量。`summary.json` 给出联合统计。此流程不是新的图检测分数，也不接收 route_graph 的 features.jsonl。

完整图流程继续使用 `scripts/run_route_evaluation.sh`。采集显示样本进度、当前前向、逐 token 算子进度及 checkpoint 记录进度；参考拟合、评分、评价也显示进度，stderr 会随脚本 tee 保存到日志。

## 本次核验

删除旧功能专属测试后，现存 21 项测试全部通过，包含真实 tiny Llama 的完整脚本和进度日志验证，以及首次错误不可被后续错误替代、正常差分基线覆盖、同来源多事件保留。执行了 `python -m pytest -q --show-capture=no --tb=short`，模型测试使用 transformers 4.57.1 的本地隔离环境。Ruff 检查和格式检查通过，现存三个 shell 脚本逐个通过 `bash -n`。

另用真实子进程检查 onset shell：从仓库外运行、输入输出路径带空格、成功输出 4 个匹配首错并显示进度。Windows 检查进程显式使用 UTF-8，避免父子 Python 编码不同导致读取进度文本失败。未重新运行 688 个自然数据来源，未运行远端 GPU；历史汇总与本次软件测试分别报告。
