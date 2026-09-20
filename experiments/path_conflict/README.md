# 同题正负配对机制审计

问题：适用证据有没有控制最终选择，多 head 如何共同作用，起点的影响能否传到后半段。
完整[检测对象、文献与实验判据](../../docs/PAIRED_MECHANISM_20260920.md)。

```bash
python -u main.py flow
```

默认读取 reanchor 原重采样路径及其模型设置，输出 `outputs/paired_head_transport_v1`，
重复命令从逐 world NPZ 续跑。没有重新生成样本，也没有训练检测器。

1. 核对同题、seed、局部事实、来源角色和历史状态；当前两对不是整答正确/错误标注。
2. 从两侧 onset 的完整候选目标取得逐 head 梯度；冻结同一组物理 head，在两侧、各阶段复用。
3. 分别删除适用来源、不适用来源、history、self、更早历史、已输入子句及整个 receiver 写入。
4. 两剂量四世界干预；比较单独作用、条件作用和 J，不以梯度符号或 J 符号命名协同。
5. 删除上游后恢复下游 head/MLP；各候选分支使用自己的基线做 sham。
6. 只在起点干预后固定后续文本，检验后半段/子句后实际词概率的变化；不冒称语义恢复。

onset 保存完整核验候选对比；其他阶段保存实际下一词 logp，二者不能拼成同尺度事实曲线。
全部 confidence 序列另存熵、无 attention 的 EWMA 和 prefix-RAUQ-style 对照。
旧采样 NPZ 可能未保存 `logit_entropy`：相关分数记为未测（CSV 空值、NPZ NaN），
并保存 `entropy_status=not_saved`；已有 surprisal 和原生干预仍可计算。原命令可直接续跑，
无需补造熵或重新采样。top-5 logits 与 log normalizer 不足以还原全词表熵。

| 文件 | 职责 |
|---|---|
| paired_cases.json / paired_inputs.py | 可扩展的核验事实对、原始索引和来源角色 |
| paired_screen.py | 只读原 NPZ 的因果置信度对照 |
| paired_plan.py | 两侧共同 head、完整模型随机 head 控制和预先固定配对 |
| paired_trials.py | 单删、四世界、下游写回、起点影响传递及逐 world 缓存 |
| paired_report.py | 同题同组件差、source 汇总和旧 v3 CSV 重算 |
| paired.py | 主流程及命令入口 |

```bash
python -u main.py flow --stage inventory
python -u main.py flow --stage screen
python -u main.py flow --stage report
```

模型搬迁可传 `--model`，必须仍是原生成器权重；数据搬迁传 `--samples`。
增加已核验事实对用 `--cases`。更改候选、剂量、选头设置时使用新 `--output`。
默认 4 个梯度候选 head、1 个随机 head、.25/1 两剂量、最多 2 组下游恢复。
这是需要多次原模型前向的机制实验，不是一次轻量检测。

输出包括 `effects.csv`、`interactions.csv`、`adaptation.csv`、`persistence.csv`、
`paired_*.csv`、`confidence_controls.csv`、`REPORT.md` 和 `paired_review.tar.gz`。
每个子句的完整原生消息与 world 评分保留为 NPZ。缺测不补零，失败 sham 不进入配对比较。

## 旧结果和独立实验

重算上传 v3 CSV，无需 GPU：

```bash
python -u main.py flow --stage review --review-input <解压后的v3结果目录> --output <新结果目录>
```

旧目标消息边实验保留为 `python -u main.py flow-edges`，写回修正后输出
`outputs/target_transport_v4`。仅查看 v3 旧报告：

```bash
python -u main.py flow-edges --stage report --output <原v3目录>
```

v2 局部 lens 见 [FLOW.md](FLOW.md)，入口 `python -m experiments.path_conflict.flow`。
固定 head 实验 `python -m experiments.path_conflict.main --study focused`；
粗扫描 `python -m experiments.path_conflict.main --study coarse`。
监督 head 统计解释 `python -m experiments.charm_structure_audit.supervised_head_roles`，
其 head 编号不能跨模型直接对应本机制实验。

本轮已重算真实上传结果并做微型真实 Llama 软件验证；本地没有用户 8B 权重和原 attention
缓存，尚未跑新协议自然审计。两来源探索不能代替独立来源确认或全 RAGTruth 检测评价。
