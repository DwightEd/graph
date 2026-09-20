# Reanchor 原生审计

检验事实生成前的回看候选，是否读取了适用证据，并经更深层、跨位置的消息影响事实选择。[完整设计](../../docs/REANCHOR_AUDIT_DESIGN_20260920.md)。本入口用于标签辅助机制审计，不训练检测器。

## 运行

在仓库根目录、原 Llama 3.1 环境运行：

```bash
python -u -m experiments.reanchor_audit.run \
  --paired-input outputs/paired_head_transport_v1 \
  --output outputs/reanchor_audit_v1
```

输入也可以是原 `paired_review.tar.gz` 的解压目录：只要求 `paired_config.json`、`pairs/*/reviewed_case.json` 和两侧的 `onset/context.json`。不要求旧 attention NPZ 或 `logit_entropy`。模型默认采用原配对配置中记录的路径；路径迁移时用 `--model /原模型新位置`。模型必须是同一生成模型，使用本地权重、eager attention、默认 `cuda:0` / `bfloat16`。软件验证环境是 PyTorch 2.6.0、Transformers 4.57.1。

默认每侧 1 个早期回看候选，另加可匹配的低流、无正回看高流、事实位置直接读取对照；各测 `.25` 和 `1` 剂量、两个等范数随机方向。不把这些数值称为学到的边界或显著性阈值。`--events`、`--top-k`、`--doses` 可显式调整；改变设置用新输出目录。运行全程前台显示进度，同一命令复用已保存的图和干预世界。

先只检查输入，不加载模型：

```bash
python -m experiments.reanchor_audit.run --stage inventory \
  --paired-input outputs/paired_head_transport_v1 --output outputs/reanchor_audit_v1
```

`--stage discover` 运行原生图采集和候选筛选，随后用相同设置 `--stage run` 继续有限干预。报告可单独在 CPU 重建，无需模型或原配对目录：

```bash
python -m experiments.reanchor_audit.run --stage report --output outputs/reanchor_audit_v1
```

## 测量及解释

1. **路径候选**：保留 layer/head/position；先反向算目标可达势，再前向传播条件流。主分析用势加权根注入，另保存均匀可达根、消息范数容量对照。`receiver < decision_query` 的回看才作为早期候选，事实位置直接读取单列对照。
2. **所读内容**：选择之后才挂接人工核验的 scope、supported_value、value_source。来源角色不等于物理 K/V，也不由 attention 大小定义。输入角色 token 重叠会中止，不能把相交集合当独立消息。
3. **原生作用**：删除完整 `W_O(A V)` 子消息，不重归一化 attention；保存单独、条件和联合效应。恢复后续消息 B 测其在入口 E 被切断时的条件作用。
4. **执行控制**：消息 sham、完整状态 sham 和 E+B 联合 sham 分别真实执行。完整状态恢复必须回到原分支的基线；恢复增益的开关差与四世界交互是同一个量，不能作为双重证据。

图是筛选代理，省略的边不重新分配质量。图中的 MLP 只保留同位置的层序连接；干预世界则完整执行原模型的 MLP/norm。非负路径流无法辨认支持、抑制或事实适用性。干预分数为两条固定候选的完整序列 log-probability 差；保存首词、尾部、总和及均值，不能把长度和不同属性槽位影响解释为幻觉机制。

头饰两侧前史/措辞不同；洋葱两候选是低火与时长，尚不构成同属性的干净语义对照。固定载荷的关系反事实及 Q/K/V donor 实验仍需单独核验和执行。

## 审阅输出

- `review.html`：候选精确 token 上下文、来源角色、逐 head 的原生时间曲线及干预表。
- `coverage.csv` / `summary.json`：未测图、缺失中继、匹配失败、空来源、实际/预期行数、失败控制分别记账，计数不会取反成负数。
- `effects.csv` / `role_interactions.csv`：全部已测入口、后续消息及 scope/value 联合作用。`support > 0` 只表示该消息提高此处的候选概率差。
- `cases/*/*/routes.npz`：原始稀疏 attention、来源索引、消息范数、保留比例、完整 attention 计算的回看增量；`flow.npz` 可在 CPU 从该图重算。
- `worlds/*.json`：每个干预世界的候选逐 token 读出；对应 NPZ 供缓存续跑，不能用审阅包代替完整缓存续跑。
- `reanchor_review.tar.gz`：回传此包即可审阅，包含 `routes.npz`、上下文、所有世界读出和图表；不包含大型激活缓存或模型权重。

## 验证

```bash
python -m pytest -q tests/test_reanchor_audit.py tests/test_target_transport.py \
  tests/test_paired_transport.py tests/test_paired_report.py
```

测试使用随机初始化的真实微型 Llama，仅证明软件的计算/干预语义，不代表自然语言机制已验证。上传样本的实际输入覆盖记录在 `results/reanchor_audit_preflight_20260920/`，该记录明确没有新原生图或有限干预结果。
