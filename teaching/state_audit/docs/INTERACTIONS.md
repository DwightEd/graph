# 从定点采集到条件交互

## 轻量观察

`capture_targets(model, {name: Target(...)})` 只复制选中的坐标，返回 `states[name][layer]`；
全局站点的 layer 为 None。一个 Target 可以跨多个层，各层的身份分别保留。
装载次序为 intervene → capture_targets → model.forward，才能观察干预后的真实状态。
原来的 capture_sample/run 与 ModelState 仍负责整段回放和逐层落盘。

```python
from state_audit.capture import capture_targets
from state_audit.operations import Target

targets = {"selected": Target("head_readout", (12, 16), positions=(30,), heads=(0, 2))}
with capture_targets(model, targets) as states:
    hidden = model.forward(token_ids)
print(states["selected"][12].shape)
```

## 消息与读出

MessageSite 是某层、某组物理 head、若干绝对 query、一个可选来源集合。
keys=None 表示整头；指定 keys 时计算 `attention[..., keys] @ value[keys]`，保留各头。
GQA 映射由 ModelAdapter.head_layout 给出，输出投影由 ModelAdapter.project_heads 提供。
消息写入不含 o_proj 的共享 bias，不能把共享 bias 分摊成各头贡献。

```python
from state_audit.experiments.messages import MessageSite, measure_messages

site = MessageSite("source_a", layer=12, heads=(0, 2), positions=(30,), keys=(4, 5, 6))
scores, messages = measure_messages(model, prefix_ids, [candidate_a_ids, candidate_b_ids], [site])
changed, _ = measure_messages(
    model, prefix_ids, [candidate_a_ids, candidate_b_ids], [site], [site.deletion()]
)
```

只有固定 prefix 内的状态会被采集。候选 continuation 分别 teacher-force；score_contrast 复用 score_targets，
输出逐 token logp、sum_margin、mean_margin。库只定义候选 0−1，不知道哪一个事实正确。
实验层可以用审核的 preferred 标注方向，但必须保留固定候选方向以比较反事实条件。
较短候选在末尾填充未评分的 token，使两次前向的矩阵形状相同；因果 mask 阻止尾部影响评分位置。
填充不计入候选长度、logp 或均值，减少不同 GEMM 形状导致的前缀数值差异。

messages[name] 中：

- readout / write：当前 A·V 来源分量及其各头 W_O 投影。
- total_readout / total_write：选中头的实际总状态，包含人为替换或注入。
- mass：指定来源的原始 attention 质量；整头模式未另外测来源质量，返回 None。
- attention / values：来源权重及按 query head 展开的 GQA value，用于原生消息恢复。

来源消息恢复使用 `ReplaceSource`：在当前完整 A、V 的副本中替换指定来源的权重和值，
按原生 dtype 重新计算完整 A @ V，只把结果写回选定 query。其他来源保持当前值，不重归一化。
不同 query 单独计算，不会修改共享 GQA value 后污染其他 query 或 head。
donor 来源数不同时先清空目标来源权重，按来源顺序占用选定槽位；较长 donor 使用额外外部槽位。
槽位只用于反事实消息计算，不是新发现的 attention 边，也不改变已观察的原始路由。
因此 readout/mass 仍是当前路由的观测，实际替换体现在 total_readout/write；参照应来自未替换的运行。
所有向量按位置、头、特征保留，不用相加后的范数替代各头数据。

0.3.0 的 `Inject(reference − current)` 依赖分别舍入的来源和总状态可相加还原。
上传的 Llama-3.1-8B BF16 数据证实该假设不成立；0.3.1 保留 0.01 nats 控制阈值，修正计算路径。

## 条件与恢复

`analysis.interactions.factorial_effects` 输入明确的 F11/F01/F10/F00；
1 为原生，0 为预先定义的删除或 donor 替换，不能混用两种参照。

    A_with_B = F11 − F01
    A_without_B = F10 − F00
    interaction = A_with_B − A_without_B

不是 Shapley 联盟平均。负交互也可能是冗余/饱和，不自动命名为抑制。
same-layer heads 并行计算；跨层恢复才检验上游改变后的下游适应。

`restore_messages(evaluate, sites, reference, operations, current)` 按层恢复。
evaluate 是普通函数，接收操作列表，返回 `(scores, messages)`，可直接闭包调用 measure_messages。
每恢复一层就重新前向，因此下一层的 current 包含之前的恢复。
不能对多个相互依赖的层一次注入从旧运行计算的静态差，声称它们都恢复到了基线。

    total_effect = F(delete_A) − F(base)
    fixed_downstream_effect = F(delete_A, restore_B) − F(base)
    adaptation = total_effect − fixed_downstream_effect

adaptation 的符号需要结合 fixed_downstream_effect；正值本身不等于“发生补偿”。
若它反向抵消删除造成的变化、且总偏离变小，才支持所选站点起到缓冲作用。
这是指定干预下的功能分解，不是证明该组件在未干预时唯一必要。

## 有意保留的边界

源位置与 head 编号的合法性、干预不触及候选未来位置、引文的精确对齐、续跑配置、mask 不能打开，
都可能改变科学结论，因此在输入/模型边界验证。内部张量形状交给 NumPy/PyTorch 报错。
新模型只扩展 model 适配，新的来源角色只扩展输入，普通删除/替换仍用 operations。
