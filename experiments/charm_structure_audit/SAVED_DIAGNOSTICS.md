# 从已保存的 CHARM 预测查验高分来源

这是对 `charm_structure_audit` 现有结果的只读分析，不增加模型、训练目标、
图特征或报警规则。只加载 `samples/*.npz` 中的分数、标签、offsets 和原审计记录；
不加载 checkpoint、embedding、原 attention、LLM 或 tokenizer。

## 运行

在 graph 仓库根目录运行：

```bash
python -m experiments.charm_structure_audit.diagnose_saved \
  --predictions outputs/charm_structure_audit_qa/QA/seed_0/charm_out/test
```

默认写入上面目录下的 `saved_diagnostics/`：

- `diagnosis.json`：上下文分组、回答内排序、冻结控制的同覆盖比较及实际重连比例。
- `answers.csv`：每份回答的排序指标、正负例数量和首错分数。
- `onsets.csv`：所有唯一 token 起点（不是只挑高分例子），包括首错/后续起点、
  文本、原分数、是否报警、各已保存控制的分数与变化。

原 `report.json`、NPZ、校准阈值和续跑文件都不变。只分析中断前的最终 NPZ 时加
`--completed-only`；`.partial` 不计入，结果标记为预览而不是全量。
原始预测文件可以整体拷贝：manifest 内的旧服务器路径只用于取得样本文件名。
样本格式仍使用现有审计的固定 schema，不接受原 attention NPZ 代替预测。

## 四个明确问题

### 1. 是识别首次错误，还是回答出错后的状态？

`context_rankings` 的正例分别是 `first_error`、`later_onset`、`continuation`、
`nonfirst`、`all_error`。负例分别为全部正常 token、全正常回答的 token、
含错误回答中首次错误之前的正常 token、首次错误之后的正常 token。

每格打印正负数量、AUROC/AP，以及原阈值下的命中和误报。
这不是在线首报警实验；later onset 不等于前一个错误的因果后果。
原标注 span 与 token 起点可能多对一，因此分别记录 raw_annotation_spans 和
unique_onset_tokens；不把多个标注映射到同一个 token 当成多个分类样本。

### 2. 是回答整体区分，还是回答内部定位？

`answer_rankings` 同时报 pooled AUC、混合标签回答内的 macro AUC、回答内
正负配对数加权 AUC，以及不同回答之间的配对 AUC。恒等式为：

```
AUC_all * all_pairs = AUC_within * within_pairs + AUC_cross * cross_pairs
```

ties 计半分。全正常或全错误回答没有回答内 AUC，不补 0.5；报告覆盖数量。
回答内 AP 的阳性比例与 pooled AP 不同，不直接比较大小解释优劣。
回答多时跨回答配对天然很多，配对数量占比大本身不证明混杂或机制。

### 3. 冻结模型真的依赖哪些消息？

`controls` 自动读取现有的 `score_no_rr`、`score_no_rp`、`score_zero_edge`、
`score_zero_mark`、`score_shuffle_nodes`、`score_rewire_in`、`score_prefix` 等。
每个控制只在它与原分数共同非缺失的 token 上比较；再分首错、后续起点、续写。
同时输出相同子集的 baseline/control AUROC/AP，不能从子集 delta 扣全量基线。

对各类 token 保存平均有符号分数差、绝对差、裁剪到 [1e-6,1-1e-6] 后的
logit 差，以及在原阈值下丢失/新增的报警数。缺测不当成零分或漏检。
prefix 只覆盖旧审计的均匀位置和金标起点，不代表全部 token 的前缀表现。

注意原审计的条件：

- no_rr/no_rp 等冻结控制保留完整原图 divisor。charm_out 的 no_rr 仍含未来
  RR 读取次数，不是“完全不使用历史/后文”的模型。
- 删检测器的 prompt 边不等于删 LLM 的 prompt；缓存中的节点/边特征早已
  在包含 prompt 的上下文中计算。
- 重连保留边向量、目标入度和 RP，并在冻结预测时保留原出度 divisor。
  它主要检验历史源节点表征与边属性的具体配对，不是摧毁全部结构。
- 冻结控制是敏感性与分布偏移检查，不是独立重训的性能贡献分解。

### 4. 对照到底改动了多少？

`rewiring.changed_fraction_rr` 的分母只用 RR 边，另存全图分母，避免大量未动的
RP 边稀释对照强度。这里只检查已经保存的 seed，不生成新的图对照。
`adjacent_transitions` 描述 NN、NH、HH、HN 相邻位置分数的升降及报警切换，
不平滑分数，也不把同一标注 span 自动扩成预测。

## 验证与边界

```bash
python -m pytest tests/test_charm_saved_diagnostics.py -q
```

合成测试检查回答内/回答间配对恒等式、prefix 共同覆盖、起点角色、固定阈值、
原文件不被改写，以及真实 CLI 不导入 torch/transformers。
没有真实服务器预测时，不能声称已测得回答内 AUC 或各通道的作用。
本入口不做神经网络消息路径干预；如果需要解释某条边/哪个 head 的作用，
还须使用对应 checkpoint 在独立后续实验中检验，不能用相关统计冒充。
