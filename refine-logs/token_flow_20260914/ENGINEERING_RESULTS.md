# Token 图实现与验证记录（2026-09-14）

本轮完成后验、目标条件化的 RoutingResidual 工程候选。自然检测有效性未验证。

## 实现

- 完整节点属性＋稀疏响应attention边，保留缺失质量，禁止未来边、矩形query错位与静默hidden回退。
- hidden要求冻结observer/tokenizer revisions、layer、state_location schema，跨来源一致；diagonal仅显式工程对照。
- 来源等权岭残差；graph、mass_matched_uniform、no_neighbors、permuted_weights四组。
- 来源块最大值尾部校准，5%预算少于19来源时无限阈值；不是正常来源FPR或token FDR保证。
- 输入/代码/环境/模型/评分hash冻结，评价token身份绑定、标签最后加入、不可评价覆盖、首错/首错后、来源内和配对bootstrap。

## 已实际执行的证据

1. research环境 `python -m pytest -q`：143 passed，25.89秒，actual exit 0。
2. `python main.py token-graph --help`：actual exit 0。
3. 单真实训练cache `10011.npz`，source14768：976节点，296响应token，1024维diagonal，84479聚合边；CSR1017656bytes，对应稠密float64阵7620608bytes；质量误差0，5.52秒；没有hidden。原始记录 `outputs/token_flow_engineering_20260914/real_cache_check.json`。
4. `python -m examples.token_flow_demo --output outputs/token_flow_demo_20260914_v2`：16参考、19校准、4测试来源；64合成测试token；四版本完整评分冻结后写标注再评价，actual exit 0。其检测指标只描述人为状态扰动，不是RAGTruth成绩。
5. v1示例保留；后续增加schema、配对统计和覆盖后由v2替代，v1不作为当前代码可复验冻结版本。

## 核心解释修正

query i的attention受token i影响，即使无直接z_i输入，也不满足历史滤过可测性；因此不称严格innovation或提前预警。attention diagonal与历史质量存在softmax闭合，不能作为语义主目标。Gaussian残差编码代价仅是工作模型解释，不是真假似然比。

## 未完成的科学证据

现有正式attention缓存不含研究版本必需hidden；没有新自然确认评分、自然AUROC或图必要性结论。正式hidden roster尚需按冻结采样规则采集并测峰值RSS。默认1024MiB是属性预算，不是总进程RSS上限。用户授权本轮合并main并push；发布工程候选不等于宣称方法已验证。

发布检查额外修复：requirements声明已有测试所需torch；CI先安装与本地通过测试的2.8.0同版本CPU wheel，避免默认CUDA依赖下载。此变更不改变评分算法或冻结示例。
