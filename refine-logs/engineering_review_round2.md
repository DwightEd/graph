# 第二轮工程复审

日期：2026-09-12。范围：`route_graph/archive.py`、`archive_evaluation.py`、`context.py`、`alarms.py` 及对应测试，`reanchor/src/decoding/binding_validation.py`、运行脚本、机制测试、冻结输出和方法文档。审查只运行 CPU 检查，没有编辑正在执行的实验代码或使用 GPU。本报告是工程复审，不是外部科研裁决。

## 结论

结论为 **REVISE**。上一轮的归档字段白名单、跨回答宽度、原子 marker、候选上下文严格覆盖、各读出独立有效标志、标签区间检查、分数/校准归档绑定、首错截断、每回答至多一次报警和来源 bootstrap 均已落实。现有文档明确说明 C1/C2 未获支持、全流 95% fit 分位不等于测试集等实际报警预算，也没有把 128 个条件冒充 128 个独立模板。

没有发现会推翻“当前主张未获支持”这一负结论的问题。仍有一项会改变已执行置换对照数值的 Required 问题；在修正或降格该对照的解释前，不能把表中的 shuffled 列称作充分随机化的端点归属控制。其余问题不改变本轮已报告数值，但应在通用入口复用或提出 C1/C2 主张前修正。

## Required：当前置换对照在所有同端点数条件中重复同一弱置换

`route_graph/context.py:70` 在每次调用时重新构造 `np.random.default_rng(seed)`，而 `binding_validation.py:139-141` 对每个条件和层都使用默认 seed `20260912`。因此随机状态没有跨条件推进，也没有用模板、布局、对象、世界或层派生条件 seed。

当前 128 条冻结结果中，66 条有 3 个端点、62 条有 4 个端点。默认 seed 对 3 个端点始终产生 `[0, 2, 1]`，对 4 个端点始终产生 `[0, 2, 1, 3]`：前者固定最早端点，后者同时固定最早和最晚端点，所有条件都只交换中间两个端点。端点按 source 位置排序，因此这个控制系统保留相同位置，而不是在模板间独立地随机重配上下文。`tests/test_source_context.py:79-87` 只检查可复现性和它是一个排列，捕捉不到恒定固定点。

最小修复是从全局预声明 seed 和稳定的条件键（template/layout/subject/world/layer）派生每条件 seed，或由调用者持有一个固定 seed 的 RNG 顺序消费；同时记录跨候选归属实际移动的比例，并测试常见的 3/4 端点情况。若控制的定义要求每个端点都改变归属，应预先声明 candidate-aware derangement，而不是在看过本轮结果后选择一个有利置换。本轮原始结果必须保留为“固定索引置换”结果，修复后使用新输出目录。

这会改变 shuffled 列，但不会改变当前总判断：主层 G 没有优于 D，且 128 条原生 top-1 全部正确，所以 C1 仍不能通过。

## Required：冻结 AUROC 评价入口没有验证 score manifest

`route_graph/archive_evaluation.py:98-110` 的 `evaluate` 子命令直接调用 `RouteEvaluator`，没有读取相邻 `complete.json`，也没有验证 `complete["scores_sha256"] == digest(scores)`、`labels_used is False` 或 score 所属 archive。报警入口已经在 `route_graph/alarms.py:127-133` 做了这项检查，但全流 AUROC/AP 路径没有。

这不会改变当前数值：现有 `outputs/recovered_baseline_20260912/scores.jsonl` 的哈希与 `complete.json` 一致。但 `evaluation.json` 自身也不记录 score 或 archive 哈希，因而不能从产物证明“先冻结 scores，再接标签”。最小修复是在 `archive_evaluation evaluate` 调用 `RouteEvaluator` 前复用报警入口的 manifest 校验，并把 score、archive 和 label 文件哈希写入评价报告。

## Required before reuse with actual model errors：机制错误子集仍依赖四方法共同有效集

`binding_validation.py:237-249` 先用顶层 `readout["valid"]` 构造 `common`，该值是 context/direct/shuffled/count 四个有效标志的合取；随后所有方法的 `error_correct` 都只在这个共同集合中的原生模型错误上统计。这样，一个仅因 shuffled 无质量而无效的条件，也会从 G 对 D 的错误条件比较中消失，与独立有效集的接口目标不完全一致。

当前 128 条每个方法都有效且模型没有任何 top-1 错误，所以该问题不改变本轮数字。未来出现错误时，应分别报告全部实际错误数、各方法有效错误数，并为 `context vs direct`、`context vs shuffled` 建立各自两方法交集；不要让无关对照决定 G 与 D 的错误样本集合。`tests/test_binding_validation.py:29-59` 目前只覆盖总体方法有效数，没有覆盖这种交叉缺失。

## Required before any C2 claim：配对 bootstrap 未覆盖全部已计算的非图对照

`route_graph/alarms.py:279-301` 只生成 residual 相对 `entropy`、`negative_margin`、`position`、`null`、`signal` 的配对区间。冻结阈值和方法表还包含 `null_knn`、`signal_knn`；它们有完整的报警统计，却没有配对来源区间。因此当前函数不能机械地执行“相对最强非图对照”的判据。

本轮明确不声称 C2 通过，且已报告的 residual 对 entropy 比较不受影响。最小修复是显式冻结非图对照名称集合，至少包含 null/signal 的 direct 与 kNN 版本，并对同一来源重采样输出全部配对差；报告中再明确哪个对照是预声明的主对照。不要在测试标签上事后选择对 residual 最有利的对照。

## Optional：报警绑定还没有核对 k/cap 配置

`alarms.py:127-133` 已核对 archive hash、score 文件哈希和无标签标志，但 calibration metadata 没有显式保存 `neighbors=3`、`per_source=8`，评价也没有比较 score `complete.json` 中的这两个值。当前运行中 score complete 明确是 3/8，校准器也在 `alarms.py:42-44` 固定为 3/8，所以本轮数值一致。

最小加固是在 threshold metadata 写入 `neighbors` 和 `per_source`，评价时要求它们等于 score complete 中的值；或把完整 reference 配置规范化后保存一个配置哈希。

## Optional：归档仍可接受与捕获协议不一致的候选数和字符顺序

`archive.py:69-77` 只要求候选列表在单个回答内等长且至少有两个，不要求整个归档候选数一致或等于本轮声明的 4。`archive.py:79-103` 只验证每个字符区间自身有界，没有验证 token 顺序上的区间单调且不重叠。针对性构造证明，两个完整回答分别使用 2/3 个候选，或同一回答两个 token 使用相同 `[0,1]` 区间，都会被计为 complete。

这不影响现有恢复结果：独立扫描 217 个完整回答、47,880 个 token 后，候选数全部为 4，且没有回答出现重叠或逆序字符区间。建议把候选数作为 archive-wide invariant 写入 index，并验证相邻 `char_span` 单调不重叠，使以后无需另写扫描才能证明这些事实。

## 候选竞争覆盖边界已正确补充

算子的 `covered_candidates >= 2` 只表示至少两个原生 top-4 token 在 source 出现，不表示两种构造数值都进入 top-4。冻结结果的后验 evaluation 显示只有 62/128 条同时包含两种构造值：clean 5/32、reverse 6/32、distractor 28/32、misbound 23/32。其余 top-4 常由空格、标点或其他数字补足，所以原始/逆序的 32/32 多数不是关系竞争测试。

新增 `reanchor/scripts/inspect_binding_coverage.py` 在冻结预测后才读取构造事实，校验 cases 哈希和 expected token，不改变候选、读出或原始分数；输出 `constructed_value_coverage.json` 的 62/128 统计可复现。`docs/METHOD_ITERATION_20260912.md:79-102` 已明确区分算子 token 覆盖与两种构造值覆盖，并据此维持“不支持 C1”的结论。这一处理保持了检测输入的标签边界。

## 已验证事项

- 归档严格字段白名单位于 `archive.py:17-31,58-77`，跨回答宽度在 `archive.py:159-160` 拒绝；评分元数据改为白名单复制。
- settings 和逐回答 marker 使用 `atomic_json`；中断临时文件的回归测试通过。
- 候选上下文完整性在 `context.py:81-89` 严格检查；D 和 count 在 `context.py:84-93,108-113` 保留独立有效性。
- attention 只读取 `query` 及以前，显式拒绝上三角未来质量；历史和候选上下文只沿过去方向。
- 报警阈值只由 leave-one-fit-source 评分生成，test source 不进入参考；标签只在评价阶段接入。
- `alarms.py:169-180` 在首错当步截断再计算报警，首错后的任意分数不会改变命中、误报或冷却状态；budget-one 也在相同因果前缀上取得第一次报警。
- 来源级 paired bootstrap 已实现，报告同时声明相同 fit 分位和“至多一次容量”均不等于相同实际测试报警数；当前没有 C2 通过声明。
- 机制 settings 的当前 code SHA256 与 128 条输出目录记录一致；四布局各 8 模板、每模板 2 对象×2 世界完整，所有原生 top-1 均正确，因此 `error_detection_tested=false` 合理。

本次独立重跑的定向 CPU 测试为 graph 17 passed、reanchor 2 passed；未使用 GPU。主流程记录的全套结果为 graph 38 passed、reanchor 54 passed。
