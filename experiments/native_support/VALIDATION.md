# 本轮实际验证

## 外部成熟研究驱动的值路径分解（2026-09-23）

```bash
python -m pytest -q tests/test_transport_pipeline.py tests/test_transport_audit.py \
  tests/test_token_detection.py tests/test_native_support_evaluation.py \
  tests/test_native_support.py tests/test_dynamics_pipeline.py
```

结果：50 passed，14.97 秒。使用本地随机小模型，未运行服务器 8B 或全量自然数据。

- 原生候选 logits 保持；输入根有符号作用之和还原候选差；没有把多个层的作用重复相加。
- 实际 Mistral 滑动窗口注意力与保存行一致，禁止边为零；使用原生掩码而非猜测模型属性。
- 修改未来输入、分批候选反向、稀疏续跑不改变当前归因；关闭迭代器恢复钩子和参数状态。
- 块内正负分开后再求和，改变自动分块宽度不改变总来源作用或根读出。
- capture/score/evaluate 真正执行；修改标签不改变分数；完整缓存续跑不加载模型。
- 原始 R/A/H、旧预算审计及独立 dynamics 的受影响路径通过回归。

另行小模型精度检查：float32 输入根账本最大误差约 `6e-8`；一个两层 bfloat16
检查的最大误差约 `9e-4`，不是可外推到 8B 的误差界。完整反向比旧单 query
冻结 KV 的范围更大，FFN 重计算节省激活内存但增加计算；没有实测速度或 24GB 保证。
`capture/*/timing.json` 保存真实运行耗时与 CUDA 峰值，`ledger_error` 保存每词误差。

以下保留历史验证记录；其中退役实现与命令请查 Git `9d4ba17`，不代表当前入口。


## 固定分段的经验读出与新增 RAGTruth 入口（2026-09-22）

```bash
python -m pytest -q tests/test_state_readout.py tests/test_joint_state.py tests/test_native_support_evaluation.py
```

结果：22 passed。仅针对性 CPU 软件检查；无服务器 8B 或全量实验。

- 固定 posterior 下只平均已观察的 R；恒定 R 保持原值，独立状态退化为原始分数。
- 未来观测不改变过去；当前观测系数与保存的 E[1/n] 一致。
- v4 的分段及所有旧分数保持不变；禁止状态重算后仍完成 readout。
- 改变标签不影响新分数；readout/evaluate CLI 实际执行。
- 来源选择与标签、输入文件行顺序无关，参考/测试/已检查 source 不重叠。
- 使用随机小 Llama 和官方格式 fixture 跑通 prepare-only、前向、外部参考、评价、报告及断点复用。

这里的 fixture 不是 RAGTruth 自然数据。当前环境没有用户服务器模型/数据或四答缓存，
没有计算 joint_observed 的自然 AUROC。联合模型 0.756062 与普通均值 0.779200 来自用户报告。

## 多观测切换状态（2026-09-22）

```bash
python -m pytest -q tests/test_joint_state.py tests/test_route_filter.py \
  tests/test_native_support_evaluation.py tests/test_native_routes.py
```

结果：34 passed；ruff 与 git diff --check 通过。未运行服务器 8B 或全量数据。

- 共轭预测密度与 SciPy 多元 Student-t 独立实现一致。
- 因果递推与枚举所有可能分段的后验一致；新段/延续贡献精确还原总分。
- 修改目标未来不改变过去，改变辅助观测的单位并同步变换先验不改变推断。
- A/熵改变分段后验，但不能在 R 恒为零、参考均值为零时单独创造风险方向。
- 低熵持续高 R 不被清零；每词切换时严格退化为独立更新。
- 同源回答全部排除，参考 source 等权；外部参考检查 observer 身份。
- 真正运行 model/evaluate CLI；禁止原始缓存提取及模型加载仍完成小缓存评分与报告。
- 修改标签不改变分数，旧 v3 评价保留，缺少标签不覆盖有效评价。

合成输入仅验证数学和软件，不是自然机制证据。当前未取得四答 836-token 的原始/紧凑缓存，
新模型尚无自然 AUROC；0.779200 是用户已报告的普通均值基线，不是新模型成绩。

## 因果状态滤波与紧凑缓存（2026-09-22）

```bash
python -m pytest -q tests/test_route_filter.py tests/test_native_support_evaluation.py \
  tests/test_native_routes.py tests/test_native_support.py
```

结果：45 passed；ruff 与 git diff --check 通过。仅针对性 CPU 软件验证。

- 原始功能路由逐项等于 v2；head×region 联合分布使用同一层内消息范数预算，能还原底分。
- 状态恒定时滤波严格等于普通因果均值；改变未来分数或状态不影响此前分数。
- 一致重排全部时刻的头编号不改变结果；只改变当前头分配可以改变权重，合并头对照不变。
- 随机小 Llama 真实采集后运行默认 CLI，完成评分、评价和报告；缓存评分禁止加载模型。
- 改变标签不改变任何分数；紧凑缓存复用时禁止重新提取原始大缓存仍能完成。
- 旧 NPZ/v2 评价保留；缺少标注不覆盖有效评价；恢复排名排除不连续的观测位置。

当前环境没有四回答的 836 个原始 token 缓存，未运行新的自然 AUROC，也没有服务器 8B
或全量实验。0.754983 仍是用户已报告的功能路由基线，不能写成新滤波器的成绩。

## 路由恢复与统一比较（2026-09-22）

本轮命令：

```bash
python -m pytest -q tests/test_native_support.py tests/test_native_support_evaluation.py \
  tests/test_native_routes.py teaching/state_audit/tests/test_native_trace.py
```

结果：57 passed；ruff 通过。仅 CPU 针对性软件回归，没有全量实验或服务器 8B 前向。

- 消息范数而非平方能量；官方来源块和整个 prompt 分开，self/历史/特殊位置明确。
- 同源头与不同源头的 Gram 秩、已观察锚点结构、同一头同一窗口的有符号写入。
- source-disjoint fit/calibration/test；改变被测回答的未来观测不改变此前因果分数。
- 不足三个来源时缺测；pooled 与同答 AUROC 区别显式报告。
- 随机小 Llama 真正采集后复用 NPZ，禁止加载模型仍可比较和补评。
- 改变标签不改变分数，旧缓存/旧结果保持不变，缺标签不覆盖有效评价。

另直接提取历史提交 `f7344e2` 的 `_prompt_carriers` 与 `detect.py`，与当前实现核对：

| 核对输入 | 项目 | 实际差异 |
|---|---|---|
| 15 个合成来源、705 token | 完整历史 offline 校正分数 | 最大绝对差 `2.936155996113854e-08` |
| 用户已有机制审计的 40 个真实 NPZ | attention/功能消息的来源支持、头秩及 log-volume；锚点 ID 相同 | 最大相对差 `1.309162598772673e-06` |

40 个位置来自 14315/14375 的旧原生机制审计，**不是**当前 11907/12015/12045/12219
四回答的 836 个检测 token。当前环境未取得这四回答的原始 NPZ，未计算新版自然 AUROC。
历史 0.721235/0.712162 的原日志/文档已核对；约 0.7337/0.7333 仅保留会话报告来源，
不冒充此次复验。公式一致、账本闭合和软件回归不等于检测有效。

## 以下为前序批次记录

2026-09-22，CPU 软件验证，未运行用户服务器的 8B 或全量自然数据。

```bash
python -m pytest -q tests/test_native_support.py teaching/state_audit/tests/test_native_trace.py
```

结果：32 passed。新检测检查及共享采集钩子的既有回归均通过。

- 实际小型 Llama 原生 logits 与逐 token 缓存回放对齐，覆盖 float32/bfloat16。
- 禁止 `autograd.grad` 后仍能采集；没有参数梯度，异常后钩子清理。
- 当前目标 token 未进入自身预测输入；修改未来 token 不改变此前采集。
- 保存 attention、逐头逐源写入及 value energy；账本可还原最终 logit gap。
- 局部项加历史项严格组成总分；正向历史支持与反对历史分别处理。
- 新 prompt 支持能抵消旧风险；来源模式不匹配时不传播。
- 缩放写入不改变归一分数；FFN 正负不能直接成为错误标签。
- 远处 prompt 区域之间的读取增强可观察，首行的变化明确缺测。
- 断点补采与一次采集结果一致；标签仅在独立评价阶段读取。

另使用 teaching 的 `build_demo` 创建本地随机权重 Llama，通过真实命令
`python main.py support` 完成 6-token 采集、评分、NPZ、CSV、HTML 和 summary 写入。
最大绝对账本闭合误差为 `2.773035817638103e-08`。
这是入口与软件集成检查，不是自然样本成绩或幻觉机制证据。

默认真实档案只有 4 个自然前缀、322 个已观察回答 token。
本轮没有计算它们的新检测分数，也没有把候选文本拼成“真实错误续写”。

## 评价入口修复批次

命令：`python -m pytest -q tests/test_native_support.py tests/test_native_support_evaluation.py`。
结果：21 passed。未运行服务器 8B 或官方 RAGTruth 模型实验。

- 复现不存在的 `token_labels.json`，现在返回明确 unavailable 状态，不造正常标签，也不加载模型。
- 使用官方 JSONL 格式的合成 fixture 验证字符标注→token 标签，保留相邻 span 的不同起点。
- 改变同一回答的标签不改变模型输入；分层选择行为单独记录。
- 不同样本集合不能覆盖既有缓存；无标签状态不会覆盖之前有效的评价结果。
- 真正调用 run 入口，以随机小模型完成四回答的准备、采集、评分及自动 AUROC/AP 评价。
- 无需传 annotations 参数即可重新评价，且禁止模型加载时仍能完成。
- 特殊位置按有效 token mask 排除；单一类别不会输出伪造的 AUROC。

合成 fixture 的 AUROC 仅用于验证评价代码，不能作为自然检测效果。
