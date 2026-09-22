# 本轮实际验证

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
