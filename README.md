# Unsupervised entropy seeds → local token reuse

## 单样本 token 图：无监督路由残差

本轮新增 `python main.py token-graph`，完整方法、数学假设及信息论边界见
[方法说明](docs/TOKEN_GRAPH_INFORMATION_FLOW.md)。主输入是完整 token hidden，
用 prompt/history 实际连接预测当前节点属性，按无标签来源参考残差排异常。
这是看到当前 token 后的路由条件异常，attention query 已受当前 token 影响。

```bash
python -m examples.token_flow_demo --output outputs/token_flow_demo_new
python main.py token-graph score --roster /path/to/roster.json --output outputs/token_flow_new --attribute hidden
python main.py token-graph evaluate --output outputs/token_flow_new --annotations /path/to/observer_token_labels.jsonl
```

包含 `graph`、同维同质量 `mass_matched_uniform`、`no_neighbors` 删除对照和
`permuted_weights` 诊断。评分无幻觉标签，来源隔离，输入/代码/预测冻结；
评价核对 observer token IDs，报告首错、首错后、回答内和覆盖指标。
现有 attention-only 正式缓存缺 hidden，只可显式 `--attribute diagonal` 做工程对照。
**自然幻觉检测效果尚未验证；图异常不等于事实错误。**
默认 `main.py` 仍为下方已有的局部复用流程。

默认 `main.py`：无标签参考熵 → 疑似入口 → 实际局部 attention 多跳复用。
不训练入口/延续分类器，不加载 S10/S11 监督权重，不用正常样本标签校准。
标签只在全部预测冻结之后用于评价。**尚无新自然 RAGTruth 检测成绩。**

## 一键运行

在 graph 根目录及 research 环境：

```bash
python -m pytest tests/test_unsupervised_reuse.py -q
python main.py unsupervised \
  --population ../reanchor/outputs/ragtruth_population_20260912 \
  --output outputs/unsupervised_local_reuse_v1 \
  --tasks all --generators llama-2-7b-chat \
  --device cuda:0 --query-chunk 16 --window 16 --resume
```

首轮每答一次 teacher-forced backbone 前向，按层分块重建 response-query attention，
保留全部物理 layer/head 的局部端点。模型与数据路径从 population/settings.json 读取。
之后复用逐样本 NPZ。完整上下文不裁剪；局部窗口仅限制被保存的历史边，不重归一化。
这是 observer 回放与 attention 依赖代理，不是原生成器的 WV/WO 因果消息追踪。

阶段：`--phase capture` 仅采集；`--phase score` 无标签评分；`--phase evaluate` 补评；
默认 all。无监督允许最后用金标检验效果，不允许金标参与种子、传播、选头或阈值。
旧 S11 的 `outputs/s11_local_attention_all` 与本版缓存结构不同，保留但不混用。

| 文件 | 用途 |
|---|---|
| reuse_detector/core.py | 无标签参考、种子、端点继承及同质量/距离对照 |
| reuse_detector/capture.py | SDPA 正常前向 + 只读 Q/K/RoPE 重放及数值核验 |
| reuse_detector/run.py | 全任务 roster、缓存续跑、混合校准来源异常预算、冻结 |
| reuse_detector/evaluation.py | 首错、span 起点、延续、停止评价，完整来源固定权重 |

[完整方法与边界](docs/UNSUPERVISED_LOCAL_REUSE.md)。评价重点比较 reuse 与 seed_only、
single_hop、mass_matched_uniform、lag_group_permuted；特别看 continuation_vs_normal、
strict_post_first 和错误结束后正常 token 的误报。高 attention 可能是纠正而非沿用，
该方法不具备语义否定/绑定判定。高置信首错没有熵种子时也会漏检。

## 旧监督基线保留，但必须显式选择

- `python main.py supervised-s11 ...`：入口+延续的监督读出，原参数不变。
- `python main.py supervised-s10 ...` 或 `python -m structural_detector.experiment ...`：S10。
- `python -m structural_detector.audit ...`：对已经冻结的 S10 分数补评，不重训。

旧命令 `main.py transport ...` 会停止并说明它是监督 S11，避免继续误运行。
[S10 结果](docs/S10_RESULTS_20260914.md) 的 .7492 是全错误、.7791 是**所有 span 起点**，
不是每答第一次错误；这两项都不是新版无监督成绩。

## 信息论相关的另一个原型

`binding_detector/projection.py` 实现显式关系下的 `-log Q(合法绑定)`，需要可靠来源匹配
和关系数据包；它不参与本默认流程，不能从五列统计中自动恢复事实图。
当前默认流程使用熵和经验尾部，不估计真假两类密度、条件互信息或贝叶斯幻觉后验。
率失真定理解释高置信碰撞的可能性，不为当前 attention 传播提供有效性保证。

历史结果与归档均不删除。原 S11 说明见 docs/S11_LOCAL_PROPAGATION.md；
其中旧 transport 命令需改成 supervised-s11 才会运行。
