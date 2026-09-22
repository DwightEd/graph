# State Audit

一个独立的教学与机制研究项目：同题重采样 → 保存内部表征 → 比较回答 → 干预并重新运行模型。
组织单位是模型表征，不预设只有 evidence/history 两条消息，也不把任何审计结果当检测器。
本目录可以单独复制和安装；不导入仓库里的其他研究项目。

## 安装与最小闭环

```bash
cd teaching/state_audit
python -m pip install -e '.[model,test]'
python -m state_audit demo --output runs/demo
python -m state_audit check --run runs/demo
python -m state_audit pair --run runs/demo
python examples/intervene.py --run runs/demo
python -m pytest -q
```

`demo` 用本地创建的随机权重小模型，不下载 checkpoint；样例标注是教学构造。
默认 Llama，也可 `--family mistral` / `--family qwen2`。它验证软件行为，不提供自然数据结论。

## 职责

| 位置 | 负责什么 | 不负责什么 |
|---|---|---|
| `dataset/` | Example、JSONL、RAGTruth 转换与字符区间核验 | 模型推理、自动判断事实真假 |
| `model/` | 原生模块与表征轴映射、forward、读出 | 数据集解析、特征设计 |
| `state.py` | ModelState / LayerState，逐层读取与绝对位置查询 | 把所有状态同时装进内存 |
| `capture.py` | 选择表征、观察、逐层保存、移除 hook | 写入干预、挑幻觉 token |
| `attribution.py` | 一个目标的逐 head/来源有符号敏感度、消息能量 | 用真值拟合方向、跨层相加成因果总量 |
| `operations/` | Target、Delete、Replace、Inject、Steer、ReplaceSource | 固定两个 head、固定来源类别 |
| `intervention.py` | 将操作临时接入真实模型执行 | 规定具体研究假设 |
| `generation.py` | 同题多次采样、原回答回放、身份与 seed | 继承旧答案的幻觉标签 |
| `pairing.py` | 导入审阅结果，按同题配正常/错误回答 | 把未标注回答当正常 |
| `analysis/` | 纯数组比较、reanchor/角色/标注审计 | 加载或改变模型 |
| `experiments/` | 固定回答的条件比较、两组干预交互 | 充当全部干预能力的接口 |
| `storage.py` | JSON/NPZ 与完成标记 | 用缺省值掩盖缺测 |
| `cli.py` / `pipeline.py` | 参数与流程编排 | 数学公式 |

先读 [架构与接口](docs/ARCHITECTURE.md)，再按 [教学顺序](docs/LESSON.md) 阅读代码。
具体轴和文件契约见 [FORMATS](docs/FORMATS.md)。

对一个已保存回答 token 计算显著性式贡献：

```python
from state_audit.attribution import capture_target_attribution

result = capture_target_attribution(
    model, answer["token_ids"], answer["prompt_length"], target=3,
    layers=(8, 12, 16, 20), special_token_ids=answer["special_token_ids"],
)
```

只输入目标之前的前缀，目标为实际 token 相对其余词表的 log-odds；
`contribution[layer, head, key]` 是消息 gate 的一阶导数，保留正负。
`aggregate_sources` 按调用者给出的来源分组，排除特殊 token 后汇总，不平均 heads。
每个目标单独反传；参数和 hooks 在结束后恢复。这是目标敏感度测量，不是真伪概率。
历史 KV 默认以 256 token 分块、无梯度预填充，只为最后预测位置保留梯度图；
`prefill_chunk_size` 控制计算块大小，完整前缀仍参与读取，不是截断历史。

## 数据与重采样

任意数据集先转成统一 JSONL。最小一行：

```json
{"id":"q1", "source_id":"source1", "prompt":"What does Mira wear?", "evidence":[]}
```

`evidence` 可选，为 prompt 中的 `{id,start,end}` 字符区间。
回放已有回答时附 `response`、`labels`；`labels: null` 表示未审阅，`[]` 表示已审阅且无标注错误。
其他信息进入 `metadata`。没有证据也能采集 hidden/MLP；来源 reanchor 分析则不适用。

```bash
python -m state_audit convert-ragtruth --data /path/to/RAGTruth \
  --output data/ragtruth.jsonl --split test

python -m state_audit generate --data data/ragtruth.jsonl --model /path/to/checkpoint \
  --output runs/resampled --device cuda --dtype bfloat16 \
  --samples 4 --seed 10 --temperature 0.8 --max-new-tokens 128 --max-length 4096

python -m state_audit capture --run runs/resampled --layers 12 16 20 \
  --representations attention query key value head_readout attention_write \
  residual_before residual_after mlp_write final_hidden
```

生成时按相同 source、prompt 和 evidence 合并已有回答，避免 RAGTruth 同题的多条原回答触发重复重采样；
保留全部 `parent_ids`。每题使用 seed 10、11、12、13，回答 ID 与 draw 单独保存。
`--mode replay` 则逐条回放原回答，不合并。
采集默认只保留回答预测位置；加 `--scope all` 也保存 prompt 内部状态。`run` 是 generate + capture，分析使用独立 `audit` 命令。

可给不同 checkpoint 和数据集各自的输出目录；[resample_models.py](examples/resample_models.py) 提供一个直接的双循环，
每个模型只加载一次。新增数据集只需产生 Example；不改模型或采集代码。

重采样不保证一定产生正负答案，也不自动核实事实。审阅 JSONL 每行包含 `sample`（本 run 的整数索引）、
原样 `response`、`labels`（字符区间）。导入并配对：

```bash
python -m state_audit pair --run runs/resampled --reviews reviews.jsonl
```

结果在 `pairs.json`；未审阅回答只计入 `unreviewed`，同 source 且实际 prompt IDs 相同的已审阅回答才配对。
`review.json` 单独保存，原始回答与生成 IDs 不变。后续 audit 会读取审阅覆盖层。
默认输出所有正常×错误组合；统计时这些组合并不独立，应按 source 聚合/重采样。

## 读状态，而不是传一串固定特征

```python
from pathlib import Path
from state_audit import ModelState
from state_audit.analysis.vectors import compare_vectors

state = ModelState.open(Path("runs/demo/samples/000000/trace"))
layer = state.layer(0)
attention = layer["attention"]
comparison = compare_vectors(
    {
        "attention": layer["attention_write"],
        "mlp": layer["mlp_write"],
        "residual": layer["residual_after"],
    }
)
print(comparison.groups, comparison.norms.shape)
print(comparison.pairs, comparison.cosines.shape)
```

比较结果保留 group/pair 轴，不为每个研究问题手写一批 `result["xxx_norm"]`。
只有旧 reanchor 报表的 CSV 导出边界会展开固定列名，以保持结果可读。

## 表征干预

```python
from state_audit.operations import Target, Delete, Replace, Inject, Steer
from state_audit.intervention import intervene

# 绝对输入位置 30；不限于两个 head，也可跨任意层。
target = Target("head_readout", layers=(12, 16), positions=(30,), heads=(0, 2, 5))
operations = [Delete(target)]

with intervene(model, operations):
    hidden = model.forward(token_ids)
```

四种操作都接收 Target。Replace 接收 donor 数组，Inject 加指定向量/张量，Steer 加 `amount × 单位方向`。
参数广播遵循 PyTorch；K/V head 是原生 KV head，不能拿 query head 编号冒充。
方向如何获得由实验定义；Steer 本身没有训练，也不意味着这个方向代表事实性。

原生 attention 的删除是把指定边置零，不自动把剩余质量重新归一化；这是来源消息消融的含义。
Replace/Inject 可修改合法边，必须非负且不能打开未来/滑窗 mask。
修改发生在 `A @ V` **之前**，后续 o_proj、残差、MLP 继续原生执行。
同一 site 按操作列表顺序执行，跨 site 按模型的 forward 顺序执行。全部 hook/临时 forward 在退出时恢复。
不要嵌套同一个 attention site 的干预上下文；把多个操作放在同一个列表中。

完整可运行的四类操作、同状态替换、任意 head 组见 [intervene.py](examples/intervene.py)。
命令行采用计划文件，替代旧版 `--head-a/--head-b`：

```bash
python -m state_audit intervene --run runs/demo --sample 0 \
  --plan examples/delete_heads.json --output runs/demo/interventions/delete_heads.json
```

计划里的 `targets` 是回答 token 索引；`Target.positions` 是绝对输入位置。两者不同：`query=P+target−1`。
`positions` 省略时操作所有当前输入位置。固定回答比较的分数是干预减基线的实际 target logp（nats）。
两组条件交互是 `experiments.factorial_interaction` 示例，每一组可含任意数量的头、层、操作；不再叫 world。

自由生成也能干预：`sample_answer(..., operations=operations)`。普通生成使用 KV cache；
带干预生成使用完整前缀重算，以保持绝对位置含义。固定位置必须已存在；
`positions=None` 表示对每步当前前缀持续应用操作。速度会慢于普通缓存生成。

## 观察审计与验证

```bash
python -m state_audit audit --run runs/demo --roles
python -m state_audit check --run runs/demo --sample 0
```

`audit` 保留原 reanchor、key swap、标注连续性报告，是一个具体分析实例，放在 analysis 目录。
它需要完整的 attention/QKV 等状态；hidden-only capture 应直接用 ModelState 分析。
`check` 明确验证 QK/RoPE/mask → attention、A@V → head 输出、投影 → attention write。
`complete.json` 只表明采集步骤完成，不能替代重建验证或机制证据。

观察视图剔除特殊 token；原生状态与模型内干预保持真实计算。
零普通 attention 质量可以存在，原始值保留为零；“以普通质量为条件”的分布在此处未定义，记 NaN。
历史消息与证据消息余弦、几何抵消本身不证明错误，也不证明多头协同失败。

## 检查为什么保留

没有给每个内部函数增加类型/维度/空值兜底，不捕获错误后补零继续。
保留的主动检查只覆盖会破坏研究解释的边界：

- 字符标签错位、证据归属含糊，不能生成貌似有效的标注。
- 模型结构/动态 RoPE 不匹配，不能悄悄按另一种架构读取。
- 干预选择了不存在的轴、重复/负索引、零 steering 方向或打开未来边。
- 输入被截断、续跑配置改变、review 对应另一个回答，不能复用旧身份。

维度或普通索引错误直接交给 NumPy/PyTorch。原生重建的数值检查集中在 validation 与测试，
不再混在每一层的采集代码里。更多 `raise` 不等于更可靠，全部删掉也不等于更清楚。

## 支持范围与迁移

0.3 新增固定前缀候选对比、轻量定点采集与逐层消息恢复，保留原 capture/operations API。
独立教学示例（使用上面的 demo run，不依赖仓库 experiments）：

```bash
python examples/compare_messages.py --run runs/demo
```

核心接口与计算公式见 [消息交互](docs/INTERACTIONS.md)。项目中的真实审计入口只是这些接口的调用方，
实验专用的来源角色、头名单和标签不进入模型适配器。

原始轨迹接口 `state_audit.native_trace.iter_native_traces` 复用同一 `ModelAdapter`，
记录逐边实际写入、FFN 与 attention 前/后的残差，以及原始激活上的局部依赖。
新增 `residual_mid` 对应 attention 残差相加后、FFN 归一化前的位置。
它使用分块 KV 预填充和单查询求导，不修改模型消息，也不要求消融。
真实案例、候选口径和一键运行见 [原始轨迹审计](../../experiments/native_trace_audit/README.md)。

当前原生适配器支持 Llama、Mistral、Qwen2（共享 decoder 布局，包含 GQA 与滑窗 mask）；
Transformers 4.57.x。其他布局需在 model 中显式映射，不声称任意 `AutoModel` 都支持。
批大小为 1，支持单设备加载与逐层落盘；全 attention 内存成本仍为二次方。
新架构加入时应实现 forward/score/bind 接口，并验证原生等价，不在 capture 或 operations 中写模型分支。

0.2 调整了 Python 导入路径和 CLI，旧 `models/datasets` 改为 `model/dataset`，分析模块进入 `analysis/`。
既有 v1 完整 trace 可由 ModelState/离线 audit 读取；新的 capture 配置使用 schema 2，不在旧目录直接续采。
原研究结果不迁移、不覆盖。先拉取 main，再在本目录重新安装即可。
