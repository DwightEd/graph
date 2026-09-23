# 原生逐 token 检测与来源传递

当前只保留明确分开的入口：

| 入口 | 作用 | 模型成本 |
|---|---|---|
| `main.py support --stage score` | 原始 R/A/H 固定基线，复用缓存 | 不加载模型 |
| `main.py transport --stage run` | 来源值路径分解、候选作用和评价 | 首次需要新采集 |
| `main.py transport --stage score` | 复用 value-path 缓存读出 | 不加载模型 |
| `main.py transport --stage evaluate` | 只评价保存的分数 | 不加载模型 |
| `main.py transport --stage audit` | 只读旧 source_transport 预算结果 | 不重算图/状态 |
| `main.py transport-pack` | 打包当前 value_transport 的逐头、来源与时间审计数据 | CPU；不加载模型 |
| `main.py transport-state` | 条件来源读出、历史状态候选及同缓存对比 | CPU；不加载模型 |
| `main.py dynamics` | 独立历史状态模型与其审计 | 显式调用，非新方法依赖 |

## 来源传递

详细公式、论文依据、近似范围和数据轴见 [VALUE_PATH_TRANSPORT.md](../../iclr/VALUE_PATH_TRANSPORT.md)。
采用 DecompX/ALTI-Logit 的来源分解与输出相关读出思想；保留原生 attention、残差与 FFN 值路径。
原生 forward 不改；归因固定 attention/RMS，并对 SwiGLU 使用割线与乘法等分。
输入根的支持/抑制与当前层读取地址分别记录，不再先压成预算再统一平滑。
该分解不包含 Q/K 路由变化；对实际答案的支持不自动等于事实正确。

已有四答目录先运行：

```bash
git pull --ff-only origin main
python -u main.py transport --stage run --output outputs/native_support_ragtruth4 --resume
```

新输出在 `value_transport/`：`summary.json`、`evaluation.json`、`comparisons.csv`、
`tokens.csv`、`source_choices.csv`、`onsets.csv`、`high_risk_normals.csv` 和逐 token NPZ。
旧 source_transport/state_dynamics 缓存保持不动；旧 detached-KV 数组不含完整根来源，不能直接转换。
首次每答一个完整前向及多次独立候选反向；之后 `--stage score` 只读缓存。
SDPA 与 FFN checkpoint 控制显存；默认 gradient-batch=1，可明确调大。没有真实 8B/24GB 实测承诺。

## 已完成采集：直接打包

在项目根目录和现有 research 环境中运行，不需要下载额外脚本：

```bash
git pull --ff-only origin main
python -u main.py transport-pack \
  --output outputs/native_support_ragtruth4 \
  --archive value_transport_head_review.zip
```

上传项目根目录的 `value_transport_head_review.zip`。输入目录必须已有完整采集和
`settings.json`、`annotations.json`；此命令不启动模型、不改分数或原始文件。
输出文件已存在时不覆盖；再次打包可明确更换 `--archive` 文件名。

全部层/头、候选 ID、正负作用、FFN 作用、根归因、账本和评分均保留。
完整 attention/value-energy 替换为精确的逐头来源组质量与消息范数总量；额外保存每头
最强的 8 个历史 key、完整历史质量及 top-8 保留质量。没有历史时保存空边，不伪造端点。
这不是完整历史边图。`audit_pack_schema.json` 记录省略范围及 key/query/target 对齐。

当前熵、平滑和动态状态的实际范围，以及下一步设计见
[VALUE_PATH_STATE_REVIEW.md](../../iclr/VALUE_PATH_STATE_REVIEW.md)。

## 条件选择与历史状态：复用已采集数据

```bash
git pull --ff-only origin main
python -u main.py transport-state \
  --input outputs/native_support_ragtruth4 \
  --output outputs/native_support_ragtruth4/choice_state_v2
```

输入需要已有 `value_transport/capture/` 与 `value_transport/responses/*/scores.npz`。
也可把 `--input` 改为 `value_transport_head_review.zip`，直接读取打包文件，不必解压。
输出目录必须为新目录；模型、原始分数、旧结果均不改动。标签缺失时仍保存状态和分数，
evaluation 明确不可用。不要为了此命令重新运行 `transport --stage run`。

结果在 `choice_state_v2/summary.json`、`evaluation.json`、`ranking_audit.json`、
`state_audit.json` 及 `responses/*/state.npz`。FFN 不重复相加；实际候选 ID、
全部头读取、完整根历史边和未捕获候选质量均保留。
`risk` 默认仍为 raw_route；新增分数是独立探索候选，没有自动替换主基线。

评估结束自动生成旁边的 `choice_state_v2_review.zip`，包含结果、标注快照、逐回答
状态、原始基线及全部已捕获 token 的逐头正负作用、FFN 作用和根归因。
原始 dense attention/value-energy 转为精确组总量及 top-8 历史地址；这不是完整 attention 图。
包内 `review_manifest.json` 列出覆盖量、保留字段和缺失结果，不改分数、不重新运行模型。
新运行若使用 `--annotations`，会在全部评分后把实际使用的标注保存到结果目录。

**已经跑完的本次结果，只打包：**

```bash
git pull --ff-only origin main
python -u main.py transport-state --stage pack \
  --input outputs/native_support_ragtruth4 \
  --output outputs/native_support_ragtruth4/choice_state_v2
```

上传 `outputs/native_support_ragtruth4/choice_state_v2_review.zip` 即可。
`--input` 也可为原有 value_transport ZIP。`--stage pack` 不重新评分或评价。
同名包已存在时拒绝覆盖，可用 `--archive 新文件名.zip` 指定另一个名字。
四答完整联合包实测约 485 MiB；体积主要来自已有逐头/候选作用数据。
打包失败不发布最终 ZIP，评分结果保留，可修正路径后使用 `--stage pack`。

四答探索中直接来源缺口 AUROC/AP 为 0.769226/0.218225；历史递推为
0.752383/0.203478，未改善直接读出。这不是独立数据上的验证。
公式、数据轴和限制见 [CONDITIONAL_CHOICE_STATE.md](../../iclr/CONDITIONAL_CHOICE_STATE.md)。
本轮证据总结、正负作用的可辨识边界及条件异常检测方案见
[DETECTION_FINDINGS_AND_NEXT_MODEL.md](../../iclr/DETECTION_FINDINGS_AND_NEXT_MODEL.md)。

## 原始基线与数据准备

```bash
python -u main.py support --stage score --output outputs/native_support_validation32/test
```

首次使用 `support --stage prepare --dataset ... --model ... --output ...` 准备官方回答与标注，
或 `support --stage run` 采集原始 R/A/H。没有标注时保存分数，不伪造 AUROC。
`support --stage compare` 保留历史同数据公式比较，`--stage evaluate` 只评价默认逐 token 基线。
标签不进入采集与评分；人工语义证据类型不需要。

## 清理

已删除旧 optimize/model/readout/fuse/validate 执行分支及其私有滤波/融合代码，
删除旧共享预算图评分；历史实现见 Git `9d4ba17`。源隔离 cohort 准备函数仍供 dynamics 使用。
用户模型、缓存和结果没有删除。历史预算审计命令保持可用。

验证记录见 [VALIDATION.md](VALIDATION.md)。真实检测增量须看同样本 AUROC/AP，
不能把小模型账本测试当成真实自然数据效果。
