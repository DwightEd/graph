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
