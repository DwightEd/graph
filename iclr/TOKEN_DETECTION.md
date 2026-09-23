# 逐 token 检测主流程

2026-09-23。按用户要求，检测不再以消融、语义重锚发现或完整机制解释为前提。
默认入口已经改为这一流程；这不是新增一个需要接到v3–v6之后的阶段。

## 问题、观测与可检验判断

问题：给定prompt与已经输入的回答前缀，当前位置的内部观测是否能对下一个token的幻觉标签排序？
对完整回答每个token都计算，不挑选金标首错、不预先检测事件、不按金标span切换算法。
预测回答token t的query固定为 P+t-1；当前目标token不进入路由输入。

| 观测 | 实际测量 | 用途与边界 |
|---|---|---|
| A：attention_displacement | 读取质量偏向history还是source | 判断读取分配；不是正确证据识别 |
| R：routing_imbalance | 投影后消息范数预算偏向history还是source | 固定主风险分数；不是有符号支持或反对 |
| H：entropy | 原生下一token分布的熵 | 独立不确定性指标；高熵正常词也会得高分 |

仍使用原生逐边逐头消息算子：

    m_(j,l,h) = W^O_(l,h) [a_(q,j,l,h) W^V_(l,h) x_normalized_(j,l)]
    R = mean_l [Σ_(h,j) d_j ||m_(j,l,h)|| / Σ_(h,j) ||m_(j,l,h)||]
    A = mean_(l,h) Σ_j d_j a_(q,j,l,h)
    H = -Σ_v p(v|prefix) log p(v|prefix)

d_j为history记+1、source记-1、其他记0，精确定义沿用routes.py，包括self处理。
先算每条实际消息的范数再汇总，不能先平均头向量再取范数。原始逐头缓存不改写。
这个标量读出确实汇总层头，因此它不表达完整头协同，不能声称已经解决该问题。
没有官方来源区间时明确使用prompt_*变体，不能冒称evidence指标。

R作为固定主分数，是因为已有同样本结果提供了直接检测证据：32答AUROC=0.711589。
A与H独立输出，检验其错误起点与延续的表现，不为了凑一个新公式强制混合。
v6的max融合仍未得到自然效果证据，所以本轮不把它设为默认，也不宣称它已失败。
这是流程和研究逻辑收敛，不是一个已经提升AUROC的新算法。

## 唯一默认路径

1. 首次运行：已有teaching原生前向按因果query分块采集；每个前缀不做删边或重跑消融。
2. 读取原缓存；逐token算R/A/H，保存NPZ和一张tokens.csv。已有v3标量缓存直接复用。
3. 全部分数保存后才读取annotations，统一计算AUROC/AP。

没有参考集、状态拟合、图传播、窗口选择、最大分位融合，也没有自然标签训练。
新缓存不计算不使用的head_profile；不复制旧的大段长后验矩阵。
原始模型采集和检测读出分开，已有32答不需要再跑GPU。

评价保持同一套协议：all_error、span_onset、answer_first、continuation、前后半段；
附来源等权、同答、逐答指标。首错/延续仅是评分后的评价分组，不是推断可用的输入。
不根据本次AUROC翻转方向、选择指标、选择头或拟合权重。缺标签只保存分数，不伪造AUROC。
没有阈值时高分正常词只是排名审计，不能称为已判定误报。

## 运行和输出

现有32答，只需：

```bash
git pull --ff-only origin main
python -u main.py support --stage score --output outputs/native_support_validation32/test
```

输出在该目录的 `token_detection/`：
`summary.json` / `evaluation.json` 是总体与详细AUROC/AP；
`tokens.csv` / `responses/0000/scores.npz` 是全部token观测和主risk；
`report.html` 是统一报告；`onsets.csv` / `high_risk_normals.csv` 供可选查看。

首次数据采集沿用 `--stage run --dataset ... --output ...`，其默认后处理现在也是上述检测。
`--stage evaluate` 优先评价已完成的token_detection，不因旧融合文件存在而切回旧方法。
旧 v3–v6 的 optimize/model/readout/fuse/validate 分支已清理，历史实现见 Git `9d4ba17`。
来源归因候选改用 `main.py transport`；详见 `VALUE_PATH_TRANSPORT.md`。原缓存/结果保留。

## 验证边界

软件检查：缓存复用与原R/A/H逐值一致；标签变化不改变评分；无需状态拟合或参考集；
默认采集到评价的小型模型集成；token与query对齐。
没有本地自然缓存，未产生新的真实AUROC。预期同数据同公式的R成绩保持原值，不能称为优化增益。
