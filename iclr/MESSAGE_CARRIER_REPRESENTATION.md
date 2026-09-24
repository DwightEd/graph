# 关键历史消息干预与逐 token 表征

实现：2026-09-24。入口：`main.py transport-carriers`。

输出是每个**原回答 token**一个固定维度向量，外加被选历史边的身份表。
它是原生干预响应表征，不是训练得到的 embedding，也不直接表示推理步骤正确率。
本次实现方法采集与冻结候选评分，没有新8B/自然数据 AUROC。

## 1. 测量对象与筛选

复用已完成 `transport-contrast` 或其 aggregation 目录/ZIP 的原 token、来源删除视图、
文本单元和基线。不重新划分单元，不读取真假标签，不生成候选文本。
目标单元 (U=[a,b)) 的目标函数为原文平均 log probability：

\[
F(U)=|U|^{-1}\sum_{t\in U}\log p(y_t\mid P,y_{<t}).
\]

候选历史节点 (j<a)，限定为该单元开始前的回答 token。
每条候选 carrier 的身份是 `(layer, head, history_index)`；实际干预是一束边：
在该单元所有预测 query (q=P+t-1) 上，门控同一历史位置的消息
\(m^{l,h}_{q,j}=W_O^{l,h}(A^{l,h}_{q,j}V^{l,h}_j)\)。
不删除输入 token、不删除整个 head，不改 softmax 的归一化分母。
单元内已经生成的词仍在上下文中；选择整个单元会用到其中的未来词。

在有来源、无来源两个条件各做一次完整原生前向/反向，计算
\[
c^{s}_{l,h,j}=\sum_{t\in U}
 \langle\nabla_{o^{l,h}_{q}}F_s(U), A^{l,h}_{q,j}V^{l,h}_j\rangle,
\qquad r_{l,h,j}=\max_s|c^{s}_{l,h,j}|.
\]
`o` 是 W_O 前的原生 head readout，因此梯度已包含 W_O 和下游变换。
Q/K、RMS、SwiGLU 均保留原生导数；不是旧值路径近似。
梯度用于候选排序，**是单元目标梯度，不是独立逐 token 梯度**。

每个物理头先取最高 `r` 的一个历史 key，再从所有层头选前 `K=8` 个。
这是明确的干预算力预算及每头一个 carrier 的限制，不是所有边的无约束 top-K。
保留两个条件的有符号梯度；负梯度同样可能入选，不按符号判断真假。
当前模型没有读取的约束不会因此自动被发现。

## 2. 真正的有限删除与条件交互

两个来源条件使用相同的历史 token 身份 `j`，对应绝对 key 为 `P_s+j`。
分别测量不删、每条单独删、整组一起删、随机对照组一起删的逐 token log probability。
在层内减掉被选 `A*V` 后再进入 W_O；下游层重新计算 Q/K/V、FFN 和残差。
删除多个层时，后面的层使用前面已受干预的状态，不能用冻结 donor 替代。

记 \(\ell_{s,c}(t)\) 为来源 `s` 有/无、选中消息 `c` 保留/删除时的 log probability：

\[
D_+(t)=\ell_{1,1}(t)-\ell_{1,0}(t),\quad
D_-(t)=\ell_{0,1}(t)-\ell_{0,0}(t),\quad I(t)=D_+(t)-D_-(t).
\]

冻结的检测候选：
\[
S_{\rm selected}(t)=-[\ell_{1,0}(t)-\ell_{0,0}(t)].
\]
表示删除选中历史消息后，原 token 的剩余来源作用取负。
同一公式用于随机删除 `source_sham`，不删除重放为 `source_full_replay`。
每种都有独立 token 分数及同单元平均 `*_unit_mean`。
交互 `I` 保留为表征，不直接充当真假风险。原 `risk=raw_route`。

随机对照逐边保持层、头与数量，从两个条件至少一个可读取的历史 key 中抽另一个位置。
只有一个 eligible key 时必须重合，`sham_overlap` 显式报告。
这不是能量/距离/语义完全匹配，单次随机对照也不提供显著性检验。
单边作用不相加冒充联合效应；联合删除实际重跑，并报告与单边之和的差异。

SDPA 保留完整原生前向，只重建被测单元的 attention 行，避免物化全长 attention 矩阵。
稀疏 `A*V` 使用原生概率精度，减法与 eager 直接删边存在 kernel/dtype 舍入差异。
每层保存完整 head readout 重建误差；float32 小模型直接删边对照已验证。
bf16 的小效应应结合误差审查，不能称 bit-exact 或完全隔离的事实因果效应。
来源删除仍改变 prompt 长度与位置。

## 3. 每个节点的向量

`responses/NNNN/node_features.npz` 中：

| 数组 | 形状 | 含义 |
|---|---|---|
| `node_features` | `[T, 11+4*L*H]` | 一个回答 token 一行；float32 |
| `history_index` | `[T,L,H]` | 该头被删的回答历史位置；`-1` 为未测 |
| `target`, `token_id`, `unit_id` | `[T]` | 原回答位置、原 token ID、固定文本单元 |

前11维是有/无来源 logp、来源作用、联合删除双条件效应与交互、
联合删除后来源作用、随机删除双条件效应与交互、随机删除后来源作用。
之后固定按 `[layer,head,channel]` 展平；每头4维：
`single_delete_with_source`, `single_delete_without_source`, `single_interaction`, `measured`。
头身份固定，不按“第几个入选”混用特征轴，不跨头取平均。
未入选头的数值以0占位，但 `measured=0`，不能当成测到零效应。
32层32头的模型对应 **4107维/token**。

同一单元共用选边，但各 token 的有限效应分别测量，因此向量不靠复制单元均值生成。
由于在单元所有 query 上删除，后面 token 的效应可包含前面 query 的干预传播；
它不等于“仅删除当前 query 的一条边”的效应。
首单元没有之前的回答历史：仍保存向量，头全部未测，联合/随机删除等于不删。
有历史但受 sliding-window mask 完全遮挡的头也不伪造候选。

`feature_schema.json` 记录轴、通道名、维数、flatten 公式和缺测语义。
`unit_AAAAAA/measurements.npz` 保存：

- `edges`, `sham_edges`：`[K,3]` 的层/头/历史位置；K可以小于预算。
- `approximation`：`[K,2]` 的两个条件有符号单元梯度。
- `keep`, `joint`, `sham`：`[2,|U|]`，条件轴固定为 with/without source。
- `single`：`[2,K,|U|]`；不删减原始测量，只读评分可重建全部表征。
- `queries`、`edge_keys`、`prompt_lengths`：两个来源条件的实际位置映射。
- `eligible_counts`、`reconstruction_error`、目标位置与 token ID。

读取示例：

```python
import json
import numpy as np
from pathlib import Path

root = Path("outputs/native_support_ragtruth4/message_carriers_v1")
schema = json.loads((root / "feature_schema.json").read_text())
with np.load(root / "responses/0000/node_features.npz") as saved:
    X = saved["node_features"]                 # [T, D]
    heads = X[:, len(schema["global_channels"]):].reshape(len(X), *schema["head_shape"])
    measured = heads[..., 3].astype(bool)      # 未测的头不能直接当真实0
```

## 4. 执行与成本

```bash
git pull --ff-only origin main &&
bash experiments/native_support/run_carriers.sh
```

脚本默认读取 `evidence_contrast_v1`，输出独立的 `message_carriers_v1`。
也能直接使用已经汇报的 aggregation 目录：

```bash
python main.py transport-carriers \
  --input outputs/native_support_ragtruth4/evidence_contrast_aggregation_v1 \
  --output outputs/native_support_ragtruth4/message_carriers_v1 \
  --top-k 8 --resume
```

这是新增采集，旧四列概率不足以恢复消息梯度与删除效应。
K≥2时每单元2次前向/反向加 `2K+4` 次无梯度删除前向；K=8即22次前向、2次反向。
只保存当前单元计算图，参数不求梯度，FFN checkpoint；无 GPU/8B 显存或耗时保证。
逐答/单元/单边都有进度；完成标记最后写入，`--resume` 只补未完成单元。
修改 top-K、seed、dtype、输入或协议必须换输出目录。

```bash
# 模型采集，暂不打开标签
python main.py transport-carriers --stage capture --input CONTRAST_DIR --output NEW_DIR --resume
# CPU重建向量与全部分数，然后读取标签评价；不加载模型
python main.py transport-carriers --stage score --output NEW_DIR
# 只评价已保存分数，或重新打包
python main.py transport-carriers --stage evaluate --output NEW_DIR
python main.py transport-carriers --stage pack --output NEW_DIR
```

所有回答的向量和分数保存后才读取标签。结果包括原基线、三种新候选及相同单元均值，
token AUROC/AP、首错/延续、来源平衡/答内指标、单元等权、单元内定位及整单元预算。
`intervened_tokens_evaluation.json` 额外仅比较存在历史边干预的 token，同样本并列所有方法。
`intervention_diagnostics.csv` 报告梯度/有限删除偏差、联合非加性、随机重合及重建误差。
`coverage.json` 报告计划/表征/实际干预 token 数、维数、重放与旧来源基线的差异。
自动生成同级 `message_carriers_v1_review.zip`，包含表征及完整有限干预测量。

## 5. 如何使用结果与当前验证边界

先比较 `source_selected` 与 `source_sham`、`source_full_replay`、旧 `source_local`，
以及 raw/observable 路由和同单元均值。关注真正干预的子集和首错，不仅看总体 AUROC。
向量可供后续独立表示读出或图分析；本入口不新增真假分类器、不按标签挑头/改符号。
关键消息、来源依赖和条件交互不是事实支持的定义，不能据此认证每个推理步骤。
先前四答41单元全部标签齐性，仍不能据该批数据检验单元内部定位收益。

软件验证覆盖 Llama/Mistral/Qwen2 小型随机模型的 GQA、Mistral sliding-window，
稀疏与 dense attention 删边、中心差分梯度、双来源位置映射、原生钩子恢复、
无历史/无来源退化情况、未读未来后缀、向量轴、ZIP输入、标签隔离及中断恢复。
现有 contrast/aggregation/function 测试用于回归。没有真实新8B结果，不声称提高检测性能。

梯度筛选再有限干预是已有 attribution/activation-patching 思路；
相关原始研究：[AtP*](https://arxiv.org/abs/2403.00745)、
[Hydra](https://arxiv.org/abs/2307.15771)。这里不是完整 AtP* 复现，工程实现本身不构成新颖性证明。
