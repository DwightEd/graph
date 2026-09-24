# 从单元消息删除到逐 token 的条件选择回溯

2026-09-24。新增 `transport-carriers --mode token`，协议 `message-carriers-token-v2`。
旧 v1 入口、缓存、结果及默认 `risk=raw_route` 保持。本文将上传缓存的实测、
新版已经实现的软件行为和仍待验证的研究假设分别说明。

## 1. 当前发现与设计动机

上传 `message_carriers_v1_review.zip` 含四答、836 token、81 标错、41 文本单元。
37 个单元存在前文候选；其296条入选边的梯度与实际单删效应，双来源条件下
Pearson 相关约 .935/.923。37/37 单元的联合删除双条件绝对平均效应大于随机对照。
这说明梯度找到了会影响当前目标的消息，不能据此推出这些消息承载适用的事实证据。
相关性只在入选边上计算，也不排除筛选漏掉重要边。

一个具体例子是 L31H14：37 个有历史单元中入选23次，每次 key 都是上一单元末尾。
其双条件绝对删除效应总量的97.9%落在本单元第一个目标 token，主要关联边界续写。
回答12045的 `[91,106)` 单元为“5. Preheat the grill to high heat (unspecified in passages)”。
有来源条件下，联合删除使首 token `5` 的 logp 下降6.102，而 `grill`、`heat` 的下降
仅约 .00137/.00101；单元均值 .375，去掉首 token 后为 −.0339。
整句平均 logp 的选边目标因此可能主要选择编号/边界机制。

v1 的 `source_selected_unit_mean` 相对重放的总体 AUROC 从 .82317 升到 .85597，
但答内正负 token 对加权 AUROC 均为 .94937。13392 个答内正负对的排序没有改善或退化；
改善来自2006个跨回答正负对。这个收益不能解释成更精准的答内事实定位。
41 个单元全部标签齐性，当前四答也不能检验单元内部的错误定位。

**边界消息不等同于 attention sink。** Attention sink 指高注意力聚集现象，
可能承担 softmax 的额外质量，不要求其 value 对具体输出有大作用；边界消息却可能
真实控制编号、措辞或续写状态。删除效应大不能完成这种分类。
依据：[StreamingLLM](https://arxiv.org/abs/2309.17453)、
[When Attention Sink Emerges](https://arxiv.org/abs/2410.10781)。

## 2. 独立的输出选择目标

对原回答每个 token `y_t`，只输入原 prompt 和 `answer[:t]`，不采样新回答。
先在有来源条件下取最高 logit 的非原词 `b_t`，随后两个来源条件及全部删除
都固定使用同一个 `b_t`。记录实际 token 的 logp 和选择 margin：

\[
M_t^c=z_t^c(y_t)-z_t^c(b_t),\qquad c\in\{+,-\}.
\]

梯度目标为单个 `M_t^c`，不再用整句平均 logp 代替每个 token 的梯度。
最高竞争 token 是模型词表候选，可能只是词形/格式替代，不能称正确答案或语义反事实。
原词本身也可能低于这个竞争词；margin 可以为负。

从这个目标反传到原生各层 head readout。对边
`e=(layer,head,receiver,key)` 的消息
`m_e=W_O^h(A^h[receiver,key] V^h[key])`，保存有符号门导数
`c_e^c = ∂M_t^c/∂g_e`，原生门值为1。
默认用 `|c_e^+ − c_e^-|` 排序：目标是来源条件改变了多少输出选择作用，
两个条件中相同的局部导数会抵消。另保留 `--selection magnitude`，
按 `max(|c_e^+|, |c_e^-|)` 排序，作为明确的消融。

这是**条件敏感性候选**，不是已完成的 sink 消除或语义解耦。
来源删除也可能改变格式和位置；真实证据也可能在两个条件中作用近似。
本方法可能遗漏后一类消息，所以必须与 magnitude 及旧来源/路由基线并列评价。
梯度筛选接实际删除参考已有 attribution patching 思路，
不是完整 [AtP*](https://arxiv.org/abs/2403.00745) 复现或新颖性证明。

## 3. 从当前输出回溯更早的接收位置

每个物理 head 始终包含当前预测 query，即回答历史位置 `t−1`。
另在每个来源条件选最多 `receiver_budget−1` 个较早的回答 query，
按该位置 `||∂M/∂head_output|| × ||head_output||` 排序，再取两个条件的并集。
默认 `--receiver-budget 2`，每头因此最多3个不同 receiver；设为1仅测当前 query。
梯度经原生 Q/K/V、FFN、残差和归一化传播，允许较早层/位置影响最终选择。

对入选 receiver 枚举其可见的回答历史 key，包括当前文本单元内部的历史。
两个来源条件共用 receiver/key 候选身份。每个物理 head 至多留一条边，再取前 K 头。
receiver 范数筛选是计算预算；头内抵消、多边协同和梯度饱和可能导致漏选。
当前仍是**回答历史边的有限回溯**，没有枚举 prompt 来源边或恢复完整因果图。
`t=0` 没有回答历史，保存双来源观测和未测 mask，不伪造边或重建误差。

对同组边，在有/无来源条件分别实际执行：逐边删除、联合删除、随机对照删除。
删除只减去指定 `(layer,head,receiver,key)` 的原生 `A*V`，不重归一化 attention。
后续网络重新计算；联合删除直接运行，不能把单删效应求和当作联合结果。
随机对照匹配层、头及 receiver，改取其他可见 key；仅有一个 key 时允许重合并记录。
这不是距离、能量或语义完全匹配的对照。

SDPA 前向与稀疏重建减法有 dtype/kernel 舍入差异，逐层重建误差一并保存。
来源删除沿用原输入并压紧位置，因此测到的是来源条件干预效应，不能称纯事实因果效应。
这里仍是 observer 的 teacher-forced 机制，不是原回答生成器当时的内部轨迹。

## 4. 分数与每个节点的表征

令 `D_e^c = M_keep^c − M_cut(e)^c`；保存两个条件及交互 `D_e^+ − D_e^-`。
新候选 `choice_selected = −(M_joint^+ − M_joint^−)`，
即删除选中历史消息后的剩余来源选择作用取负。
另存 `choice_sham`、`choice_replay` 和相同测量的 logp 分数
`token_source_selected/sham/replay`。候选方向冻结，不根据标签翻转。
它们不是校准后的幻觉概率，选择作用也不等于事实正确性。

`node_features.npz` 的 `node_features` 为 `[T, 22+7LH]`；32层32头为 **7190维/token**。

| 部分 | 内容 |
|---|---|
| 前22维 | logp 与 margin 各11个全局观测：原生双条件、来源差、联合/随机删除的双条件效应及交互、删除后来源差 |
| 固定 layer×head×7 | 单边 logp 的双条件效应/交互、单边 margin 的双条件效应/交互、measured mask |
| `history_index`, `receiver_index` | `[T,L,H]` 的边端点；未测为 −1 |
| `foil_id`, `target`, `token_id`, `unit_id` | 竞争词及原回答身份 |

未测头填0但 mask 为0，不等于测到零效应；不把不同 token 的“第一个入选头”混成一个轴。
逐 token 保存 `token_TTTTTT.npz`，包含单删/联合/随机原始观测、门导数、
attention mass、投影后消息能量、竞争词文本及重建误差。
`feature_schema.json` 明确各轴，`selected_edges.csv` 支持检查具体输出/竞争词和边端点。

每个 token 独立选择和测量，不依赖文本分单元方式或未来后缀。
旧 baseline、v1 选边控制和 `*_unit_mean` 仍可能使用单元未来信息，不能混称在线方法。
`ranking_scope.json` 分解答内与跨回答排序；单元等权、首错/延续、同单元定位等沿用既有评价。
全部答案的向量和分数保存后才复制/读取标签，不按标签选词、挑头或设边界。

## 5. exp 位置权重：独立的缓存对照

“压低前期权重”有两个不同操作。按历史 key 距离衰减会额外偏好近处消息，
可能压低真实的远程证据；本版没有把这个先验混入选边或 attention。
当前测试的是**输出单元内的聚合权重**：

\[
w_i={\exp(\beta\,i/(n-1))\over\sum_{j=0}^{n-1}\exp(\beta\,j/(n-1))},
\qquad \bar r_U=\sum_i w_i r_i.
\]

单 token 单元权重为1。`beta=0` 为普通均值；固定 `beta=1` 时末/首权重比为 e。
β 没有训练、搜索或按结果选优。对每个原始候选都计算同样权重，不改原分数和模型。

上传四答 v1 缓存的实际 CPU 重算如下；表内是聚合分数广播回原 token 的总体指标：

| 分数 | 普通均值 AUROC / AP | exp(β=1) AUROC / AP |
|---|---|---|
| source_selected | .855973 / .340470 | .825869 / .333094 |
| source_full_replay | .823171 / .310212 | .781163 / .364041 |
| source_local | .914856 / .451496 | .854861 / .479659 |
| source_pair | .893582 / .372711 | .880991 / .484879 |
| raw_route | .770746 / .228008 | .772463 / .228659 |

`source_selected` 答内加权 AUROC 从 .949373 升到 .964456，但总体 AUROC/AP 均下降；
`source_local` 答内 AUROC 从 .957736 降到 .848716，AP 却上升。
所以结果存在指标与排序范围上的取舍，不支持把 exp 自动升为主评分，也不能判为普遍无效。
位置权重不会识别某个 token 的语义角色，格式与事实词都可能处在单元任何位置。
这些是反复使用的四答探索结果，不是独立泛化证据，也不是 v2 新模型结果。

CPU 阶段另存 `unit_exp_evaluation.json`（单元等权）和 `ranking_scope.json`。
通用 `unit_budget.json`、`aggregation.png` 仍对应普通单元均值，不冒充 exp 的预算/图表。

## 6. 运行与验证

```bash
git pull --ff-only origin main &&
bash experiments/native_support/run_carriers_token.sh
```

默认读取 `outputs/native_support_ragtruth4/message_carriers_v1`，
保留 `unit_carrier_v1` 及其均值作为控制，写入独立 `message_carriers_token_v2`。
也可覆盖 `--input` 为 contrast/aggregation 目录或 review ZIP。
程序逐答、逐 token 展示进度，完成标记最后写入；`--resume` 补未完成 token。
结束自动生成 `message_carriers_token_v2_review.zip`，包含向量和原始有限干预测量。

```bash
# CPU：复用现有 v1/v2 缓存，不加载大模型
bash experiments/native_support/run_carriers_position.sh
# CPU：已采集 v2 的重新读出、评价和打包
python main.py transport-carriers --stage score \
  --output outputs/native_support_ragtruth4/message_carriers_token_v2
```

可单独用 `--receiver-budget 1` 或 `--selection magnitude` 检查两个新假设，
每种设置须指定新的 `--output`。协议变化不能在原目录中假续跑，不自动遍历参数挑最高分。

K≥2 时，每个 token 需要2次带梯度前向/反向及 `2K+4` 次删除前向；K=8为22次前向、2次反向。
旧 v1 是每个单元这个次数，新版836个独立目标相比41个单元明显增加计算，
不能套用 v1 的耗时/显存。梯度图及时释放、观测暂存 CPU、FFN checkpoint 和 SDPA 保留。
当前只有小型随机 Llama/Mistral/Qwen2 的软件验证，没有运行真实8B v2 或获得其自然 AUROC。

针对性验证包括：精确 query 删除与 dense 对照、早期 receiver 传播、固定竞争词双来源
中心差分、零门强度原生一致性、GQA/sliding-window、分单元/未来后缀不改变目标，
表征与未测 mask、标签隔离、单 token 中断恢复、缓存 exp(0) 与均值一致。
旧消息删除、contrast、aggregation 和 functional transport 回归同时保留。
