# 原始回答的原生功能审计

2026-09-24，协议 v3，入口 main.py transport-functions。

## 为什么删除候选过滤器

v1/v2 为了构造语义对照，先自动改写原文，再让同一观察模型检查组内同义、组间差异。
错误在于把 bank.valid 作为采集原始回答的前置条件：候选生成或评判失败，会让原文也没有测量。
检查候选的质量不能决定是否测量实际回答；同一模型的语义评判也不是独立真值。

用户上传的 v2 包中，12219 有 269 个回答 token、16 个单元，接受和完成均为 0。
59 次保存的生成调用没有产生任何 candidate NPZ。25 个可解析检查判决全部为 false。
唯一标注错误区间 [218,231) 也被 no_propositional_paraphrase 拦住。
旧包里的 baselines 是复制的既有分数，不能作为此次机制测量或新的 AUROC。

v3 删除候选提案、修复、语义自评、重复候选检查及有限 bank 读出整个执行链。
不是把 valid 强制改成 true，也不将不可靠的候选当作有效对照。
旧方法、结果和缓存保留；历史候选实现可在 Git be57906 查阅。

## 当前流程

1. 从 settings 读取原始 token IDs、prompt 边界和自动来源分组。
2. 按标点切分目标区间，保留编号；过长区间按计算预算拆块，不跳过。
3. 每块原文使用完整历史 prefix，一次 teacher-forced 原生 forward/backward。
4. 保存每个 query、物理层、头、来源的读取和三条敏感性通道，以及 FFN/RMS 数据。
5. 汇总实际覆盖，自动打包；原有标签和基线只在采集后复制，不进入测量。

标题、标点、重复文本、错误句均参与采集。没有文本质量或语义接受门槛。
token 对齐、来源坐标、模型上下文上限、续跑设置一致性仍检查；异常直接报错，不将失败补零。
原文过长而超过模型上下文上限也不静默截断。

默认覆盖整个所选回答。start-target/stop-target 精确选择半开 token 区间；
max-units 是显式的每答计算块上限，默认 0 表示全部。coverage 单独报告未选择 token。
max-unit-tokens 默认 96；分块会改变梯度目标范围，所以是协议的一部分。
标点边界和预算边界都不是模型发现的命题、reanchor 或幻觉边界。

## 测量的含义

对原文区间 [a,b)，目标为：

    L_[a,b) = sum(t=a..b-1) log p_theta(y_t | prompt, y_<t)

每个位置的概率、完整词表熵保持原生因果前向；query=P+t-1。
一次反向使用区间总目标，因此一个 query 的梯度可受区间内后续目标影响。
这些是原文区间的敏感性切面，不是彼此独立的逐 token 梯度，更不是 token 真假概率。
允许离线利用未来 token，但不跨越实际区间冒称测量整个回答的作用。

prefix_root_sensitivity[j] = embedding[j] dot dL/dembedding[j]，
包含原生 Q/K、RMS、SwiGLU 的导数。within_unit_root_sensitivity 保留区间内历史根。
这是局部敏感性，不是加和后还原 L 的守恒账本，也不反传穿过过去的离散采样。

对当前 attention 后残差 u 和包含 post-attention RMSNorm 的 FFN 算子 F：

    v = u + F(u)
    dL/du = dL/dv + J_F(u)^T dL/dv

对当前头从 key j 读到的消息 m_jh=A_qjh W_O,h V_jh，保存：

    head_residual       = m_jh dot dL/dv
    head_ffn_mediated    = m_jh dot J_F(u)^T dL/dv
    head_total          = head_residual + head_ffn_mediated

沿接收者 OV 消息路径读出，不单独把来源对 Q/K 的影响归给该边。
prefix 根包含 Q/K 路径，两种观测不能相加。层间观测也不能当独立贡献相加。
同层 FFN write 敏感性独立保留，不能再重复加入 head_total。
正负值分别在 key 上分开后按来源聚合；保留组内抵消信息和所有物理头。
负值只表示对当前目标的局部方向，不能直接命名为抑制事实、反证据或表达组织。

来源轴前 G 个组沿用 sources.json，G=len(blocks)+3；
最后一个组 G 为当前区间内已输入的历史 token。
key_group_ids 保存完整 key 坐标，因果 mask 排除未来 key。
没有把新组视为语义证据类别，也不把根作用当作一次新回看。

SDPA 原生前向，FFN checkpoint 保留原生函数和导数。
选定 query 的 A/V 重建只用于读边，head_reconstruction_error 单列低精度差异。

## 末层 FFN 与 RMS

令 m 是末层 FFN 输出，s(x)=sqrt(mean(x²)+epsilon)，A=W_U diag(gamma)：

    z(u+m)-z(u) = A m / s(u+m) + A u [1/s(u+m)-1/s(u)]

保存原始目标 token 相对最终最大非目标 token 的 margin：
direct、rescale、delta，以及 float32 代数恒等式误差和原生低精度舍入差。
pre_ffn_entropy 使用原生 FFN 前残差经最终 RMSNorm/unembedding 读出；
entropy 是原生最终 logits 的完整词表熵，单位 nats。
FFN 前读出不是删除 FFN 后的重跑，也不等于已经解耦事实与措辞。

## 一键运行

在原项目、模型环境中执行：

    git pull --ff-only origin main &&
    bash experiments/native_support/run_functions.sh

默认输入 outputs/native_support_ragtruth4，回答 12219，设备 cuda:0，bfloat16。
结果写入 outputs/native_support_ragtruth4/function_audit_v3，
同时产生同级 function_audit_v3_review.zip。v1/v2 目录不动，不作跨版本 resume。
脚本可从任意目录调用；附加 CLI 参数覆盖默认值；
FUNCTION_AUDIT_PYTHON 可指定解释器，失败退出码原样返回。

仍需从 settings 指定的真实模型补采集；旧拒绝 bank 不能替代这些观测。
每块一次原生 forward/backward，没有额外的候选生成调用。
--stage capture 与 run 相同，可直接采集，无需 prepare。
已完成块续跑不加载模型；未完成块重新采集，不将半份文件当作完成。

只汇总／打包已有数据：

    python main.py transport-functions --stage report \
      --output outputs/native_support_ragtruth4/function_audit_v3

report 不加载模型，对中断结果也报告缺测并打包。
--stage pack 仅打包现有目录。无选中 token 时模型阶段返回非零，不伪装成功。

## 文件

| 文件 | 内容 |
|---|---|
| protocol.json / settings.json / plan.json | 原始输入、测量口径、实际目标区间 |
| responses/*/sources.json | 自动来源块及原始位置分组 |
| responses/*/unit_*/observed.npz | 原始目标 ID、query/key 坐标、根敏感性、全层头来源通道、FFN、RMS、概率和熵 |
| responses/*/unit_*/complete.json | 区间身份、计算时间和峰值显存；最后写入的完成标记 |
| observed_tokens.csv | 每个已测 token 的身份、目标区间、概率、熵及 RMS 数据 |
| ffn_layers.csv | query 对应 token × 层的 FFN write 敏感性，目标区间明确记录 |
| coverage.csv / coverage.json | 所选回答中每个 token 是否纳入计划／实际测量，缺测位置与未选择数量 |
| units.json / summary.json | 完成／计划数量、数值误差；不产生新的检测分数或 AUROC |
| annotations.json / baselines/*.npz | 采集后复制的既有标签和基线，供独立分析 |

head_total/head_residual/head_ffn_mediated 及其 positive/negative、head_attention
均为 [query, layer, head, group]；ffn_write_sensitivity 为 [query, layer]。
原始多维数组保留在 NPZ，不为 CSV 展开复制大量头数据。

## 验证边界

15 项针对性 CPU 测试通过：原生概率／全词表熵、根有限差分、FFN Jacobian 分解、
RMS 缩放、GQA/滑动窗口、钩子恢复、全部文本类型、预算分块、准确范围、
中断缺测与续跑、禁止生成调用、标签不参与、缓存对齐、旧版本保护、一键脚本。
没有全量测试或真实 8B 重跑，没有新增自然数据 AUROC。

对用户上传 12219 的实际 token 数据只做计划核验：
默认 16 块覆盖全部 269 token，包含 [218,231) 的全部 13 个标注错误 token。
这不代表已完成这 269 token 的模型测量。
测量支持后续分析消息怎样转化；它尚未解耦真实语义支持与普通表达组织。
