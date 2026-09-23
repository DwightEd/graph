# 完整短语的功能读出与 FFN 条件传递

2026-09-23，候选协议 v2。入口 `main.py transport-functions`。这是已实现的机制测量与读出更新，
没有新自然数据 AUROC，不替换 `raw_route`，不对四答标签拟合系数。

## 问题与架构

旧 choice-state 把不同 token 的正向选择作用沿历史边继承，但这些选择的含义并不一致。
例如同一个 `folding` 可以出现在对 Passage 1 的正确说明或对 Passage 3 的错误说明中。
本实现用当前完整短语的同一组意义对照，直接询问所有可见历史输入的作用，
取消本路径中的“旧词正作用 = 当前命题支持”的递推假定。旧命令及结果保持可复现。

流程直接分成五步：

1. 从已保存回答取 token 对齐的标点单元，编号与后续句子合并，冻结原始 token IDs 和共同 prefix。
2. 自动提出同义改写、极性/数量/范围变化、对象绑定变化；每个意义类别两种表达。
3. 同一观察模型另外检查组内同义、组间差异和 prefix 适配性；保存原始输入与输出。
4. 每个候选完整 teacher forcing，一次原生前向和一次独立短语目标反向。
5. 在同一 prefix 坐标上组合候选导数，保存来源 × 层 × 头 × 对照及 FFN 通道。

标点单元不是模型发现的 claim/reanchor，可能是不完整语义单元。候选可以被拒绝；
拒绝数、长度预算跳过数和完整测量数均报告。没有人工证据类型、自然标签训练、
SVD、消融或另训真假分类器。自动检查来自同一模型，不属于独立语义真值验证。

候选任务只收到 `answer_prefix`（已生成回答）和当前 `unit`，不把原问答指令嵌套进去。
payload 中的特殊 token 字面量使用可逆 JSON Unicode 转义，不能成为新的对话控制 token。
这只改变候选生成/检查的输入；所有概率和导数仍使用完整、逐 ID 冻结的原生输入前缀。
若仅凭回答前文无法消歧某个对象，自动分组仍可能不可靠，不能据此声称完整覆盖命题。

## 候选概率和功能作用

对候选完整短语 c，实际分数为：

    L_c = sum_k log p_theta(c_k | saved_prefix, c_<k)

保留每个 token 的 log probability、ID、候选总长度。实际观察短语直接沿用保存的 ID，
替代短语仅自身分词，绝不重新分词共同 prefix。不同长度和表达先验仍会影响 L；
不使用事后按标签选择的长度校正。

在有限候选集合内 p_c=softmax(L)_c，意义组质量 q_f=sum_{c in f} p_c。
这里没有枚举全部可能回答，也没有 EOS 终止事件覆盖，因此它是有限 bank 的分数分布，
不能称为大模型完整的意义概率或事实正确率。

    H(p) = H(q) + sum_f q_f H(p(.|f))

同一 bank 的 FFN 前后读出还保存 KL 的组间/组内精确链式分解。
“前”是最后 FFN 之前的原生状态，经同一个最终 RMSNorm/unembedding 读取；
它不是删掉 FFN 后重跑的世界，也没有重算前面 token 的 KV。

观察命题 f 对另一个命题 g 的目标：

    G_fg = logsumexp(L_c, c in f) - logsumexp(L_c, c in g)
    dG_fg/dL_c = softmax(L within f)_c 或 -softmax(L within g)_c

先独立求每个候选的原生梯度，再按上述系数组合，得到对 G 的局部敏感性。
同义表达对照另外保留，避免把所有输出竞争都称为事实冲突。
`prefix_root_sensitivity[j] = embedding[j] dot dG/dembedding[j]`。
这是原生梯度读出，包含 Q/K、RMS、SwiGLU 导数；它不是守恒归因账本，
各输入项之和不要求还原 G，也不保证事实支持。

候选后缀的坐标/长度可能不同，因此跨候选只合并共同 prefix 根和共同边界 query。
每个分支内部所有 query 的数据单独保存，绝不把“第 k 个候选词”跨分支强行对齐。
其中实际回答保留全部 query 的逐头数据，替代分支仅保留共同边界 query 的逐头数据，
减少无法跨分支对齐的冗余；替代分支仍保存全部 query 的 FFN/RMS 数据。
`head_query` 与 `query` 分别给出两类数组的绝对位置轴。
历史根针对当前 G 读取；不能声称原生反向穿过了此前离散采样。

## FFN 作为依赖当前状态的传递算子

设当前 attention 后残差为 u，F 包含 post-attention RMSNorm 和原生 MLP：

    v = u + F(u)
    dG/du = dG/dv + J_F(u)^T dG/dv

对头 h 从 key j 读取的消息 a_jh=A_qjh W_O,h V_jh：

    residual_channel = a_jh dot dG/dv
    ffn_mediated_channel = a_jh dot J_F(u)^T dG/dv
    total_channel = residual_channel + ffn_mediated_channel

实现用原生 head-readout 梯度与 FFN-output 梯度作伴随读出，不显式建 Jacobian，
不删除消息。两条通道依赖同一实际上下文，FFN-mediated 正负描述的是它对这个
具体语义对照的局部转化作用，不能当作“FFN 支持/反对事实”的通用标签。

head 通道沿当前接收者 OV 消息路径测量；它不是把该来源对当前 Q/K 的路由效应
另外归给 OV 边。prefix 根梯度包含原生 Q/K 路径。不得把两者相加。
各层是观察切面，不能跨层相加成总信息量，也不能重复加 FFN write。

所有物理头独立保存。每分支先在 key 上分正负，再分来源组，保留组内抵消信息。
跨候选合并后的 head group 数值是有符号净作用；分支正负数组另存，不能把分支
正值简单相加并冒称完整语义对照的正向边质量。

SDPA 执行原生前向，重建选定 query 的 A/V 仅用于读边。
`head_reconstruction_error` 记录它与实际 head-readout 的差别，尤其检查低精度误差。
FFN 使用 checkpoint 降低宽中间激活占用；其函数和导数保持原生。

## 最终 RMS 缩放单独核算

令 m 为最后 FFN 输出，s(x)=sqrt(mean(x²)+epsilon)，A=W_U diag(gamma)：

    z(u+m)-z(u) = A m / s(u+m) + A u [1/s(u+m)-1/s(u)]

按实际词与同一原生 foil 的 margin 保存 direct、rescale、delta 和重建误差。
这是指定末层端点下的代数恒等式，不是唯一的因果交互分配。
float32 恒等式误差和 native bfloat16 的 margin 舍入差单列。
归一化重缩放通常不改变 logits 排序；direct 项仍可能混合表达、事实、绑定等作用。

## 输入、运行与成本

需要原来的 `settings.json` 和 `value_transport/capture_settings.json`、各答 sources.json。
也可直接读取 transport-pack/choice-state review ZIP，不读取密集旧 attention。
旧缓存没有这里的原生 Q/K/RMS 导数或语义候选，无法转换得到；首次需要补采集。
原模型路径来自 settings，仍使用 teaching 的 Llama/Mistral/Qwen2 适配器。

v1 的候选与切分协议已改变，v2 必须使用新结果目录；不能沿用 v1 的拒绝 bank 做 resume。
v2 一键运行／续跑，默认回答 12219：

```bash
git pull --ff-only origin main &&
bash experiments/native_support/run_functions.sh
```

脚本用当前环境的 `python` 在同一进程内准备缺失候选、采集有效单元、汇总并自动打包。
默认 `--stage run --resume`，使用原 `function_audit_v2` 目录；已完成结果和拒绝记录复用。
可以从任意目录调用脚本，额外 CLI 参数覆盖默认值；相对数据路径按项目根目录解析。
如需指定解释器可设 `FUNCTION_AUDIT_PYTHON`。脚本透传失败退出码，不把错误当成功。

只准备候选或只补采集仍可分别运行：

```bash
bash experiments/native_support/run_functions.sh --stage prepare
bash experiments/native_support/run_functions.sh --stage capture
```

v2 内续跑添加 `--resume`，参数和 settings 必须相同；完整候选复用，完整单元不加载模型。
候选按实际 token IDs 检查组内与跨组重复；结构或语义检查失败时，带具体反馈重写一次。
原 `proposal.json` / `validation.json` 保留，修复另存 `proposal_repair.json` /
`validation_repair.json`；修复过的候选不复用原候选的语义判决。
第二次仍不合格则在 `bank.json` 记录拒绝原因和冲突位置，继续后续单元，不做静默去重。
`summary.json` 的 `rejection_reasons` 汇总原因；被拒绝单元不会产生概率或梯度读出。
零接受时记录 `status=no_valid_banks`，保存失败详情和 review ZIP 后返回非零退出码。
完整 run 可用默认 `--stage run`；不能因进度条跑完而把零测量当作实验成功。

遇到 `Unterminated string` 等生成 JSON 错误时，保留原始输出并作一次格式重试，
输出另存 `*_json_retry.json`。续跑会复用成功的格式重试，不反复读取坏缓存后崩溃。
仅接受完整 JSON（允许外包完整代码围栏）；不补齐截断字符串，不从半个对象猜测 `valid`。
模型输出字段也在此边界检查：`valid` 必须是 JSON 布尔量，候选必须是文本／文本列表。
格式重试仍失败则记录 `invalid_generated_json` 或 `invalid_generated_schema`，继续下个单元。
原始输出另记实际生成长度及是否达到 token 预算；模型执行和文件读写错误仍正常抛出。
这项修复兼容已有 v2 缓存，无需换目录或重跑已完成采集；它不保证语义候选通过率提高。

仅修改读出/报表时：

```bash
python -u main.py transport-functions --stage report \
  --output outputs/native_support_ragtruth4/function_audit_v2
```

`--stage prepare` 只生成并检查候选；之后同参数 `--stage capture --resume` 补采集。
`--start-target / --stop-target` 选择与目标范围相交的完整单元；
`--max-units` 默认 0 表示全部所选单元；`--max-unit-tokens` 默认 96，过长单元明确跳过。
这些是审计计算预算，不是幻觉 span 长度假设。默认不是全量 RAGTruth。

每候选一次完整 forward/backward；候选按序处理，不同时保留多个计算图。
相较旧缓存 CPU 评分会慢，24GB/8B 的真实显存和速度未实测。没有假称免费地
复用旧梯度或已经提高 AUROC。报告阶段为 CPU 数组计算，不加载模型。

## 结果

| 文件 | 数据 |
|---|---|
| protocol.json / settings.json | 完整范围、观察模型、分组与梯度契约 |
| responses/*/unit_*/proposal*.json / validation*.json | 自动候选、可选修复与每次检查的原始请求/输出 |
| bank.json | 精确 prefix、原始和替代 token IDs、意义分组、拒绝原因 |
| candidate_*.npz | 每分支 prefix 根、后缀根、逐 query/layer/head/source 三通道、FFN write、RMS 分解 |
| contrast_*.npz | 共同边界 query 的语义/表达对照；输入根仍逐位置保存 |
| units.json / readout.json | 组间/组内熵及 KL、候选长度、数值误差 |
| head_contrasts.csv | 物理层头来源对照，不平均头 |
| observed_tokens.csv | 实际回答的 RMS/置信度变化，不把候选分支贴自然标签 |
| summary.json | 接受、拒绝、跳过、完成数；status 区分准备/采集/零接受；new_auroc 明确为空 |
| annotations.json / baselines/*.npz | 所有测量完成后复制的既有标签/基线，仅供审阅，不进入新读出 |

所有阶段结束后自动生成 `function_audit_v2_review.zip`（结果目录旁），包含上述数据。
`--stage pack` 可重新打包。结果不修改输入缓存、旧风险分数或自然标注。

## 验证与尚未解决的问题

用户上传的 v1 包（回答 12219）实际有 23 个单元、23 个原提案、11 个修复提案和
13 次语义检查，总计 47 次生成调用；接受和完成均为 0。8 个单元最终因 token 重复拒绝，
2 个因组内表达数不合格拒绝，13 个被语义检查拒绝，其中 7 次理由转向原问答的信息充足性。
7 个独立编号 `1.` / `2.` 等被误当候选单元。包中的 baseline NPZ 是旧结果，不是新测量。
这些数据不能说明 FFN 支持或抑制证据，也不能计算新的 AUROC。

已修复嵌套角色控制 token、原问答指令混入候选任务、编号切分、语义拒绝无法反馈重写、
零测量仍按成功退出的问题。修复后的真实 8B 候选接受率尚待 prepare 验证；没有降低
组内同义要求来通过无效候选。该回答采用新编号切分后为 16 个单元，不改任何原始 token。

小随机 Llama/Mistral 核验原生短语概率、有限差分根敏感性、原生 FFN Jacobian、
GQA/滑动窗口、纯 RMS 缩放、熵/KL 链式恒等式、准确 token 对齐与钩子清理。
流水线测试使用固定候选 fixture；未验证真实 8B 候选生成质量、语义分组正确率或检测性能。

这一步解决的是可观测作用的混淆。它尚未提供来源适用性的真值、完整的表达/事实
神经子空间分离或可识别的隐状态真假模型。先检查正常组织表达与错误对象绑定
是否能被这些对照区分，再决定如何冻结无标签检测读出；不会事后挑对照方向来提高四答 AUROC。

## 文献依据

- Promote, Suppress, Iterate (EMNLP 2025): https://aclanthology.org/2025.emnlp-main.815/
  正确候选也会被抑制以避免重复；本实现没有复制其 knockout。
- Confidence Regulation Neurons (NeurIPS 2024): https://arxiv.org/abs/2406.16254
  归一化路径调节置信度，不能由直接投影符号概括。
- How do Language Models Bind Entities in Context? (ICLR 2024): https://arxiv.org/abs/2310.17191
  区分实体/属性与绑定；此处未假定其绑定子空间可以直接迁移。
- Semantic Entropy (Nature 2024): https://www.nature.com/articles/s41586-024-07421-0
  区分含义与表达；这里使用受控有限 bank，而非复现其采样估计器。
- Tuned Lens: https://arxiv.org/abs/2303.08112
  跨层读出存在漂移；本实现只做最终 FFN 同一读出，不假称已训练逐层语义 lens。

软件验证：本次 33 项针对性测试通过，覆盖旧截断缓存的有限重试和复用、失败后继续下一单元、
完整围栏 JSON 解析、不接受截断对象／字符串布尔值，以及脚本路径、参数和失败退出码。
保留真实 FastTokenizer 的控制 token 隔离、编号切分、候选任务与原生前缀分离、
语义修复判决独立保存和零接受失败打包验证。
保留 token 重复、有限修复、失败缓存续跑、拒绝后继续采集及完成缓存复用验证。
前次 18 项旧 choice-state/pack 回归在独立进程通过。
旧测试含全局 `torch not in sys.modules` 断言，不能与加载小模型的测试混在同一进程。
运行环境为 CPU PyTorch 2.14.0、Transformers 4.57.6；未下载或执行真实 8B 权重。
新增核心代码按采集、候选、读出、编排拆分；没有修改旧基线公式。
