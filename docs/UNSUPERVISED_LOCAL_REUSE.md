# 无监督双阶段局部复用检测（2026-09-14）

本版本基于 main `915c0bd` 核验并发布，不删除 S11。S11 主流程的入口/延续训练及正常回答阈值仍使用标签，故只保留为显式监督基线；本版不导入这些权重。

## 问题与旧版缺口

S10 的监督 error/onset 头分别预测全部错误和每个 span 起点，不是“首错后切换延续模型”。
原方法把远历史质量（排除近16token）和近期熵记忆相乘，无法知道当前到底读了哪个 token。
用户补评的 strict_post_first AUROC .6124 不能当作连续片段定位已成功；84起点也不是52个整答首错。
本版直接检验“疑似起点→读取它或其中继→持续/停止”这一假设，不沿用监督权重。

## 1. 起点：参考混合分布中的高熵异常，不是真值标签

从原始 teacher-forced 前缀得到 H_t=-sum_v p_t(v) log p_t(v)。固定官方train来源80%作为
无标签reference，20%作为无标签calibration；官方test来源不参与任何拟合。
参考中允许存在幻觉，不挑“正确样本”。按task/generator分组，绝对token位置分箱为
0、1–3、4–15、16–63、64+。样本不足时仅退回同task/generator总体参考。
每个来源等权，每来源每箱最多固定均匀取512个位置。位置箱只依赖当前t，不依赖答案总长。

记参考加权经验分布的左侧质量为 F_<(H)，保守的尾部量为：

```
p_tail(t) = [1 + N*(1-F_<(H_t))] / (N+1)
onset_rank(t) = 1-p_tail(t)
e_t = max(0, 1-p_tail(t)/alpha), alpha=.05
```

并列熵按 >= 处理；恒定参考不会把全部token制造成异常。该尾部量只是经验异常排序，
不是以正常类为条件的p-value，也不是贝叶斯幻觉概率。raw_entropy与连续onset_rank同时保留。
稀疏e_t只负责播种，可能漏掉高置信/低熵错误，不能强称覆盖全部幻觉。

## 2. 延续：真实局部读取的折扣递推，不做风险累积

response零基下标t的预测query是q=P+t-1。局部lag k=1..W的来源为response[t-k]，
实际key是P+t-k；t-k<0时边为0。尤其lag1是query的self key（已经生成的上一个token），
并非未生成的当前token，不能误删。默认W=16，保存所有层/头的这些物理权重：

```
A[t,l,h,k] = native_attention[l,h,P+t-1,P+t-k]
```

权重经过完整因果softmax后截取，不重新在局部集合内归一化。因此source、特殊token、
远历史和未标记历史可以吸收权重；仅对<=1e-5的浮点行和溢出做数值修正。

```
u[t,l,h] = gamma * sum_k A[t,l,h,k] * r[t-k]
i[t] = fixed_quantile_0.90({u[t,l,h]})
r[t] = max(e[t], i[t]), gamma=.90
```

每个head独立产生继承读数，最终用固定次序统计量汇总，不在建图前平均head。
前一token的r是共享标量，所以中继允许切换layer/head；每一步仍保存其物理通道。
这是生成位置上的诊断依赖图，不是Transformer沿层的精确计算图，也不是JVP。
多个局部步骤可连接超过W的span；没有固定8步错误标签，也不使用真实span结束点。

无新种子时满足：

```
r[t] <= gamma * max_{t-W<=j<t} r[j]
```

因此不会像noisy-OR/累计max一样越生成越高。若不再读取任何带风险的节点，继承为0；
若只反复传同一风险，每次传递都会衰减。但gamma是预先固定的折扣，不是测出的事实控制半衰期。
分数减小只表示这个局部读取代理减弱，不能据此证明语义已纠正。

每token保存完整channel_risk/channel_inherited、dominant_parent/dominant_origin。
parent/root只说明最大加权项的来源，不等于所有因果路径或唯一的语义来源。

## 3. 必须能失败的对照

- seed_only：只用相同起点，不传播。
- single_hop：只读取原始e_j，不读取已继承的r_j。
- mass_matched_uniform：每行保持完全相同的局部质量，但在已有局部位置均分。
- lag_group_permuted：固定lag1，在2–4、5–8、9–W中置换端点权重；每head/行质量保留。
  所有heads使用同一位置置换，保持通道间对应关系。很短的窗口对照可能不改变，不能强称充分随机化。
- raw_entropy、raw_negative_margin（native top1-top2的负值）、总历史减来源、远历史减来源和位置基线。

若真实端点不比同质量对照好，拒绝“精确局部复用提供增量”的假设，不通过测试标签翻向。
本版不使用gold-onset播种；所有labels只进入最后评价。也不使用S10监督输出作种子或融合权重。

## 4. 原生采集与计算成本

输入复用 reanchor/outputs/ragtruth_population_20260912/inputs.jsonl 的原始token IDs与source mask。
不重新套chat template、不重采样、不把错误回答替换为正确回答。
Llama SDPA执行正常的无缓存整前缀backbone forward；hook读取原来的Q/K/V和post-attention pre-WO值。
复用传入的RoPE cos/sin，在response-query块上重算完整因果softmax，只保存局部端点及来源质量。
逐层核对重算A*V与原生pre-WO context的相对L2误差（最大允许.02），失败不输出有效缓存。
hook不修改原输出；当前层的Q/K/V在处理后释放。最终LM head也按块计算，避免全prompt×词表logits。
保存的local_attention为float32 [T,L,H,W]；例如T=200,L=H=32,W=16约12.5MiB（压缩前）。
每个响应只需一个backbone forward，不随种子数增加，不存所有层的全T²注意力。
首次仍须补采真实局部边：旧S10五列或population的分组总量不能推断出缺失端点。
已有完整新缓存逐样本复用；捕获和评分设置分别冻结，失败样本原子保存，不伪造完成状态。

支持普通dense Llama/Llama3 GQA，不假定支持QK-norm、sliding-window或tensor-parallel改版。
模型、dataset路径默认从population/settings.json获取，加载local_files_only，代码不下载模型。
参考/评分代码不依赖transformers；采集需要用户research环境的torch和transformers。

## 5. 无标签阈值与评价

无标签calibration来源上每个来源取其回答/位置的最高分，阈值是这些最大值的95%分位数。
这是混合样本异常预算，不是利用正常标签取得的5%FPR，也不保证测试集或正常回答的实际误报率。
混合参考可能很保守：高置信低熵错误会漏检，而语言组织的正常高熵可能误报。
所有模型/阈值/预测落盘后才打开response.jsonl，重新运行evaluate绝不改变分数。

主要评价views与既有补评一致：
all_error、span_onset_full_stream、span_onset_vs_normal、first_error_full_stream、
first_error_until_first、continuation_vs_normal、strict_post_first。
首错/起点主读出预定为onset_rank，其他主读出预定为reuse；所有方法在所有views完整报告。
每个视图报告pooled和source_fixed_full_answer：来源权重先用完整回答token数确定，之后应用
gold截断mask，绝不在截断后重新赋予早错来源更大的每token权重。
来源bootstrap对paired指标差使用同一source重采样，按已有排序计算，不反复跑模型。

还评价：正常回答任意报警、首次错误前提前误报、精确首错召回、连续错误run召回、
命中run的延迟，以及错误run结束后8token内正常位置的风险残留。后者帮助检查停止问题。
这些都用gold仅做事后评价，不能作为无监督特征输入。

## 6. 运行命令

```bash
# 测试：其中HF tiny Llama在有transformers时实际运行；本地无该包会明确skip。
python -m pytest tests/test_unsupervised_reuse.py -q

# 全部3任务、官方train/test，原回答生成器固定Llama2-7b；observer从已有settings读取。
python main.py --population ../reanchor/outputs/ragtruth_population_20260912 \
  --output outputs/unsupervised_local_reuse_v1 --tasks all --generators llama-2-7b-chat \
  --device cuda:0 --query-chunk 16 --window 16 --resume

# 只补评现有新分数；不会训练、不打开GPU。
python main.py --population ../reanchor/outputs/ragtruth_population_20260912 \
  --output outputs/unsupervised_local_reuse_v1 --tasks all --generators llama-2-7b-chat \
  --phase evaluate --resume
```

默认捕获和评分全部所选样本，不能用completed-only悄悄改分母。中断后同命令resume；
若要工程smoke，用新目录+`--phase capture --limit 2`，不得称自然实验结果。
要改变窗口或科学设置，使用新输出目录；query-chunk也属于捕获数值契约。
all默认读settings指定dataset下的标签做最终评价；完全不碰标签可只跑capture和score阶段。

## 7. 已验证与未验证

本轮本地：35项测试通过，1项HF tiny Llama因没有transformers跳过；Torch GQA/RoPE/
SDPA测试、bfloat16数值一致性、hook不改输出、前缀不变性、多跳/跨head/端点对照/衰减、
标签隔离、样本续跑及完整合成评估通过。真实8B与自然数据未跑，不报告新AUROC/AP。

依然没有解决：高置信首错无熵峰、读错误内容是在纠正而非赞同、长距离非局部依赖、
以及首token与延续token间语义绑定的精确判定。当前新模块检验的是用户提出的低成本结构机制，
不是宣称任何两阶段设计都天然有效。若lag对照同样好，则保留负结果，不能再将时间平滑命名为信息流。

实现参考（仅API，不是检测效果依据）：
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
- https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook


## 8. 与信息论研究的准确关系

代码已经实现的是下一词 Shannon 熵、无标签经验尾部和局部复用递推；没有实现或估计
`p(score|hallucination)`、`p(score|supported)`，也不把风险叫作真值后验。
First Hallucination Tokens Are Different from Conditional Ones 为区分入口与延续提供经验动机，
并没有证明后续高 attention 的语义是“支持错误”。

Hallucination is a Consequence of Space-Optimality 在其随机事实/稀疏成员判断设定下，
刻画事实与非事实分数分布 KL 的最优存储代价；它不证明正确和幻觉的 attention 图必然
可分，也不保证本模型的局部 risk 递推。高置信分数碰撞提醒我们：没有熵异常的错误入口
不能靠传播器凭空发现。

未来信息论检验目标可写为 I(Y;G|C,Z)：在置信度 C、任务/位置条件 Z 相近时，
来源端点/绑定信息 G 是否仍增加判别信息。这里 Y 的金标只能进入独立评价，不能回流
调参数。real-uniform 的 AUROC/AP 差只是结构增量检验，不等于已经估出了该互信息。
另一个 binding_detector 原型计算明确合法绑定集合上的最小 KL 修正量，但需要额外语义
关系输入，不应和本条纯原生 attention 流程混为已完成的算法。

一个当前递推的严格限制：因为 `gamma<1` 且局部行和不超过1，
`max_t reuse[t] == max_t seed_only[t]`。因此同一阈值下“整答任意一次超阈值”的判定与
seed-only 相同；传播只会改变答内定位、token召回及误报延续，不会改善基于全答最大值的
回答检测。当前同时保存 onset_rank 与 seed_only，前者不应与后者的尺度混同。
窗口内的风险可因重新读取旧高风险节点而回升；严格约束是不能超出历史最大种子，
不是每一步必然单调下降。不将此性质包装成已验证的事实控制半衰期。

文献：
- https://arxiv.org/abs/2507.20836
- https://arxiv.org/abs/2602.00906
