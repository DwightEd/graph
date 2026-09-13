# source 上下文一致性：本轮方法与证据边界

更新时间：20260912_115530 UTC。状态：本轮已完成，当前读出不获支持（pending Codex review）；旧路径残差为冻结负结果基线。

## Problem Anchor

在不预先标注实体关系、不使用幻觉标签训练检测器的条件下，从冻结模型的内部计算识别重新读取远处证据的事件，以及约束上下文是否与当前对象保持一致；在来源隔离数据上评价首错当步或提前报警，并用同输入非图对照检验连接结构是否必要。主线代码在 graph，reanchor 只承载机制实验。

## 文献依据与差异

- Feng & Steinhardt, ICLR 2024, [How do Language Models Bind Entities in Context?](https://arxiv.org/abs/2310.17191)：用实体/属性位置的激活替换研究 binding ID。归属敏感性与激活替换本身不是本项目的新贡献。
- [Representational Analysis of Binding in Language Models](https://aclanthology.org/2024.emnlp-main.967/), EMNLP 2024：归属表征可能携带出现顺序。必须交换顺序、保留归属真值；不能把顺序或实体清单当检测输入。
- Frasca 等，[CHARM](https://arxiv.org/abs/2509.24770), ICLR 2026：attention 与激活组成属性图并训练 GNN。图加隐藏状态不是新贡献。
- [TOHA](https://arxiv.org/html/2504.10063)：包括无需幻觉标注的 copying-head 选择变体。无监督 attention 拓扑本身不是新贡献。
- Karim 等，[Attention Deficits in Language Models](https://arxiv.org/abs/2602.19239), 2026 预印本：在已知单 token 候选集合的程序任务中研究 gating/binding 与读出失败。其 probe 或 oracle checkpoint 不等于自然生成中的无监督归属检测。

研究缺口是检测时不提供正确实体—属性关系，仍能从计算连接读出对象上下文失配；这个缺口是否能被当前最小算子填补，必须由实验决定。

## 路线选择与复杂度

路线 A：复用现有 candidate-conditioned path residual，先取得完整样本上的真实自然检测结果。
路线 B：对可在 source 中找到的候选 token，比较其 source 上下文与近期回答通过历史中继读取的 source 上下文。
本轮同时保留 A 作为已实现基线、B 作为一个可删除的机制假设；不拼接两类分数，不训练 AE/GNN/probe，不增加人工实体解析器。

## B 的精确定义

attention 记作 A_l[h,i,j]，接收者 i 读取 j。默认只用相邻两层，前层 head 平均、末层 head 先平均作为最小实现。主层对固定 (15,16)，(30,31) 仅报告敏感性，不能按自然标签选层。

对预测位置 q，候选 C 仅从 q 的原生 logits top-4 得到。S 为非特殊 source token，H 为 [q-16,q) 内的 response token。

1. 候选出现位置 J_c={j∈S: token_id(j)=c}，不使用正确值或对象清单定位。
2. 每个 j 的上下文 B_j(k)∝A_(l-1)[j,k]，仅 k∈S、k<j，剔除所有候选 token 身份后归一化。只承认向过去的层方向。
3. 当前对象上下文 T(k)∝Σ_(u∈H) A_l[q,u] A_(l-1)[u,k]，在与 B 相同的 source key 集合上归一化。
4. 候选上下文兼容度 G_c = Σ_(j∈J_c) A_l[q,j] cosine(B_j,T)。归一化得到候选间相对读数，仅作支持分布，不命名为正确概率。
5. 无路径对照 D_c=Σ_(j∈J_c) A_l[q,j]；端点对照将 B_j 在候选出现位置间置换，保持 D；数值词对照仅计数 J_c。
6. 风险方向预声明为 1-G_top1/Σ_c G_c；相同定义用于 D。无覆盖、任一已匹配候选端点缺上下文、零历史上下文质量或仅一个有覆盖候选都标记不可判定。各对照独立报告有效覆盖；端点随机置换seed固定20260912。

这是 attention 路径兼容性，不是 value/O/MLP 的真实信息贡献，不声称纯二阶因果交互。主要失败风险：对象上下文可能已失配、顺序/词形仍混杂、source 只有单句时前向上下文不完整、合法综合多处证据导致低兼容度、候选不能精确复制 source。

## 回看定位（本轮未实施）

下一轮拟沿用 local_to_remote 的共同可见 key 与当前远近划分，并验证特殊 token 排除；窗口 8/16/32，主窗口16。与旧 W1+过去基线比较。只使用过去或 fit 来源确定阈值，固定事件预算与去重间隔。先报告连续分数及边界误报；已有手工自然窗口不能用于声称自动定位率。

## 两项待证主张

C1：上下文路径读出在未输入归属标签的情况下跟随构造事实的归属交换，并比纯数值词/直接读取更稳健；必须覆盖顺序变化、干扰前缀及实际错误条件。
C2：在完整来源隔离自然回答上，结构信息提高固定报警预算下的首错召回，且超过 entropy/margin/position 及无路径对照。
本轮两项均未获支持：C1主层没有优于直接读取且128条件无模型实际错误；C2未取得检测优势，等实际预算仍待验证。B保留为诊断，停止接入主线。详见 [本轮完整报告](../docs/METHOD_ITERATION_20260912.md)。

## 数据与原始设置

旧 graph 运行计划256条，feature 每种表征1920维、候选4。缺完成 manifest，层/路径布局不能由维度反推。先校验并保存完整样本，恢复存储与原始特征逐项一致；保留缺失/部分样本与未知 provenance。此批只能作为有明确边界的探索性基线，不伪造原捕获 manifest，也不与新配置混拼。

reanchor 13 个已有案例复用完成记录；新增对照用独立目录。Llama-3.1 replay Llama-2 回答与原模型采样机制分别报告。

## 实施接口

graph/route_graph/archive.py：按回答校验、原子保存、恢复读取，保留数据/文件哈希和失败清单。
graph/route_graph/context.py：纯数组输入的两层上下文路径读出与对照。
reanchor/src/decoding/binding_validation.py：构造对照、一次模型加载、逐条件持久化与同设置复用，调用 graph 的纯算子。
来源隔离参考复用 SourceReference；标签只在分数落盘后由评价入口接入。

## 评审状态

本环境未提供 Codex MCP reviewer 工具，也没有 codex CLI。REVIEW_UNAVAILABLE；不冒充跨模型审查通过。执行可逆的小型证伪实验，并通过接口测试和独立代码审查弥补工程检查；这些不替代研究有效性证据。

最终候选核验：仅62/128条件同时覆盖两种构造数值；原始/逆序仅5/32、6/32。其余source候选可能是空格/标点，不能把32/32读出当作两数值归属识别。此边界进一步限制C1，详见完整报告。
