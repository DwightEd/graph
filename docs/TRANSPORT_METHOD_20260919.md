# 从完整候选回查原生消息，再检验多头关系

本轮基于 graph `bd5ae7d`，继续使用 reanchor 的原始 token IDs 与同题重采样。
文献核查见 [18 项相关工作的采用边界](LITERATURE_TRANSPORT_20260919.md)，
此前观察量与负结果见 [证据台账](DETECTION_CONVERGENCE_20260919.md)。
这次实现的是原生功能测量与干预流程；自然无监督检测的训练和成绩仍未完成。

## 1. 问题怎样收敛

服饰错误前缀中的 L25H4/L26H31 仍可增强有依据候选；L31H14 的符号受前缀和面板影响，
原 history 大效应主要由当前 query 自身复现。洋葱则有多处支持消失，未呈现同样强烈的负向末层效应。
这些观察不支持一个固定的“坏 head”或统一的“历史更多就错”。两题的自然前缀不同，不能把它们
当作只改变真假的因果对照；72 行也不是 72 个独立来源样本。

上一版仍有四个缺口：按 local lens 和人工来源组选点；只在最后一个 query 删除；
只配对同来源组的 head，漏掉 evidence/self 等跨来源组合；首子词选点容易定位语法/措辞因素。
本轮把主线改为：**完整候选 → 原生反向敏感性 → 具体消息 → 联合效应 → 下游恢复**。
来源角色在定位之后追加解释，不参与挑选 head 或 source。

## 2. 测量对象和公式

每条消息有完整身份 e=(layer, head, receiver, source)：

\[
m_e=W^O_{\ell,h}\big(A^{\ell,h}_{r,s}V^{\ell,h}_{s}\big),\qquad
V^{\ell,h}_{s}=W^V_{\ell,h}\operatorname{LN}(x^{\ell-1}_s).
\]

GQA 的 KV head 按原模型对应到各 query head；保留 route mass、value energy、message norm，
不平均 head。当前直接 self 指 s=r，和“来源是最终 query”分别记录；二者在早期 receiver 上不同。

默认目标为完整候选：

\[
M=\log p(c_{1:k}\mid P)-\log p(w_{1:m}\mid P).
\]

首词统一来自同一次 prefix 前向，两个 continuation 只补上各自第 2 个词起的 logp。
这是现有 `scoring.py` 的规范口径，可避免把不同长度前向的首词舍入差混入比较。
完整候选不等长时仍有长度偏好，`mean_margin` 和各词 logp 也保存，但不把平均值叫概率。
首词相同、后缀不同的候选现在仍能定位消息。

令 g_e 是仅缩放这条原生消息的 gate，在原计算 g_e=1 处求导：

\[
a_e=\frac{\partial M}{\partial g_e}
=A_{r,s}^{\ell,h}\left\langle\frac{\partial M}{\partial z^{\ell,h}_r},V^{\ell,h}_s\right\rangle.
\]

z 是进入 W_O 前的 head 输出。反向方向来自实际最终目标，经过后续 Q/K/softmax/V、
归一化、残差及 MLP。并非固定下游 attention 的 rollout，也不是局部 unembedding 投影。
在同一接收位置，不同来源的 a_e 可相加；跨层相加会重复计算同一影响，不能当总因果贡献。
消息 gate 改变 value 写入，不重归一化 attention；它不同于删除输入 token。

每条候选续接单独前向、逐层 VJP，随后对同一 prefix gate 的导数求和。
候选后缀提供评价目标，**不允许把后缀 token 当干预位置**。
CPU 保存 decoder 输入和原调用参数，GPU 每次只保留一层反向图。不会构造完整 Jacobian，
也不会将所有 layer/head/source 的向量消息同时展开。逐层 hook 读取实际 attention，
关闭模型返回全部层 attention 的收集器，避免一次保留整叠二次方大小的矩阵。
三条分支增加计算量及 CPU checkpoint，
不是零成本方法；没有实测用户 8B 在 24GB 卡上的峰值或速度。

## 3. 选点与有限验证

先对所有层、head、prefix 接收/来源位置计算 gate 导数，再为每个物理 head 保留正负各 2 条边。
默认从中选择 8 条消息，正负交替且主候选的物理 head 不重复；另抽 2 条未选候选作为控制。
控制池也经过每 head 的梯度筛选，因此它不是所有原生边的均匀抽样，不能给出整体漏检率。
保存 node_signed/node_positive/node_negative 等 NPZ，保留候选截断前的逐接收位置统计。

候选和配对在任何有限干预之前写入 plan.json。默认 4 对，覆盖 opposed、co_support、co_suppress
中实际存在的类型；不同来源、不同接收位置可以配对。这些名称来自干预前的梯度符号，
不是根据结果重新挑选的正确性类别。梯度可能因饱和或抵消漏检；缺失候选不能证明无机制。

对每条消息使用删除剂量 d∈{0.25,1}：

\[
D_A(d)=M(1,1)-M(1-d,1).
\]

再执行同一前缀、同一目标、同一剂量的四世界比较：

\[
J_{AB}(d)=M(1,1)-M(1-d,1)-M(1,1-d)+M(1-d,1-d)
=D_A(d)+D_B(d)-D_{AB}(d).
\]

同时保存 A 在 B 存在/缺失时的作用，B 同理；首词与整段候选分别报告。
正 J 表示该候选对比下的正交互，负 J 表示负交互；并不自动对应“语义协同/错误抑制”。
不同层联合删除时，后层的待删消息在已干预状态上重新计算，因此 J 包含后缀网络响应。
同层 head 也可能因共享的非线性后缀而产生 J，不代表两头直接通信。

每个候选另有一次同位置、同范数的随机方向扰动。它检验幅度特异性，没有语义匹配保证。
不根据这些两题的结果重新优化选点阈值、头集合或检测方向。

## 4. 区分基线作用与删除后的补偿

当 A 在更早层、接收位置不晚于 B 时，执行删除 A 后将 B 的指定来源消息替换回原基线消息：

\[
R_{A\to B}=M(-A,B\leftarrow B_{full})-M(-A).
\]

此外，恢复 B 所在层、接收位置的整个 MLP 输出，单独保存结果；它的范围比一条消息更大。
若 D_A>0 而 R<0，说明恢复原组件会使删除后的 margin 更差，与该组件的适应性补偿相容。
这是有条件的诊断，不能将该 head/MLP 永久命名为纠错器，也不能把多个 R 相加。
同层或时间上不可能的 A→B 不执行这项恢复。

每个恢复同时有未删 A 的同世界 sham。基线恢复向量使用与删除完全相同的原生 source_write，
不使用低维 head-block 的近似投影替代。bf16 下不同长度前向仍可能有舍入差，
sham_delta 和 gradient_check 必须与小效应一起看。只检查所选 B 及其 MLP，阴性结果不排除其他补偿位置。

## 5. 检测方案：已经排除什么，接下来究竟学习什么

方案经过三个判断。其一，来源总量、集中度、异常簇不能提供稳定真假方向，已有自然结果也不支持继续
简单加权。其二，有符号多头竞争比总量更贴近计算，但正常选择也有竞争，J 不能直接变成幻觉分数。
其三，保留“当前候选的适用关系被表示了没有、在运输中被采用了没有”这一语义问题，
用本轮流程定位应当读出的原生状态和消息，而非先训练任意 GNN 再为分数找机制。

候选检测模型要学习的是对象—阶段/范围—值的联合关系 z，不是自然幻觉二分类：

1. 在受控数据中保持值不变而改变适用对象/阶段，并用同关系改写作 nuisance 对照；
   再做固定关系、只改变值的控制。所有 2×2 单元保持同一回答前缀和可比较候选。
   不以错误回答的已有说法重新定义原材料的合法关系，不启动任意来源错接训练。
2. 训练关系读出 Q_θ(z|原生状态/逐头消息)，监督来自程序构造的关系真值；
   留出实体、模板、措辞和来源。称为任务关系监督或自监督任务，不能声称零语义监督。
   绑定子空间/双线性关系读出是可检验起点，不先假定注意力里的最大地址就是该角色。
3. 要求读出对同关系改写稳定、对关系更换敏感；在明确对齐的位置换入关系表征，
   候选偏好应朝接收世界要求的方向变，而非仅复现 donor 的表面答案。
   比较只看值、只看条件、独立 head、联合 head，判定关系增量是否存在。
4. 冻结后，对来源独立给出的合法集合 Ω 计算 `-log Q_θ(Ω)`。
   复用已有 `binding_detector/projection.py`；关系缺失保留 unknown 及上下界。
   每个自然 token 都保留输出和缺测状态，不能只报告成功解析子集；不使用金标边界启动评分。

上述第 1–3 步的可靠自然关系提取和可迁移读出尚未实现，本轮没有伪造这些模块。
目前的两案例输入仍有人工候选与事实位置，所以它不能自动处理任意自然 token。
反向扫描虽不读取来源角色，也不等于整个实验无语义监督。
现有无标签 HMM 和其他基线仍可独立验证，不以完成全部机制研究为其前置条件。

停止条件明确：如果关系读出只跟词面或值变、干预只搬运 donor 答案、联合 head 不胜独立 head，
或自然来源上的首错/恢复没有增量，就不把该关系写成检测主机制；不靠再叠加异常特征挽救它。
后续确认要用未反复探索的独立来源，分首错、延续、恢复评价，并与纯过去分数、节点读出比较。

## 6. 代码与运行

| 文件 | 职责 |
|---|---|
| target_gradients.py | 原生 decoder checkpoint、逐层 VJP、完整候选导数、带符号来源消息 |
| transport_plan.py | 干预前选消息与跨来源 head 配对；无角色/标签读取 |
| transport_trials.py | 复用 native/scoring，运行单删、双删、随机方向和恢复，保存 NPZ |
| transport.py | 编译原案例、冻结设置与 token IDs、断点续跑、追加文本角色 |
| transport_report.py | 汇总四世界及恢复结果，打包复核文件 |
| native.py | 增加明确 receiver/source 的干预和原生消息替换；旧默认行为保留 |

在 graph 根目录：

```bash
git pull --ff-only origin main && python -u main.py flow
```

默认 `outputs/target_transport_v3/`，读取原 reanchor 样本，不重新采样。
有逐案例/逐层/逐实验进度条。重复同一命令续跑，已存世界不再计算；
CSV 浮点往返采用 round_trip，避免精度变化触发 plan 变更。
配置或原始 token IDs 变化仍拒绝混用；不将 v1/v2 缓存伪装成 v3。

输出 `transport_edges.csv`、`transport_interventions.csv`、`transport_interactions.csv`、
`transport_mediation.csv`、`REPORT_TRANSPORT_zh.md` 与 `flow_review.tar.gz`。
NPZ 留在 `panels/*/discovery.npz`、`baseline.npz` 和 `worlds/`；复核包包含 CSV/JSON/说明。
只重建报告：`python -u main.py flow --stage report`。

旧 v2 复算入口继续存在：

```bash
python -u -m experiments.path_conflict.flow --output outputs/evidence_target_flow_v2
```

本轮使用本地随机初始化的微型 Hugging Face Llama/GQA 验证完整反向与逐层 VJP、
有限扰动、相同首词的完整候选梯度、过去接收位置、消息/MLP 同世界恢复、因果顺序、
无模型调用续跑以及 float32/bfloat16/float16 导出。72 项针对性测试通过；
首词目标逐层 VJP 与独立全模型反向的接收位置汇总最大绝对差为 1.49e-8。
相同首词、不同后缀的候选测试中，整段目标重放差为 1.88e-8，
所选 gate 的中心差分与梯度差为 7.46e-5。
bf16 的来源求和与原生 head 输出投影仍有约 4.92e-4 的最大舍入差，
不能忽略它而宣称浮点计算精确守恒。验证记录见
`results/target_transport_20260919/validation.json`。这些是软件和数值验证，
没有运行用户 8B 自然样本，没有新增 AUROC/AP 或宣称发现普遍机制。
