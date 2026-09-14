# S11：熵入口与真实局部端点继承

状态：可执行实现和本地软件验证；尚无新的自然 RAGTruth 成绩。保留 S10 与绑定投影代码、历史结果，不把新方法的构造例或 oracle 诊断当成检测成绩。

## 为什么改

S10 的 M[t]=max(H[j] exp(-(t-j)/4)) 只是最近八步的时序记忆；不知道后续读了哪个 token。两个逻辑回归目标是 error 和所有标注 span onset，不是“先用 onset 推断再传播”，也没有独立 continuation 模型。其 0.7791 是 span onset 而非每答第一次错误。论文 First Hallucination Tokens Are Different from Conditional Ones 支持分开检查起点和延续，但没有证明注意力高就导致幻觉。

本版真正测两个问题：
1. 熵及其变化能否提名错误片段的入口？
2. 控制当前局部关注总量后，读到的具体历史 token 是否携带一个尚未衰减的错误入口风险？

这不是事实语义验证器。正常的纠正、否定也可能读取错误历史，所以结果允许否定第二个假设。

## 1. 明确索引和数据

完整输入是原始 prompt + 已有 response，沿用 reanchor 的 inputs.jsonl 精确 token IDs、来源 mask 与字符 offsets，不改模板，不重新生成答案。生成 response[t] 的 query 是 P+t-1，历史来源 response[j] 是 P+j。

```
A[t,layer,head,d-1] = attention[layer,head,P+t-1,P+t-d]
d = 1..min(W,t)
```

d=1 对应 query 自身（刚输入的 response[t-1]），是有效的前一个回答 token，不是偷看当前 target。第一回答 token 没有历史边。默认 W=16，保存所有 layer/head 的局部带；不截断原模型上下文。

每条回答只做一次新的正常前向。hook 在每层读出带状权重后，仅将返回的诊断 attention 项换成 None，释放大矩阵；不改 attention 输出或残差。必须用 eager；缺权重报错，不把五列标量伪装成图。最终熵、top1/top2 negative margin 在小块完整词表 logits 上计算。与旧脚本的最大差异是记录端点而非只存总量；BF16 行和用完整 attention 行归一化纠正，**从不单独归一化局部窗口**。

原缓存只有五列/来源总量，因此无法零次前向恢复这些端点。新缓存逐样本 NPZ，可续跑；模型/输入/window 不一致拒绝复用。teacher forcing 对其他生成器的回答是 observer 分析，不是那些生成器的原生轨迹。

存储规模 O(T×L×H×W)，没有 T×T×D 的向量消息，也没有逐事件 JVP。eager 前向自身仍需要一个层的密集 attention，长输入可能有显存成本。只在原始输入超过 --max-tokens 时明确失败，不静默截断。

## 2. 入口模型

```
a[t] = sigmoid(b + w_H H[t] + w_delta (H[t]-H[t-1]))
```

第一位置差分为零。只有这两项，**不使用当前实际 token、真实历史错误或真实起点**。标签是每个标注错误 span 的首 token；整答首错只在评价时单列。

使用自然 train 标签训练线性读出，不称无监督。为第二阶段训练产生 seed 时，按来源做三折 cross-fit：每条训练回答的 seed 来自未见其来源的入口模型。最终入口模型在全部 fit 来源上训练，应用于 calibration/test。没有训练期 gold-error history、测试期 predicted-error history 的替换。

## 3. 每个物理 head 独立传播入口风险

```
U[t,k] = sum_d A[t,k,d-1] R[t-d,k]
R[t,k] = a[t] + (1-a[t]) U[t,k]
```

k=(layer,head)。U 是继承量，R 是当前入口与继承量组成的风险状态。每个 head 内先计算，最后才进入读出，不提前平均 heads。

这是依赖图上“遇到入口即吸收”的递归概率代理：A 的剩余质量离开所研究的局部历史子图，因此不需要额外制造长度惩罚或固定 span。R 始终在 [0,1]。没有入口时永远为零；只有一个入口、每步继承质量为0.8的单链上，R 为0.8^距离。链可通过中继延长到远超16个 token。不能把它解释成原 LLM 的 WV/WO 有符号消息或严格跨层因果路径；不同层的通道在此独立运行，不是原生层序的消息 DAG。

不对每个 head 做独立 noisy-OR，避免“头越多风险越接近1”；不对所有早期熵作累积和。正常转向 prompt 或其他局部 token 时，原入口继承量可降低。若模型持续读取错误 token，量可持续。语义纠正的错误指代仍可能保持高风险，本方法没有预先解决这点。

## 4. 延续读出及两个目标的连接

每个位置输入 [U[t,k], local_mass[t,k], evidence_mass[t,k]]。其中 local/evidence 质量控制一般局部阅读强度，检验 U 是否增加具体端点信息。使用最多64个物理通道：只在 fit 来源、无标签地按局部质量的来源均衡方差选取。候选、顺序在真实/对照图中固定；--max-channels 0 保留全部。原始缓存始终保留全部通道。

用非 onset 的训练位置拟合条件延续概率 c[t]。对 U 对应的系数施加非负约束，不能靠把“继承更多意味着更正常”翻转后仍宣称验证了错误自强化。其余质量项可正可负。标准化加L2的有界线性逻辑回归，不训练GNN或LLM。

```
p_error[t] = a[t] + (1-a[t]) c[t]
p_continuation[t] = (1-a[t]) c[t]
```

生产递归永远只使用预测 seed；**不把最终 p_error 再无条件滚动喂回去**。最终读出与 R 的含义不同：R 是入口祖先的传播量，p_error 是从自然标签学习的风险估计。此设计避免读出自己的假阳性作为新事实无限重复计数。概率形式不保证跨数据集已校准。

## 5. 同等输入的对照

- real：真实历史端点，允许多跳。
- uniform：逐行/逐head保持局部总质量，对可见历史端点均匀分配。检查是否只需要时间邻近。
- permuted：逐行重排现有端点权重，所有head共享该行置换；保留每head质量、权重集合。检查具体指向。
- one_hop：只读取过去入口 a[j]，不读取继承状态 R[j]。检查中继是否有用。
- no_edges：U恒为零，保留相同的local/evidence质量。检查读出是不是只利用静态统计。

每个对照有相同训练样本、相同读出维数和非负约束。另训练同队列 instant 基线，避免拿重采集数据与旧0.7492直接作差。

另外输出真正无需幻觉标签评分的 raw control：来源均衡的训练熵参考分布，最高5%尾部线性映射为seed，然后相同 real/uniform 传播，最终取最强通道读数。该分数是无标签排序对照，不是校准幻觉概率；参数固定，不按test选择。

## 6. 诊断不是生产分数

所有预测、模型及校准阈值保存冻结后，才加载test标签计算：

1. oracle_seed_real：只在真实 span 起点注入1，沿真实边传播。
2. oracle_seed_uniform / permuted：同样真实起点，不同端点。
3. oracle_history_real：用每个过去token的真实error作为单跳来源状态。

单独写 *.oracle.npz / oracle_diagnostic.json，永远不进训练特征或正常评分文件。它们不是可部署检测器，也不是形式上的性能上界，只用于定位瓶颈。重点看 continuation_vs_normal 与 strict_post_first，不能拿 oracle 的起点识别率做成果。

判定：oracle真实边也不如uniform/置换，则具体历史连接没有提供预期信号；oracle有效而预测seed无效，则优先修入口；real不胜one_hop，则多跳未获支持；real不胜no_edges，则收益只是静态质量。不能只拿最终一个AUROC解释所有情况。

## 7. 数据与评价

按 task×generator 各自训练和报告，official train来源80/20分fit/calibration，official test来源互斥。所有gold span起点/整答首次错误/延续/严格首次后分别评价，沿用binding_detector.evaluation的七个口径。正常回答完整保留；输出 pooled 和source-balanced AUROC/AP及共同来源bootstrap差。

报警阈值改为**完全正常校准回答的最大分数分位数**，而不是正常token的5%分位数。报告正常回答至少一次报警、首次错误准确位置召回、首次前任意报警和延续召回。正常校准回答不足时阈值为null，不伪造5%控制。历史数据的复用、模型偏移不满足严格交换性时不作误报保证。

## 8. 一键执行

在 graph 根目录，使用现有 research Python，QA默认与S10保持同一generator队列：

```bash
python main.py transport \
  --population ../reanchor/outputs/ragtruth_population_20260912 \
  --features outputs/s11_local_attention_qa \
  --output outputs/s11_two_stage_qa \
  --tasks QA --generator llama-2-7b-chat \
  --device cuda:0 --window 16 --max-channels 64 --bootstrap 200 --resume
```

默认从 population/settings.json 取原observer权重路径，可用 --model 显式指定。annotations 默认沿用本项目服务器路径；可用 --annotations 覆盖。`--tasks all`覆盖三类任务，`--generator all`覆盖所有原回答生成器；全量耗时随回答数增加，不需要逐事件干预。

已完成capture，重跑CPU方法比较：

```bash
python main.py transport --phase fit \
  --features outputs/s11_local_attention_qa \
  --output outputs/s11_two_stage_qa_repeat --bootstrap 200
```

结果目录不可覆盖，capture可续跑。输出每个task__generator下：models.json、thresholds.json、prediction_freeze.json、evaluation.json、oracle_diagnostic.json、每答NPZ。每答保存a、各臂error/continuation概率、实际物理通道、R/U及最大末跳父位置，支持逐样本解释。不输出假定的事实真值。

测试：`python -m pytest tests/test_two_stage.py -q`。无下载的tiny Llama集成测试仅在transformers已安装时执行；其他测试包括真实PyTorch运算的读取钩子等价、多跳闭合、因果截断、source-cross-fit、test标签改变不影响分数、数值范围、完整合成训练与oracle隔离。本地未具备真实8B权重或RAGTruth缓存，不声称自然有效。

依据：First Hallucination Tokens Are Different from Conditional Ones (arXiv:2507.20836)；Lookback Lens (EMNLP 2024, 2024.emnlp-main.84)。它们不提供本方法的有效性保证。
