# IRIS / RAUQ 方法与公开实现细读（2026-09-13）

范围：只提取方法、训练目标、推理接口与可复用结构；未运行作者代码，未开展 GPU 实验，不给科学有效性审查结论。使用 research-lit Step 2 的独立文献提取流程。来源贡献为 web（原文与作者仓库）。确定性 citation helper 不可用：两篇均保留 `UNVERIFIED (helper unavailable)`；下表的人工核验不是 helper 验证状态升级。

| 文献 | 人工一手核验 | 代码版本 | 确定性状态 |
|---|---|---|---|
| Ponhvoan Srey, Xiaobao Wu, Anh Tuan Luu. *Unsupervised Hallucination Detection by Inspecting Reasoning Processes* | [EMNLP 2025 正式页面](https://aclanthology.org/2025.emnlp-main.1124/)，DOI 10.18653/v1/2025.emnlp-main.1124；[arXiv 2509.10004v1 全文](https://arxiv.org/html/2509.10004v1) | ponhvoan/iris @ `d7e81561a7f262fcb2da662a9a3e07e2e6daf83f` | UNVERIFIED (helper unavailable) |
| Artem Vazhentsev, Lyudmila Rvanova, Gleb Kuzmin, Ekaterina Fadeeva, Ivan Lazichny, Alexander Panchenko, Maxim Panov, Mrinmaya Sachan, Preslav Nakov, Timothy Baldwin, Artem Shelmanov. *Efficient Hallucination Detection for LLMs Using Uncertainty-Aware Attention Heads* | [arXiv 2505.20045v3](https://arxiv.org/abs/2505.20045v3)，2026-06-17 更新，journal reference 为 ICML 2026；[会议目录](https://icml.cc/Downloads/2026)亦列出该题目。旧 v1 题目为 *Uncertainty-Aware Attention Heads: Efficient Unsupervised Uncertainty Quantification for LLMs* | mbzuai-nlp/rauq-hallucination-detection @ `12dbc0afe173c5987db11eec06a8db4a0cdc32f3` | UNVERIFIED (helper unavailable) |

## 1. IRIS：论文方法（§2.1–2.3、Eq.1）

输入是待判断陈述；代理 LLM 另生成事实验证文本。特征取验证响应末位置的末层向量，不是原始生成过程的逐 token 图。软标签取验证的不确定性，候选是归一化熵或口述正确概率；后者表现更好。小 MLP 为 d→256→128→64→输出。目标 t=β·伪标签+(1−β)·当前预测；损失为正向 CE 加权反向 CE。Adam lr=.01、weight decay=1e−5。文中报告 True-False 平均准确率 90.38%，不等于 RAGTruth token 定位结果。[全文 §2 / Table 1](https://arxiv.org/html/2509.10004v1#S2)

## 2. IRIS：公开实现中的额外事实

以下分别对应不同的一手代码文件；是代码阅读结果，不代表已复现实验。

- [generate_embeddings.py](https://github.com/ponhvoan/iris/blob/d7e81561a7f262fcb2da662a9a3e07e2e6daf83f/generate_embeddings.py)：先执行验证生成；`--verb` 是默认 True 的 store_true，又独立执行 verbal-confidence 生成。因此默认实现不只有一个生成调用。数值解析失败的条目会缺失 verbal 软标签。
- [prompts/verb.txt](https://github.com/ponhvoan/iris/blob/d7e81561a7f262fcb2da662a9a3e07e2e6daf83f/prompts/verb.txt)：另问原陈述正确概率，并有固定示例；并未把前一次验证文本传入此模板。不能把这个标签称为原生成过程内部事实判断的直接观测。
- [train_classifier.py](https://github.com/ponhvoan/iris/blob/d7e81561a7f262fcb2da662a9a3e07e2e6daf83f/train_classifier.py)：梯度目标由软标签构造，但真实标签参与 train/val 分层划分，并以真实验证准确率控制早停；默认 epochs=5、β=.8、φ=1。bootstrap 分支未 detach。`--normalize` 在划分后只改 X_arr，现有 X_train/X_val 不随之更新。这些细节不能原样作为本项目“完全不读幻觉标签进行训练与选择”的协议。
- [utils/losses.py](https://github.com/ponhvoan/iris/blob/d7e81561a7f262fcb2da662a9a3e07e2e6daf83f/utils/losses.py)：entropy、perplexity 分别对已生成文本再 forward；CE 为 soft-target 对 log prediction 的求和。默认总成本应区分生成次数与额外完整前向次数。
- [utils/dataset.py](https://github.com/ponhvoan/iris/blob/d7e81561a7f262fcb2da662a9a3e07e2e6daf83f/utils/dataset.py)：末层索引默认从后数 1；entropy/perplexity 标签归一化代码被注释，不能仅依据论文描述假设当前原始标签已落在 [0,1]。缺失 soft_label 时只跳标签，特征数组未同步过滤，需复用前修正。
- [models/simple.py](https://github.com/ponhvoan/iris/blob/d7e81561a7f262fcb2da662a9a3e07e2e6daf83f/models/simple.py)：默认 MLP 三个 ReLU 隐层，最终 2 logits，在训练处 softmax。仓库虽还有 pooling/conv/Transformer 备选，主方法不是 GNN，也没有时间状态或证据选择器。

## 3. RAUQ：最新论文方法（v3 §4、Eq.1–4、§5.1）

逐回答、逐层选 h_l=argmax_h mean_i A^{lh}_{i,i−1}。c_1=s_1；c_i=αs_i+(1−α)A^{l,h_l}_{i,i−1}c_{i−1}；u=max_l[−mean_i log c_i]。base 模型 s_i 为 token 概率、α=.2；Instruct 用 s_i=log|V|−H_i、α=.9。选中间三分之一至三分之二层；32 层示例为 10..22。无拟合损失、无幻觉伪标签；论文承认 α 在小型 OOD 子集选择。输出为整回答不确定性，主评估为任务质量的拒答排序。§4 未建模证据归属或正负采纳；有递推但无纠正/换题状态。[v3 §4–5](https://arxiv.org/html/2505.20045v3#S4)

## 4. RAUQ：公开实现约束

- [rauq.py](https://github.com/mbzuai-nlp/rauq-hallucination-detection/blob/12dbc0afe173c5987db11eec06a8db4a0cdc32f3/rauq.py)：输入 attention_all、greedy_log_likelihoods（entropy 变体还需 entropy）；提取 offset=−1 对角，按整回答均值选头。`single_head=False` 为逐样本选择；True 才从训练注意力选共享头，不涉及真值标签。构造器默认 aggregation=mean、token_aggregation=meanmin，并非论文主配置。复现需显式指定 max/meanlog；meanlog 的分数另加常数 1，不影响排序但影响绝对阈值。entropy 信号未归一化到 [0,1]，不应解释为事实正确概率。
- [run_polygraph.py](https://github.com/mbzuai-nlp/rauq-hallucination-detection/blob/12dbc0afe173c5987db11eec06a8db4a0cdc32f3/run_polygraph.py)：包含多配置网格和消融；直接运行默认入口不等于仅执行论文单配置。本项目若纳入基线，应固定配置与版本再评估。

## 5. 对本项目结构的推导（下面是我们的设计分析，尚未验证）

两篇最有用的贡献是“用不同视图形成可训练监督”和“显式传递过去的不确定性”。它们没有解决当前任务最重要的可识别性：模型可能自信地检查错事实，或者一直沿着流畅的错误绑定生成。仅重建模型已有激活、只最大化两视图一致、只蒸馏 entropy，都会可能把错误绑定学习得更稳定。没有来源约束锚点，图的重建准确率不等于事实判断准确率。

建议将我们的最小可训练模型固定成三个有明确接口的部分，而不是先要求一个回看启发式覆盖 100%：

1. **适用约束头**输出 π_t(c)，c 是全来源候选约束或 NULL。输入为当前陈述状态和来源高维节点，必须联合比较实体、动作/属性、阶段/条件，不能只对数字或实体表面做相似度。训练锚点来自来源自身的可回溯事实：遮盖值时保留其拥有者和条件，恢复原位置；困难负例使用同来源、同值类型的其他拥有者/阶段。删除该完整约束后，目标指向 NULL。此目标监督“来源能否支持这一声明”，不是监督 RAGTruth 幻觉标签，也不是由模型当前选择反向定义正确证据。自然语言等价表达仍有监督噪声，应给适用概率而不冒称确定真值。

2. **信息传递头**输入原回答高维状态与有类型的 source→response、history→response 关系，预测该证据对当前声明各候选输出的有符号影响。训练目标可来自少量批量干预的实际输出变化，而不是 attention 本身；来源映射 π_t 与实测采纳向量分开。这样能区分“看错约束”“看对约束但输出未采用”和“来源根本无此信息”。Attention/A·V 是消息输入，不能直接充当支持标签。

3. **声明状态递推**以来源支持、正向历史继承、反对/纠正、声明转移分别建模。可以借鉴 RAUQ 的显式历史传递，但传播权应依赖当前边的语义和符号，并允许恢复/重置。所有历史节点仍可访问；span 是状态变化后形成的分组，不是物理截断。后验回看位置是对状态变更具有解释作用的一组节点，可有分布式集合；不必把单节点定位成功作为检测器推理的硬前置条件。

训练与评估的防循环约束：source self-supervision 与因果影响自监督分别声明目标含义；分组按来源划分，NULL 构造不得只用容易辨认的删除痕迹；不能用真实 QA test 或作者式真实验证准确率选 checkpoint；模型平均 loss 下降也不能替代自然幻觉检测。至少保留“同来源错误归属但高注意力”“正确引用错误历史后明确纠正”“来源缺失但生成自信”“换陈述仍需跨段来源”四类端到端失败类型。选择性地验证完整链条，比继续展示关系词会改变数字有用。

## 6. 可复用结论与不可宣称部分

| 部分 | 可复用 | 不能由这两篇解决 |
|---|---|---|
| IRIS | 小型高维 probe；不同验证视图形成软训练信号；稳健软目标思想 | 不保证伪标签正确；不输出约束拥有者、回看位置、连续幻觉 span；现成代码不是严格无标签模型选择 |
| RAUQ | 固定文献配置的廉价基线；逐层保留不同功能头；递推依赖概念 | 只使用一阶近邻边；无正确证据选择及正负方向；整回答选头不是严格 prefix-only 在线定位 |
| 我们候选结构 | 源约束恢复锚定适用性；实际干预锚定采纳；有符号关系驱动声明状态 | 尚未实现/通过自然数据验证；不能称主线收敛或图增益已成立 |

线上时序若需要严格定位，必须在第 t 步只使用 prefix≤t 的状态；不能把 RAUQ 的全回答头选择或 IRIS 事后验证隐藏在“自动回看节点”接口里。离线 prompt+response 检测可以使用完整回答，但须明确定位结果是事后分析。
