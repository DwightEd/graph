# 约束绑定、来源冲突与原生采纳：一手文献核查

research-lit只读检索子任务constraint_flow_literature；本地现有说明与失败结果优先，未发现Zotero/Obsidian工具。下列发现用于假设，非本项目已验证机制。

|论文|机制/可借鉴内容|适用边界|
|---|---|---|
|[How do Language Models Bind Entities in Context?](https://arxiv.org/html/2310.17191v2), ICLR2024|实体与属性可由binding ID关联；实体后位置也可能载有绑定信息|受控任务、存在direct binding替代机制、冻结其他上下文的存储试验不等于自然生成传播|
|[Understanding Parametric and Contextual Knowledge Reconciliation within Large Language Models](https://proceedings.nips.cc/paper_files/paper/2025/hash/94f7c80260eea8d52a5c0ec7b1637c04-Abstract-Conference.html), NeurIPS2025|冲突信号可持续存在，同向信息累积；不同头传递不同来源|使用目标实体标记及rank-8 LoRA probe，不直接移植成无标注检测器|
|[AtP*](https://arxiv.org/html/2403.00745v1)|梯度近似受Q/K饱和及直接/间接抵消影响，有限干预必需|没有实现其GradDrop；本轮fixed-past native梯度不代表跨时间全部路径|
|[Towards Best Practices of Activation Patching](https://arxiv.org/html/2309.16042v1)|logit差比饱和概率更适合测有符号干预效应|margin作为量尺不意味着作为真假分数有效；改变消息幅度与改写输入不是同一操作|
|[On Mechanistic Circuits for Extractive Question-Answering](https://arxiv.org/html/2502.08059v1)|路径patching分解上下文/参数来源，部分头可定位答案来源|用已知答案和受控probe，关注答案位置不证明关系执行正确|

本轮据此保存四件事：按实际key的读取；逐候选即时响应；原生下游响应；来源块单独/联合小幅干预。约束不先标为对象/阶段/实体，潜在载荷由原上下文化value表示。它包含主题、语言等非约束信息，不能把一个高响应key自动称为正确证据。来源分块只依token表面的标点/换行，用于干预实验，不是完美语义划分。

正负配对用于机制发现，检测分数不能把正确回答或人工候选作为输入。所有1024物理头原始因子留存；末端聚合是可替换读出，不销毁原信息。符号/幅度一致只证明工具能测作用，需另检查旧FN/FP和首错位置。

补充核查：[Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)，ICLR2024，说明初始token可获得很高attention，即使语义重要性很低。本文研究流式KV保留，并未提出本项目的真假评分；本轮仅据此检查BOS混杂，新的常驻地址零和交换由原生梯度与有限干预另行核验。
