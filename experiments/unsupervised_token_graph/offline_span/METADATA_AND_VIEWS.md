# 输入身份修复与片段视图说明

## 输入检查不是检测成绩

`missing_source_ids=answers`、`unknown_tasks=answers` 表示旧适配器只读取了六字段
attention 缓存，附近又没有 inputs/records 索引。不能把这些回答全部当成一个来源训练。

本次新增的 `metadata.py` 按以下顺序工作：

1. 优先使用 NPZ 与原 inputs/records 索引里的身份。
2. 身份不完整时，从缓存祖先目录查找 `response.jsonl` 或 `dataset/response.jsonl`；
   也接受 `--dataset` 指定原目录或原文件。
3. 用回答编号关联 `source_id`、`split`、`model`，再用原 `source_info.json/jsonl` 的
   `task_type` 补任务。不会根据源目录的模型名猜原回答生成器。
4. 只把白名单字段留在训练样本中；原文件中的 response、labels、标注边界不进入模型。
   原文件可能同时存有这些字段，不能声称“训练前从未打开这个文件”。
5. 不凭编号复制回答哈希或 offset 来冒充内容核验。缺失 offset 仍在最终评价时通过
   原 tokenizer 与保存 token IDs 的精确匹配恢复。身份冲突报错，不覆盖。

没有产生新的 metadata 数据集，也不修改原 attention 缓存。原字段完整时无需读取原
response 文件。元数据缺失在 prepare 开始就报告，不先构完所有图再让 fit 报错。

```bash
python -m experiments.unsupervised_token_graph.offline_span --phase inspect
```

检查新增 `metadata_files`、任务和生成器数量、`ready_to_fit_metadata`。
`hidden=[N,0]` 表示本次 tokens 模式没有加载 hidden；是否在当前选定 NPZ 里保存过 hidden，
看 `hidden_fields_in_selected_archive`。这不搜索服务器上的其他隐藏状态目录。
coverage、hidden、offset、entropy 的显示仅描述 `first_id`，不是全体样本统计。

用户给的两个覆盖率恰好是295/296、164/165；需结合新输出的
`uncovered_response_positions=[0]` 核验。canonical q=P+r 行预测下一位置，
没有保存 q=P-1 时，首个回答词没有对应预测行；不靠错位一格补齐。

## 当前片段视图的具体构造

`regions.py` 首先枚举最长 `max_length` 范围内的连续区间 [start,end)，仅限原观测
覆盖区间，包含单 token，不使用 gold span 或高熵门槛。每个视图是四组节点的索引：

- 段内内容：这个回答区间的所有 token。
- 选择候选：区间第一个、最后一个词的预测位置，以及区间内来源读取质量最大的预测位置。
  三者去重；不是从 Q/K 读出了指针，也不保证包含发生在区间前面的 reanchor。
- 证据：统计这几个位置保留边的来源质量，选择最大的一个 source group，保留该组所有词。
  没有 source_groups 时就是固定32-token prompt窗口；没有 source_mask 时也包含指令。
  此处跨head累加仅用于选组，模型的边编码仍保留head；没有声称该组是完整、正确的事实。
- 后文：沿保留的读取边，找后面实际读过段内词的位置，按时间取最早 `future_budget` 个。
  当前只收集直接读者，不会自动找齐所有间接后代，也不会自动识别语义支持或反驳。

模型分别汇总这些节点的词项/图编码，再学习证据侧和回答侧的配对相容性。
现版本只选一个证据组，不是此前设计中已经完成的多事实联合绑定模型。

## 来源错接的意义与限制

训练对照保持段内内容、选择候选和后文节点不变，只替换检测器收到的 evidence_nodes。
当前从同一材料的其他组中按长度接近、token集合重叠高选择一组。它不改原模型输入，
不重跑LLM，也不修改保存在图中的原始attention。

目的：避免模型只学习“片段自身流畅、聚集”，迫使它比较来源与这段回答的对应。
标签只表示“原始配对/人工错接”，不是“事实正确/幻觉”。

必须承认：
- 原配对也可能本来错绑；把它作为观测正例可能让检测器学会错误的对应。
- 替换的另一组也可能支持同一陈述，不能把所有重接都叫人工幻觉。
- 当前对照只匹配长度和词项重叠，没有完整匹配head质量、粗距离与语义角色。
- 证据侧当前是词项/hidden的均值汇总，不保证区分年份、否定和对象之间的关系。
- 因此这只是待验证的代理任务，不是理论上保证有效的真假监督。

只有真实关系比无边/错接对照更能区分自然标签，并且超过相同单词分数的普通平滑，
才能主张图结构及片段模型有增量。本次修复不改变损失、构图或模型定义。

## 软件验证

44项 CPU 测试通过：此前30项回归，新增14项元数据测试（含参数化来源文件格式）。
测试包含六字段缓存复现、原dataset自动发现、白名单访问、标签改变不影响输入、
来源/划分冲突、缺失元数据及时停止、hidden存在但tokens模式不加载，以及真实小模型
prepare/fit/score。使用的五个父适配模块与原Git blob一致。

没有访问服务器的2497/449个真实文件；不能把构造接口测试说成真实数据已全部接通。
