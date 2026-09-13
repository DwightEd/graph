# O5：关系路由的时间与层定位

O4结果驱动的追加协议，必须与O4分开记录。2026-09-13。
O4四个受控世界均按阶段选对；最终query E与XE分别4/4翻转，X为0/4。
因此X扫描不能替代关系路由E扫描。N/A/G均值读出分别3/4、2/4、2/4，未获图增益。

固定O4 real_after world0为接收世界、world1为捐赠世界；不重新选择样本、关系词、
候选12/14或模型参数。直接读取O4已保存完整高维A/V及输入，核验manifest及基线。

逐项执行：
- 全部132个回答预测query，每次只在一个query、全部32层替换source内E。
- 最终query逐层E，共32条件，其他层保持本次真实计算。
- 此前全部132query联合E，以及去掉最终query的131query联合E，共2条件。
- 同世界最终query全层E sham，1条件。

共167条件。全部保留其他上下文和间接中继，E保持当前接收head的source质量。
各条件保存端点完整logits、固定候选差、JS和此前所有query精确不变性检查。
最终query E须复现O4同一条件；不得把结果改变归咎于新的分词或不同前向。

判断：单query翻转仅说明此具体E干预充分；多query联合而任一单query均不充分
说明分布式作用。去掉最终query后是否翻转用于检验最终query在联合干预中的必要性。
仍不能推断唯一最早的信息筛选事件，或泛化到全部幻觉。
比较O4执行前冻结的source读取增量top8对充分query的召回，若漏掉最终query就如实
报告；不重新选分数或预算。若充分集为空，召回为不可定义，不报告0%/100%。

使用与O4相同环境，估计少于5 GPU分钟。独立见证运行以下调度命令，结束后立即
恢复同一RAGTruth目录（当前PID19668）；只增加调度器参数，population代码仍冻结。

```bash
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python scripts/interleave_relation_only.py --population-pid 19668 --experiment relation_routing --output outputs/relation_routing_20260913
```

内部执行：`bash scripts/run_relation_routing.sh --output outputs/relation_routing_20260913`。
科研审稿REVIEW_UNAVAILABLE；不把本追加探索性实验当作预先确认性检验。
