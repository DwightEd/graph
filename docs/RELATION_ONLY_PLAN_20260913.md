# O4：固定数值，交换阶段归属

执行前冻结，2026-09-13。目标是检验真实决策前缀中，来源节点内容和连接能否
传递阶段归属；不声称已有无监督事实检测器。

## 对照与真值

实际00012回答是“Remove the bratwurst from the grill and cook the onions in the
beer mixture for 10 to 12 minutes”。此前资料把这里简化成从beer mixture取出，
不够准确。原始来源没有为这一组合明确给时长，12→14不能称为自然纠错。

因此保留原始完整聊天prompt的其余文本，将passage1最初两步替换为明确的两条
阶段规则：Before removing the bratwurst from the grill, cook the onions in the
beer mixture for 10 to 12 minutes. After removing the bratwurst from the grill,
cook the onions in the beer mixture for 10 to 14 minutes.

另一世界只互换Before/After。两世界数值ID/位置、token多重集合、总长度不变；
除了两个关系词，其余prompt token必须逐项一致。回答token跨来源世界逐项固定。
这引入了构造事实，属于真实前缀上的受控机制实验，不能作为新自然幻觉样本。

两个查询：real_after保留真实回答；contrast_before仅把上述动作前缀改成
“Before removing the bratwurst from the grill, cook the onions...”，作为相反阶段
控制。它是构造查询，不冒充真实采样。世界0的正确上界分别14/12，世界1分别12/14。
数值候选12/14及其来源节点预先固定，用于读出和端点干预；正确答案与阶段映射
只在评价中使用，不进入读出、不训练。必须报告原生argmax、
并列、完整分布JS和固定候选差，不能只报告两候选归一化概率。

## 冻结读出与干预

保留每层高维source V/K、query残差/Q及逐head A；Q/K记录投影后、RoPE前的
实际向量，不能误称为已旋转的attention点积。在两个固定数值节点上比较：
N：query与O投影后的单位权重V的余弦；
A：数值节点的attention质量；
G：真实A×V/O消息在query方向上的投影，除以该节点单位消息范数。
每层/每head原量保存；端点最终读出固定平均全部32层，不挑最佳层、不拟合权重。
这些是兼容度候选，G不自动代表输出采纳；它们可能失败，失败不得改方向掩盖。

两个查询各做双向全32层最终query的X/E/XE/MLP交换及同世界X sham，另外当前
数值A端点互换（保持每head质量）。每个世界还做此前所有回答query的全层X，
合计28个固定patch。干预后query之前的全部预测logits要精确不变。
X换全source V，E换source内A分布且保持接收质量，XE联合；使用现有ownership算子。
保存全部输入世界逐query logits、原始属性/边、每patch端点完整logits和前缀误差。

对real_after、world0额外穷举从首回答预测到最终数值的每个query：只在该query
做全层X，记录最终候选差与原生翻转。这检验此前窗口外是否也有充分节点，不
把最终端点已知的扫描叫作自动首错定位。以原世界source读取质量正增量排序的
top8与穷举结果比较，仅作这个明确干预的候选召回；不能据此承诺100%定位。

## 决策门槛

若两阶段均正确随关系词交换而翻转，则支持这组受控输入中关系信息被模型采用；
若仅一阶段成功，保留另一阶段失败。X/E/XE的差异分别报告，不能先指定唯一路由。
N/A/G均报告4个世界/查询的原始值和选择；样本仅一个来源，不能报告泛化准确率。
G若不优于N/A，不接入默认检测器。若模型能翻转而读出失败，下一步改读出目标，
不推断内部信息不存在；独立来源验证、连续错误终点、图增益仍单独待证。

预期少于10 GPU分钟；唯一GPU被全量任务占用，准备与CPU检查完成后短暂中断其
当前未发布回答，保留所有已发布/部分结果，执行本组后立即按原冻结设置--resume。
调度事件另记，模型和全量科学参数不变。不得同时加载两个8B模型。

在reanchor目录执行独立见证：

```bash
bash scripts/run_relation_only.sh --output outputs/relation_only_20260913
```

本次全量PID16928仍运行，独立见证用以下调度入口；它核对PID及冻结代码后发送
SIGINT，保留当前未发布条目的partial，在finally中恢复原全量目录。内部执行上面
同一个机制命令。日志和调度事件分别写reanchor/runs，不覆盖原启动日志。

```bash
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python scripts/interleave_relation_only.py --population-pid 16928 --output outputs/relation_only_20260913
```

科学审稿REVIEW_UNAVAILABLE。设计借鉴实体/属性因果分离的控制要求：
[RAVEL](https://aclanthology.org/2024.acl-long.470/)及
[How do Language Models Bind Entities in Context?](https://arxiv.org/abs/2310.17191)。
这是实验设计参考，未声称本项目复现了其方法或具有同样结果。
