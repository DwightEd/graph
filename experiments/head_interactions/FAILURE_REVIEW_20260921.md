# v1 来源恢复失败：实际上传结果复核

输入：用户上传的 `head_interactions_v1_review.tar.gz`，Llama-3.1-8B-Instruct，BF16。
2 个 source，4 个自然前缀，8 个 panel，136 个 trial；阈值 0.01 nats。
本记录分析真实上传文件，修复验证使用本地 CPU 小模型；两者不混为一次实验。

## 已确认的失败位置

zero_strength：8/8 通过，最大 token logp 与 margin 误差均为 0。
self_restore：8/8 通过，最大 token logp 与 margin 误差均为 0。
delete_restore：0/8 通过。未执行 factorial、单头剂量及 adaptation 条件。

|案例|前缀侧|panel|最大 token logp 误差|sum_margin 误差|
|---|---|---|---:|---:|
|14315 headwear|supported|selected|0.020913|0.012152|
|14315 headwear|supported|random_0|0.113455|0.142517|
|14315 headwear|unsupported|selected|0.121448|0.207075|
|14315 headwear|unsupported|random_0|0.056884|0.191641|
|14375 onion|supported|selected|0.088390|0.188177|
|14375 onion|supported|random_0|0.079203|0.055846|
|14375 onion|unsupported|selected|0.026740|0.012373|
|14375 onion|unsupported|random_0|0.015528|0.000203|

最后一组 margin 误差虽小，单 token 误差仍超过阈值，不能只用候选求和抵消误差。
有 5/8 组的 margin 误差超过计划的 0.05 nats 最小效应。

## 消息向量的直接证据

取每组 trials 中 baseline、round_trip/delete、round_trip/restore/step_0，
读取最早 L21 站点的 total_readout 与 baseline 的来源 readout。
八组保存的第一步恢复向量均逐元素等于：

    float32(BF16(deleted_total) + BF16(baseline_source_readout))

八组也均不等于 baseline_total。首层最大坐标误差范围为 0.00048828125～0.00390625。
来源 readout 是 CPU float32 重建的部分 A @ V；原生完整 A @ V 已在 BF16 舍入。
因此分别计算、舍入后相加不是原生删除的精确逆操作：

    round(A_not_S V) + round(A_S V) != round(A V)

仅把最后一次加法提升到 float32 也没有让这八组第一层全部恢复。
首层向量已证实恢复路径有数值问题；现有记录没有独立分解后续各层与 GEMM 形状变化的贡献。
不能把最终全部 0.207 nats 的误差精确归因到某一个舍入步骤。

旧测试只检查低幅值随机小模型的最终误差低于 0.01，未检出这类偏差，验证覆盖不足。

## 修复与边界

teaching 的 ReplaceSource 在当前完整 A、V 副本中替换指定来源，使用原生 dtype 计算整个头输出。
只写回目标 query，保留其他 query/head 及未替换来源。它在实数算术下仍是
`current_complement + reference_source`，但避免分别舍入再注入。
GQA value 按 query head 展开后处理，不改共享 value 投影；多来源可组合，donor 可不同长度。
同时在两个候选末尾做因果填充，使矩阵形状相同；填充不进入评分。

协议升为 v2，输出到新目录；原始 v1 证据、0.01 控制阈值、0.05 效应门槛均保留。
controls.csv 增加实际头总状态的最大恢复误差与站点，summary 增加失败控制类型计数。

软件验证共 108 项通过：三种模型布局、小模型 BF16 增大 value 幅值后的逐层精确恢复、
直接张量测试证明旧加回不精确而新收缩精确、不同长度 donor、非目标 query 隔离与多来源组合。
未运行用户的 8B CUDA 模型；需要新结果确认实际 8 组控制通过。

这份 v1 数据不能判断多头协同/抑制/补偿是否存在，也没有检测 AUROC。
此外，onion 候选是温度与时长，headwear 自然前缀不同且 supported 侧已有其他过度概括。
即使 v2 控制通过，自然配对仍是探索性候选审计；语义对照与独立来源验证仍需保留。

```bash
git pull --ff-only origin main
bash experiments/head_interactions/run_all.sh
```

新报告：`outputs/head_interactions_v2/REPORT.md`。
新压缩包：`outputs/head_interactions_v2_review.tar.gz`。
