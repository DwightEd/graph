# O4 relation-only 独立执行见证

2026-09-13，Asia/Shanghai。执行者为 fresh agent `relation_only_witness`。
这是工程运行与产物完整性见证，科学审稿仍为 `REVIEW_UNAVAILABLE`；不判断方法效果是否通过。

## 文档执行

先只读 `docs/RELATION_ONLY_PLAN_20260913.md`、`.aris/compute/local.md`，
核查工作目录及祖先路径，无适用 `AGENTS.md`。复用现有 research 环境，无安装、
重建或代码修改。在 `reanchor` 目录逐字执行以下调度命令一次：

```bash
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python scripts/interleave_relation_only.py --population-pid 16928 --output outputs/relation_only_20260913
```

仅增加 stdout/stderr 到 `/tmp/relation_interleave_witness_20260913.log` 的保存。
内部 O4 日志为 `reanchor/runs/relation_only_20260913.log`。
wrapper 和机制子进程均退出 0，无自行重跑或 GPU 并发加载。

## 全量任务暂停与恢复

- 执行前独立快照：PID 16928 正常运行，4458 条已发布，失败 0；保存这 4458 个
  response manifest 的 SHA256 和 population settings SHA256 到
  `/tmp/relation_interleave_witness_before_20260913.json`。
- 调度记录的停止前进度为 4472 条，失败 0；01:14:36 停止旧进程。
- O4 子进程 PID 19518；01:17:19 以原冻结参数和 `--resume` 启动 population
  新 PID 19668。停止旧进程至启动恢复进程间隔 162.96 秒。
- 新 PID 先校验已有结果、重新载入模型。01:19:32 独立首次观测 `running`
  为 4485 条；01:19:37 再观测为 4489 条，失败 0。
- 另核验 17 个新增 response manifest 的 mtime 晚于调度 `resumed_unix`。
  这验证了恢复后实际新增发布，不只依赖暂停前 progress 的可能滞后计数。
- 4458 个快照 manifest、population settings 及 8 个冻结执行代码 hash 均未变。
  O4 的 13 个冻结代码/输入 hash 也均未变。

调度事件：`reanchor/runs/relation_interleave_20260913.json`。
恢复日志：`reanchor/runs/ragtruth_population_resume_20260913.log`。
独立恢复采样：`/tmp/relation_interleave_witness_resume_20260913.json`。

## 产物核验

`reanchor/outputs/relation_only_20260913/manifest.json` 列出的 **190/190 文件**
逐文件重算 SHA256 全部相符。4 个世界、28 个固定 patch 和 132 个扫描 patch 齐全；
160 个条件名唯一，扫描 steps 恰为 0…131。

两个查询对分别只有 prompt 位置 67、92 的关系 token 发生变化。
数值节点位置均为 89、114，ID 分别为 717、975；数值位置/ID、完整 token 多重集合
以及回答 token 序列在对应两个来源世界之间全部保持不变。两个查询的 prompt
长度均为 549；完整输入长度分别为 745、746。以上由独立数组比较确认。

4 个世界均包含以下全部 32 层数组，独立检查形状、float32 和有限性通过。
其中 Q 为响应预测 query 数：real_after 为 132，contrast_before 为 133；来源节点数 450。

| 原始数组 | 每层形状 |
|---|---|
| 来源 V、RoPE 前来源 K | `(450, 8, 128)` |
| RoPE 前 query Q | `(Q, 32, 128)` |
| query 残差、MLP 更新 | `(Q, 4096)` |
| 各 head 来源 A | `(32, Q, 450)` |

世界 logits 为 `(Q, 128256)`；N/A/G 读出各为 `(32, Q, 2)`。
160 个干预端点均保存完整 `(128256,)` logits，全部有限。
Q/K 是投影后、RoPE 前向量，不能把它们直接当作实际旋转后的注意力点积。

4 个 sham 端点与对应 baseline 最后 query logits 独立逐项比较相等。
runner 记录的 4 个 sham 全 query 最大 logit 误差均为 0；160 个 patch
对干预位置之前 query 的最大 logit 误差均为 0。后两项是对运行时检查结果的核验，
没有另行重跑前向。通用 JS 公式在相同 logits 上留下最大 1.39e-8 的浮点残差，
不能解释为 sham 产生了效应。

runner 记录 134.66 秒、CUDA allocated 峰值 17,502,954,496 bytes（16.30 GiB）。
该运行时长不是完整调度及恢复的墙钟时长，也不是干净环境重建证据。
详细核验摘要：`/tmp/relation_interleave_witness_verification_20260913.json`。

本见证不把受控事实改写当作自然幻觉新样本，不把这个单一来源上的条件结果
当作泛化准确率，不宣称 N/A/G 已构成无监督约束归属检测器。
