# RAGTruth全量机制运行状态

2026-09-12；本文件记录真实执行状态，详细参数固定在
[执行前协议](RAGTRUTH_POPULATION_MECHANISM_PLAN_20260912.md)。

当前：17790条全量后台作业已启动，PID 16928；实际完成数以输出目录progress.json为准。
两个仓库的既有修改、未跟踪文件和旧结果全部保留。

## 范围与已有证据

过去没有在RAGTruth全量执行机制实验。O1–O3主要覆盖一个来源上的真实正误
窗口及构造对照，见[已有机制结果](OWNERSHIP_MECHANISM_RESULTS_20260912.md)。

本批覆盖全部2965来源、17790已有回答，6生成器；按QA、Summary、Data2txt排序。
使用本地Llama3.1-8B重放原始prompt/response。每个回答的全部预测token分别
测量基线、同输入替换、来源消息10%/100%削弱、远处回答历史消息10%/100%
削弱、来源边端点置换、MLP分支10%削弱，共8条件。

这批测量来源/历史依赖、拓扑变化和分支更新的真实输出效应。标签仅在结果保存
后的独立评价阶段使用，按任务/生成器/官方split分组报告。它不是原生成模型
内部轨迹，也不是已验证的自动归属检测器。保持数值不变而只交换动作/阶段
归属的O4仍未运行；本批不能替代该实验。高维节点+全边图的捕获算子仍保留，
本批保存紧凑逐层/head统计及逐token效应，不全量缓存原始高维图。

## 当前验证

- 两个tiny Llama核心测试通过：原生基线、sham、因果范围、未来token隔离、
  固定baseline候选和实际干预效应。
- 全部2965来源都能在原始prompt中精确、唯一定位，未使用幻觉标签选择样本。
- 独立工程审查已推动修复恢复清单缺失、评价seed遗漏、额外目录重复计数等问题。
- 真实GPU小批次已完成：6/6回答、48条件、1491回答token，最大输入2616，失败0。
  最高CUDA allocated约17.15GiB；sham全状态误差与prompt前缀误差均0。
  5/6评价组包含正负token，固定seed排序指标实际执行成功。
- 正常resume与缺input_manifest恢复均退出0；30个完成文件hash不变，未重新加载
  模型、未重跑前向。缺失manifest与保留备份的SHA256一致。
- 独立工程审查通过，见[审查记录](../refine-logs/ragtruth_population_engineering_review_20260912.md)；
  GPU/恢复见证见[执行记录](../refine-logs/ragtruth_population_smoke_20260912.md)。
- 独立科学审稿：REVIEW_UNAVAILABLE，工程检查不代替方法有效性证据。

## 入口与进度文件

在reanchor目录执行（完整任务使用新输出目录）：

```bash
bash scripts/run_ragtruth_population.sh --output outputs/ragtruth_population_20260912
```

中断后同命令加`--resume`；代码/模型/输入/参数身份改变时拒绝混跑。
失败条目另记，需显式`--retry-failed`重试；不得把失败跳过称为全量成功。
每条完成后原子发布，进度在输出目录`progress.json`，设置在`settings.json`，
输入清单在`input_manifest.json`，实际长度/超限ID在`length_preflight.json`。
正常完成与评价成功后才写`COMPLETE`；存在失败则写`COMPLETED_WITH_FAILURES`。

## 后台任务

PID：16928；启动记录：`reanchor/runs/ragtruth_population_20260912.process.json`。
输出：`reanchor/outputs/ragtruth_population_20260912`。
日志：`reanchor/runs/ragtruth_population_20260912.log`。

在reanchor目录查看：

```bash
cat outputs/ragtruth_population_20260912/progress.json
tail -f runs/ragtruth_population_20260912.log
```

运行中保持settings.json列出的执行文件不变；修改会导致最终评价或恢复的
代码身份校验拒绝。独立新方法可以另建文件；不要把新结果混入当前冻结目录。

最近核查快照：2026-09-12T23:35:58+08:00，完成54/17790，失败0，
正在执行QA。输入清单已核对哈希，全部输入最大2616token，
超限0条；首3条的12个发布文件哈希通过，执行代码哈希未变。
完整快照：`graph/results/ragtruth_population_launch_20260912.json`。此计数为检查时值，
后续以progress.json为准，不把后台启动描述为全量完成。
