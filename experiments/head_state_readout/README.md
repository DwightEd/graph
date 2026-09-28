# 完整头状态、JS与Jacobian联合读出

四轮48答/6463token的CPU缓存实验已完成；不是重新跑旧全测试。见[完整结果、位置与失败分析](RESULTS_20260928.md)。固定来源融合候选v3检95/134错、误报147/1353；比原路由好但未解决头饰/养老金等，不替换默认。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.head_state_readout.run --output-prefix outputs/head_state_readout_new
```

依赖现有route_complement_20260928、message_js/operator、source_first和自然trace缓存；入口不安装环境、不采集GPU。每task4fit/4dev，来源互斥；评分不用标签，设计参考已暴露误例。pair_diagnostic明确用标签拟合，只用于机制研究，最终评分不调用该拟合方向。

已有结果不必重算：outputs/head_state_readout_20260928_v3/TOKEN_AUDIT.html逐词着色TP/FP/FN；token_positions.png与natural_local_scores.png为静态图。各轮alarm_spans.json列出全部具体位置；address_audit.json给出自动来源地址，NPZ保留全部1024头。

features保存59401原坐标，不压低秩；density执行整来源排除的条件参考；association比较读取与输出采纳；boundary检验窗口与来源广播。原始全key/V/tangent另保留。预算ROC文件是事后分析，不部署标签选择的阈值。

默认创建新目录，完成标记存在则跳过；中断的部分目录保留，重新实验使用新前缀。该命令复现本轮固定方案，不能用于参数修改后覆盖旧结果。v1包含完整块，v2条件关联，v3 --head-only保留所有物理头并排除整体语义块，v4独立边界消融。所有候选完整报告，不能自动把最大本地成绩当默认。
