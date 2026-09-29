# 片段维持与来源对输出的控制

实现见 [代码与公式](../experiments/span_source_control/README.md)，实测见 [12答结果](../experiments/span_source_control/RESULTS_ZH.md)。

2026-09-29新增：1024头、3206token的来源attention-logit竞争导数及逐头正负来源记忆；真实8B完整前缀干预336次，包含24无标签选择事件、10posthoc位置和2缩剂量检查。后续Q/K/V、FFN及残差原生重算，文字固定。与旧fixed-past-KV当前query缓存读出严格区分。

它是机制量化实验，不是已验证的新幻觉检测器，不替换默认risk，不宣称全量AUROC提升。正负作用、相同来源复用、隐藏状态变化均不自动等于事实正确/错误；GSM后续步骤unknown保留。共享交接在lys/codex/research/refine-logs/span_source_control_20260929。
