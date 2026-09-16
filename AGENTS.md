# 本项目当前工作约定

修改代码前读取 `iclr/CODING_GUIDELINES.md` 与 `iclr/MECHANISM_FIRST.md`。
当前检测验证是 `experiments/unsupervised_token_graph/fixed_graph`；先读 `iclr/FIXED_GRAPH_VERIFICATION.md`。
`span_audit` 保留为标签辅助解释；不再要求完成所有因果机制后才能检验无标签检测。
先使用真实错误与正常对照，区分入口、复用、退出；不要启动来源错接训练来替代这个问题。
最终无监督检测是后续独立任务，不能将金标构图/金标窗口的结果称为检测成绩。
代码采用短函数、直接名字、空行分逻辑；不要一行多个语句、层层封装和散落的校验。
复用已有缓存；不使用 `set -e`；不后台执行实验；不改用户模型和结果文件。
不得用“source影响输出”“Q改变attention”等必然或过弱的现象宣称发现幻觉原因。
计划和实际实现分开报告。仅有软件测试，不声称已运行自然数据或发现新机制。
