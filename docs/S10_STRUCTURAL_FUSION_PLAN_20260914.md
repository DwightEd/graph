# S10 自然RAGTruth结构与熵的因果组合

用户授权实现、清理冗余、重构、运行、合并推送两个仓库。已有N9完整代码先提交archive/pre-structural-20260914；历史数据/结果/执行快照不删除。

数据固定：population_20260912的QA、llama-2-7b-chat全部839 official-train回答＋150 official-test回答；读取Llama3.1 observer缓存，不冒称原生成器机制。按source SHA256(seed20260914:source_id)排序，official-train来源前80%train，其余calibration；官方test所有source与两者互斥。整体旧数据曾评价过，测试是本方法冻结后的来源留出，不称从未见过的数据。

缓存source_attention与remote_history_attention（排除最近16token）提供逐层逐头统计。D=remote_history-source，明确不是历史总量位移；B=entropy、negative_margin、D。完整组合另外使用source/remote质量、跨头source标准差、相邻entropy/D差、历史8步衰减熵记忆与当前D交互。所有窗口仅t及过去，无真首错位置、未来文本或总回答长度；t=0定义充分，没有17token冷启动。log(1+t)只作独立位置对照，不进入主组合。

固定L2逻辑回归C=1，无超参数搜索。两独立目标：原RAGTruth错误token与标注span起点。主比较raw entropy、raw D、raw negative-margin、瞬时组合、完整时序组合；另固定去entropy族和去D/attention族消融。训练source等权，校准在calibration拟合正斜率缩放＋截距；固定calibration正常token 5%FPR阈值，测试不调。不强制回看门控，不把真实首错作为输入。

先导出无标签特征、冻结分割与代码；仅读取train/calibration scoped标签训练两目标；全部模型和test预测写入freeze后才读取test scoped标签。报告全token/起点AUROC AP logloss Brier、响应内AUROC、正常回答anyalarm、固定阈值recall/FPR、来源bootstrap完整vs瞬时/单特征增量。不根据结果改名最优方法。

运行：只复用真实GPU产物做CPU抽取/训练/评价，不重复GPU。输出S10独立目录；先12行无标签工程sanity，再完整989行。复核代码/因果前缀/分割，未解决问题不扩大数据、不调test。

重构：graph只保留新structural_detector主入口、必要测试/运行脚本。native测量公共模块迁往reanchor/src/route_graph并保留其依赖闭包，避免删除活跃测量能力；删除被替代的graph旧检测/外部核验/一次性编排代码及对应测试，旧代码可从archive分支恢复，历史结果完整保留。reanchor负责原始内部测量与无标签特征导出。
