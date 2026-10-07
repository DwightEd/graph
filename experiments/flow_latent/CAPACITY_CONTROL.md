# 同容量节点历史对照（审计后追加，运行前冻结）

2026-10-07。原node-only模型只有截距，不能称与图条件同参数量。新增lag_pair：每head的predictor完整A@V（将source/history/other相加）及前一predictor节点拼接；t=0将边界节点重复。对t>=1，首节点对应已有carrier节点t-1，第二节点对应t-2（t=1重复边界）。不使用显式来源划分/图端点/多跳特征。该节点表征本身来自attention模型，不是去attention干预。

head16各块16384坐标，两块32768，与v3/root+other完全同输入宽；fit-only PCA128，同冻结node PCA128、K4/rank8、固定30步MAP、seed42/43/44，回归参数与协方差参数形状完全匹配。只新增一个对照模型/seed，旧node/native/rewired/NLL分数、阈值、拟合文件保持不变。3个独立新目录保存旧分数的逐数组一致性检查、新模型、后验、冻结分数及评价。

24 fit来源无标签学习；12 reference决定同样95百分位预算；17旧regression只在新分数冻结后读标签。主比较native-lag_pair按来源bootstrap，报告全部seed及首错/正常报警。无论结果方向均不改分数符号、不追加挑优训练。此项属于既有样例上的审计补充，非新数据确认。

缓存内容manifest在原21次拟合之后建立，不能倒推声称原运行已受全缓存哈希保护；新对照在manifest完整校验后运行。固定30步只承诺MAP目标非降，不声称已收敛。

## 绑定修正重现

首次3个对照完成后，fresh审计发现调用者cache未与基础协议地址明确绑定、基础node PCA模型未哈希。修正为cache路径与输入字节一致、完整cache manifest校验、基础native_model.pkl先SHA256再从相同字节反序列化。三个seed在新control_bound_seed目录全部重跑；原control_seed目录保留。没有改变特征/模型公式/参数/阈值预算/评价输入。audit.control_contract复核基础模型SHA、所有旧分数数组和参数形状。
