# 约束载荷与候选条件采纳：评分前pilot方案

新增真实native fixed-past重放。RAGTruth选同prompt自然配对00005/00006(头饰适用范围)、00012/00013(洋葱阶段)，局部已复核31token及前后邻域；未复核处unknown，不能把supported答案整体视为正确。追加旧养老金15604及WiFi9022全文做难例。GSM8K选49/243、43/240、123/313三个同题正负对照，共六答；逐步骤最多等距8个query，无gold位置挑token，首错后unknown。选题是已暴露机制pilot，不是随机盲测。

所有prompt token均参与，不读旧source_roles/evidence进行评分，自动标点/换行分块仅用于干预诊断。固定3个最终最强非实际词，逐物理32x32头保存完整key attention、128维原生head value、native下游及即时logit-lens的head梯度。它们精确重建每key对三个实际词−候选margin的贡献，无随机投影、无低秩压缩。即时lens仅作局部读出，不是中间层真实概率。native梯度包括当前query下游attention/MLP，但past KV固定，不声称跨时间完整因果传播。

主固定风险为同一头/候选中prompt反对实际词且history支持实际词的共存比例：2 sqrt(P_minus H_plus)/(P_abs+H_abs)，每候选取完整1024头中top8均值，候选最大。对照prompt反对比例、source local正→native负的反转、冲突总量、source attention及最终负margin。全部高尾风险方向预先固定，不依标签拟合。评分后报告配对局部与官方span/首错步骤AUROC、已知样本逐位置分数；不从这些样本拟合阈值或宣称全测试FPR。固定0.5工作点仅为探索报警，对照排序与每例错误，不能称校准5%FPR。

有限干预：按无标签响应选高冲突位置及固定均匀位置；同一物理头自动来源块最正/最负，单独/联合±.05、±.25消息幅度改变，不删除完整来源、不人工注释绑定。核验一阶斜率、零剂量、配对块非加性。另选匹配大小source块作位置对照。干预一致只验证测量，不证明检测有效。若第一轮失败，依据具体FN/FP只开展明确固定的第二轮读出，不按test自动优化。

## v2：针对v1弱头比例饱和的修正（v1评价已知后固定）

v1主分数几乎全部>.5，RAG配对.3291、GSMstep.4921；top8风险头只有约.5–.9%作用量。第二轮固定改为所有1024头按各自原生绝对作用量加权，不按标签学习。主energy_conflict = Σ_h energy_h*conflict_h / Σ_h energy_h，再按原最终竞争词条件概率加权三个候选，去除max候选挑极端；对照energy_conflict_max分离head权重与candidate权重。另从原始key精确重建每head来源/历史净效应，net_conflict = 2sqrt(max(-Pnet,0)max(Hnet,0))/(Pabs+Habs)，相同全头与candidate权重。energy_reversal与energy_opposition亦保留。分母零对应没有测得效应，置0；没有增加标签拟合或改动原v1。

保持首轮GSM最多8点每step，在第二轮先隔离读出修改；不将未采样token的行为编成观测。若后续密集采样另开目录。固定探索阈值.5不重新按标签找阈值；报告连续排序、正负配对均值及误报/漏检。评分前预定v2主energy_conflict，不根据评价选择其他候选作为主结果。

## 密集GSM复验（v2结果已知，公式不再变化）

v2主排序：官方两个旧难例.351979、自然配对.606838、GSM24已知步.666667；固定.5阈值三个群均无报警，不能称检测成功。来源正→下游负的energy_reversal在自然配对.730769，但GSM.428571，不能挑其作为统一主线。为了排除每step只采8点的测量遗漏，对同六个GSM回答全部token做相同新native捕获，再用原v2已固定公式整步均值；RAG完全复用原v2分数。不增加方向/阈值调整。这是密集观测验证，不是新候选搜索。

## 定位后的交互/传播诊断

密集数据完成后，对两个自然配对局部声明中间位置、三个GSM首错步及正确对照同序号步骤中间位置，共10处做标签辅助机制审计。这些点不参与拟合评分；同序号不保证相同语义计算。来源仅按连续16token自动分块，补充单独、联合与positive+matched-control联合干预；不人工标注证据。5种处理×5剂量=250前向。每次保存完整32×4096的实际hidden差以及跨层候选lens投影；可观察某次消息扰动的当前query层间传播，但不声称词表投影等于事实，也不声称跨时间总效应。自动选择的是最大/最小有符号块，名称positive/negative不保证该块数值真的正/负。

## v3：attention sink基准下的零和地址交换（诊断后固定）

10个标注定位的传播诊断7次选中L31H5，负块多为BOS/系统模板；读取量与消息幅度干预混入通用载荷。参照attention sinks现象（arXiv2309.17453），不手动标实体或证据，也不移除来源：每层头以当前回答已测query平均prompt attention最大的地址b作常驻参考，完整数据保持。对key j与b作零和交换 δA_j=epsilon A_b A_j，δA_b=−epsilon A_b A_j；多个j联合时相加，±.25保持合法非负分布。效应精确一阶为e_j=A_b A_j g^T W_O(V_j−V_b)。这测内容差异的传递，不把BOS强作用误称正确约束。

重算全部head的相同conflict/opposition/reversal/net读出，原candidate权重和.5工作点不变。v3主仍energy_conflict；完整GSM逐位置，不从第二轮结果挑方向。分块计算仅减内存，不低秩压缩。旧原始因子足以精确重建相对差，保存每head参考地址及全部逐头字段。再在原10个诊断点做零和单块/联合/控制联合有限干预并保留完整hidden路径；数据与旧幅度试验各自保存，不覆盖。
