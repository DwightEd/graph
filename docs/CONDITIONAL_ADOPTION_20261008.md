# 原生候选响应、真实过去中继与损失弹性：已执行，检测未通过

2026-10-08。本轮新增真实8B原生测量与两次候选验证，没有复跑旧模型冒充新方法，也没有将新方法默认为成功。四来源八个历史QA回答，共1139个有效token、77个官方错误token；3错误答、5正常答。默认旧方法保持，未新增三任务全量/GSM成绩。

## 做了什么

1. 对64个fit来源/384答/53767词审计自然关系伪目标。现parser仅提出9关系/34责任位置，还含未解析代词和不可靠阶段对齐；qualified自然方向/责任/q均0。未训练q/π，G0 FAILED不是检测recall0。
2. 对八个完整回答逐token执行两目标原生反传：actual logp，以及actual相对native最高非actual候选的margin。保留32层×32头×5通道的有符号响应与完整128维消息/梯度。1139词/2278 backward，733.026秒，峰值18.024GiB。参数与软件环境均未变。
3. 固定L15与+.25幅度执行当前/严格过去/联合来源增强，过去改动的全部后继KV原生重算。3491无梯度forward，272.387秒，峰值18677317120字节。prefix文字不变，是teacher-forced条件内部状态中介，不是离散生成的完整因果图。
4. 第一阶段失败后预定一次开发迭代，将目标换为log预测损失，ε=1e−4，无权重/参数搜索；明确post-pilot探索，不回写原primary。H从未进入新风险，没有span均值/标点清零。
5. 新审计17个高分正常词与错误标注位置（含已检出的错误词），复用已存梯度获取完整逐层/头/key的有符号AV作用；与五通道作用最大重建误差9.5367e−7，0新backward、0新风险评分。

## 实际结果

下面全部同1139词/77错误，AUROC采用原始分数。macro答内AUROC只平均三个同时含正负token的错误回答。历史病例已暴露，正常配对全GPT4、错误来自Llama2，pooled不能作为生成器无关的机制证据。

|候选|总体AUROC|AP|macro答内AUROC|
|---|---:|---:|---:|
|冻结主：source_candidate_regret|0.476398|0.067847|0.485431|
|预定对照：source_margin_rejection|0.471360|0.070604|0.504916|
|预定对照：source_local_countervote|0.523088|0.067773|0.465473|
|开发迭代：source_loss_elasticity|0.480996|0.075627|0.524237|
|有限当前来源：regret_current|0.456074|0.063879|0.456779|
|有限过去来源：regret_past|0.453469|0.063450|0.458745|
|冻结中继主：regret_both|0.442830|0.062589|0.451905|
|有限联合损失弹性|0.427862|0.069050|0.470809|
|旧source/route固定对照|0.430271|0.079948|0.486145|
|旧开发标签选型unit对照|0.413921|0.181319|0.623504|

表中数字已逐项核对raw evaluation_v2.json。旧unit用自然dev标签选方法，不能叫本轮严格无监督对照；旧fixed的无标签fit参考包含15521这两答。新增候选0自然标签拟合，参考/方法选型/已暴露病例分别记账。

以评价集正常词95分位数作**oracle诊断**（不是部署校准）：原生signed检出3/77，损失弹性6/77，联合中继4/77；三个候选都在五个正常回答上报警。新损失弹性检出了12219的not@223，但不是首错起点，且12297的2@106仍漏。不能称“误报下降”“首错有效”或“几乎检尽”。

## 具体原因：源码和真实数组支持的结论

- **正常corner@12219:99**：来源corner key195的margin作用−1.3383，其他来源corner也为负；当前native竞争词是and。来源确实影响措辞/顺序，不意味着正常corner是错误。
- **正常fold@12216:35**：来源bias key210的margin作用−3.5881，native竞争词bias。阅读动作及修饰词之间的选择也产生来源拒绝，不能按负作用直接报警。
- **错误normal@15604:103**：来源normal key216作用+1.8426，native竞争词state。取到了源文中的normal rates，却没有认证它适用于当前断言；源文第4项是工资（Wages）按正常税率征税、第5项是私人养老金（private pension income）不征税；回答却生成“private pension income is taxed at the normal state tax rate of 5%”。这是把工资规则绑定到养老金，来源助推actual不能等同事实支持。
- **错误not@12219:223**：真实完整输入source regret .159155、NLL .121177左右，弹性1.312344；oracle cutoff1.136290下可检出。来源反对不是没读取，但正常词也能更强反对，raw cutoff .568148时仍漏。L15联合增强反而略帮助not，说明一个局部层不能代替所有信息作用。
- **错误引用2@12297:106**：native竞争词3，actual NLL7.6677；整体source regret只有.173219，弹性.022590。引用处源文空白/标点正负作用相消，来源选择、内容归属和词预测拒绝不是同一信号。它所在官方49-token错误span多数内容本身源文可支持，不能把这个观察偷偷重标后报更高token成绩。

逐key诊断不是任意挑最大attention token：分别保存正/负最终候选作用和attention首峰，完整贡献保留。当前key导数只测当前AV替换作用；不能据此称已经追完某个过去key的最初源读取。

## 投稿水准的判断及切入点

这轮算子保真通过，检测主线失败。当前不能作为顶会级有效方法，不能通过加入图/谱/JS把失败变为成立。下一主线的对象必须是**来源约束与实际断言的绑定状态**：自动识别主体、谓词、极性、阶段/条件、归属及token责任；再用已实现的有符号算子和真实KV中继验证该绑定如何控制候选及延续。缺的是关系适用方向与责任，不是没有更复杂的传播公式。

实现边界必须清楚：约束/责任现在尚无可用自然q，不把未知判正确/错误；不人工圈prompt实体；源程序伪目标只能作自动自监督并验证自然迁移；监督自然标签只用于机制发现/独立诊断，不能把监督head权重改名无监督。首先必须在normal rates错用、正常corner/fold，以及引用归属上提高答内区分与正常词控制，再决定是否全量。

## 代码与原始证据

入口 `experiments/conditional_adoption/run.py`；说明同目录README。函数分别在mechanism.py、relay.py、readout.py、elasticity.py、evaluate.py、key_audit.py；不是只有提案。11项科学CPU检查通过，8B联合方向20forward finite检查最大relative .2061%、最大absolute .001007；所有1139 token身份、严格预测位置及有限性通过。原始向量/前后代码快照、四套分数、v1与v2评价、逐token/逐key审计都保留。

raw：`graph/outputs/conditional_adoption_20261008/{complete_roster,complete_capture,relay,scores,elasticity_scores,relay_scores,relay_elasticity_scores,key_audit,figures}`。四套`evaluation_v2.json`更正了旧baseline方法级监督选型元数据；原分数不变。最初base评价在添加旧fixed对照时曾被覆盖，独立审计记录了前后哈希；不能宣称所有历史评价版本均完整保留。后续评价改为拒绝覆盖、写新文件名，现有evaluation.json和v2均保留。figure comparison.png/PDF与三个逐token轨迹使用原始分数，不缩放AUROC。

独立experiment-audit已完成，结论WARN（same-family/provisional）。51方法块、612排名块、4556逐token行及物理数组复算一致；官方GT、评分构造、指标执行通过。局限包括历史样本/生成器混杂、oracle阈值、旧标签选型与历史模型依赖哈希不完整。审查运行时型号未独立认证，不称跨家族科学PASS。审计后修改仅清理元数据与文档，见POST_AUDIT_REMEDIATION.md；不追认历史依赖保证。所有本轮GPU已退出，最后读回1MiB/0%。观察器Llama3.1-8B不是这些回答原生成器，不能据此声称解释原始生成因果机制。

复现命令和输出新建规则见 [模块README](../experiments/conditional_adoption/README.md)。共享研究目录中的EXPERIMENT_AUDIT和POST_AUDIT_REMEDIATION保留独立WARN与事后修正，原始大体量数组不入Git。
