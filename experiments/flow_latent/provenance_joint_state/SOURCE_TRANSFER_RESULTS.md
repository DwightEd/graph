# 出错前完整内部状态：本轮实测与失败

当前结论：旧post-token监督marker的局部可读性成立；提前首错能力弱得多。新source-program全状态读出可读取结构化绑定，但在真正confident-wrong的旧否定样例上也排序错误；尚未得到无自然标签的有效token预警、类型识别或生成干预。

## 首错预测与完整矩阵

149QA答、30619token、2307错，原生成器Llama2，观测模型Llama3.1-8B。旧监督signed marker post-token AUROC .844678；只用提前1词状态，包含错误延续时 .838923，排除任何已经错误的prefix后 .715071/AP .004990（23924token/51首错，缺prompt-end导致漏1）。提前2词clean .589899；提前8词 .542202。没有重新训练，也未计算此次预测CI。gold首词可能早于实际关系决策词，不能把精度视为语义operator预警。

8×1024完整self-head矩阵的rank1/2/4/full AUROC分别.760762/.819907/.845400/.844678；rank4−full来源CI[−.002406,.004132]，rank1/2明确损失监督方向。rank4总体保留不证明首错更好，或之前4维native边代理有效。这个SVD只检验保留率；不把它解释成信息运输或新检测方法。

## 原生信息传递与当前选择

2旧人工核验case。20完整prefix＋20正确词分支forward，layer17历史t−2正确Passage3消息±.05/.1扰动，重算后继native KV。保存全1024self-head/4096residual；centered finite响应两剂量相对差.00303/.00137。identity与末层future canary为0。尚无exact8B JVP/VJP；fullchain direct/transport/endpoint分解只有toy精确验证。

同prefix正确/错误词分支pre-choice hidden和source方向差0；post-marker错减对gap1.8601/.5079。source+.1对gap影响.000530/.000445，小于该equalnorm random .001171/.001072，未证明来源语义特异性。post-marker可以在正确来源增强后提高对已输入错误词的冲突识别，因此不能把它的方向直接当防错方向。

再130完整forward，输入严格在目标词前结束。保存所有过去预测步的32层×residual-before/attention-write/MLP-write×4096，以及全prefix preRoPE Q4096/K1024/V1024，均float32，无窗口/SVD。两case状态分别[107,32,3,4096]、[224,32,3,4096]；跨层r_next=r+a+m重构0。CPU独立final RMS/unembedding候选margin误差≤1.91e−6，128干预概率/margin/delta复核通过。

| Case | 原生预测尚未提交的错误词 | 错误词概率 | 正确候选概率 | 正确−错误margin | 解释 |
|---|---|---|---|---|---|
|12297|2|.000475|.999269（3）|+7.651581|observer本就选正确，不能解释Llama2旧错|
|12219|not|.883993|.056355（provide）|−2.752783|真实prefix条件confident wrong，尚非自产整答实验|

每层当前MLP写入削10%与equalnorm random，共128干预。0 greedy翻转。12219 L30削弱改善margin+.22334，L31削弱却恶化−.54676；不支持统一削MLP或按正负写入判幻觉。raw logit lens未校正层基底，不能用其正负证明正确fact先存在后覆盖。

## 一个真实source-self-supervised读出pilot

完整z[l,s,d]通过fit-only标准化，候选e(c)为冻结lm_head行单位向量。唯一兼容读出：g=b+Σα[l,s]〈d⊙z[l,s],e(c)〉/√4096。4193trainable参数，另有786432fit-derived均值/尺度；全部96site及4096坐标保留，参数共享本身不是创新。

仅source角色控制中的bool路径，original/swapped×left/right四prefix，True/False单token，候选不进入模型。23fit/8dev来源，124forward、1程序监督fit，400fixed Adam steps，无dev选型。来源等权BCE+pair logistic+ridge。cache195,035,136 bytes。fit/dev pair排序100%，8dev四world全对；native两候选排序同样100%，词面only理论平衡50%。并非full-vocab生成准确率，也不是自然token AUROC或prevention收益。numpy独立复算分数最大差1.7531e−6、分割/四world反转/hash通过。

冻结该模型，再零forward/零fit应用到两旧pre-choice手工候选（超出True/False训练支持，未校准）：12297正确−错误g差+.131390，12219差−4.757120，仍偏错not。numpy/torch独立误差≤1.852e−6。这是适应性跨任务小试，不能报全自然检测，也不能事后挑权重救这两例。它说明程序满分尚未带来该实际错例的提前否决。

## 规模与边界

本轮20+20+130+124=294完整8Bforward；1source-program fit、0natural detector fit。native阶段11.04/6.80/38.06秒，程序阶段73.14秒（时间统计包含项不同，不能合成统一吞吐）。最高18,194,494,976bytes（18.194GB/16.946GiB）。模型/环境未改变，GPU释放。

13科学测试：完整矩阵SVD/低方差功能反例/因果行；toy fullchain分解/末层切点/finite及JVP-VJP；pre-choice不含当前未来词/clean首错历史/时间移位；全坐标候选导数/fit-only统计/source真值反转。独立数值复核是同agent另一算术实现；5轮同家族provisional方法审查，终审8.11/REVISE，非fresh跨家族科学接收。原始数据/失败/评分前协议均保留。

现有自然fit/dev只有36/6个error-bearing答，不能据此选大型probe/多个类型模型。无训练的typed detector、无online pause、无同模型自产确认、无exact8B JVP/VJP。来源约束“未读出”不等于不存在；post-marker、α权重和raw lens均不足以定位错误机制。

下一步冻结native-discordant门：当原生选择错时，内部source-compatible信息是否仍可读且超过matched-native-margin/final-residual-only基线。必须留出来源、模板、candidate词，而不重复饱和lookup题；通过后才做source根→中间原生状态→当前选择的物理路径与暂停。类型按读取/绑定/覆盖/历史错复用/采样偏离作可证伪诊断，允许混合或unresolved，不用一个risk分数自动命名。

原始：graph/outputs/source_transfer_matrix_20261008、source_transfer_native_prefix_20261008、source_transfer_native_branches_20261008、source_transfer_prechoice_states_20261008、source_transfer_boolean_prechoice_20261008、source_transfer_boolean_to_oldcases_20261008。共享FINAL_PROPOSAL.md及EXPERIMENT_TRACKER.md为当前交接。
