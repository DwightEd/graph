# 2026-09-29 完整实验结论

重复头状态可补齐部分错误段，但本轮没有获得三任务一致的AUROC提升，不替换原固定无监督基线。已完成9轮局部读出/传播迭代，以及5套完整2700答评价；独立的新8B采集是一套2700答/450来源/424408token，后续版本复用其完整1024头响应与JS。

|方法|QA AUROC|Summary AUROC|Data2txt AUROC|
|---|---:|---:|---:|
|原完整参考固定无监督|.890665|.752177|.758040|
|新响应与原基线融合|.864729|.734673|.762453|
|融合起点+双起点重复|.872556|.734646|.754596|
|原基线起点+双起点重复|.882804|.749103|.752481|

新响应融合只在Data2txt小幅增加AUROC（来源簇名义95%增量区间[.000584,.008424]），QA/Summary明确下降。QA双起点重复相比新融合有增量，但仍低于原基线；仅距离对照QA .880525，高于头重复 .872556，所以局部头身份对照不能证明完整QA优势。Summary打乱头 .736616也略高于原生双起点 .734646，未发现统一头身份收益。

原基线双起点把Summary完整错误连续段33/239→76/239，Data2txt80/1021→190/1021，但正常token误报分别3.67%→12.00%、4.02%→9.92%。QA新阈值下完整段21/213→4/213，错误召回25.51%→9.67%；这不是成功降低误报，而是校准与召回的严重取舍。新融合双起点的Summary正常回答any-alarm76.44%，不可忽略。

局部8答中，融合双起点54/134错、87/1353误报→63/70，3/9整段覆盖。第一轮70个FN：15个整答无高分起点，9个24token内无起点，46个有附近起点但无合格路径；新增41个FP中28个由错误起点跨界传播到正常文本。稳定错误绑定、遗漏否定/限定、错误与正常解释混杂，以及低风险整段，仍未解决。具体token和测量路径见RECURRENCE_RESULTS_20260929.md及输出propagation_tokens.csv/propagation_head_paths.json。

结论对应的设计边界：JS/Jacobian是消息及当前选择的测量工具，不能直接充当事实真假；高维异常不保证错误，重复状态不保证共享真假。将风险起点、段内覆盖、事实边界与校准拆开审计有用，但本轮实现的max-min路径和双起点还没有形成能替代基线的完整检测器。扩大窗口或降低阈值会放大误报，本轮不继续用已暴露test搜索这些参数。

局部8答是开发回归，不是独立留出：15604/219/9022/7305属于官方train，11907/12015/12045/12219属于test。新的4fit/4dev参考与官方test来源互斥；原完整参考基线的train案例不能冒称原参考留出。4个重叠test回答×13方法的局部/新采集比较：最大分数差2.97e-5，报警差0。自然31token局部标签单列，缺来源分数的融合方法不可评价。

所有5套全量官方标签/数值复核共108000个case-method组、120个task-method组（包括复用控制），均为同agent另一实现，不是外部独立评审；11项不同针对性测试通过。原采集进程在全部预测冻结后评价中收到SIGTERM143，CPU恢复exit0；末尾报告原假设8答均属test导致检查失败，按真实ID交集修正并仅重跑报告，退出0。失败日志和原始数据保留，未重跑模型。当前无活动实验。

生成器分组共252行，不能只看总体；例如QA/gpt-4仅19个错误token，分组AUROC极低但样本很稀，不能外推为稳定生成器规律。完整原始目录为outputs/context_response_full_20260928_v1–v3、outputs/context_response_recurrence_full_20260929_v2–v3。所有结果是历史暴露总体上的探索复验，原起点追加设计还参考了本轮初步full结果。

下面为最终同分数总体上的完整比较与来源簇区间。

# 完整三任务：重复模式与风险起点对照

2700个回答、450个来源、424408个token，所有1024物理头的新响应/JS已经采集。评分冻结后评价；同一测试总体历史已暴露，原起点追加对照还参考了本轮早期结果，属于探索复验。检测评分/校准无自然标签拟合，不声称盲测。

|任务|方法|AUROC|答内AUROC|错误召回|正常token误报|正常回答任意报警|完整错误连续段|
|---|---|---:|---:|---:|---:|---:|---:|
|QA|original_full_reference_fixed|0.890665|0.857399|0.255073|0.017007|0.105405|21/213|
|QA|strong_fused|0.864729|0.811258|0.287513|0.031889|0.410811|3/213|
|QA|recurrence_offline|0.869870|0.823605|0.357873|0.032507|0.144595|24/213|
|QA|recurrence_shuffled|0.845903|0.772680|0.380388|0.064252|0.377027|5/213|
|QA|distance_offline|0.880525|0.833220|0.541253|0.058120|0.144595|56/213|
|QA|corroborated_only|0.872556|0.830992|0.329581|0.026892|0.187838|21/213|
|QA|barrier_corroborated|0.871813|0.828567|0.325433|0.026451|0.187838|20/213|
|QA|original_recurrence|0.880475|0.833152|0.088579|0.003786|0.008108|3/213|
|QA|original_corroborated|0.882804|0.839368|0.096726|0.003913|0.010811|4/213|
|Summary|original_full_reference_fixed|0.752177|0.760802|0.264849|0.036718|0.252874|33/239|
|Summary|strong_fused|0.734673|0.743477|0.312482|0.085163|0.764368|9/239|
|Summary|recurrence_offline|0.732426|0.743057|0.393113|0.109754|0.669540|20/239|
|Summary|recurrence_shuffled|0.736616|0.711917|0.308752|0.075321|0.568966|8/239|
|Summary|distance_offline|0.724625|0.750036|0.465710|0.149996|0.429598|86/239|
|Summary|corroborated_only|0.734646|0.748681|0.405165|0.124139|0.764368|24/239|
|Summary|barrier_corroborated|0.735883|0.751490|0.403156|0.121906|0.764368|24/239|
|Summary|original_recurrence|0.747204|0.754839|0.457963|0.126118|0.495690|75/239|
|Summary|original_corroborated|0.749103|0.758565|0.445624|0.119995|0.508621|76/239|
|Data2txt|original_full_reference_fixed|0.758040|0.732719|0.143648|0.040237|0.299065|80/1021|
|Data2txt|strong_fused|0.762453|0.749138|0.314432|0.081440|0.878505|97/1021|
|Data2txt|recurrence_offline|0.749403|0.730342|0.348589|0.110609|0.778816|109/1021|
|Data2txt|recurrence_shuffled|0.726347|0.709962|0.315242|0.117108|0.816199|82/1021|
|Data2txt|distance_offline|0.731452|0.714084|0.227217|0.065498|0.308411|142/1021|
|Data2txt|corroborated_only|0.754596|0.738533|0.343594|0.100307|0.819315|108/1021|
|Data2txt|barrier_corroborated|0.753238|0.736792|0.328743|0.096656|0.819315|95/1021|
|Data2txt|original_recurrence|0.751008|0.723653|0.317132|0.107730|0.510903|187/1021|
|Data2txt|original_corroborated|0.752481|0.725417|0.300797|0.099248|0.514019|190/1021|

完整错误连续段以每答官方token正标签的连续区间统计，和单个原始标注对象的数量可能不同。离线双向传播可用后续token；过去向消融单独保存在完整指标中。

来源簇bootstrap300次；以下是AUROC增量的名义95%区间，未对多候选作多重比较校正：

- QA / recurrence_offline − strong_fused: [-0.002780183658509491, 0.013131636280656073]
- QA / recurrence_offline − original_full_reference_fixed: [-0.027978699440656767, -0.013085140244355635]
- QA / corroborated_only − strong_fused: [0.0013296904340619976, 0.014125484656780209]
- QA / corroborated_only − original_full_reference_fixed: [-0.025365929899469238, -0.01043214459835808]
- QA / original_recurrence − strong_fused: [0.004289767830906649, 0.029073151535045187]
- QA / original_recurrence − original_full_reference_fixed: [-0.013410096987229276, -0.006837838118848768]
- QA / original_corroborated − strong_fused: [0.006584749697980188, 0.0314746540319779]
- QA / original_corroborated − original_full_reference_fixed: [-0.010912537576932202, -0.004661470187008355]
- Summary / recurrence_offline − strong_fused: [-0.015482819940364994, 0.0073724748804480525]
- Summary / recurrence_offline − original_full_reference_fixed: [-0.03308433054742131, -0.01128595272168625]
- Summary / corroborated_only − strong_fused: [-0.010142793119506097, 0.007238139409716471]
- Summary / corroborated_only − original_full_reference_fixed: [-0.02849289401047817, -0.00993703444861357]
- Summary / original_recurrence − strong_fused: [0.00018257885072801044, 0.023024554949475192]
- Summary / original_recurrence − original_full_reference_fixed: [-0.012216671057955153, 0.000695909329442146]
- Summary / original_corroborated − strong_fused: [0.00396338854665301, 0.023296965381418493]
- Summary / original_corroborated − original_full_reference_fixed: [-0.008303011037641809, 0.0007817834592498236]
- Data2txt / recurrence_offline − strong_fused: [-0.01736097122518384, -0.008791192343626072]
- Data2txt / recurrence_offline − original_full_reference_fixed: [-0.013952282612095478, -0.0032009005160499046]
- Data2txt / corroborated_only − strong_fused: [-0.011458415165224885, -0.00453217001411365]
- Data2txt / corroborated_only − original_full_reference_fixed: [-0.008096767550285863, 0.002236298447182051]
- Data2txt / original_recurrence − strong_fused: [-0.01714202142827605, -0.005733453127216128]
- Data2txt / original_recurrence − original_full_reference_fixed: [-0.010456063047130575, -0.003558260901852346]
- Data2txt / original_corroborated − strong_fused: [-0.015348768416806612, -0.004572668518744949]
- Data2txt / original_corroborated − original_full_reference_fixed: [-0.00813780441637162, -0.0026514490996401513]

原生响应直接影响当前原词相对竞争词，不是事实证据符号。重复图表示机制状态相似，不证明语义事实相同；max-min与双起点并非后验概率。

四来源无标签dev95不是正常FPR=5%的保证；原固定基线使用原完整训练/dev参考与阈值。原评分数组未改写，已有对照逐元素核对后复用；full_verification.json为同agent另一实现的官方span和数值复核。

所有14方法/任务的完整指标见RESULTS_ZH.md；错误连续段见error_run_metrics.csv；生成器分组见generator_metrics.csv；新采集与局部缓存的分数一致性见pilot_stream_consistency.json。
