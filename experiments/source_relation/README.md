# 来源关联、消息几何与生成采纳

五轮真实局部实验已结束，新增36个prompt的完整原生Q/K，复用48答6463token的A/V/门导数。不是旧测试集复跑。见[结果与失败分析](RESULTS_20260928.md)。旧默认保持，当前没有达到几乎全检且低误报。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.source_relation.run --output-prefix outputs/source_relation_new
```

依赖现有route_complement_20260928、message_js/operator、head_state_readout_v1/v3、source_first和自然trace缓存，以及本地Llama3.1 8B模型。新目录约15GB；单4090采集prompt，之后矩阵计算/CPU读出。入口不安装环境、不改旧缓存；完成标记存在则保留，中断的部分目录保留，新实验用新前缀。不是全RAGTruth运行入口。

- `capture.py`：完整旋转后Q/K，36来源去重，逐层V和原FP32重放比对。
- `measure.py`：每头D为来源读取，H为历史读取，B为来源内部注意力。比较D B与H D_previous的JS；反向邻接使用row-normalize(Bᵀ)，不冒称原生反向因果传递。读取及绝对margin门导数分别计算，保留1024头。
- `kernel.py`：精确W_Oᵀ W_O保留原生输出消息内积；全部来源key之间RBF核，计算加权分布MMD²。无低秩/随机投影。
- `score.py`：头坐标条件近邻、逐头fit秩高尾、相对direct的条件增量；固定来源融合独立列出。
- `refine.py`：以读取/来源关联/生成条件约束旧完整JS+Jacobian采纳状态。
- `combine.py`：旧完整头、前向/反向来源关联、双侧异常的统一max/mean再校准，不按样本选模块。
- `audit.py`、`addresses.py`、`report.py`：逐词误报/漏检、来源正负作用地址、真实竞争token、完整候选/目标核验及位置图。

每task4fit/4dev，整来源互斥；fit/dev只用于无标签参考及混合95分位。标签用于冻结后评价与暴露样本上的研究设计，不用于评分参数、选头或阈值拟合。混合95分位不保证正常FPR为5%，分数不是真假概率。自然只有31个局部标签，其余未知；自然轨迹缺少旧来源条件基线，所有相应融合保留NaN，不算正常通过。

旧完整头融合95TP/147FP；新backward-JS固定融合40/50，消息MMD融合36/38，统一max融合67/85。分母134错误/1353正常，均是局部探索而非全量AUROC。纯内部统一max仅12/79，不能用融合成绩代替。forward-JS高尾在头饰21token AUC .9182、4错/0误报，但全回归该读出AUC .3029；统一校准未保住头饰报警，不能按样本拼接最优成绩。

已有结果：`outputs/source_relation_20260928_v1`至`v5`。各轮`TOKEN_AUDIT.html`可切换全部候选，`FP_tokens.csv`/`FN_tokens.csv`列全部位置；v1的`address_token_ledger.json`包含1518个官方/自然局部token的读取地址和实际竞争词。v5的`all_round_metrics.csv`包含所有候选，`goal_audit.json`记录未达目标。
