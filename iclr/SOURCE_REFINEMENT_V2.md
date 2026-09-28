# 来源优先的 token 细化 v2

入口 `main.py source-refine`，复用 `outputs/probabilistic_detection_20260928_full/packs`。
它将标量读出从 `graph-anomaly` 的 123 维变换、IsolationForest、图训练中拆出，
不需要 torch/CUDA 或新的 LLM 前向。图历史实现和原结果仍保留。

## 输入与公式

`scalar.py` 统一旧标量公式及拟合来源的 10001 分位点；旧 fixed 分数不变。
设 local/full 单元来源风险为 `context[:, :2]`，单元内原生偏差为
`observations[:, :2]`，则 token 来源风险为二者相加后取两通道均值。
高风险表示去来源后原 token 的 logp 相对更高，不等于错误的事实概率。
route 使用原缓存的离线 16-token 窗口，不重算、不更改方向。

新 source5 对 token 来源风险取原始 target 距离 ±2 的均值，同时限制在同一个
回答、同一个保存文本单元内；排除的特殊 token 不计入分母、不用更远 token 替代。
这是离线局部平滑，不是模型发现的 reanchor。

每个通道以 fit token 的经验分位点作秩变换 R，非 source 等权、非真值校准。
对细化通道 d，按当前单元内有效 token 居中：

    delta_i = R(d_i) - mean_{j in same answer/unit} R(d_j)
    residual_score_i = R(anchor_i) + lambda * delta_i

lambda 固定为 0.025/0.1；修正绝对值不超过 lambda，单元内均值不变，但可能
改变跨单元排序。token_source/source5/route 三种细化，local/full/pair 三种锚点。
固定无监督主候选 `pair_source5_residual_0.1`；另单列 `pair_source5_tie`。

## 严格同分细化与可迁移阈值

严格按 `(raw_anchor, delta)` 字典序比较。每个待评分批次的不同 raw_anchor
加上已冻结阈值的 anchor 构成整数档位，再加 `0.25 * delta`。
不同来源锚点的顺序不变；只有完全相同的 raw_anchor 才按 delta 排序。
不能用 fit 档位的 searchsorted 替代这一步，否则不同新 anchor 会落入同档。

报警阈值保存为开发混合分布字典序的第 95% 分位二元组 `(anchor, delta)`；
在当前批次同步编码此阈值，再作严格大于判断。数值分数随批次不同，
但相同 token 对的顺序及报警判断不变。不要跨批次拼接这些整数分数，
应合并原始二元组重编码后计算总体 AUROC；分数不是概率。
原有 `transport-refine` 已有同分/残差思想，这里修复 inductive 档位问题并改变平滑边界，
不把这些成熟读出当作新机制或图方法。

## 执行与隔离

```bash
python main.py source-refine --phase prepare --output outputs/source_refinement_20260928_v2
python main.py source-refine --phase select --output outputs/source_refinement_20260928_v2
python main.py source-refine --phase cases --output outputs/source_refinement_20260928_v2
python main.py source-refine --phase score --output outputs/source_refinement_20260928_v2
python main.py source-refine --phase evaluate --output outputs/source_refinement_20260928_v2
python main.py source-refine --phase cases --include-test --output outputs/source_refinement_20260928_v2
```

每次复验用新输出目录。prepare 不读取标签数组；三任务开发输入先评分，再 select 读取
原开发来源的自然标签按 token AUROC 选型，固定原融合也参与候选。34 个预先固定候选。
固定版拟合/评分/阈值无自然标签；selected_dev 是标签辅助选择，不能叫严格无监督。
全部方法报警阈值统一使用无标签开发混合分布 95% 分位，不承诺正常 FPR=5%。
score 前冻结全部三任务的参考、校准和选型；全部 test 分数完成后才 evaluate。
评估报告完整三任务、全部生成器、AUROC/AP、答内/单元内定位、首错、误报、source bootstrap。
历史 test 已暴露；本轮为探索性复验。已知 fit 案例是 in-sample，dev 案例参与过开发。

运行环境复用 research conda Python；CPU threads=4。科学不变量测试见
`tests/test_source_refinement.py`；实际结果见共享 research 目录的
`refine-logs/source_refinement_20260928_v2/RESULTS_ZH.md`。

## 本轮实际结果（全部官方测试）

|任务|上轮dev选择AUROC|本轮dev选择AUROC|本轮AP|
|---|---:|---:|---:|
|QA|0.890665|0.890665|0.353886|
|Summary|0.767623|0.761094|0.145833|
|Data2txt|0.785537|0.787538|0.175266|

固定无标签主候选（统一pair/source5/0.1）分别为0.879552/0.748534/0.787538，
不能把选型版成绩称为统一固定无监督方法的成绩。
Data2txt对上一轮AUROC增量95%来源bootstrap CI为[0.001453, 0.002476]；
Summary为[-0.012431, -0.001315]。不覆盖旧Summary结果，不事后按test拼接“最佳”系统。
养老金/WiFi/时长等仍漏，7305检出同时误报邻近全部正常token。
同协议阈值下QA正常例0误报不代表新排序成功，QA选型分数与旧fixed逐元素相同。
2700答/450来源/424408token完整评价actual exit0，56测试通过；无新GPU前向。
全部生成器及首错/误报指标见原始输出；不宣称已解决事实绑定或优于旧监督检测器。

完整回答/单元先确定细化残差后，严格同分的顺序与阈值判断才具有批次不变性；
把同一单元拆成不完整token批次重算其均值，会改变定义，不在本协议内。

进一步官方标签计数验证：若严格保留原始pair来源顺序，AUROC最大增益不超过
0.5×同分正负对比例。本测试QA/Summary/Data2txt上界为1.32e-5/2.90e-5/3.19e-5。
因此严格同分适合作定位对照，不能作为大幅提升全局AUROC的主线；上界仅对应本批原始pair锚点。
