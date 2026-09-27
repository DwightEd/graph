# 条件响应监督检测：2026-09-28完整实验

用户授权轻量监督或伪标签；本轮使用官方自然训练标签。新入口
`python main.py probabilistic`复用source-first四条件原词概率及原生route/attention/entropy，
不生成新答案、不调用外部事实核验。源码说明见
[probabilistic_detection](../experiments/probabilistic_detection/README.md)。

## 完整测试结果（历史test上的探索性复验）

| 任务 | 旧source_refine AUROC/AP | 冻结条件交互 AUROC/AP | 同输入HGB AUROC/AP |
|---|---|---|---|
| QA | .880220 / .286478 | .918765 / .509635 | .916670 / .461740 |
| Summary | .763045 / .137236 | .795918 / .260765 | .797821 / .219717 |
| Data2txt | .788506 / .160782 | .828858 / .227465 | .830690 / .232674 |

15,090 train答按来源拆分fit2,011/dev504；test2,700答/450来源/424,408有效token。
所有六生成器保留，每来源全部回答在同一分区。先冻结全部预测再评价。
三任务在开发集均选conditioned，C分别1/.01/1；测试后不改选择。
旧基线只有开发选分数，新模型还使用训练标签，增益不能全部归因于交互结构。

来源簇bootstrap300次，新方法减旧基线的AUROC差95%区间：
QA [.02689,.05064]；Summary [−.01145,.08948]；Data2txt [.03192,.05051]。
Summary区间包含0；相对同输入HGB的区间三任务都包含0。
统计比较未校正多重检验，官方test已在既往设计中暴露，不称独立新确认。

## 实现和消融

- `data.py`：6个来源/位置context，11个概率/路由/熵response观测；原token对齐，离线窗口逐答计算，无标签特征。
- `model.py`：共同训练分位变换、context spline、类条件ridge均值、收缩Gaussian LR；对角/共享covariance/共享correlation对照。
- `readout.py`：直接L2 log-odds估计，linear→每维平方→观测交互→来源anchor调节交互，全部列训练加权标准化。
- `iteration.py`：固定小C池，开发选择，固定linear特征消融；另工程选择已有模型族最优开发分数，不包装成新算法。
- `run.py`：prepare/develop/iterate/freeze/score/evaluate分阶段；冻结输出拒绝覆盖。
- `evaluation.py`：AUROC/AP、答内/单元内、onset/first、开发阈值误报与召回、来源bootstrap。
- `cases.py`：已暴露难例逐token报告，区分fit/dev/test，训练内例不计泛化。
- `report.py`：原始冻结结果导出CSV、Markdown和SVG/PNG比较图。

Data2txt的test阶梯为linear .812936→squares .818875→interactions .824588→conditioned .828858。
交互−平方CI [.00386,.00737]，条件−交互CI [.00139,.00708]；有当前表示下的预测交互证据。
QA conditioned−linear也为正，但Summary没有一致的非线性优势。
Gaussian不是赢家；不能因其数学形式更复杂就声称创新有效。

## 未解决的问题

- QA单元内AUROC由旧.831193降至.771404；不能称定位全面改进。
- 新方法test正常回答any-alarm为QA20.41%、Summary36.78%、Data2txt80.06%；Data2txt旧值47.98%。
  token FPR约5%不等于回答级报警可用。
- 9022营业时间由旧1/13检出至7/13，intimate由0/3至1/3；WiFi仍0/3。
- 15604养老金税率仍0/15（fit诊断）；219时长仍0/3（dev）。7305时刻15/16检出但邻近正常全误报（fit）。
- 12045两个test错误段分别12/15、31/53检出；12219由旧整段检出退为8/13。
- 单observer标量cache无法证明多头协同、FFN语义或生成器自身的因果信息流。
- 相对HGB没有明确优势，不能据此认定已形成顶会方法。

## 值得继续检验的机制假设

给定来源支持后，attention的历史/来源分配与消息幅度加权分配是否失配？
在单层归一化attention分布pi(h,k)=a_hk/H下，令u=||W_O v||，
q为历史/来源/其他的+1/−1/0指标，则

    A=E[q], R=E[uq]/E[u], R−A=Cov(u,q)/E[u].

这是幅度重加权偏好的恒等式，不是语义或事实正确性的证明。
原实现历史包括query自身的key；有限精度和首query排除的prompt token需相应处理。
当前分类器对R/A分别做经验分位变换，相反系数不等于直接测量原生R−A。
下一轮须预先冻结原生差分/共同分量/来源条件差分对照，并在新observer或新数据上检验。
只有对检测有效才进一步做reanchor消息内容与约束保持的干预，不穷举无收益消融。

[HaloProbe](https://arxiv.org/abs/2604.06165)已有条件先验和内部Bayes判别；
[CORTEX](https://arxiv.org/abs/2606.31033)已有文档差分、历史传播修正及连续span处理。
Bayes、Gaussian LR、二阶logistic不是本项目原创贡献。

## 执行证据

全部主实验实际exit0；41项针对性测试通过。奇异常量输入测试保留4条sklearn数值警告，
输出有限、正定性测试通过；实际自然训练未出现对应警告。
无新增大模型前向；本轮CPU重用缓存，不表示未来新回答免除原有四条件测量成本。
原始结果在`outputs/probabilistic_detection_20260928_{pilot,full}`。
共享方案、逐例HTML、完整统计和审查在
`/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/probabilistic_detection_20260928/`。
新fresh审计线程因agent thread limit不可用；已有非实现者的独立数值复核有设计背景，
仅同家族provisional，不冒充fresh或外部科学评审。
