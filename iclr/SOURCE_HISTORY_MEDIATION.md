# 来源与历史状态中介：先测有限交互，再做原生状态交换

日期：2026-09-24。入口：`main.py transport-mediation`。

用户要求尝试有理论依据的技术方法，并结合之前的成功与失败结论。
本轮实现两种明确区分的实验：`cache` 复用真实 carrier 删除缓存；`native` 新采集
保持位置的 prompt × 历史 K/V 四世界。前者已运行四答，后者已通过小模型软件验证，
**没有新的真实8B中介检测成绩**。当前全量强基线不改，不把这次实验称为已成立的创新方法。

## 1. 已有结论怎样约束设计

| 已有观察 | 本轮选择 |
|---|---|
| local来源均值四答AUROC .914856，强于复杂图 | 同token、同单元保留来源/路由基线；不自动替换 |
| v2完整图 .866748，去图 .881694，随机图 .872962 | 不再假设生成依赖等于真假同质性；无图风险平滑 |
| 逐头正负效应会抵消，物理头身份可能重要 | 单头有符号作用、端点、联合删除分开保存 |
| 删边可能测到边界/格式，强作用不等于证据选择 | 联合随机端点对照、首token作用份额与位置相关保留 |
| opposition/条件异常/状态递推均未稳定胜过基线 | 不把负作用、异常或持续状态直接当事实标签 |
| choice margin单独检测较弱 | 主读出使用原词logp；明确log-softmax本身也带来非加性 |
| 来源删除改变位置；固定历史可能携带已经写出的答案 | 新实验用同长source-key屏蔽；明确只测连续K/V中介，文本不变 |

本轮不自动提取或改写实体关系。上轮讨论的“事实绑定是否适用”仍未由source-key屏蔽解决；
不能把来源敏感状态叫作已经验证的事实表示。此实验先验证比标量路由更具体的功能分解能否
改善检测，同时为后续受控关系变化提供可复用的状态交换算子。

## 2. 共同理论：两个干预因素的有限差分分解

令两个因素的状态为a,b∈{0,1}，测量同一目标token的log概率 F_ab。
精确恒等式为：

\[
D=F_{10}-F_{00},\quad M=F_{01}-F_{00},\quad
I=F_{11}-F_{10}-F_{01}+F_{00},
\]
\[
F_{11}-F_{00}=D+M+I.
\]

I描述一个因素的作用怎样依赖另一个因素，不能仅由边强度恢复。
如果使用两因素Shapley分配，则
\(\phi_a=D+I/2,\phi_b=M+I/2\)，两者之和严格等于总差。
这是已有的有限交互/合作博弈分解，**不是本项目原创理论**；把交互平均分配也不意味着
找到了唯一真实的路径贡献。源于非线性的I既可能在正确计算中出现，也可能在错误中出现。

最近的activation-patching研究明确讨论了恢复/破坏背景下的交互、backup机制和排序差异，
并指出去掉交互也会失去依赖上下文的机制。因此本轮将交互单列，不把它统一删除或判错：

- Vaidyanathan et al., *The Curse of Multiple Mediators: Hidden Interaction Effects in Activation
  Patching*, arXiv预印本，2026-06：https://arxiv.org/abs/2606.27510 。
- Zhang & Nanda, *Towards Best Practices of Activation Patching in Language Models: Metrics
  and Methods*, ICLR 2024：https://arxiv.org/abs/2309.16042 。
- Stolfo et al., *A Mechanistic Interpretation of Arithmetic Reasoning in Language Models using
  Causal Mediation Analysis*, EMNLP 2023：https://aclanthology.org/2023.emnlp-main.435/ 。

## 3. 可立即复用缓存的检验

a为来源有/无，b为已选择历史消息保留/联合删除。旧数组的condition轴是[有来源,无来源]，
新world轴显式改为[0,1]，不混淆两套顺序。

每个来源条件c内，单边作用和联合作用分别为
\(d_{c,e}=f_c(keep)-f_c(cut\ e)\)、\(J_c=f_c(keep)-f_c(cut\ all)\)。
联合非加性剩余为

\[
C_c=J_c-\sum_e d_{c,e},\qquad C_E=C_1-C_0.
\]

以删除指示变量作多线性展开时，C_c是二阶及以上交互系数总和的负值。
只有单边和全组删除，**不能进一步识别每一对头的交互**。对logp的非加性也可能仅来自
输出log-softmax曲率；不能直接解释成头之间的语义冲突。

评价前固定六个token候选及各自同单元均值：

| 候选 | 分数 | 性质 |
|---|---|---|
| gate_source_shapley（固定主候选） | -phi_a | 两种历史门背景下来源作用的对称分配 |
| gate_source_controlled | -D | 与旧source_selected完全同义，明确作为恢复对照 |
| gate_interaction_size | abs(I) | 交互强度与错误有关的待检验假设 |
| head_coalition_size | abs(C_E) | 条件联合非加性是否提供新信号 |
| head_cancellation | 1-abs(sum_e I_e)/sum_e abs(I_e) | 逐头作用抵消程度；零分母定义0 |
| gate_sham_adjusted | -(g_joint+g_keep-g_sham) | 用已有联合随机端点变化校正联合删除 |

g是对应干预下的有来源减无来源logp。单边sham未采集，因此不伪造逐头随机对照。
空选择集合的交互定义为0，`selected_edge_count`明确区分没有干预的token。
v1是整个单元query上的历史边束；v2是独立目标的精确receiver/key。两者保存的receiver轴
不能混用；v1记录为-1。本轮实测仅v1，用户已运行的v2结果并未被否认。

## 4. 新原生实验：上下文状态与历史内部状态独立交换

旧的有/无来源前向同时改变prompt和所有历史内部状态，因而无法拆开两者。
新的两个donor世界使用**同一套原token、同一位置**：

- 世界1：来源可读，正常原生前向。
- 世界0：每层每个query均屏蔽来源key，原生softmax重新归一化；其他token和位置不变。

首token及最后prompt预测位置必须不属于来源；否则明确报错，不偷偷改干预。
屏蔽后，非来源节点不能通过其他prompt节点间接读取来源；单独只屏蔽最终query是不够的。

对目标y_t，预测query是前一个输入token的位置q。按两个因素构造F_ab：

| 部件 | 状态来源 |
|---|---|
| 严格位于q之前的prompt K/V | donor a |
| 当前query能否读来源key | a |
| 严格位于q之前的回答K/V | donor b，所有层一起换 |
| 当前query自己的K/V、Q、残差、FFN、最终读出 | 在该混合世界重新计算 |
| 回答的离散token与绝对位置 | 四世界均不变 |

这里a包括整个来源条件化prompt状态，所以D不能窄称“直接来源attention边效应”。
M是指定全层历史KV中介集合的受控交换效应，不是已经排除残差旁路的唯一自然间接效应。
I记录两者耦合；不按同层独立贡献相加。历史里的字面事实仍然存在，不能声称已经检验
`E→生成的离散历史文本→当前词`这条完整因果链，也不能区分所有错误事实绑定。

固定主候选`mediated_direct=-D`：在来源被屏蔽时形成的历史内部状态背景上测来源作用。
它检验“来源已经进入历史状态，掩盖新的来源依赖”这一具体假设；该背景仍有原历史文字。
同时保留`-M`、负总差、两种负Shapley项和abs(I)，全部先冻结、后评价，均值只作独立对照。
本轮未按这四答选择其中某个候选或调任何融合权重。

## 5. 一次矩阵块计算多个目标

原生donor前向分块构建K/V。对混合世界，Q行对应不同目标，每行可以共享donor过去K/V，
但不能读取其他目标**重算后的混合状态**。

实际注意力对每行拼接两个部分：
1. 共享donor过去K/V，按绝对位置严格限制j<q及原生sliding window；
2. 一个独立self槽，仅使用这行重算的K_q/V_q。

将两部分logits一起softmax，再读取消息。Q/K/V、RoPE、GQA、W_O、残差、RMS和FFN保留原生
操作；只改变指定的历史/上下文状态。所有物理头独立到W_O，不先平均头；没有反传。

成本为两个donor前向（各一次prompt+回答）及四个回答query重放，后两条对角重放用作身份控制。
与缓存单边删除比不需要每token×每边重算。donor K/V暂存在CPU，重放逐层搬到GPU；
长输入的CPU内存和传输仍有成本。默认prefill128、query16；小显存可减小分块。

`worlds.npz`：[prompt状态,history状态,token]，附原生对角logp及恢复误差。
`--save-heads`额外保存四个[token,layer,head,width]的W_O之前head-readout数组，体积较大；
不是逐key图，不能从中恢复任意边干预，也不是训练好的节点embedding。
同世界重放误差超过`--identity-tolerance`（默认0.05 nats）则保留测量、终止评分，不写完成标记。
小模型float32误差<1e-6；真实8B bf16还没有验证。该阈值是软件身份检查，不是检测阈值。

## 6. 真实缓存实测：未获得提升

输入：已上传carrier v1四答，836 token、81标错、41单元、0混合单元。
原始cohort按回答是否含错误做过平衡抽样；这是重复使用的探索数据，不能称独立泛化测试。
本轮6候选在读取标注前冻结，未翻转弱候选符号或根据成绩换主方法。

| 方法 | token AUROC/AP | 同单元均值后token AUROC/AP |
|---|---|---|
| gate_source_shapley | .619197 / .138752 | .846881 / .333163 |
| gate_source_controlled（旧selected对照） | .626588 / .139579 | .855973 / .340470 |
| gate_interaction_size | .551067 / .108516 | .498749 / .123577 |
| head_coalition_size | .526972 / .105950 | .487205 / .122045 |
| head_cancellation | .587589 / .165432 | .823939 / .293431 |
| gate_sham_adjusted | .619295 / .138337 | .855973 / .341355 |
| source_local | .622680 / .146458 | **.914856 / .451496** |
| source_full | .609558 / .135924 | .823171 / .310212 |
| raw_route | .755425 / .199692 | .770746 / .228008 |

四答中联合source×gate交互绝对值的单元首token份额分别为41.9%、15.3%、18.1%、18.9%；
单位首词数仅占各回答约3.4%–5.9%。这提示边界集中，但没有matched语义对照，不能定性为sink。
多头非加性首错AUROC .79735来自仅2个首错，不能挑出这一结果宣传成功。

本轮结论：多头非线性确实可被测量，但这些强度/分配方式没有改善检测；
不能继续把大交互、头抵消或Shapley分配包装成已有效方法。原生状态交换是**另一种测量**，
其自然数据有效性尚待验证，不能从这些旧删边结果推出成功或失败。

## 7. 运行与交付

先做CPU缓存试验（可直接对用户已有v2缓存运行）：

```bash
python main.py transport-mediation --mode cache \
  --input outputs/native_support_ragtruth4/message_carriers_token_v2 \
  --output outputs/native_support_ragtruth4/source_interactions_v2
```

再采集四世界原生中介（默认读取同一个v2目录中保存的输入）：

```bash
git pull --ff-only origin main
bash experiments/native_support/run_source_mediation.sh
```

默认输出`outputs/native_support_ragtruth4/source_mediation_v1`，自动生成同级`_review.zip`。
需要修改模型位置用`--model /path/to/model`，分块用`--query-chunk-size 8 --prefill-chunk-size 64`。
旧contrast或carrier完整目录/ZIP均可作为native输入，不必重跑旧carrier采集链。
该小实验入口不直接解析全量官方数据；全量基线入口仍是`transport-benchmark`。

`--stage capture`只采集；`--stage score`只保存分数并停止，不读标签；`--stage evaluate`仅CPU重评；
`--stage pack`只打包。原命令`--resume`按回答跳过已完成的新采集，协议不可暗改。
输出包括原基线、逐token新分数、组成量、逐头有限效应或可选readout、AUROC/AP、答内/
单元内/首错/延续/前后半段评价、同单元均值对照图、诊断及完整打包。

针对性测试覆盖：有限差分恒等式、非加性/抵消构造例、三模型原生对角恢复、独立原生混合KV
参考与矩阵重放一致、source屏蔽内容不变性、无未来读取、v1/v2缓存轴、标签隔离、原文件不改、
完成标记与恢复误差失败路径。随机小模型只验证实现，不能证明检测或事实绑定有效。
新增15项测试与已有全量benchmark9项回归，共24项通过；CLI及shell语法检查通过。
