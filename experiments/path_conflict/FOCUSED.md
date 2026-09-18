# 指定证据支持如何量化：固定 head 的后续干预

## 本轮问题与输入

前一轮实际轨迹中，服饰题 L22H28、L23H6 的证据组局部读出差异较大；L31H14 的历史负向读出两侧共有，最终效果又受 MLP 重算影响。因此，本轮验证这些**固定坐标**的功能，不重新按本轮结果挑最好 head。

仍读取原 Llama-3.1-8B-Instruct 的16条多采样轨迹，只使用 cases.json 中两项局部陈述的两侧前缀。不训练检测器、不改原文、不重新采样、不使用 CHARM/LDA，也不把局部标签冒充整答标签。服饰有依据侧的前文并非完全无错。

## 1. 三种量必须区分

固定预测前位置 q、层 l、head h 和原文位置集合 S：

```
route_mass = sum_{j in S} A[h,q,j]
m_S = W_O,h @ sum_{j in S}(A[h,q,j] * V[h,j])
M = logit(correct first token) - logit(wrong first token)
```

A来自这次原生 eager attention；V来自同一次v_proj；W_O来自原模型。GQA按query-head到KV-head的固定映射展开，所有head独立。无head平均、无attention阈值、无重新softmax。

- **关注质量** route_mass：读取了多少权重，没有语义正负。
- **局部支持** D = Lens(r) - Lens(r-m_S)：r为attention后的真实残差。Lens应用原终层norm和两个候选的lm_head。它只衡量该读出器在中间状态上的变化，不运行下游。
- **最终路径支持** C = M(full) - M(cut m_S)：在原attention输出中减去m_S，保留残差、MLP、其他head，并让剩余层重新计算。C>0表示该写入在这次定义下支持正确候选相对错误候选。

局部D和最终C可以异号。二者不能相加；C包含下游非线性反应。非线性作用下，多head单独C的和也不等于联合C。

具体代码：operators.source_write提取消息；native.NativeRun.apply在原模块输出中减去；scoring.read_prefix计算最终概率。核心删除只有：

```python
_, write, mass = source_write(attention, values, module.o_proj.weight,
                             [query], source_indices, [head])
changed = attention_output.clone()
changed[0, query] -= write[0]
```

这是**来源位置上某条读取边的支持**。V_j已经含上下文，scope语义可能已进入其他token；删scope位置的边不等于抹掉该条件全部语义。分类器正权重也不用于这里的支持方向。

## 2. 原文角色拆分与来源轨迹

| 角色 | 服饰题 | 烹饪题 |
|---|---|---|
| scope | Only the Inca could wear | Remove the bratwurst from the beer mixture; |
| supported_value | 普通帽子及折叠布的原文描述 | reduce heat to low, and continue cooking the onions. |
| value_source | 特殊头饰与金羽的原文描述 | 10–12分钟所在原文句子 |

supported_value是一整段直接描述，仍含主语和句法，并非纯净的“候选词面因子”。上述角色由人工按材料提议；支持性标签仍需结合具体陈述理解。

历史分成互斥的query_self={q}、recent_history=[max(P,q-10),q)、remote_history=[P,max(P,q-10))。其他prompt另列。角色与token映射在启动时统一检查并保存source_tokens.csv.gz；不把self混入历史。

对四个指定head还保存q之前10步至q的完整逐来源attention，及最终q上的每来源V范数、W_O消息范数、局部候选支持。source_routes.html显示每步前三个地址用于阅读；NPZ保留所有来源，不用显示截断计算效应。较早q的来源角色仍按最终决策划分，self标志另按query==source计算。

## 3. 预定对照

固定坐标 L22H28、L23H6、L31H14、L31H21。每个head分别切scope、supported_value、value_source、self、近历史、远历史和其他prompt。

1. L22H28和L23H6分别切证据组，再联合切；scope与supported_value联合分别测。半剂量0.5与全剂量1并列，不假定响应线性。
2. 联合早期证据切断与L31H14历史切断，保存 `M_AB-M_A-M_B+M_full`。这只是明确操作下的交互差，不叫协同百分比。
3. 早期证据切断后，将L27–L31各层MLP输出分别恢复到**同一个前缀的未干预值**；另外恢复L31H14或L31H21。比较恢复前后变化，不跨自然前缀移植激活。
4. 切L31H14历史时，固定原L31 MLP输出。对比“MLP自然重算”与“MLP输出固定”，定位前轮两侧净效应不同的来源。此恢复控制改变了输入—输出一致性，是受控路径检验，不是自然中介比例。
5. 同层相邻编号head作固定控制。早期联合与末层历史各加3次随机残差方向；每层随机改变量范数使用**相应真实cut运行**记录的范数。bf16最终相减仍有舍入，实际范数全部输出。
6. 所有关键操作在两条自然前缀重复。正确侧同样有冲突并不意外；比较具体效应，不能把冲突存在作为幻觉判据。

默认每面板62种条件（含baseline），服饰两种候选表述×两侧、烹饪原表述×两侧，共372次条件评分；每次评分包含一个前缀前向及两个固定候选分支前向。这里只是2个探索案例，不作总体p值、AUROC或自我锁定已证实声明。

## 4. 修正候选与数值问题

之前两种候选长度不同，分别前向取得首词logp，二者相减最多与同次margin差约0.2493。本轮将两个首词概率都取自**同一次prefix-only前向**的同一个归一化常数。

读取实际终层norm输出，以FP32分块乘lm_head，避免已经bf16舍入过的词表logits再转float32。Transformer内部仍使用指定dtype；不能宣称整网FP32或消除全部数值误差。记录torch/transformers版本与输出重投影误差。

对数归一化使用float64。next_margin恒等于correct_first_logp-wrong_first_logp。完整候选续写也复用这两个首词概率，其余词独立teacher-forcing；分支在首位置的偏差另存，不混入首词指标。

同时保存 `tail_margin = sequence_margin - next_margin`。最后一层只改q而后续词未受影响时，该tail变化应接近零，不能再把首词改善当成整段纠错。序列长短不同，平均logp仍不是真实的序列概率。

服饰额外增加parallel_singular表述：

```
 a cap with a folded piece of cloth tied on top
 a headdress with a special fringe of gold and feathers
```

将共享的token前缀a先teacher-force，再在第一个不同token前干预。这减少原caps与a的词性/限定词差异，但属于**人为控制的候选分叉**，不冒充新的自然采样或完全消除词面混杂。两侧原历史仍不同。烹饪题不编造一个不存在的正确时长，保留原对照作为局限明确的控制。

## 5. 运行与读取

```bash
python -u -m experiments.path_conflict.main --study focused
# 只按保存结果重新汇总
python -u -m experiments.path_conflict.main --study focused --stage report
```

默认新目录 `outputs/same_question_path_conflict_focused_v2`，不覆盖原实验。每种条件完成保存小NPZ，同配置可续跑。原全组扫描仍通过 `--study coarse` 使用，重算输出到coarse_v2；旧目录的 `--stage report --output <旧目录>` 仍读取旧记录，不重写原NPZ。

| 输出 | 核验问题 |
|---|---|
| effects.csv | 最终支持、两候选绝对概率、词表KL、首词与后续词效应 |
| joint_and_restoration.csv | 联合剩余项、恢复相对原cut的变化 |
| source_tokens.csv.gz | scope/候选描述/self/历史到底包含哪些token |
| head_source_writes.csv.gz | 每组每头的原始质量、V/写入范数和局部支持 |
| routes/*.npz、source_routes.html | 是否始终读同一地址，还是跟随最新词的self/近历史 |
| trajectories.csv.gz | 下游每层attention与MLP后的候选读出 |
| runtime.json、replay.json、status.json | 运行版本、原采样回放误差、完成范围与同前向恒等式 |

自动生成 `path_conflict_review.tar.gz`。新程序不读取旧结果的浮点表当成新干预；已有自然数据只用于固定候选与头。真实GPU运行尚待用户服务器执行。

## 方法依据

- Scaling Reasoning Hop Exposes Weaknesses (arXiv:2601.21214)：区分局部写入读出与上游处理头的最终效果；这里只借鉴实验区分，不转移论文中的机制结论。
- Towards Best Practices of Activation Patching (arXiv:2309.16042)：同一评价指标、自然对照及干预方式影响解释。本轮报告绝对候选概率、同幅度随机控制与恢复局限。

前轮探得的局部模式是候选定位信号。仅当特定来源切断、候选对照、下游恢复及正确侧保持形成一致证据时，才进一步讨论证据如何失去选择优势。
