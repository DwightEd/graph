# 连续标注解释：先遮住当前观测，再干预训练权重

本轮目标不是证明预设结论，而是检验“监督检测主要在识别持续错误状态”。
保留原输入、标签、配对、模型和结果；不跑原LLM。新增计算写入独立输出目录。

## 分开三条因果链

1. **分数后处理**：只看过去分数，能否替代当前分数？含当前词的EWMA不是纯过去。
2. **训练权重**：长错误span有更多BCE项，是否把模型推向容易的续错？改变权重、从头重训。
3. **输入已含历史**：当前x本身由上下文模型生成，可能编码错误历史。前两项不能消除它。

pointwise模型为z_t=f(x_t)。固定参数及逐样本计算时，共同置换(x,y)不改变总BCE和梯度。
独立打乱y会破坏类别对应，不是只改变连续性。不能用这两个实验声称去掉了原LLM历史。
监督模型的“主要有效来自连续性”必须限定具体模型、具体人群、具体操作；不报告AUROC因果百分比。

## 已保存分数：纯CPU、无重新训练

每条回答独立计算，所有排序比较使用共同覆盖，首token无历史记缺测。

| 名称 | 当前词信息 | 操作 |
|---|---|---|
| current | 有 | 原分数p_t，完全不改 |
| previous | 无 | p_(t-1) |
| past_mean | 无 | 之前最多10词均值 |
| past_ewma | 无 | s_(t-1)，s_t=.5*p_t+.5*s_(t-1) |
| causal_ewma | 有 | s_t，复现原node_only平滑公式 |
| current_increment | 有 | p_t-past_mean_t；不能解释为独立真值信息 |
| prefix_mean | 无 | 全部过去分数的均值，控制回答级共同风险 |
| sampled_past | 无 | 从全部过去随机取同样数量的词，控制近邻顺序的额外价值 |

sampled_past不跨回答、不读未来、不要标签，固定seed；这仍是观测对照，不是原模型因果干预。
原输入是概率分数，不先变logit再平均；current_increment单位为概率差。

输出：共同覆盖全流/首错/续错/给定上一标签的排名，同回答且给定上一标签的排名；
固定原配对前后半分数差、配对AUROC、后半减前半、相对current的source重采样差；
四种标签转移的均分；错误游程结束后正常词的误报/撤警，不按金标边界重置分数。
所有标签只用于分组/评价，不能进入分数公式。

### 阈值的可比性

已保存原阈值只用于current。新的变换若没有独立calibration分数，只报告排名和均分，
不在test上找阈值，也不把原阈值生搬给新评分后声称相同FPR比较。
若已有 `<model>/calibration/tokens.csv`，自动读取；必须对应training.json中原calibration IDs，
且与test来源不重叠。新阈值只由该集合正常文本token的95%分位数给出。
没有calibration表时明确记录；训练阶段会同时保存它，后续重算不用再运行小模型。
观察raw分数高也不等于概率已校准。

## 训练目标：复用已有四种权重

token：原逐token权重。span_equal：FIT每个错误span总权重相同且总正权重不变。
onset_half：每span一半正权重给起点，其余分给续错。
random_onset_half：同一权重多重集在span内置换，隔离“特别强调起点”与“强调任一词”。
正常词不改标签/权重，原分区及输入不变。每seed的初始化、数据顺序、步数一致。
主checkpoint统一固定最终epoch，selection AP仅记录；不能拿旧早停模型充当训练干预的基线。

除了各方案减token，也必须报告onset_half减random_onset_half，包含首错、续错、退出正常词、
固定配对前后半的变化。训练每个完整epoch保存模型/optimizer/scheduler/RNG，支持中断恢复。
这是损失干预而不是消除全部历史信号。若重加权仍有较高排序，不能宣称已排除全部标注偏差。

## 解释判据

- 过去分数接近原模型，但退出误报高：部分成绩来自状态延续，不能称为当前支持性判断。
- 同样过去状态下，current或increment仍能区分：排除那个指定历史代理的完全解释，不排除更复杂历史。
- span_equal显著减弱后段优势：支持重复计权影响；若起点强调不优于随机强调，不能归因起点身份。
- raw current本来就后段高：不是额外EWMA造成。持续弱信号可使EWMA逐渐升高；单次高分后全低只会衰减。
- 一个个案的连续平滑成功，不证明模型从训练中学到了span边界。

最终需要独立操纵H=历史是否错、C=当前陈述是否被材料支持，保持候选词和位置可比，
以同一观察模型重新提取输入，才能区分“读取当前事实”与“读取历史状态”。这项原LLM实验本轮未做。

## 运行

现有入口仍有效：

```bash
python -u -m experiments.charm_structure_audit.main --mode continuity \
  --continuity-stage observe --models node_only charm_in \
  --output outputs/charm_structure_audit_qa/QA/seed_0/audit_continuity_v2 --bootstrap 2000
```

每模型新增 `history_controls/`，无checkpoint、原图或原LLM前向。
也可单独执行新模块，显式控制窗口：

```bash
python -m experiments.charm_structure_audit.continuity_history \
  --root outputs/charm_structure_audit_qa/QA/seed_0 --models node_only charm_in \
  --output outputs/charm_structure_audit_qa/QA/seed_0/audit_history_v2 --window 10 --bootstrap 2000
```

训练使用原命令、新目录；先node_only，完整CHARM需另外明确指定：

```bash
python -u -m experiments.charm_structure_audit.main --mode continuity \
  --continuity-stage train --models node_only --continuity-seeds 0 1 2 \
  --output outputs/charm_structure_audit_qa/QA/seed_0/audit_continuity_train_v2 --bootstrap 2000
```

不删除旧结果；原50轮是完整实验，小轮数只检查接口。参数变化使用新output，不调整test阈值。

## 本轮已实际复算：已有CHARM QA分数（2026-09-18）

输入为用户上传的 `charm_audit_review_20260917_182703(1).zip` 中原charm_in/test/tokens.csv、
spans.csv、原阈值与固定50对cluster配对。只变换已保存分数，不重跑CHARM或原LLM。
没有取得原node_only全流CSV或自然训练checkpoint，不能把这次结果冒称node_only或多seed重训。

所有全流对照都只去掉每答第一个没有过去分数的词：30470词、2306错、149来源。
原全流30619词的AUROC为0.884368，共同覆盖后的current为0.884152；不是原模型被改坏。

| 分数 | 共同覆盖全流AUROC | 同答且固定上一词标签的条件AUROC |
|---|---:|---:|
| current | 0.884152 | 0.799128 |
| previous | 0.879069 | 0.646655 |
| past_mean (10) | 0.870446 | 0.555264 |
| past_ewma (beta=.5) | 0.882432 | 0.610899 |
| causal_ewma (含当前) | 0.889116 | 0.754771 |
| current_increment | 0.618871 | 0.798404 |
| prefix_mean | 0.796949 | 0.483329 |
| sampled_past | 0.771648 | 0.486854 |

条件统计：96个有正负两类的同回答/同上一标签格，11362词、52来源；格内按正负配对数
聚合至回答，再来源等权。其余单类格不提供AUROC，不能把11362词称为全部测试。
纯过去EWMA减current为-0.188229，2000次配对source区间[-0.252998,-0.125366]。
这是对已知标签的观察性分层，不是部署时读取历史标签，更不排除所有复杂历史表征。

相同50对、40来源、每侧904词的后半段配对来源均值AUROC：current=0.813371，
past_mean=0.843513，past_ewma=0.845138，causal_ewma=0.844254。
past_ewma减current为+0.031766，探索性配对source区间[0.004507,0.063817]。
错误侧后半减前半的平均分增长：current=+0.057885，past_ewma=+0.134077。
原raw分数本来就后段高，平滑的滞后又放大了这种现象；不能由增长推出网络学习了边界。

同一批69个从错误游程退出的正常词，current均分0.401136，past_mean为0.607399，
past_ewma为0.588697。纯历史分数会保留旧风险。此包没有独立calibration分数，
所以新增变换只报均分/排序，不冒称已经比较了相同校准FPR下的误报率。

结论：过去分数足以复现大量全流及后段排名优势；但不能解释当前正常退出/继续错误的全部区别。
“后段高分证明主要从连续标签排列学习而非内部表征”未获证明：历史分数也来自监督内部表征。
已有四损失重训可检验重复计权，本次只完成代码与小型真实训练路径测试，尚无自然多seed结果。

软件验证：`python -m pytest tests/test_charm_history_controls.py -q`，22项CPU测试通过。
包括时序无当前/未来泄漏、单峰衰减/持续输入增长、标签不进入分数、同覆盖、独立校准、
原node-only代数、共同置换的BCE/梯度不变、四种权重守恒、原格式训练/评价、断点恢复精确一致。
GPU和全部自然训练未执行。统计区间为探索性，未作多重比较校正。
