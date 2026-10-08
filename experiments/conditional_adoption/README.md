# 原生候选响应与来源中继：研究验证入口

本模块保存真实模型测量、冻结的无标签风险假设和独立评价。当前不是已经有效的完整幻觉检测器。熵仅供事件分析，不进入新风险；不按标点切断状态，不给语义span平均广播标签。

2026-10-08实际八答/1139词/77错误：原生导数AUROC 0.476398，损失弹性0.480996，真实过去/当前联合中继0.442830，均未通过有效性验证。损失弹性在正常词oracle95分位诊断下检出6/77、五个正常回答全报警；该阈值不是独立校准。完整结果及范围见 [实验报告](../../docs/CONDITIONAL_ADOPTION_20261008.md)。这是保留失败证据的研究入口，不替换原检测默认。

每个实际生成token在P+t−1预测位置采集全部32层×32头的五组消息AV与输出作用：source、local history、remote history、self、other prompt。目标包括实际词logp和实际词相对原生最高非实际词的margin。模型参数冻结，当前query反传经过真实后继网络；过去KV固定，当前算子不包含过去读取的中介作用，也不包含该层AV之前的QK寻址导数。

`source_candidate_regret` 是来源消息同时小幅增强时实际词logp的一阶降低。跨层相加表示联合冻结方向干预，不是守恒归因。`source_margin_rejection` 检验softmax饱和；`source_local_countervote` 保留同一物理头上的方向冲突。正常语法、正确改写、内部自洽错绑也可能产生相同信号，因此必须进行同token基线与误报审计。

单卡完整复现（需要项目中已有source-first缓存和本地模型，输出路径必须新建）：

```bash
PYTHONPATH=.:teaching/state_audit/src \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
-m experiments.conditional_adoption.run \
--output outputs/conditional_adoption_new_run
```

追加来源当前/过去/联合有限干预：在同一命令后增加 `--relay`。此阶段固定L15、幅度+.25，真实重算改变后的后继KV；过去来源的作用与当前读取分开。实际prefix文字固定，不冒称已经干预离散生成过程或证明来源相容。

本轮已有完整缓存后只做CPU读出/评价：

```bash
PYTHONPATH=.:teaching/state_audit/src \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
-m experiments.conditional_adoption.readout \
--capture outputs/conditional_adoption_20261008/complete_capture \
--output outputs/conditional_adoption_readout_review

PYTHONPATH=.:teaching/state_audit/src \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
-m experiments.conditional_adoption.evaluate \
--scores outputs/conditional_adoption_readout_review \
--source-root outputs/native_support_ragtruth_all/source_first_v1
```

上面的读出目录必须不存在；已有评价不能覆盖，需用新的 `--report-name` 和 `--token-report-name`。本轮冻结结果可直接读取 `outputs/conditional_adoption_20261008/*scores/evaluation_v2.json`，无需重新运行模型。

复现失败后的开发迭代（只CPU计算，明确为post-pilot探索，输出目录也必须新建）：

```bash
PYTHONPATH=.:teaching/state_audit/src \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
-m experiments.conditional_adoption.elasticity \
--scores outputs/conditional_adoption_readout_review \
--output outputs/conditional_adoption_elasticity_review

PYTHONPATH=.:teaching/state_audit/src \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
-m experiments.conditional_adoption.evaluate \
--scores outputs/conditional_adoption_elasticity_review \
--source-root outputs/native_support_ragtruth_all/source_first_v1
```

评分先保存数组及hash，评价再读取官方标注。报告pooled与错误回答内部AUROC/AP、native-winner分组和逐token错误表。评价集正常词95分位数只作明确标出的oracle阈值诊断，不是可部署校准。

冻结八答：12219/12216、12297/12294、15604/15600、11907/11904，共1139原token、四来源。官方3错误回答和5正常回答，11907按官方正常；对照按既有正常回答ID升序选定。正常对照皆GPT-4，错误来自Llama-2，这使pooled指标存在生成器/风格混杂，必须看回答内部区分与正常词误报。历史样本已暴露，不称盲测，不称三个任务全量结果。观察器Llama-3.1-8B不是这些回答的原生成器。

当前自然关系伪目标资格失败，未训练q/责任模型，亦无自然标签拟合。原始实验在`outputs/conditional_adoption_20261008`；研究计划与审查在共享`codex/research/refine-logs/conditional_adoption_20261008`。
