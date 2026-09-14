# B8：原文关系与中间草稿关系的正交控制（准备阶段）

目的：检验当前外部核验器是否依赖“词在场”，还是实际受对象/阶段/条件—值归属约束，
以及错误中间草稿能否在完整原文仍可见时改变同一目标词判别。
这是输入级因果控制；不等同原Llama2内部回看节点干预，不证明普遍幻觉机制。
不改正在执行的P7，不读取P7确认标签、分数或生成草稿。合成真值只由模板构造，不能替代自然AUROC。

## 固定范围

8个预写模板：阶段、对象、实体、极性、时间、量词、适用条件、税率归属。
每个模板2种原文关系×2种原文顺序×(无草稿+2种草稿关系×2种草稿顺序)=20条，总160。
同一模板回答/目标词始终固定；原文和草稿关系独立交叉，不能只给正确草稿。
交换前后所有词及标点的多重集完全相同。不改变原文事实数量，不加入模型生成“真值”。
真实Qwen tokenizer对每个模板的16条有草稿查询须长度完全相同，4条无草稿查询内部也须相同。
若不满足，保留完整失败，不删模板或填充到刚好出结果；先报告设计问题，不宣称关系因素已隔离。
无草稿与有草稿长度不同，仅作上下文基线，不作为纯语义因果对照。

## 模型接口

仅使用已核验的P7最终逐词判别接口（非thinking、原始zB−zA）；输入为构造SOURCE/ANSWER及可错draft。
没有审计采样或best-of-K，没有训练、分数拼接或标签拟合。先CPU生成完整输入和tokenizer审计。
未来GPU评分必须等P7确认实际完成且空闲，冻结160条输入/提示/代码后单独一次运行；当前尚未执行。
数值精度沿用不变BF16存储权重+逐层FP32计算+math SDPA/TF32off；保留raw A/B与完整输入token。
GPU执行器现已实现但尚未执行；此文不是启动授权/已运行证据。
逐条单样本完整prefill、不缓存、不采样；每模板第一条重复一次检查数值差≤0.005，
160条评分+8次重复，共168模型前向。全部保存原始A/B，不用构造标签选取分数或调参。
执行器在P7全128实际exit0前拒绝GPU启动；仍须独立静态复核和主审明确启动决定。
产物绑定完整固定160输入、真实tokenizer预检、完整输入token流、原P7提示代码SHA。

未来一次性命令（仅在P7完成、GPU空闲、独立检查与主审启动决定之后）：

```bash
bash /share/home/tm902089733300000/a903202310/lys/research/graph/scripts/run_binding_factorial.sh
```

输出graph/outputs/b8_binding_factorial_scores_20260914_v1；GPU raw输出manifest.json。
上述sh只运行GPU评分，不自动生成对照。独立执行见证必须在实际子进程exit0后记录
graph/outputs/B8_BINDING_LAUNCH_20260914.json，字段要求：
status=completed、attempt=1、actual_subprocess_returncode=0、output绝对路径、
prediction_manifest_sha256、scorer_sha256、contrast_code_sha256、
model_shard_sha256（全部5片）、p7_completion（launch_sha256/manifest_sha256/freeze_sha256）、
log绝对路径/log_sha256；另保留实际UTC起止、PIDs、原始命令及启动前全部文件哈希。
该独立记录不是由对照程序自行生成的“完成”证明。
完整160及实际exit0记录齐备后，main另行执行一次：

    cd /share/home/tm902089733300000/a903202310/lys/research/graph
    CUDA_VISIBLE_DEVICES='' /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m next_iteration.binding_factorial_contrasts --inputs outputs/b8_binding_factorial_inputs_20260914_v1/inputs.jsonl --predictions outputs/b8_binding_factorial_scores_20260914_v1 --launch-record outputs/B8_BINDING_LAUNCH_20260914.json --output outputs/b8_binding_factorial_scores_20260914_v1/contrasts.json
失败保留，不删难模板、不改符号、不重复到同目录。模型仍为research@03909e02中同一Qwen3-8B；
不安装环境、不改变权重。若P7失败退出，先处理P7故障；本脚本不得绕过门控抢占GPU。

## 预先规定的对照（不选最佳模板）

风险r=zB−zA，构造source_assignment=0支持固定回答，=1不支持。

1. 原文关系效应：固定草稿关系和两种顺序，r(source=1)−r(source=0)。
   若正，模型会响应原文关系；若仅个别词模板成立，不泛化。
2. 草稿关系效应：固定原文和两种顺序，r(draft=1)−r(draft=0)。
   草稿不是证据；这一项刻画错误中间表述如何影响决策，不能自动称为有益读取。
3. 关系交互：上述原文效应在两种草稿关系下的差。报告全部8模板和所有顺序，不用平均掩盖符号反转。
4. 固定0阈值的构造正确率、被错误草稿诱导的翻转、正确草稿纠正的翻转；不调阈值。
5. 原文/草稿顺序敏感性，以及无草稿参照。完整160输出逐项保留；不报告合成AUROC为自然检测成绩。

可以支持的结论仅限：这些构造输入上的关系/顺序/草稿改变对外部核验器输出有何功能影响。
若要说“内部读到了但绑定错误”，还需原模型表征读出、内容保真的干预与无关路径对照，当前不声称完成。

本阶段代码：binding_factorial_fixture.py、binding_factorial_preflight.py、binding_factorial_score.py、
binding_factorial_contrasts.py。初版20项CPU测试及补充6项入口测试实际通过；独立审查后另增防伪/派生数值测试，
最终实际测试数量以修复检查记录为准，不将单元测试人工分数计作模型输出。
测试里的人工分数只验证数学，不是模型实验；不会写入正式结果目录。
真实远端tokenizer预检actualexit0/PASS，160条全部保留：8组有草稿长度321–370，
对应无草稿273–308，各组内部所有关系/顺序组合长度完全相同。模型前向0、自然标注访问0。
产物graph/outputs/b8_binding_factorial_inputs_20260914_v1，preflight.json；
tokenized_queries.json SHA0020027a473160ff573a147dcf8923ddb5d1c06acb31f7c2697db4fd4a613ff0。
fixture manifest SHA1f3a1c765f43b09f9686e1cb24801a598189aa98cbec19e9ee55f5e5b10c6331。

## 启动前独立审查与修复（没有B8效果结果）

初版静态审查WARN：构造/长度/320个对照数学通过，但伪造manifest可被消费、P7完成门不严、
权重与import依赖未全绑定、空闲快照有并发窗口、派生inf/sidecar覆盖检查不足。
旧版与原报告保留，不将WARN改写成初始PASS。
修复后消费者要求精确ordered/unique160、当前审查的scorer及执行快照hash、
全部输入/模型/依赖身份、168前向、固定8重复检查、逐条输入token流hash及独立真实exit0记录。
P7门绑定预定128 IDs、freeze、方法、无error、实际exit0日志及完成manifest；
只读取这些完成元数据和无标签名单，不读取P7预测分数或标注。
评分前重新核实既有5片模型权重的已批准SHA，父提示文件和grounding_contrast依赖均绑定。
同一B8任务使用Linux flock协作锁，模型释放后解锁，加载前再次检查GPU；
这是本任务合作进程的互斥，不是平台排他预留，无法禁止不遵守锁的外来任务。
风险/派生差分均要求finite，JSON禁止NaN/Inf，对照及代码sidecar都独占创建。
上述修复不改变任何输入、提示、阈值或数学对照；尚需独立复审和明确主审决定才能启动。
