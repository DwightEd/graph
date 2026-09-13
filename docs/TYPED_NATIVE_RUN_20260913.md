# Typed native：真实决策的证据路径与query/layer定位

该批直接测冻结自然回答的完整B/A决策，不再测“关系会影响决策”。来源provider只覆盖Data2txt明确七天营业时间。它没有闭合通用QA/Summary归属，不代表新检测器有效。

## 冻结协议

父输入 `outputs/typed_hours_population_prepare_20260913` 包含全部17790回答及所有拒绝状态。只从其official train原生合格对照，按seed20260913+bridge SHA排序，每来源第一条、最多12来源；不按observer偏好/效果或RAGTruth标签替换。清单和分母冻结于settings。

完整事件F=logP(原B)-logP(支持A)，保留前文、精确token前缀和所有非目标字符；F≥0.5 / ≤-0.5 /中间分别报告B-preferred/A-preferred/near-tied。A-preferred不叫修错。当前observer为Llama3.1，不是原6种生成器的因果解释。

每对照最多128次实际forward，包含两个baseline、≤64的root/layer/query/非邻接union搜索、position和输入→同A-V地址的origin full/half/sham/repeat，以及可用的固定controls和补集。A固定为全部7天目标端点，不能按效果挑某天。全部原prefix可见query和全部层进入root；保存父组、测过单层/联合层与pending frontier，不能声称100%或唯一回看点。

control census在模型加载前枚举全部来源leaf及两个完整、不重叠且schema无关leaf的全部组合。组合仅用于控制，一次联合干预，不当作事实或owner，也不把成员效果相加。这样可以匹配跨七天A自身约0.5的输入因果可达性；阈值和语义排除不改变。

controls在baseline之后、任何search干预之前，根据来源schema、长度/质量/embedding norm/因果可达性冻结实际ID。selected scope只复核这些ID，缺失不替补。无两个合格controls仍执行raw测量，强证书关闭。没有独立B来源时不造错owner，也不从MLP干预默认推出聚合失败。

高维节点取自已计数的原始B baseline：各层decoder输出在全部共享query和A来源token上的原生bf16值。npz保存原始字节及shape/dtype；来源字段、query坐标、测量的层/query超边给出拓扑。组效应不拆成虚构的每条边效应。

## 运行前状态

`typed_hours` core/prepare与native measure/runner独立工程复核C0/R0。相关CPU检查包含非首句完整事件边界、禁止postprompt A、联合根保留、baseline A仍测量、固定control IDs不替补。CPU检查不是机制结果。

环境research@03909e02，既有Python/torch/transformers/模型，复用不重建。全量CPU父编译完成后，由root执行下列prepare，冻结模型/代码/语义链；此步不加载GPU模型：

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.typed_native_runner --preflight outputs/typed_hours_population_prepare_20260913 --output outputs/typed_native_v2_20260913 --max-sources 12 --stage prepare
```

## 独立文档执行命令

仅在父编译/上面的prepare完成后，fresh witness执行一次以下命令。实际执行之前root将核对PID并记录冻结settings digest；以下PID为当前原population进程，wrapper会严格核对cmdline/cwd/progress，任何不符应停止而非猜PID。

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m experiments.interleave_typed_native --population-pid 162289 --output outputs/typed_native_v2_20260913 --journal ../reanchor/runs/typed_native_v2_20260913.json
```

复用冻结 `interleave_soft_graph.execute_and_restore` 独占事务：preflight先于暂停、pidfd中断正确population、保存全部已发布manifest和冻结文件、持原population输出锁执行native、finally核验并恢复、见到真实增长才报告恢复。不能同时运行另一个GPU模型；不能kill wrapper。保留失败与已有输出，不自动重跑或改变参数。若population已自然完成，本文这一固定命令应报身份/状态不符；由root安排适用于完成状态的单独明确调用，不由见证自行改命令。

见证记录session、wrapper/child实际PID、逐对照进度/forward、实际退出码、完整raw/node/donor哈希与调用总数、原population恢复真实增长、所有异常。报告 `refine-logs/typed_native_doc_witness_20260913.md`。一次执行结束以前保留session，每次poll≤60s，及时向root报告。

当前文件是方法与运行说明，不是已执行成功的证明。没有RAGTruth标签读取/评价，没有wrong-source路由或聚合证书。未知和未覆盖项保留到最终完整方法评价。

## CPU预检修订记录

旧v1 prepare实际exit1，未加载GPU、未生成settings/snapshot：live contrast tuple与JSON list直接比较触发拒绝。真实9273重算核验canonical digest与sealedSHA均相同，仅prefix/continuations的tuple/list表示不同。v2改为完整canonical digest比较，继续逐项验证全部7天A/源mask/原token。这是存储类型修复，不改变任何语义判断或阈值。v1目录保留。另修复非连续control pair的原始span重叠排除及成员ID/坐标一致排序；v2随后冻结。

## v2已冻结CPU准备状态

prepare实际exit0，selected1（全部train合格对照只有1条：9273/source14637，不是从多条native结果中挑出）。settings_object_digest `36d838b6fe5b52469aaff687f1492a280bab51526069a28b32f4fb6db95acb8a`；settings_file_sha256 `895966d2f2ded77df3efad7b6508d3afefa662768e3ef96356c9850fd895dde1`。54份代码冻结。最新独立复核C0/R0，目标CPU检查7passed。原population读取17133/17790、failed0、PID162289、running。GPU尚未执行，此后不得改冻结代码。
