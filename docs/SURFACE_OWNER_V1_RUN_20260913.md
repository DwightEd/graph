# Surface owner v1：独立运行见证指令

目的：在冻结36个自然RAGTruth回答上执行SRL-free图节点/辅助高维owner匹配/有限单槽位B/A验证。不是效果评价；不读取标签，不执行native干预。设计见 `SURFACE_OWNER_METHOD_20260913.md`。

## 已准备状态

- 输出：`graph/outputs/surface_owner_v1_20260913`。
- 父预检：`graph/outputs/surface_owner_preflight_v2_20260913`，36回答/6来源/237 Base/1619 slots。
- settings parsed digest：`51e2ff407e2b34dfe880820ec040554791864f1fe2808279e4b183502ce11973`。
- settings file SHA256：`3b268fa460dad8540d3ba7998956e92a0adf53b4e9947870ffe8faf8d6492e9d`。
- 55份执行代码已复制并验证，运行时不得修改这些文件。新native迭代请用未被本次snapshot引用的新模块。
- CPU prepare exit0；2377 masked documents共108701 tokens、最长812，无截断；36个原observer行精确对齐、有限标签均单token。
- Surface graph/owner/verifier相关CPU共36 passed；模块独立复核C0/R0。Runner启动链复核已闭合C0/R0；独立CPU/mock验证summary首次写入、相同重入与变更拒绝。必须用下列module命令。chunk在`.npy`落盘后、receipt前中断时会保留不完整产物并拒绝同目录恢复，不会覆盖。

## 实际启动命令

从以下工作目录启动。必须使用 `-m experiments.interleave_surface_owner`，直接执行脚本路径不具备正确module import路径。

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m experiments.interleave_surface_owner --population-pid 159091 --output outputs/surface_owner_v1_20260913 --journal ../reanchor/runs/surface_owner_v1_20260913.json
```

PID159091在准备后读取到running、14750/17790、failed0；执行前wrapper会再次核对进度/PID/cmdline/cwd，通过pidfd只向这一原population进程发SIGINT。任何不一致应停止，不猜PID，不启动第二个worker。preflight发生在暂停之前。

wrapper复用冻结 `interleave_soft_graph.execute_and_restore`：持有scheduler和同一个phase文件描述符，等待population优雅退出并接管原输出锁，记录所有旧manifest hash，运行features→finite，finally校验原population元数据和全部旧产物后恢复。不得kill wrapper，不覆盖任何旧输出或journal。

## 应记录的运行事实

独立见证只按本文启动，不修实验代码或替换命令参数。若缺少文档/导入失败/显存不够/快照不符，应留下错误并报告；实际成功前不写“运行中”。记录exec session、wrapper与child PID、features文档/前向进度、B/C完成数、reader实际调用、所有异常、退出码及恢复PID。新snapshot相对路径可由settings读取。

若进程仍在跑，不要结束自己的执行session。每次poll仅等待≤60秒；给父任务简短更新。结束后先核对全部B/C、feature_summary、reader计数，确认native0/labels未读；恢复还需看到population实际进度增长和锁状态，而非只有loading_model。

如果父任务基于新设计证据明确要求早停，只中断经身份复核的本批child，让wrapper完成finally，保留所有部分产物并标注主动提前停止。不要做部分标签评估，不将部分批次称为完成。

见证报告写入 `graph/refine-logs/surface_owner_v1_doc_witness_20260913.md`。本文件是运行说明，不是已执行成功的证明。
