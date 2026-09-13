# Reader CPU document-execution witness — 2026-09-13

Fresh agent 执行文档命令一次，session **29651**，审计 PID **163204**，退出码 **0**。开始于 2026-09-13 02:49:08 UTC，完成于 02:57:42 UTC，命令墙钟约 **514 秒**；逐请求计时合计 **480.712 秒**。最终 **10/10 请求、5 类各 2 条、30 次真实 model forward**，每条均为 1 次全词表评分 forward 加 2 次自然生成 forward。设置的自然续写上限仍为 48 token；实际均生成指定字母与 EOS 后停止。

这是冻结自然请求的有限读出故障诊断。没有读取真假 GT，没有真假准确率、幻觉检测效果或 native 干预结果；`labels_read=false`、`native_interventions=0`。所有字母均为协议输出符号，不是数据集真值标注。

已使用 run-experiment skill 的 agent-follows-doc 流程，先读 skill、共享 compute-env-contract、项目 local ledger、env-spec、当前运行文档与脚本。已有环境 spec hash 为 **03909e02**，本次无安装、无重建。检查发现源 C 目录为 36 条，五类候选数为 target_original 42，其余四类各 145；独立按每类 request digest 字典序取前 2 条，恰好得到 10 条。输出目录原先不存在。AST、generation 参数、CPU 设备路径、单次写入方式和 10 份 receipt 哈希预检无阻塞问题；未修改运行协议或源码。`rg` 在宿主机不可用，文件检查使用 Python 标准库完成，不影响文档命令执行。

实际执行的文档命令原文：

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m next_iteration.reader_output_audit --run outputs/surface_owner_v1_20260913 --output outputs/reader_output_cpu_audit_20260913 --per-kind 2 --max-new-tokens 48
```

运行结果如下；续写列省略相同的 `<|im_end|>`，每条完整原始输出仍保存在对应 JSON。条件概率差是 CPU 与旧 GPU 缓存在同一组枚举字母上的最大绝对差，没有再归一化全词表标签质量。

| 文件 | check | 输入 token | 自然字母 | 全词表标签总质量 | CPU/GPU 条件概率最大绝对差 | forward | 秒 |
|---|---|---:|:---:|---:|---:|---:|---:|
| 00.json | edit_kind | 949 | K | 0.999997139 | 0.000000238 | 3 | 24.04 |
| 01.json | edit_kind | 866 | K | 0.999992073 | 0.000700059 | 3 | 23.85 |
| 02.json | edited_base | 1850 | S | 0.999998271 | 0.000262141 | 3 | 74.07 |
| 03.json | edited_base | 509 | C | 0.998963296 | 0.000178277 | 3 | 9.02 |
| 04.json | preservation | 2312 | F | 0.999999046 | 0.000016570 | 3 | 96.47 |
| 05.json | preservation | 2216 | F | 0.999999821 | 0.000035042 | 3 | 92.94 |
| 06.json | selected_occurrence | 930 | I | 0.999998689 | 0.000000004 | 3 | 17.41 |
| 07.json | selected_occurrence | 2243 | S | 0.999886513 | 0.045864105 | 3 | 95.95 |
| 08.json | target_original | 602 | N | 0.999707341 | 0.029908895 | 3 | 12.35 |
| 09.json | target_original | 1210 | N | 0.999718547 | 0.026919533 | 3 | 34.63 |

10/10 自然输出都严格为该请求允许的单个字母加结束标记；标签总质量范围为 **0.998963296–0.999999821**。10/10 CPU 与旧 GPU 条件概率的 argmax 一致。最大条件概率差为 **0.045864105**（第 07 条 selected_occurrence 的 S：GPU 0.919031322，CPU 0.873167217），最大选定字母原始 logit 差为 **1.0**。因此保留 CPU bf16 SDPA 与原 GPU 数值实现的差异，不能把这些 CPU 分数视作原 CUDA 分数的复现。

在这 10 个固定请求上，观察不支持“自然输出格式跑偏、枚举字母总质量很低而被条件归一化掩盖”作为解释。该范围内的格式遵从不建立语义正确性，也不能外推全部请求或模型效果；本次没有改提示词、阈值、真假标注或原缓存。

完整性复核：新输出目录恰好包含 settings、00–09 和 summary 共 **12 个完整 JSON 文件**；summary 为 complete，逐条计数和总计一致，top12、概率与 logit 均有限。独立复算新 settings digest，逐条与 summary 全部吻合；10 个 request digest 与 receipt 文件 hash 都正确，所选列表与预检选择完全一致。runner 在每条执行前还核对了原 rendered prompt hash。新结果文件的 SHA-256 清单已记录在 verification JSON。

独立复核 **16 个模型文件**的实际内容 SHA-256 全部匹配原冻结 reader_files，总计 **16,397,461,896 字节**。预检记录的 **52 个原 settings、C 记录、所选 reader receipt、相关代码、当前文档与 env-spec 文件**在结束后全部 hash 不变。主审计脚本 SHA-256 为 `4f620a24e98ba590161c53a50cd71e541ee1b801dd365391dda3c377be2e33c4`；新 settings 文件 SHA-256 为 `d77082bedc06df28c1651b758194cf8f6242573b4bf3d355edfb0b9c756385b8`；新 settings canonical digest 为 `282e636e6459577b20ba3b8fb2d2ff74e92223ccd204a49caac1c166cf220ff8`。

设备与现有任务：runner 明确在 CPU 上加载 bf16 模型并断言参数在 CPU；OMP/OPENBLAS/MKL 和 torch intra-op 为 4，torch inter-op 为 1。运行中多次 `/proc/163204/fd` 检查均无 nvidia/dri 设备文件；审计期间 `nvidia-smi` 只显示原 PID 3598631 占用 20002 MiB，前后相同。操作仅为只读查询，没有启动 CUDA 模型或 forward、没有暂停或发信号给 population；审计 PID 结束后不存在。观测到的 CPU VmHWM 为 18,012,736 kB（约 17.18 GiB），这是采样值而非持续监控峰值保证。进程总 OS 线程为 14，包含库辅助线程；计算线程设置按文档为 4。

Transformers 日志有“generation defaults 修改 do_sample=True”的提示。独立核对所装 4.57.1 的 `generation/utils.py`，该提示后在第 1801–1802 行最终应用显式 kwargs；本脚本的 `do_sample=False` 因此覆盖默认值，仍为贪心。温度/top-p/top-k 在贪心模式被忽略的提示符合这一设置，无需修复或重试。

本 agent 没有变更任何 repo 分支，graph 始终为 `agent/graph-structure-audit`；没有清理任何既有 dirty/untracked 项。仅新增本次审计输出及本见证证据文件，未进行第二次模型回放。

证据：

- [原运行文档](../docs/READER_OUTPUT_AUDIT_20260913.md)
- [完整结果目录](../outputs/reader_output_cpu_audit_20260913/)
- [预检与原文件哈希](reader_cpu_doc_witness_20260913_preflight.json)
- [捕获的原命令 stdout/stderr 与退出码](reader_cpu_doc_witness_20260913.log)
- [会话时间与运行设备采样](reader_cpu_doc_witness_20260913_runtime.json)
- [独立完整性、模型哈希与逐条数值复核](reader_cpu_doc_witness_20260913_verification.json)
