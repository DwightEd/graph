# Surface owner v1 独立文档执行见证（2026-09-13）

独立见证已完成：文档命令执行一次，prepare、完整 features→finite child 和 wrapper 均 exit0；B36/C36。原 population 从14883恢复至14892→14899→14961，同一恢复PID162289、failed0。此报告验证执行与恢复，不证明主线效果、所有权机制或实验方法有效。

## 范围与精确执行

见证者为未参与本实验实现的 fresh subagent `/root/surface_owner_doc_witness`。先阅读 `/root/.agents/skills/run-experiment/SKILL.md`、其 compute-env-contract、`docs/SURFACE_OWNER_V1_RUN_20260913.md` 及 `.aris/compute/local.md`，按独立 agent-follows-doc 要求执行。没有改实验代码、替换参数、安装包、下载模型、重试或覆盖旧输出。

工作目录：`/share/home/tm902089733300000/a903202310/lys/research/graph`。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -u -m experiments.interleave_surface_owner --population-pid 159091 --output outputs/surface_owner_v1_20260913 --journal ../reanchor/runs/surface_owner_v1_20260913.json
```

- exec session：`82104`，持续持有至工具返回 exit0；各次poll不超过30秒。
- wrapper PID：161808；prepare child PID：161814；正式 child PID：161896。
- 原 population PID：159091；暂停前实际14883/17790、failed0。文档14750为准备时读数，运行前已增长133。
- 恢复 population PID：162289，命令为 `python -u -m decoding.ragtruth_population --output outputs/ragtruth_population_20260912 --resume`，cwd为原 reanchor。
- 环境spec canonical digest仍为 `03909e02`；复用已有 research环境，没有环境重建。启动前GPU0为21896/24564 MiB，由原population使用；模型经wrapper暂停后顺序运行。
- settings文件SHA256：`3b268fa460dad8540d3ba7998956e92a0adf53b4e9947870ffe8faf8d6492e9d`。
- settings parsed digest：`51e2ff407e2b34dfe880820ec040554791864f1fe2808279e4b183502ce11973`。

## 实际时间与完整产物

所有时间为UTC。journal开始于2026-09-13T02:23:01+00:00；features loading_model为2026-09-13T02:24:35+00:00，features phase_complete为2026-09-13T02:25:35+00:00；finite loading_model为2026-09-13T02:25:38+00:00，finite phase_complete为2026-09-13T02:33:09+00:00；child complete为2026-09-13T02:33:09+00:00。这些是落盘阶段时间，不是独立GPU计时基准。

| 项目 | 独立核对结果 |
|---|---:|
| 冻结自然回答、B、C | 36、36、36 |
| feature chunks | 75个JSON + 75个数组 |
| feature documents / capture records | 2377 / 2377 |
| feature实际forward | 595 |
| target检查 | 1619 |
| target C/N gate通过 | 44 |
| 候选对有限检查 | 145 |
| reader实际forward | 2199 |
| reader返回 / cache hit | 2344 / 145 |
| reader唯一receipt | 2199 |
| reader_error | 0 |
| native forward | 0 |
| labels_used / labels_evaluated | false / false |

独立遍历全部features/B/C：36个B/C文件ID分别与冻结roster完全一致，147份JSON的settings/data digest全部通过，75个数组文件SHA256全部通过。feature_summary与分块记录计数一致。2199个reader cache文件名与完整request digest一致，全部有限请求max_new_tokens=1；2344个C内reader引用对应2199个唯一receipt，其文件SHA256全部通过。reader的有限输出标签集合计数为SCNU1764、VKU145、PFU145、SCIU145；这些是reader枚举输出符号，不是真值标签。

所有36份C均声明native_forward_calls=0、labels_used=false、not_ground_truth=true；最终progress为complete/all、native_forward_calls=0、labels_evaluated=false。见证过程中没有读取真值标签或运行部分/完整标签评价。

145个真正有限检查的候选对终态分别为：edited_full_base_not_supported_or_multislot_error124、non_target_meaning_or_grammar_not_preserved16、value_only_edit_not_validated3、selected_owner_or_constraint_not_validated2。quoted_edit_closed等closed计数是拒绝/关闭原因，不是成功候选。上述执行计数不构成主线效果或机制成功。

## 恢复与冻结完整性

wrapper journal最终status=population_resume_verified、audit_exit_code=0；exec session最终exit0。wrapper的首次恢复判据在loading_model/14883时满足，见证继续观察到真实running和完成数增长：

| 观察UTC | PID | completed | failed | status |
|---|---:|---:|---:|---|
| 2026-09-13T02:23:01+00:00（暂停前） | 159091 | 14883 | 0 | running |
| 2026-09-13T02:37:19+00:00 | 162289 | 14883 | 0 | loading_model |
| 2026-09-13T02:37:47+00:00 | 162289 | 14892 | 0 | running |
| 2026-09-13T02:38:03+00:00 | 162289 | 14899 | 0 | running |
| 2026-09-13T02:40:31+00:00 | 162289 | 14961 | 0 | running |

恢复后的PID/cmdline/cwd重新核实。population输出锁持有，scheduler锁与本批phase锁均已释放；原159091及wrapper161808、prepare161814、child161896均已退出。恢复存在约208秒preparing阶段，随后加载模型并实际推进，没有把loading_model当作最终恢复证据。

独立验证14883份暂停前manifest的SHA256和4份冻结population元数据全部保持不变；另核对8份population执行代码和2份source输入文件无差异。实际进度增长后再次核对14883份旧manifest、4份元数据及55份surface当前代码/55份snapshot，零差异。这里只声称这些明确枚举文件的hash核对，不声称重新哈希每个population manifest内部引用的全部大型数组。

## 异常、差异与证据路径

实际调用没有导入失败、OOM、快照不符、异常退出、早停或参数补救；audit log没有Warning/Error/Traceback。只读辅助检查发现系统无rg，使用标准只读替代；research树未找到CLAUDE.md/AGENTS.md，实际环境由本次精确运行文档和已有local ledger提供。这两项未改变启动命令或实验产物。

复用wrapper的中间journal字段名running_native_audit不代表执行了native；以实际命令、冻结phase_order和最终native0记录为准。

- 完整输出：`outputs/surface_owner_v1_20260913/`。
- 原始journal：`../reanchor/runs/surface_owner_v1_20260913.json`（含14883份旧manifest基准hash）。
- 原始child日志：`../reanchor/runs/surface_owner_v1_20260913.audit.log`。
- 原始恢复日志：`../reanchor/runs/surface_owner_v1_20260913.population.log`。

报告落盘UTC：2026-09-13T02:40:31+00:00。本见证未重复运行文档引用的CPU测试；那些是运行前已有记录，不冒充本见证执行。
