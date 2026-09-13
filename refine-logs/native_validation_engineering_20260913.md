# native validation 部署前复审

2026-09-13。范围：复审 `route_graph/causal_groups.py`、`route_graph/audit_pools.py`、`tests/test_causal_groups.py` 的前次 Required 修复，并新增审查 `route_graph/audit_certificates.py`、`route_graph/audit_interactions.py`、`route_graph/audit_phase_native.py`。只读参考 `native_audit.py`、`causal_contrast.py`、`audit_runner.py`、`audit_output.py`、`audit_protocol.py`、`FINAL_PROPOSAL.md` 和修订4执行补充。未跑 GPU，未改实现。

结论：前次四个工程 Required 大部分已经落到代码中，B 阶段 native proposal/search 可以进入小样本 smoke；D 阶段完整 validation 还不应直接跑自然批量，因为 N 分支仍会执行 paired/MLP，origin control 可跨角色匹配，MLP 交互的 sham/scope 仍不足以支撑最终机制证书。整体工程分 7.6/10，verdict：REVISE before full A-D GPU validation。

## 已修复并确认

- pairwise 可见性已替代 `min(keys)`：`audit_pools.visibility()` 现在计算 `key <= query` 的 pairwise 比例，修复了混合可见 history/source key 的 control 匹配问题。
- shared-prefix guard 已加入：`CausalOracle.measure()`、`gates()`、`mass()`、`mediated()` 都会检查 query/key 留在共同 prefix 范围内。search 内部仍只从 `contrast.shared_queries` 生成 query。
- native proposal 仍无语义候选泄漏：`proposal_phase()` 只使用冻结 B/A event、source/history role keys 和模型 native F；`freeze_pools()` 不读取 reader evidence、适用性或标签。
- donor 介导接口已收敛：`mediated()` 只 capture `group["keys"]`，recipient donor gate domain 也为 selected keys，结果附带 `donor_artifacts`。
- D 的 position certificate 已改为每个具体 group 实测 sham：`validate_fixed_group()` 先跑 `oracle.group(group, 1)`，再跑 full/half/controls/repeat。
- root heap tie-break 已改成插入顺序，避免全部 root 使用 hash 顺序。
- 协议代码已更新到 `forwards_per_claim=320`、`forwards_per_response=1280`，D 阶段用 `320 - proposal_forward_calls` 建预算。

## Required before full GPU validation

1. N 类样本仍会跑 paired R 和 MLP interaction，违反当前执行规格并浪费 N 第二模板预算。

   文件位置：`route_graph/audit_phase_native.py:167-181` 先执行 paired/MLP，`route_graph/audit_phase_native.py:186-254` 后执行 N template。

   最新要求是：N 只发第二模板也通过的 primary position/origin；paired/MLP N 保留 unresolved。当前代码对所有 question 都先调用 `paired_role()` 和 `mlp_interaction()`，即使 `effective_scores["N"] >= 0.8`。`merge_response()` 后面会忽略 missing 样本的 paired/MLP 机制，但预算已经被消耗，validation artifact 里也会出现 N 下的 paired/MLP 结果。这会导致第二 withholding template 因 `remaining < 40` 而变成 `template_budget_or_event_unresolved`，而不是按规格优先复测 N primary certificate。

   最小修复：在 `validation_phase()` 中先判 `is_missing = question["effective_scores"]["N"] >= 0.8`。若为 N，跳过 paired 和 MLP，返回 `{"status": "N_scope_unresolved", "passed": False}` 或同等明确状态，并把 40 次 N template 预算放在 paired/MLP 之前。C/N 分支的预算表要在 artifact 中显式区分。

2. `origin_controls()` 没有限制 control 与 origin 同角色。

   文件位置：`route_graph/audit_pools.py:137-168`，调用位置 `route_graph/audit_phase_native.py:119-138`。

   对 source-origin 介导，controls 目前可以来自 history raw text unit；对 history previous-claim origin，controls 也可以来自 source unit。它只匹配长度、embedding norm 和 causal reach，没有匹配 `role`。这会把 source-input→V-message 的介导特异性与 history/source 角色差异混在一起，尤其会影响 `paired_role()` 里的 `origin_scope_matched` 升级。

   最小修复：`origin_controls(origin, group, frozen, semantic, origin_role)`，source content 使用 source controls，history previous-claim 使用 history controls。若需要跨角色 origin control，应作为单独诊断输出，不能参与 `origin_mediated_effect` 的主证书。

3. N 分支 alternate template 的 primary 选择没有限定到允许升级的机制类型。

   文件位置：`route_graph/audit_phase_native.py:183-254`。

   `primary = next((p for p in positions if p.get("passed")), None)` 会选择第一个通过 position 的 finalist；如果它是 route group，N second-template 仍会重测 position，但没有 origin 证书，也不应升级为“primary position/origin”。如果它是 semantic unrelated 的 source/history 位置，也可能成为 N primary。merge 层最终可能不形成 mechanism，但 validation artifact 的 `template.passed` 语义会不清楚。

   最小修复：N primary 应只从已通过且允许 N 报告的 family 中选择，至少记录 `primary_selection_reason`；若没有 content+origin 或明确允许的 position-only primary，则 `template.status=no_allowed_primary_certificate`。

4. MLP interaction 仍使用 kind-level sham，且 scope 可能比 evidence group 宽。

   文件位置：`route_graph/audit_interactions.py:80-120`，warmup 位置 `route_graph/audit_phase_native.py:93-103`。

   `mlp_interaction()` 的 sham check 是 `{"content", "mlp"}.issubset(oracle.sham_kinds)`。这比 D position certificate 的 per-group sham 弱，不能证明当前 evidence group + 当前 MLP group 的 joint sham 逐 logits 相同。另一个问题是 `mg = mlp["group"]` 可能覆盖全 shared queries/all layers，而 evidence group `eg` 可能只是 query 子集；当前 2×2 是 `eg × mg` 的全局有限事件交互，不一定是“同 query/layer 的 evidence–MLP antagonism”。slot/context 分解能降级 context-dominant 情况，但不能替代 scope 声明。

   最小修复：为 `(eg, mg)` joint sham 保存 measurement key，并在 interaction result 里记录 `mlp_scope_relative_to_evidence`。若 MLP query/layer 不被限制到 evidence 的 common scope，状态最多为 `broad_MLP_event_interaction` 或 `position_message_MLP_global_interaction`，不能升级为 `evidence_MLP_antagonism`。

## Should fix before natural-batch claims

5. `candidate_control_audit` 仍不足以审计非 finalist 的 matched-control 状态。

   文件位置：`route_graph/audit_pools.py:111-123`。

   现在返回了每个进入 family pool 计算的 candidate id、delta、pool_count、selected 和 status；这修复了“完全没有分母”的问题。但没有返回非 finalist 的 control ids、matching metrics、hash排序后的前若干 controls，也没有记录 chosen reason。若后续要证明“强候选因为无控制而 unresolved，而不是被删除”，当前摘要仍不够复查。

   最小修复：在 `candidate_control_audit` 中保存 `control_ids` 和每个 control 的 matching 字段，或返回一个只含结构信息的 `all_candidate_pools`。这仍然不把这些 controls 交给 C/D 作为回填候选。

6. search 的资源/错误计数还缺两个字段。

   文件位置：`route_graph/causal_groups.py:433-466`。

   tree measurement 现在记录 `new_forward_calls`，但 union measurement 成功/失败记录没有；返回值也没有 `invalid_groups` 或 `invalid_forward_calls`。`unsearched_tree_groups` 与 `unsearched_union_groups` 已经有了，但还不能完整解释“预算被 invalid 干预消耗了多少”。

   最小修复：union 分支也记录 before/after calls；return 加 `invalid_groups`、`invalid_forward_calls`、`measured_tree_groups`、`measured_union_groups`。这项不阻塞单条 smoke，但会影响自然批量 denominator。

7. token 资源已经计数但没有进入 phase artifact。

   文件位置：`route_graph/causal_groups.py:72-96` 已计 `tokens_processed`；`route_graph/audit_phase_native.py:28-42` 和 `route_graph/audit_phase_native.py:258-272` 未返回它。

   最小修复：B 返回 `tokens_processed`，D 返回 proposal+D+alternate token 数；merge 层汇总 response token-forwards。否则报告只有 forward calls，没有序列长度成本。

8. 320/1280 代码和主提案文档仍不一致。

   文件位置：`route_graph/audit_protocol.py:11-12` 已写 320/1280；`refine-logs/FINAL_PROPOSAL.md:171-183` 和 `refine-logs/FINAL_PROPOSAL.md:227` 仍写 256/1024 与旧 N 模板 16。

   用户已说明计划文档随后同步，因此这不是代码逻辑 blocker；但在任何实验记录或方法报告中，必须明确本次实现按 320/Claim、1280/response，而不是旧 256/Claim。否则后续 reviewer 会认为预算超规格。

9. budget 常量散落在 `audit_phase_native.py`，没有从 protocol 读取。

   文件位置：`route_graph/audit_phase_native.py:30-31`、`route_graph/audit_phase_native.py:84`、`route_graph/audit_phase_native.py:192-197`、`route_graph/audit_phase_native.py:257`。

   这次值与用户最新口径大体一致，但后续文档同步时容易再次漂移。建议从 `PROTOCOL` 读取 claim cap、B screen cap、N template cap、layer cap，或者在 phase return 中写明 exact local constants。

## 对重点项的判断

- 整个 B/A 事件 F：通过。实现仍是完整续写事件 log-odds，不是首 token 或平均 logp。
- 共同 prefix 干预：主路径通过，public API 已有 guard。仍建议为 post-divergence query 补回归测试。
- 实际 forward 预算：主计数可信，320 cap 在 D 中生效；但 N 分支顺序和资源字段需要修。
- 精确 sham：position group 已修为 per-group；MLP interaction 和部分 control/joint 操作仍需要 per-certificate sham key。
- native proposal 无语义候选泄漏：通过。B 阶段接口没有 reader evidence/label/rank 泄漏。
- 父组/并组搜索：父组保留、二分、非邻接 query union 仍合理；未测 union 计数已改善，但 invalid/union call 计数不全。
- 固定 control pool 先于语义和效应：B/C 顺序成立；但非 finalist pool audit 信息不足，origin controls 需同角色。
- paired R：公式方向和同 H/同移出质量/换 destination 对照在代码中基本自洽；origin role mismatch 会影响升级语义。
- 输入介导：selected-key donor capture 与 recipient domain 已对齐；origin controls 角色问题修完后可作为可审计介导 witness。
- MLP 4态：四状态 `base/no_e/no_m/both` 已实现，slot/context 分解已实现；scope 和 sham 仍不够强。

## 测试结果

使用指定解释器与 CPU 环境变量：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
-m pytest tests/test_causal_groups.py tests/test_native_audit.py tests/test_causal_contrast.py -q
```

结果：20 passed in 22.21s。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
-m pytest tests/test_audit_alignment_review.py tests/test_evidence_anchor_review.py -q
```

结果：4 passed in 8.63s。

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
-m pytest tests -q
```

结果：78 passed in 94.45s。

这些测试说明现有 CPU 回归未破坏，但它们没有覆盖 `validation_phase()` 的 N 分支跳过 paired/MLP、origin-controls 同角色、MLP joint sham/scope、candidate pool 全审计字段。上述问题来自静态审查，不是测试失败。

## 当前评分

| 维度 | 分数 | 理由 |
|---|---:|---|
| B/A F 与 shared-prefix 约束 | 8.6 | 主目标和 guard 已到位；还需 post-divergence 回归测试。 |
| native proposal blind search | 8.3 | B 阶段接口没有语义候选泄漏；root/order/union 计数仍可增强。 |
| control pool 与分母 | 7.2 | pairwise visibility 已修；非 finalist pool 详情和 origin 同角色仍缺。 |
| position/origin certificate | 7.3 | per-group position sham 已修；origin controls 角色混淆阻塞升级声明。 |
| paired R / MLP / N 分支 | 6.4 | pairedR 公式基本可用；N 分支顺序和 MLP sham/scope 是当前最大问题。 |
| 预算与资源记录 | 7.0 | 320 cap 实现；N 消耗顺序、invalid calls、tokens_processed 输出不全。 |
| 测试覆盖与可部署性 | 7.5 | 78 个 CPU 测试通过；新增 phase 逻辑缺专项回归。 |

加权整体：7.6/10。建议顺序：先修 Required 1-4，再跑单条 D-stage GPU smoke；再补 Should fix 5-9，最后跑自然小批量。B-stage proposal/search 单独 smoke 可以先跑，但不能把 D 证书链视为已通过审查。
