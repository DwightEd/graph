# causal_groups / audit_pools 部署前工程审查

2026-09-13。范围：`route_graph/causal_groups.py`、`route_graph/audit_pools.py`、`tests/test_causal_groups.py`；只读参考 `route_graph/native_audit.py`、`route_graph/causal_contrast.py`、`route_graph/audit_alignment.py`、`route_graph/audit_certificates.py`、`refine-logs/FINAL_PROPOSAL.md` 与 `architecture_round4_implementation_addendum_20260913.md`。本轮没有改实现、没有跑 GPU、没有新增测试文件。

结论：底层完整 B/A 事件计分、共同前缀 gate、branch-matched donor、实际 forward 计数的主接口已经接近可实现；当前不应进入完整 GPU 审计部署，因为控制池匹配和失败分母仍会让自然样本上的“不成功/不可证书”被漏记或被更容易通过的候选替代。建议先修下面的 Required 项，再跑单卡样例。

## 已确认的事实

- `ContinuationContrast.score` 计算完整续写事件的 `log P(B)-logsum P(A_i)`，没有长度平均；`CausalOracle.measure` 对两条 branch 分别取完整 continuation token logp，起点为 `len(prefix)-1`。
- `propose_native_groups` 只接收 `source_keys`、冻结 B/A contrast 和 source/history 角色范围；它不接收 reader 引文、适用性分数或语义标签。native 候选提议仍是 semantic-target-conditioned，因为 F 来自语义对比，但 evidence-candidate-blind 这一点在该函数接口上成立。
- `native_audit.NativeGate` 当前包含 `CapturedValues` 和 `expected_input_scale`；`CausalOracle.mediated` 使用 branch-matched donor，并且最新版本只 capture `group["keys"]`，recipient donor domain 也同为 selected keys。这与“只替换所选 V-message 边”的介导定义一致，不再保存无关 domain 的 donor V。
- `CausalOracle._forward` 用模型 forward pre-hook 统计真实 forward；`mediated` 在预算不足时先拒绝 4 次调用，避免预算拒绝后部分执行。
- 现有 `tests/test_causal_groups.py` 覆盖了完整事件 F、content sham cache、donor sham、预算预拒绝、搜索 forward 上限和 hook 清理。但当前环境缺少 `pytest` 与 `transformers`，只能完成 `py_compile`，不能实际执行 pytest。

## Required before GPU deployment

1. `audit_pools.visibility()` 不是规格里的可见性匹配。

   文件位置：`route_graph/audit_pools.py:10-11`，调用位置 `route_graph/audit_pools.py:22-35`。

   当前实现只检查 `queries >= min(keys)` 的比例。对于 `keys=[2, 10]、queries=[5]`，它返回 1.0，但真实因果可见的 key-query 对只有一半。底层 `native_forward` 允许混合可见性并依赖 causal mask 使未来 key 权重为 0，这一点本身正确；问题在 control pool 的匹配指标会把 causal reach 明显不同的候选和控制组当作“visibility_difference≤0.1”。这会破坏 matched-source/history control 的特异性，尤其是 history 组或 query 子组较早时。

   最小修复：把 visibility 改成 pairwise 或 per-query selected-key 可见比例，例如 `mean(key <= query for query in queries for key in keys)`，并补一个混合可见 key 的 `audit_pools` 回归测试。`origin_controls()` 的 reach 指标已经更接近 pairwise 形式，可与它统一。

2. `freeze_pools()` 会隐藏强候选缺少控制组的失败分母。

   文件位置：`route_graph/audit_pools.py:58-77`，返回值位置 `route_graph/audit_pools.py:99-107`。

   代码对每个 family 先遍历 ranked candidates 计算 pool，然后选择第一个 `len(pool)>=2` 的候选；如果没有就 fallback 到 `candidates[0]`。但返回值只保留 finalists 与它们的 controls，没有保存被跳过的强候选、每个候选的 pool 数、选择原因和 `matched_control_unavailable` 计数。这样后续 C/D 阶段无法区分“native 找到强组但结构控制不足”和“native 只找到较弱但可控的组”，会把困难样本从分母中消失。

   最小修复：返回每个 family 的 `candidate_pool_summary`，至少包含 candidate id、rank、screen_delta、control_count、control ids、是否 finalist、跳过原因。finalists 仍可限制最多 3 个，但强候选无控制必须进入失败分母，并在 D 阶段不能被视为已解决。

3. `propose_native_groups()` 的未搜索/资源计数不完整。

   文件位置：`route_graph/causal_groups.py:369-389` 与 `route_graph/causal_groups.py:409-414`。

   `unsearched_groups` 只返回 tree 队列 `len(pending)`。非相邻 query union 候选是在 tree 后临时生成的；一旦 `screen_calls` 用尽，未测 union 不进入 `pending`，也不会被计入 `unsearched_groups`。这会低估有限预算失败，和最终方案要求的“保存未发现/预算状态，不删除失败 Claim”不一致。

   最小修复：在返回值中分开记录 `screen_forward_budget`、`baseline_forward_calls_excluded`、`tree_measured`、`unsearched_tree_groups`、`union_candidates`、`union_measured`、`unsearched_union_groups`、`invalid_groups`、`budget_exhausted`。如果 B+D 共用 256 次预算，B 阶段的 search result 也应明确它的 forward_calls 是否含 baseline；当前含义是 search 增量，不含初始化 baseline。

4. `CausalOracle` 没有在 public gate 接口上强制 shared-prefix-only。

   文件位置：`route_graph/causal_groups.py:203-222` 与 `route_graph/causal_groups.py:265-278`。

   当前 search 生成的 queries 来自 `contrast.shared_queries`，所以内部路径是对的；但 `CausalOracle.group()`、`gates()` 和 `mass()` 只依赖 `native_forward` 的位置合法性。外部调用者如果传入分歧后的 query position，代码会测一个 branch-specific 后缀干预，同时 metadata 仍可能声称 `query_scope=shared_prefix_only`。这会直接违背本方法的核心目标。

   最小修复：在 `gates()` 或 `group()` 入口 assert `set(group["queries"]) <= set(self.contrast.shared_queries)`，同时在 `mass()` 对 relative query index 做同样保护。补一个测试：post-divergence query 应抛错。

## Should fix before full A-D certificate run

5. `sham_kinds` 是 kind-level 状态，不是 candidate-level 证书 sham。

   文件位置：`route_graph/causal_groups.py:199-200`；只读消费点 `route_graph/audit_certificates.py:49-57`。

   最新补丁让 `CausalOracle` 记录实际通过完整 sham 的 kind，解决了 D 证书引用缺失属性的问题。但它只保存 `content/route/donor/mlp` 这种全局 kind。若 D 把 `group["kind"] in oracle.sham_kinds` 当作某个具体候选和控制的 sham 通过，就弱于规格中的“每个证书完整 logits 逐值一致”。这不必阻塞 B 阶段 search，但不能作为最终 certificate 的 per-candidate sham 证据。

   最小修复：D 阶段为每个 finalist/control 或每个 gate digest 保存实际 sham measurement key；`sham_kinds` 可以保留为 backend smoke/status，但 certificate 的 `sham` 字段应链接到对应 group 的 `strength=1` 结果。

6. donor artifact 与 mediated measurement 的连接是旁路状态。

   文件位置：`route_graph/causal_groups.py:239-242` 与 `route_graph/causal_groups.py:257-263`；artifact 写入实现 `route_graph/audit_artifacts.py:22-77`。

   artifact writer 会把 donor V 写入 `oracle.donor_artifacts`，但 `mediated()` 返回的 result 只含 gate record 的 donor sha/provenance，不含本次 measurement 对应的 artifact handles。可以通过 sha 间接匹配，但单条 measurement record 不自包含，后续聚合或审计重放容易丢 provenance。

   最小修复：把两个 branch 的 artifact refs 写进 `mediated()` result，或在 `records` 中按 measurement key 附上 artifact refs。保留 gate 里的 `values_sha256`，不要只保存路径。

7. root 搜索顺序 deterministic 但不是显式机制顺序。

   文件位置：`route_graph/causal_groups.py:295-330`。

   content/source、route/source、content/history、route/history、mlp root 都用 `priority=inf` 入 heap，tie breaker 是 group digest。默认 64 次筛选预算下这些 root 大概率都会测到；但低预算或调试预算下，首先测哪个 family 由 hash 决定，不是最终方案里的固定机制/家族顺序。

   最小修复：给 root family 一个显式 priority tuple 或顺序号，并在 search result 中记录 root order。这个问题不影响“无语义泄漏”，但影响有限预算解释。

## Tests / execution status

- `python -m py_compile route_graph/causal_groups.py route_graph/audit_pools.py route_graph/native_audit.py route_graph/causal_contrast.py tests/test_causal_groups.py` 通过。
- `python -m pytest tests/test_causal_groups.py tests/test_native_audit.py tests/test_causal_contrast.py -q` 未执行成功：当前环境没有 `pytest`。
- 轻量 import `route_graph.audit_pools` 也未执行成功：当前环境没有 `transformers`，而 `audit_pools -> audit_alignment -> evidence_anchor -> frozen_reader` 的导入链会加载 `GenerationConfig`。这不是上述逻辑缺陷的证据，但部署前必须先安装 `requirements.txt` 与 `requirements-model.txt`。
- 没有跑 GPU，没有加载真实 8B 模型，没有验证自然样本有效性。

## 当前评分

| 维度 | 分数 | 理由 |
|---|---:|---|
| 完整 B/A 事件与共同前缀目标 | 8.4 | 主 F 与 shared query 搜索路径正确；public group 接口还缺 shared-prefix guard。 |
| native gate / donor / exact replay | 8.0 | donor 接口和 selected-key capture 已对齐；candidate-level sham 和 artifact provenance 还需收紧。 |
| 搜索预算与资源记账 | 6.8 | forward 计数主路径可信；unsearched union、baseline 口径和失败计数不足。 |
| 控制池与无语义泄漏 | 6.3 | B 阶段没有 reader 泄漏；visibility 匹配错误且强候选缺控制会被隐藏。 |
| 父组/子组/并组搜索结构 | 7.2 | 父组保留、二分和非邻接 query union 已实现；root 顺序和 union 分母仍弱。 |
| 测试覆盖 | 5.8 | 核心 gate 测试方向对，但当前环境未装依赖，`audit_pools` 无混合可见性和候选分母测试。 |
| 部署前可操作性 | 6.7 | 可继续本地实现和 CPU 级静态检查；不建议直接上完整 GPU A-D 审计。 |

加权整体：7.0/10。部署前 verdict：REVISE。不是方法规格失败；主要是工程证书接口还会漏记 unresolved / matched-control failures。修完 Required 1-4 后，可以进入单条 Claim 的 GPU smoke；修完 5-7 后再跑完整 A-D 证书链。

## 旁注：本轮不计入范围

`route_graph/audit_certificates.py` 仍在 D 阶段编排范围内；本轮只读确认它会消费 `select_controls()` 和 `oracle.sham_kinds`。若 D 后续按用户补充改为重新计算 base、核对 B/A 完整 token logps、B+D 共用 256 次预算，那么这属于下一轮审查对象，不应由本轮 `causal_groups/audit_pools` 直接背书。
