# S10 CPU 实验工程复核

日期：2026-09-14
复核角色：experiment-bridge fresh code review
结论：**PASS，可执行 `S10_RUN_20260914.md` 中一次 12 行无标签 sanity。** 完整 989 行导出与拟合仍以 sanity 的 manifest、逐文件哈希、有限值和 offset 长度全部通过为前置门。

## 复核范围与快照

- 计划：`S10_STRUCTURAL_FUSION_PLAN_20260914.md`
- 图侧：`graph/structural_detector/features.py`（SHA256 `289998ae353d238f37cc826c02bf06655ed1784ec78950a9acb15e017d88ac13`）、`experiment.py`（`1cfe7a6ac99fe0d2fd2dce39d7c33653d25aa7a782ff6943226d7ac1662d55b6`）
- 导出侧：`reanchor/src/decoding/structural_export.py`（`fb3573c2b7b5c357d92bad22f0d35889cd327273a5898754c06c5bfa0aeff65f`）
- 原生捕获：迁移后的 `reanchor/src/route_graph/population_mechanism.py`（`eff6e89c9dbf328361f80c7a46bd2f537e46c22c7fd675cf4bd5af2e054d8812`），与冻结缓存 settings 记录的原文件哈希完全相同。
- 缓存：`ragtruth_population_20260912/settings.json`、回答 `11949` 的 `record.json`、`manifest.json` 及其声明的 `metrics.npz`、`profiles.npz`、`tokens.npz`。

复核只读取无标签缓存和代码，没有运行 GPU、没有读取或计算 gold 指标、没有运行完整 CPU 实验。

## 阻断项及关闭情况

初审发现三项必须在完整导出前修正的问题，均已关闭：

1. D 及其派生量原名可能被误读为“总历史位移”。现在所有名字均明确为 `remote_history_excl16_minus_source`，包括差分、熵交互、raw 输出与 bootstrap 键；导出 manifest 同时保留 `history_exclusion=16`。
2. `source_head_std` 原先联合跨 layer 和 head 求标准差，混入层间均值差。现在固定为每层跨 head 的总体标准差，再对层等权平均：`source.std(axis=2).mean(axis=0)`。新增的层均值反例测试通过。
3. 原报告路径遗漏 raw negative-margin 的来源 bootstrap，并只为学习模型计算回答内 AUROC。现在 bootstrap 覆盖 combined 对 instant、raw entropy、raw D、raw negative-margin；回答内 AUROC 覆盖四个学习 arm 和四个 raw 对照（含独立位置对照）。

当前没有未关闭的核心工程阻断。

## 因果索引与冷启动

原始生成路径将 `ids[:-1]` 输入 observer；响应 token `t` 对应的预测 hidden state 是绝对位置 `q = prompt_length - 1 + t`。entropy、margin、source attention 和 remote-history attention 都从同一个 `q` 取得，因此特征与待预测 token 的 offset 一一对应，没有右移或未来 token 混入。

remote mask 只允许响应 key `k >= prompt_length` 且 `k <= q - 16`。换成响应内索引即 `j <= t - 17`，所以它排除最近 16 个已生成 token；`t=0..16` 的 remote mass 固定为 0，`t=17` 才首次可能非零。代码没有丢弃这些前 17 个 token：D 在首 token 为 `0-source`，相邻差分首值为 0，过去熵记忆首值为 0。对冻结 roster 前 12 个回答的无标签检查均显示 profile/metric/offset 长度一致、数值有限、remote mass 在 `t=0..16` 精确为 0 且首次非零为 `t=17`。

图侧差分只使用 `t` 与 `t-1`；8 步熵记忆只访问 `[max(0,t-8), t)`；交互只使用当前 D 与当前/过去熵。`log(1+t)` 只保存为 raw position 对照，没有进入任一学习 arm。没有真首错位置、未来文本或回答总长度特征。

## 分割、标签访问与冻结

冻结 roster 共 17,790 回答；固定筛选得到 QA、llama-2-7b-chat 共 989 回答，即 official-train 839、official-test 150。该子集有 989 个唯一 source 和 989 个唯一 response，official train/test source 交集为 0。

按 `SHA256("20260914:" + source_id)` 排序后的固定分割为：训练 671 source / 153,886 token，校准 168 source / 38,134 token，测试 150 source / 30,721 token。分割在任何 annotation 解析前写入协议冻结。

导出器不接收 annotation 路径，只读取冻结 observer 产物，并声明 `labels_read=false`、`model_forwards=0`。实验侧第一次 annotation join 只反序列化 train/calibration 的指定 ID；训练均值、标准差和 L2 logistic head 仅用 train。正斜率缩放和截距仅用 calibration；5% 正常 token FPR 阈值仅用 calibration negatives。八个模型、阈值、全 989 回答预测与索引全部写盘并由 `prediction_freeze.json` 哈希后，才反序列化 official-test 指定 ID。测试标签不参与选 arm、拟合、校准或阈值。

来源等权通过每个 source 的 token 权重和为 1 实现，再做不改变相对权重的全局缩放。来源 bootstrap 固定 seed 20260914、200 次，以 source block 重采样；本子集中一 source 对应一回答。

## 验证与边界

- 独立执行 `graph/tests/test_structural_detector.py`：8 passed。
- 独立执行新增 `reanchor/tests/test_structural_export.py`：1 passed。
- 四个目标文件 AST 解析通过；迁移后的原生捕获文件哈希与冻结 settings 中记录值一致。
- 已确认样例 `11949` 的 source/split/response 身份、child manifest 文件哈希以及 `(layers, tokens, heads)` profile 形状与 token 数一致。

这些量来自 Llama-3.1 observer 对 Llama-2 原回答的 replay，不是原生成器内部机制。official-test 整体曾被历史研究评价；当前代码已在结果 scope 中保留该限制，后续只能表述为本方法冻结后的来源留出测试。

## 执行门

fresh review 仅许可先执行一次：

```bash
bash /share/home/tm902089733300000/a903202310/lys/research/reanchor/scripts/export_structural_features.sh sanity
```

执行者须按 `S10_RUN_20260914.md` 核对 12/12、`labels_read=false`、`model_forwards=0`、records/artifact 哈希、五列有限值和 offset 长度。任一项失败即保留失败产物并停止，不运行 full。
