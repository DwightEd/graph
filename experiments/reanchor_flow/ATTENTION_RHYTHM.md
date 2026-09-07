# 原始 attention 节律审计（第 1 阶段，不是检测器）

目的：在**同一批原生完整 attention 行**上，对照原始 source 位置图、距离/WAAD/FAI 和旧四桶压缩。
不以“必然存在重锚定”为前提，不用 PCA、监督分类器、learned head transition 或 source 消融。
当前发现/监督/corridor 分支不修改；本入口是一个独立、可核验的测量对照。

## 运行

在仓库根目录，沿用 `research` 环境、原 `research_dataset`、source_info 和模型路径：

```bash
python -m experiments.reanchor_flow.attention_rhythm_run \
  --split test --task QA --sample-id 12693 \
  --query-chunk 8 --map-tokens 128 \
  --output experiments/reanchor_flow/outputs/attention_rhythm_qa_12693
```

`--map-tokens` **只限制展示窗口**；默认读取完整 response，绝不默认截为 128 tokens。
`--map-offset` 相对于第一个 response predictor。用 `--head 12:3 --head 20:7` 指定精确头。
默认按每条样本的平均 attention 距离，选 local/global 组中固定排名的代表头，不根据标签或波形挑图。
输入可用 `--cache ROOT` 指向包含 train/test 的 research_dataset；或 `--scans ROOT` 复用已完成四桶扫描
中的 token_ids（原始 attention 仍由模型重新计算，不把四桶还原成 attention）。

全任务/全样本的描述性统计：

```bash
python -m experiments.reanchor_flow.attention_rhythm_run \
  --split all --task all --samples-per-task 0 \
  --query-chunk 8 --plots-per-task 1 --evaluate \
  --output experiments/reanchor_flow/outputs/attention_rhythm_all
```

可用 `--model`、`--source-info` 指定不同路径。`--evaluate` 只在所有捕获、head 选择、峰和图名单冻结后
打开标签；不拟合检测器。显式 `--max-response-tokens N` 才截断，输出同时报告处理/原始 token 数。
同配置、同 tokens/mask 重跑会复用 NPZ；改变窗口、头、显示范围或模型请用新输出目录。
若只需总体曲线，`--plots-per-task 0` 避免第二次 map forward。

## 量的定义与坐标

依据 *Attention Illuminates LLM Reasoning*, arXiv:2510.13554v2，Eq.7/9/10：
https://arxiv.org/html/2510.13554v2

- `distance[l,h,q] = sum_s A[l,h,q,s] * (q-s)`；`distance_mean` 仅平均 response 行 `q=P..N-1`，
  不平均 heads，用于 per-sample bottom/top 30% span 排名。排名不是语义功能鉴定，也不是 train-frozen detector。
- `waad[l,h,q] = sum_s A[l,h,q,s] * min(q-s,W)`，默认 W=10。距离 1 与 8 均可被区分，
  不需要越过 local window 或重新读取 prompt。
- `fai[l,h,s]` 平均 response query `q` 对 `s` 的 attention，`s+Hlo <= q <= s+Hhi`，
  默认 `[10,100]`，上下界均包含。分母是**实际可用行数**；末尾没有未来行时 NaN，不是零。
  Hlo=0 按论文公式包含对角项；默认 10 不含。FAI 是离线后续复用，不是 onset 的在线前兆。
- `message_waad` 使用 `A*||W_O[h] V_s||` 的完整行归一化作为独立对照；四桶 message 保存未归一化幅值。
  没有 message budget 的行记为缺失。幅值不是 signed contribution，不是事实接纳。
- 同时捕获 `q=P-1..N-1`：`P-1..N-2` 预测 response tokens，`P..N-1` 是论文 response 行。
  两种视角通过坐标切片区分，绝不把 q 的已输入 token 与 q+1 的被预测 token混淆。

原始图按明确的 head/query 范围保存**所有 source 列**，无 top-k、无重新归一化。
主图逐 head 分开。`--paper-groups` 可额外保存论文式 local/global **组内均值参考图**；该参考图不进入
主统计，也不替代逐头数据。没有该选项就不做组内平均。

## 实际检验与边界

1. 同头 WAAD 峰与四桶多数翻转对照：旧规则要求 local<->nonlocal 在 50% 处翻转。
   `bucket_missed_fraction` 是 **WAAD 候选峰未被四桶规则捕捉的比例**（允许相差一位），
   不是 missed-reanchor 率、更不是幻觉率。
2. local-head WAAD 与 global-head FAI 对每一对 head 分别计算耦合，允许 FAI 同位/晚一位。
   分母是 FAI 峰数；保存整张 head-pair 矩阵。空峰集为缺失，不能人工补一个峰。
   null 是在同一可用位置集合上保持 FAI 峰数随机打乱的**精确期望**；只是占位率基线，不是因果检验。
   可用未来窗口变化会影响 FAI，所以另存 full-horizon-only 的结果与分母。
3. 峰算法是本审计显式约定：严格高于左右邻居；WAAD 相邻 prominence 至少 0.5 个位置单位，
   FAI 至少为该 head 有效范围的 10%。常数平台、端点不产生峰。论文没有提供可直接复用的峰检测
   源码，因此不声称我们的峰计数复现其 Table 2 数值。所有连续曲线均保留供敏感性检查。
4. `--evaluate` 按 `q+1` 加入标签，比较同样本、同 log2(response_index+1) 区间的 H-N，保留全部头。
   只控制粗位置，不自动排除词类、任务、generator/observer 差异。没有混合标注区间则未知。
5. 每个 split/task 单独总结：先 source 内样本平均，再 source 平均/bootstrap；不足 3 个 source 不给 CI。
   CI 条件于固定指标/峰约定，不是跨所有 head 的多重检验结果，不自动宣称发现机制。

本阶段回答“结构是否可见、何处丢失、是否有非随机时序关联”，不验证事实语义、hub 因果中介、
MLP 整合或错误吸引子。漂亮图和峰计数都不是事实正确性的证据。

## 文件与输出

- `attention_rhythm.py`：`AttentionRhythmObserver` 完整行统计；`RawMapObserver` 明确头/窗口的原始图；
  `capture_rhythm` 复用公共 native forward（GQA/RoPE），不复制 Llama 实现。
- `attention_rhythm_report.py`：峰/旧规则对照、逐头配对、FAI 耦合与 source bootstrap、图及事件文本。
- `attention_rhythm_run.py`：现有数据接口、捕获、续跑、标签后验连接；无新训练或数据打包。

`<split>/<task>/<sample>.npz`：完整时间轴逐头曲线、四桶、FAI 分母、所选原始图、token 字符与坐标。
同名 `.png`：逐头 source 图＋WAAD＋四桶＋FAI；`.full_sources.png` 保留 prompt 区的完整 source 视图。
`.events.json`：图窗口内峰的 layer/head/query、下一个词、主要 source 词、距离和旧规则是否捕捉。
`.audit.npz`：逐头/逐 head-pair统计。`summary.json`、`population_<split>_<task>.png`：总体结果。

## 测试与成本

```bash
python -m pytest -q experiments/reanchor_flow/tests/test_attention_rhythm.py
```

本次执行：15 passed，1 skipped。已测密集参考公式、chunk 不变性、1→8→1 同桶反例、FAI inclusive
窗口/末尾缺失、q→q+1、head 保留、W_O Gram 范数、空峰、source 平衡。缺少 Transformers，真实 tiny
HF Llama 对照测试未在交付环境执行（测试已附）；没有 GPU，未运行真实 8B/RAGTruth。不能以单元测试
冒充真实机制发现。部署前在 research 环境运行整组测试，tiny HF 测试应不再 skipped。

每样本一次无梯度 native forward 获得全部头曲线；需原始图的少数样本另加一次 forward。
按 query chunk 处理，只有当前层临时 A，没有全 L×H×N² 常驻；曲线仍有 O(LHT) 存储，
显示图有 O(selected_heads × map_tokens × N) 存储。默认 query_chunk=8 不是任意长度的显存保证。
FAI 使用已观察回答的未来部分；先前 token 的计算仍保持 causal mask。
