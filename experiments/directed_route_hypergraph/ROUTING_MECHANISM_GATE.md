# Attention-routing mechanism gate record

## 问题与预注册假设

当前 attention cache 能否在不训练新 encoder、不读取 hallucination label 的条件下，
观察到两类生成前异常：

1. `routing drift`：prompt-rooted lineage 逐层让位于 response-rooted lineage；
2. `routing dispersion`：endpoint 分布变散，或不同 heads 对 source role 的选择分歧增大。

第三类 `parametric bias` 不在本实验中估计。它需要新的 frozen replay 保存 OV contribution、
residual/MLP adoption 与 chosen-token logits；response-rooted attention 不能替代这组证据。

预注册的 drift 方向为

\[
g_t=\log\frac{E_t+\epsilon}{D_t+I_t+\epsilon},
\]

值越大表示 response-rooted routing ancestry 越强。Dispersion 分别读取 normalized entropy
lower/upper bounds 与 head-role JSD，不与 drift 合成一个总分。

## 数据与范围

- calibration：train split 中 checkpoint 预留、且与 test source-disjoint 的 calibration rows；
- evaluation：RAGTruth Llama-3.1-8B QA test rows；
- 默认 graph sidecar：`experiments/dbgnn_reference/outputs/qa_compare/gcn`；
- token 0 因缺少 last-prompt predictor query 而 unavailable；cached query `i` 对齐 response
  token `i+1`，最后一个 cached query 丢弃；
- 当前 prompt partition 只有 `prompt / response`，不是 evidence / constraint segmentation。

## 实现

- implementation commit: `7ee5e724005e41ae68a5e0dc36533cf19b9a5a54`
- lineage DP: `routing_lineage.py`
- censoring-aware dispersion: `routing_dispersion.py`
- controls: `lineage_controls.py`
- trace / calibration / evaluation: `lineage_artifacts.py`, `lineage_scoring.py`,
  `lineage_evaluation.py`
- one-command runner: `run_lineage_qa.sh`

代码验证：

```text
directed_route_hypergraph + grounded_route: 118 passed
compileall: passed
bash -n run_lineage_qa.sh: passed
git diff --check: passed
```

合成测试只证明数学、对齐和 artifact 防火墙符合设计，不代表真实检测有效。

## 运行命令

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph

REPO=$PWD \
GRAPH_INDEX_ROOT=$PWD/experiments/dbgnn_reference/outputs/qa_compare/gcn \
TEST_ROOT=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test \
OUT=$PWD/experiments/directed_route_hypergraph/outputs/qa/routing_lineage_gate_seed20260827 \
SEED=20260827 \
BOOTSTRAP=1000 \
bash experiments/directed_route_hypergraph/run_lineage_qa.sh
```

只冻结无标签 artifact：

```bash
EVALUATE=0 bash experiments/directed_route_hypergraph/run_lineage_qa.sh
```

## Controls 与报告项

- drift controls：ordered、reverse、fixed random-layer、last-layer、response-carrier-rewire、
  posthoc same-token；
- dispersion：entropy lower/upper、head-role JSD；
- position/length：response ordinal、absolute sequence position、prompt length、offline relative
  position、offline response length；
- statistics：source-balanced empirical high-tail probability、source bootstrap、paired metric delta；
- audits：calibration fallback/support、D/I/E/U mass、unresolved mass、carrier changed fraction、
  dispersion interval/range/role-mass error。

## 输出与结果

- output directory: `experiments/directed_route_hypergraph/outputs/qa/routing_lineage_gate_seed20260827`
- frozen scores: pending remote run
- AUROC/AUPRC and source-bootstrap confidence intervals: pending remote run
- paired ordered-minus-control deltas: pending remote run

## 停止决策

结果返回前状态为 `pending`。保留/停止规则固定为：

1. ordered drift 不优于 reverse、last-layer、same-token：停止 chronology / pre-generation claim；
2. ordered 不优于 fixed random-layer：停止真实层序 claim；
3. ordered 不优于有效 carrier-rewire：停止 exact response-relay claim；
4. drift 与 dispersion 均不优于位置/长度基线：停止扩大 attention-only encoder、embedding、
   AE/VAE 容量；
5. 只有 attention-only gate 成立后，才进入 OV/residual/MLP/logits replay 检验
   parametric-bias 机制。
