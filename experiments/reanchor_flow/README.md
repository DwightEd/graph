# Head-resolved mechanism audit

这里审计的是一条可干预的 source-effect chain，而不是从 attention 热图直接推断“模型用了证据”：

```text
selected source unit → prompt/response hub → target margin
          route screen → source-cut integration → exact patch/block
```

分析保留每一个 `(layer, head, source, target)`。不同 head 可以承担 global retrieval、local
copy 或相反方向的 residual write，因此代码从不先对 head 求平均。target 不必直接回看 prompt；
只有 selected-source cut 确实改变 hub、且 hub patch/block 对 target 闭合 mediation，才称该
operator 下的 causal-relay candidate；进一步称 grounded route 仍需 matched factual pair。

方法定义和结论边界见 [METHOD.md](METHOD.md)，artifact 字段见 [SCHEMA.md](SCHEMA.md)，更完整的
机制注册见 [MECHANISM_AUDIT.md](MECHANISM_AUDIT.md)。

## 一条命令：真实子集审计并画图

下面的纯 Python 命令会在 QA、Summary、Data2txt 各选一个无标签样本，冻结 observed token 与
native runner 的 margin，完成 native graph、source-unit screening、head-resolved route ledger、
root/carrier/corridor rerun，并为每个 target 生成四联机制图：

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/graph
conda run --no-capture-output -n research \
  python -m experiments.reanchor_flow.run subset \
    --split test \
    --task all \
    --samples-per-task 1 \
    --targets-per-sample 1 \
    --target-policy uncertain \
    --flow-signal message \
    --carrier-scope all \
    --max-response-tokens 128 \
    --edge-coverage 0.90 \
    --query-chunk 8 \
    --local-window 10 \
    --model /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct \
    --cache /share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876 \
    --source-info /share/home/tm902089733300000/a903202310/lys/data/RAGTruth/dataset/source_info.jsonl \
    --output experiments/reanchor_flow/outputs/native_mechanism_v2 \
    --plot
```

结果位于 `.../native_mechanism_v2/test/`。重复同一命令会恢复已完成的 target。长上下文的 eager
attention 仍有单层 (O(S^2)) 开销；`--query-chunk` 限制 source-cut query 重算批量，但不是固定
显存承诺。`--carrier-scope` 默认是 `all`，以保留跨层 prompt-hub 候选；若显式缩减 scope，未表示
的 prompt destination 在下一层记为 `unobserved`，不会把初始 selected-source 身份跨层硬拷贝。

如果已有 audit，只画一个 target：

```bash
python -m experiments.reanchor_flow.run mechanism-plot \
  --artifact /path/to/q123_a10_b20_message.npz \
  --output /path/to/q123_mechanism.png
```

`--tokens-json /path/to/tokens.json` 可提供仅用于显示的 token 字符串数组。绘图接口不接收
hallucination/correctness labels。

## 图中四个面板

| 面板 | 编码 | 能回答的问题 |
|---|---|---|
| A. layer-unrolled route | 保证一条 connected root→target widest backbone（含 residual steps），再补 throughput 最大的边；边宽为 throughput，颜色为 root-lineage allocated action | selected source 的候选 lineage 是否经 hub 到达 target？ |
| B. head-resolved action | layer × head × position 的 evidence/response-origin action | global/local heads 是否支持、反对或抵消？ |
| C. integration | residual、attention、MLP 的 source-cut displacement/action，加模块一致性 | source cut 在哪里改变状态，这些改变是否支持 target？ |
| D. intervention ladder | root cut、corridor rescue/block、carrier mediation 的真实 margin effect | 候选路径是否通过精确干预？ |

前三个面板提出或解释机制候选；因果命名以第四个面板的 exact rerun 为准。高跨-head/模块一致性
只表示当前 target margin 下的 functional alignment，不表示正确，也不能单独称为 attractor。

## 捕获后再看 hallucination labels

机制 capture 完成后，才可单独执行：

```bash
python -m experiments.reanchor_flow.run subset-evaluate \
  --split test \
  --model /share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct \
  --cache /share/home/tm902089733300000/a903202310/lys/research/Unsupervised-hypergraph/outputs/attention_cache/fresh_attention_c8847872bedf_20260731T074520Z_p876 \
  --output experiments/reanchor_flow/outputs/native_mechanism_v2
```

评价只检验 `native_source_mediated_observed_margin` 与
`response_origin_supporting_action_candidate`，不会回写 label 到机制 artifact。第二项不是
response-history intervention，也不支持生成自我强化的因果命名；“source-operator support 低、
response-origin supporting action 高”目前只能注册为 response-reliance candidate。

## 受控 factual pair

RAGTruth native audit 的 source cut 只能识别“observed target 对该 Value-message cut 的依赖”，
不能证明 observed token 是正确事实，也不能把残差差分称为纯语义。若要验证 grounded factual
effect，使用预先对齐的 clean/counterfactual pair：

```bash
python -m experiments.reanchor_flow.run corridor \
  --pair /path/to/paired_world.npz \
  --flow-signal message \
  --carrier-scope all \
  --query-chunk 8 \
  --model /path/to/Meta-Llama-3.1-8B-Instruct
```

pair 必须固定相同 token 坐标、相同 teacher-forced response、明确允许改变的 source units，以及
运行前注册的 positive/negative target。`attention` backend 可作为 routing 对照；无论候选如何选，
因果确认始终 patch/block 真实 pre-`W_O` message，而不是用 attention mask 代替。

## 测试

```bash
python -m pytest -q \
  experiments/common/tests/test_llama_message_intervention.py \
  experiments/reanchor_flow/tests
```
