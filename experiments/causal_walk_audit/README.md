# Causal-Walk Audit

This subproject validates the **Self-confirming Causal-Walk Lock-in** hypothesis before a final graph model is designed.

It does not assume that hallucinations simply attend less to the prompt or that every large layer transition is abnormal. Instead it asks four falsifiable questions:

1. **Non-Markov path memory:** do ordered two- or three-edge causal walks predict the next layer better than one-hop routing alone?
2. **Anchor congruence:** do direct prompt routes and response-relay routes preserve the same prompt anchor?
3. **Audit escape:** after routing becomes response-local, does the model return to an anchor-connected state before a claim token?
4. **Lock-in:** after disagreement appears, does routing persist in a response-local state with little evidence escape?

A fifth causal question—whether targeted Q/K/V, residual, or MLP interventions repair the output—cannot be answered from cached attention and is explicitly reported as not implemented.

## Scientific scope

Without an anchor manifest, prompt tokens are split into contiguous chunks. In that mode all outputs are **prompt-chunk lineage proxies**, not evidence grounding. A manifest can assign token ranges to evidence, question, instruction, and other anchors.

Example:

```json
{
  "sample_42": [
    {"name": "retrieved_passage_0", "kind": "evidence", "start": 0, "end": 96},
    {"name": "question", "kind": "question", "start": 96, "end": 128}
  ]
}
```

Ranges use prompt-token offsets in the full prompt prefix.

## Data representation

Each retained attention event remains

```text
(source token, target token, layer, head, weight)
```

The code builds a layer-event graph where one event node represents an exact token-pair edge in one layer and stores the full head vector. A De Bruijn-style relation connects

```text
(u -> s at layer l-1) -> (s -> t at layer l)
```

so the model can distinguish two routes that end with the same final edge but arrive through different histories.

The anchor-lineage state is

```text
[token, layer, head, anchor_or_response_base, relay_depth]
```

with relay depth `direct`, `one_hop`, and `multi_hop`, plus a separate unresolved sink. The propagation conserves routing mass.

## Label-free fitting

Three nested ridge predictors are fitted on the train split:

- order 1: current head-resolved prompt/response/self/unresolved routing;
- order 2: order 1 plus exact two-walk context, direct anchor state, and one-hop relay state;
- order 3: order 2 plus three-walk context and multi-hop lineage.

The target is the next layer's head-resolved role routing and anchor-lineage state. No hallucination labels are read.

Matched-dimension null models shuffle only the added order-2 or order-3 block. Positive validation gains therefore test whether the added path correspondence matters, not merely whether a wider regression has more parameters.

## Frozen token scores

```text
order1_error
order2_error
order3_error
order2_gain
order3_gain
order2_path_gain
order3_path_gain

direct_role
anchor_js_mean
anchor_js_peak
anchor_js_excess
recoupling_depth
recoupling_failure
response_persistence
evidence_escape
lock_in
known_anchor_mass
response_base_mass
```

`anchor_js_excess` compares real anchor congruence with anchor-ID permutations. `lock_in` is high only when direct/relay anchor disagreement is high, future routing remains response-local, and anchor-connected escape is absent.

## Run

```bash
DATA_ROOT=/path/to/RAGTruth/llama31_8b \
OUT=experiments/causal_walk_audit/outputs/qa30 \
TRAIN_LIMIT=100 TEST_LIMIT=30 TASK_TYPE=QA DEVICE=cpu \
bash experiments/causal_walk_audit/run.sh
```

With an anchor manifest:

```bash
ANCHOR_MANIFEST=/path/to/prompt_anchors.json \
DATA_ROOT=/path/to/RAGTruth/llama31_8b \
bash experiments/causal_walk_audit/run.sh
```

## Outputs

```text
outputs/<run>/
|-- train/
|   |-- model.npz
|   `-- training.json
|-- score/
|   |-- manifest.json
|   `-- samples/*.npz
`-- evaluation/
    |-- metrics.csv
    |-- matched_effects.csv
    |-- onset_profiles.csv
    |-- decision_table.csv
    `-- evaluation.json
```

## Stopping rules

- If order 2 does not beat order 1 and its path-shuffled null, stop the higher-order walk route.
- If anchor-aware scores do not beat `direct_role`, do not call the mechanism evidence grounding.
- If hallucinated tokens do not show lower matched evidence escape, reject the audit-failure story.
- If lock-in does not rise beyond matched pseudo-onsets, reject the persistent response-closure story.
- Cached attention can never authorize a causal MLP/QKV claim; that requires new model runs.
