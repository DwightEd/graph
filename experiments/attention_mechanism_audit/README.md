# SELECT--RELAY--OVERRIDE causal audit

This experiment tests three grounding failures with controlled fact swaps and
exact frozen-model interventions. It is a mechanism audit, not a graph encoder,
an autoencoder, a learned hallucination detector, or an observational feature
probe.

## Core hypothesis

At one candidate decision, grounding succeeds only if the model completes a
three-stage control chain:

1. **SELECT / dispersion:** the relevant answer-bearing value must exert more
   total causal influence than a matched irrelevant value.
2. **RELAY / drift:** after selection, the response history must carry the
   counter-prior evidence rather than lock onto the model's emerging text.
3. **OVERRIDE / parametric bias:** after evidence has entered the computation,
   the final decision must override the candidate preferred without context.

The stages have a common outcome variable. For candidates A and B, the replay
first measures

```text
raw = logit(B) - logit(A)
```

on the question-only branch. Its sign defines the model's prior; no prior label
is supplied by the data. Every saved branch is then oriented as

```text
M = logit(counter-prior candidate) - logit(question-only prior)
```

Thus `M > 0` always means that counter-prior evidence wins, regardless of
whether A or B was preferred initially. Exact question-only ties are skipped
and recorded rather than assigned an arbitrary direction.

## Seven fixed branches

| Saved margin | Frozen replay |
|---|---|
| `margin_question_only` | Question plus the shared neutral decision prefix; determines the prior direction. |
| `margin_prior_context` | The matched context whose relevant value supports the prior candidate. |
| `margin_counter_context` | The matched context whose relevant value supports the counter-prior candidate. |
| `margin_no_relevant` | Counter-prior context with every relevant value key disconnected from every later query in every layer. |
| `margin_no_irrelevant` | The same total-path intervention on the equal-length irrelevant control value. |
| `margin_no_history` | Counter-prior context where only the current predictor cannot attend to strictly earlier response-history keys; its diagonal is retained. |
| `margin_hybrid_history` | Counter-prior replay with the prior-context history K/V transplanted at every layer and the same absolute positions. Prompt and predictor/self states are untouched. |

The source interventions are deliberately full-path interventions. Blocking a
source only at the final predictor would miss evidence that first entered the
response history and would misclassify successful multi-hop routing as a
selection failure.

## Pre-registered readout

```text
G = M_counter - M_no_relevant       # relevant total-source gain
S = M_no_irrelevant - M_no_relevant # matched select contrast
D = M_no_history - M_counter        # history support for the prior
R = M_counter - M_hybrid_history    # evidence relayed in history K/V
O = -M_counter                      # prior capture
```

- `select_success := G > 0 and S > 0`.
- Within that domain, `self_lock := D > 0 and R <= 0`.
- Within that domain, `capture_failure := O > 0`.

RELAY and OVERRIDE are not evaluated when SELECT fails. This prevents a sample
that never used the evidence from being mislabeled as history self-lock or
parametric-prior override. Reports use no hallucination labels, AUROC, learned
probe, or fitted threshold. Means and 95% intervals are source-grouped so a
source with many rows does not dominate.

## Pair manifest

The audit accepts a JSONL file of already tokenized controlled pairs. It does
not reconstruct prompt roles, infer evidence spans, or silently fall back to an
attention cache.

```json
{
  "sample_id": "pair-1",
  "source_id": "source-1",
  "question_only": {
    "input_ids": [1, 40, 30, 31],
    "predictor_index": 3
  },
  "context_a": {
    "input_ids": [1, 10, 20, 11, 21, 30, 31],
    "predictor_index": 6
  },
  "context_b": {
    "input_ids": [1, 11, 21, 10, 20, 30, 31],
    "predictor_index": 6
  },
  "relevant_span": [1, 3],
  "irrelevant_span": [3, 5],
  "history_span": [5, 7],
  "candidate_a_token_id": 70,
  "candidate_b_token_id": 71,
  "decision_prefix_is_neutral": true
}
```

All spans are half-open token intervals. The manifest has the following
non-negotiable experimental contract:

- `context_a` and `context_b` have equal length and the same predictor.
- All IDs come from the target checkpoint tokenizer, and the supplied prefixes
  contain no padding positions.
- The relevant slot in `context_a` semantically supports candidate A; the
  relevant slot in `context_b` semantically supports candidate B.
- R/I are equal-length **answer-bearing value slots**, not whole fact
  sentences. They are exchanged exactly: `A[R] == B[I]` and `A[I] == B[R]`.
- The two contexts differ nowhere else. This holds lexical content and position
  constant while swapping which value occupies the relevant role.
- Question-only and both context branches end in the same neutral decision
  prefix. It must not reveal either answer. The boolean declaration records
  that data-curation decision; token IDs alone cannot prove semantic neutrality.
- `history_span` ends at `predictor_index + 1`. RELAY uses only
  `[history_start, predictor_index)`, so the current predictor/self is never
  removed or transplanted.
- Candidate IDs are the first different tokens after any shared candidate
  prefix already included in the branch inputs.

These constraints require purpose-built controlled pairs. Reusing the old
RAGTruth attention cache as if it contained such interventions would not test
this hypothesis faithfully.

## Run

The target checkpoint is already the script default:

```text
/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct
```

From the repository root:

```bash
PAIRS=/absolute/path/to/audit_pairs.jsonl \
bash experiments/attention_mechanism_audit/run_qa.sh
```

Optional environment variables are `OUT`, `MODEL_PATH`, `PYTHON`, `DEVICE`,
`TORCH_DTYPE`, `LIMIT`, `BOOTSTRAP`, and `SEED`. The shell script intentionally
does not use `set -euo pipefail`: each stage checks its status explicitly, the
complete Python traceback remains visible, and evaluation never runs after an
audit failure.

Outputs are intentionally separate from the old feature experiment:

- `control_chain.npz`: seven oriented margins plus the fixed derived columns.
- `control_chain.json`: schema, counts, and skipped tie IDs.
- `report.json`: source-grouped SELECT, RELAY, and OVERRIDE summaries.

## Fidelity

The model is frozen and every replay runs under `torch.inference_mode()` with
eager Llama attention. Candidate margins use the checkpoint's real `lm_head`;
there is no gradient, attention-mass proxy, operator norm, random donor, or
learned model.

For history mediation, each layer captures the prior branch's raw `k_proj` and
`v_proj` outputs only at strictly earlier response positions. The counter
branch receives those values at the same absolute positions before RoPE. On
the target Llama-3.1 checkpoint (`pretraining_tp=1`), equal positions give the
same rotary transform, and GQA K/V remain in their native 8-KV-head geometry;
nothing is averaged or expanded into the 32 query heads. The full frozen model
still executes its actual W_Q, W_K, W_V, W_O and MLP computations, so the saved
cached `operator_geometry.pt` summary is neither needed nor substituted.

Run the focused tests with:

```bash
pytest -q experiments/attention_mechanism_audit/tests
```
