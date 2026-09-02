# Dual-register attention mechanism audit

This experiment tests a narrow mechanism question: when a generated token is
unsupported, does the model carry a persistent state descended from direct
evidence, or a persistent state supplied by response history after direct
evidence attention writes are removed?

It is a frozen-model, teacher-forced audit. It does not claim to reconstruct
complete causal flow, recover full token ancestry, or identify parametric
knowledge.

## Alignment and replay branches

Let `P = response_start`. Target response token `t` is scored from its causal
predictor

    q_t = P - 1 + t

The same sample is replayed in four aligned branches:

| Symbol | Stored name | Attention writes removed at response predictor queries |
|---|---|---|
| `F` | `full` | none |
| `noE` | `no_evidence` | direct-evidence sources |
| `noH` | `no_history` | strict response-history sources |
| `noEH` | `no_evidence_history` | both |

Predictor self is never included in strict history. A deletion is applied at
each aligned response predictor query, so its state and KV consequences pass
through later layers. The intervention is deliberately narrower than “remove
all evidence”: other prompt tokens, predictor residual state, MLP computation,
and evidence already propagated into other states remain available.

## Two finite-difference registers

For every captured hidden-state quantity, the audit forms

    P_reg = F - noE       evidence-adoption register
    R_reg = noE - noEH   autonomous-history register

`P_reg` is separate from `P = response_start` in the predictor equation. The
register names describe the branch contrasts, not an assertion that every bit
of their state has uniquely known ancestry.

Each decoder layer is captured at four stages with the exact residual identity

    output = input + attention_write + mlp_write

The identity is differenced branchwise for both registers. The artifact stores
the four stage norms, the `attention_write + mlp_write` step, MLP alignment,
the evidence/history interaction norm, and closure error. This keeps MLP
amplification, cancellation, or rotation explicit instead of folding it into
an attention-only story.

For each target and register, the step vectors across layers define a Gram
matrix. One label-free graph candidate is

    provenance_takeover
      = log((lambda_max(Gram(R_reg)) + eps)
            / (lambda_max(Gram(P_reg)) + eps))

This raw quotient compares the dominant cross-layer step energy; its leading
eigenvalue reflects both step magnitude and cross-layer alignment. It is kept
as a candidate control, not promoted over an already validated raw causal
baseline and not presented as proof of ancestry. History-dependent scores are
invalid for the first two response targets, where strict history is not
available separately from predictor self.

## Signed residual-message routes

The capture also exposes how each register's attention write is assembled.
For each `(layer, target, register)`, branch-difference head-source messages use
the actual post-intervention attention coefficients and values, the GQA
query-to-KV mapping, and the matching query-head block of `W_O`. Thus the
stored magnitude is `||delta(A V W_O)||`, not `A ||V W_O||` and not bare
attention.

Every signed edge contribution is decomposed before compression into three
terms:

- `root`: the write removed by the intervention itself (direct evidence for
  the evidence-adoption register; strict history for autonomous history);
- `carrier`: content already changed in a non-root source state,
  `mean(A) * delta(V)`, including evidence propagated into response history;
- `gate`: changed routing through Q/K and softmax competition,
  `delta(A) * mean(V)`.

For non-root edges, the midpoint identity
`delta(A V) = mean(A) delta(V) + delta(A) mean(V)` is exact. It is a symmetric
algebraic decomposition, not a unique causal attribution: `gate` combines Q,
K, and softmax-competition changes. Root, carrier, and gate contributions
reconstruct the complete signed attention-register write.

Edges are ranked globally over `(head, source)` by the absolute value of their
signed projection onto the complete register attention write. The adaptive
cover, capped by `top_k`, retains for every explicit edge:

- source token and query head;
- nonnegative `delta(A V W_O)` magnitude;
- signed residual-message contribution; and
- one of four disjoint source roles: direct evidence, other prompt, strict
  response history, or predictor self.

Dense role totals are computed before compression. The artifact stores the
omitted sum of per-edge norms and the omitted signed contribution as tails;
the former is not the norm of the omitted net vector. Because tails have no
resolved source/head endpoints, the graph leaves them as row statistics. It
never invents sparse ancestry edges or connects equal-numbered heads across
layers. Explicit routes terminate at their corresponding `attention_write`
stage node; response/predictor sources link to captured `input_state` nodes,
while earlier prompt sources remain explicit external token endpoints.

The AVWO construction itself belongs to the established norm-based lineage
from Kobayashi et al. through ALTI and IFR. The experiment's novelty boundary
is the branch-defined evidence-descended versus autonomous-history comparison
and the explicit attention/MLP finite-difference registers, not AVWO tracing.

The older full-prompt collapse statistic remains only as a historical QA
audit. Its QA behavior did not generalize across Summary and Data2txt, so it is
not the present method or a task-general detector.

## Raw controls and evaluation boundary

All scores are fixed raw equations over target-token log probabilities:

| Name | Equation | Role |
|---|---|---|
| `evidence_bypass` | `noE - F` | locked primary raw baseline |
| `symmetric_route_capture` | `noE - noH` | control |
| `unsupported_history_takeover` | `2 * noE - F - noEH` | control |
| `provenance_takeover` | cross-layer Gram log spectral quotient above | graph candidate |
| `confidence` | `-F` | control |

There is no nuisance fit, cross-fitting, ECDF calibration, percentile mapping,
or label-dependent direction flip. All scores use the intersection of their
fixed validity masks for the printed comparison; every score-specific mask is
also saved. Labels remain sealed through capture, graph
construction, and score construction. They are opened only for final,
task-specific evaluation and named post-hoc audits. QA, Summary, and Data2txt
are reported separately using token-micro AUROC, sklearn average precision
(`AP`), and source-cluster bootstrap intervals.

## Files and running

- `capture.py` replays the four branches and captures register states, Gram
  matrices, and signed routes.
- `collect.py` traverses data, verifies alignment, serializes captures, and
  resumes completed samples without labels.
- `graph.py` exposes the sparse dual-register route view and unresolved tails.
- `detect.py` computes the fixed raw candidate and controls.
- `evaluate.py` opens labels for final metrics and post-hoc audits.
- `run.py` is the foreground CLI.

Schema 8 must be recaptured into
`outputs/<observer-model>/dual_register_state/{train,test}/`. Older capture
directories are preserved as historical artifacts and are not adapted or
deleted. New reports are written under
`outputs/<observer-model>/dual_register_v8/{qa,summary,data2txt}/`, so the
earlier task reports are not overwritten.

Run the complete audit once in the foreground:

    bash experiments/attention_mechanism_audit/run_all.sh

The script contains the single entry
`python -m experiments.attention_mechanism_audit.run all`, which captures the
physical train/test shards and evaluates QA, Summary, and Data2txt separately.

Run focused tests with:

    python -m pytest -q experiments/attention_mechanism_audit/tests
