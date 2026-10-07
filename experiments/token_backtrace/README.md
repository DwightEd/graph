> 2026-09-30: Current implementation and status are in [the architecture document](../../docs/ARCHITECTURE.md). Reconstruction graph commands described below are retired; their cached results remain historical evidence. Current commands: `python main.py --help`.

## Content-fixed binding verification

`binding_run` measures source7 value vectors and native edge messages in every
physical head, then replaces one edge before W_O with base Q/K/A fixed. Entity
names are disjoint across fit/dev/test; owner and fact-order conditions are
balanced. Use `--reverse-participants` to balance queried ownership for each name:

```bash
PYTHONPATH=.:teaching/state_audit/src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m experiments.token_backtrace.binding_run --model /path/to/Meta-Llama-3.1-8B-Instruct --output outputs/binding_new --reverse-participants
PYTHONPATH=.:teaching/state_audit/src python -m experiments.token_backtrace.binding_analyze --output outputs/binding_new
PYTHONPATH=.:teaching/state_audit/src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m experiments.token_backtrace.binding_controls --output outputs/binding_new --query-controls
```

`binding_controls` separates vector direction, norm/position, and attention.
Its query swaps keep all source tokens fixed: V7 must remain identical although
the queried-owner label reverses. This separates source-owner encoding from
source/query alignment in AV. If query observations already exist, resume CPU
fitting with `--cached-query-controls`; it performs no new model forwards.

The 2026-10-07 balanced experiment captured384 owner/order/style inputs and384
fixed-source query controls, then832 finite patches. Held-out source-owner
readability from full V and unit V was100% across both styles; norm/position was
81.25%/70.31%. Fixed-source query alignment was100% from AV across both styles;
V-only cannot exceed50% on those paired labels. A selected single head affected
candidate log-odds but never flipped the answer, and the effect was asymmetric.
These are auxiliary-supervised mechanism probes on synthetic facts, not natural
hallucination detection, an unsupervised classifier, or a new binding theory.
Raw results are retained under `outputs/binding_message_20261007_*`; canonical
plans, limitations and execution tracker are in shared research `refine-logs/binding_message_20261007/`.

## Source-conditioned sequence controls

`sequence.py` implements an unsupervised IOHMM with joint seven-channel
Gaussian observations and ordered behavior/support states. Shared covariance
is fixed from the fitting reference. State means solve a constrained
Mahalanobis problem; transition rows learn separate boundary/persistence
responses to lagged source and route changes. Exact forward-backward produces
token and boundary marginals; Viterbi explains state segments. State support
is an observational proxy, not a calibrated factual probability.

`sequence_run.py` fits source-balanced references on official training sources
with all 14 exposed control sources excluded. It selects checkpoints using
unlabelled development likelihood, freezes scores and mixture-quantile alarm
thresholds, then reads official annotations. Its comparisons include the
historical full-reference fixed score, a newly isolated fixed reference,
two-state HMM, homogeneous four-state HMM, no nuisance conditioning, and the
selected model's IID emission readout. Historical fixed and new references
have different exposure/sampling protocols and are reported separately.

Run from the repository root in the research Python environment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python main.py token sequence run-controls --output outputs/source_sequence_new
```

Individual `fit`, `score`, and `evaluate` stages use the same output directory;
`fit` requires a fresh path. The default pilot uses 32 fitting and 16 development
sources per task, three seeds for the main model and 12 iterations. The output
includes every valid control token, all fitted models, source lists, optimizer
history, frozen thresholds, per-case metrics and a complete token viewer.
This is a development diagnostic on previously exposed examples, with no
abstention or natural-label fitting, and does not change the default detector.

The first pilot exposed a scientific distinction: ordering Gaussian means
does not order their source likelihood ratios under a correlated covariance.
`--anchored-emissions` restricts the means to
`base + support * tau_source * Sigma * source_axis + behavior * tau_route * Sigma * route_axis`.
Positive coefficients retain mean orders and force source emission log odds
to increase along the source axis, with other emissions cancelling from that
within-behavior contrast. This is a local evidence-direction guarantee with
fixed neighboring messages, not a factual guarantee or monotonicity of the
complete input-driven chain under a source intervention. Only source and route
coordinates discriminate the anchored states; other channels contribute to
the shared density/reference rather than independent truth evidence.

`--unit-support` replaces the two raw source coordinates with existing
full/local unit means. Those historical units include future information;
they are observed preprocessing, not learned hallucination boundaries.
The raw/unit anchored runs are explicitly posthoc development iterations:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python main.py token sequence run-controls --seeds 42 --anchored-emissions --output outputs/source_sequence_anchor_raw_new
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python main.py token sequence run-controls --seeds 42 --anchored-emissions --unit-support --output outputs/source_sequence_anchor_unit_new
```

## 2026-10-06: typed-source interface pilots

The existing modules now also expose sparse native `(layer, head, query, key,
source_id, target)` amplitude-deletion derivatives. `messages.native_edge_trace`
keeps native head-output autograd live; detached Q/K/V define the observed edge
patch direction. `sparse_edge_vjp` computes an independent actual-token logp
backward per target without collapsing heads or source keys. `finite_edge_effect`
replays the same edge deletion through native descendants. These are influence
measurements, with noncausal pairs stored as NaN plus validity, and do not assign
factual truth. The physical grid pilot samples edges rather than all model edges.

The experimental relation parser separately reads complete source and answer
text. It saves raw greedy LM outputs, validates literal character quotes, keeps
up to three interpretations, and conservatively reports unresolved bindings.
Invalid records invalidate the document pair rather than silently dropping an
alternative. Raw parser outputs are frozen before constructing fixture truth
for evaluation. Natural unknown tokens remain in the full denominator. The
parser uses a frozen LM's semantic ability, and its token scopes are candidate
assertion scopes; they are not yet internally confirmed span members.

Run in the existing research environment from the repository root, using fresh
output paths (the historical outputs must remain intact):

```bash
python main.py token relations --stage typed-capture --output outputs/typed_parser_new
python main.py token relations --stage typed-evaluate --output outputs/typed_parser_new
python main.py token validate --stage sparse --keys 15604 9022 --output outputs/sparse_edges_new
```

The 2026-10-06 frozen parser pilot failed its gates: 18/32 controls correct,
with 0/8 interval and 0/6 condition cases correct, and 0/1487 natural tokens
nonunknown. Seven natural document pairs contain invalid/truncated parser
records; the remaining pair is usable syntactically but unresolved semantically.
Raw failures are retained in `outputs/typed_graph_20261006_v2`. No typed graph
detection or new full-test score was produced. See the shared 2026-10-06 tracker
for the next address/scoping interface proposal.

The sparse pilot uses `--vocabulary-chunk 8192`: the complete frozen BF16
unembedding is evaluated in FP32 output-row tiles via the existing FrozenLinear
operator. This keeps every vocabulary logit and avoids the full ~2 GiB temporary
FP32 vocabulary weight during backward. Input-gradient tile sums differ from
one GEMM only by FP32 roundoff; CPU logits/gradient tests compare them. The
historical loader default remains unchanged. All model-forward counts still
refer to complete decoder replays, not individual vocabulary tiles.

`typed-capture` defaults to 32 synthetic interface cases and all eight exposed
natural answers. `--constructed-only --case-limit 2` is an I/O smoke. Greedy
batch size and token cap are recorded. `typed-evaluate` reports case-family
correctness, unknowns, rejected records and complete natural token coverage;
it does not read natural hallucination labels or claim new detector AUROC.
Equal-length opposite/equivalent edits remain unmeasured, so the complete TG1
gate cannot pass yet. The sparse pilot measures all targets of each chosen
answer, then checks finite doses on the largest measured edge per sampled layer;
that selection is a fidelity diagnostic, not a representative stratum study.

# Independent token backtraces and exposed-case iterations

This experiment did **not** meet the detection objective. It preserves the
negative results and does not change the default detector.

`trace.py` uses a separate actual-token log-probability objective for every
target, backpropagating through native Q/K, RMS and SwiGLU to all original input
embeddings. Roots are embedding-gate sensitivities, not semantic evidence or a
conserved attribution. The model is a Llama-3.1-8B observer of existing answers.

The first readout iteration removes the historical unit broadcast and route
window. For each token, let `a` and `s` be its probabilities without and with
the source. `odds_full` uses `logit(a) - logit(s)` under the complete original
answer history. It combines a source-balanced empirical midrank of this
contrast with the raw route midrank using the historical fixed 3:1 weight.
`token_pair` and `odds_pair` also use a saved variable-unit history reset as
ablations; they never average neighbouring risks. The historical removal of
the source changes prompt positions. FP32-saturated complementary mass is
regularized at float32 epsilon, so this is not exact odds in that region.

The second iteration adds restricted copular polarity and duration interval
constraints with source witnesses. Unknown cases keep the native score;
recognized support/contradiction sets the corresponding token scope to 0/1.
The duration event alignment is heuristic, and clause support is not a proof
that the entire answer is correct. `logic_only` is a selective diagnostic:
unknown means no alarm, not verified correctness. Its fixed threshold is 0.5.
The hybrid inherits the native threshold and has no 5% false-alarm guarantee.

Both designs were informed by exposed examples. No natural-label classifier or
threshold fitting is used, but development is not label-blind. Four of the
eight cases belong to the official test set, which was already exposed in
earlier experiments. The full test is not a fresh holdout.

## Observed results

The pilot has 8 answers, 1487 tokens, and 134 official error-span members.
Historical fixed: 19 TP / 58 FP. Token pair: 23 / 77. Full-history odds:
25 / 111. Adding restricted logic: 42 / 111. Logic alone: 18 / 0 with 116
missed span members. These counts are not a separately defined key-token metric.

All 2700 official test answers / 424408 valid tokens were newly scored using
the cached native scalar measurements. Only the pilot has new root gradients.

| Task | Historical fixed AUROC | Odds AUROC | Odds + logic AUROC |
|---|---:|---:|---:|
| QA | 0.890665 | 0.799441 | 0.799592 |
| Summary | 0.752177 | 0.689958 | 0.690073 |
| Data2txt | 0.758040 | 0.620336 | 0.620343 |

The logic parser recognized only 302 tokens on the full test. Its 4 positive
alarms were all outside official spans. Full results include AP, false alarms,
within-answer AUROC, source-cluster bootstrap intervals, and custom exact
character-union/span-coverage metrics. Any-overlap span recall must not be
confused with complete-span coverage.

## Reproduction

Use the existing `research` Python environment from `.aris/compute/local.md`,
with `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4`,
`HF_HUB_OFFLINE=1`, and `TOKENIZERS_PARALLELISM=false`. Outputs below already
exist; preserve them and use fresh paths for new runs. The diagnostic currently
uses the recorded trace directories in `diagnose.py`.

```bash
python -m experiments.token_backtrace.trace --output outputs/token_backtrace_20260930_pilot --keys 15604 219
python -m experiments.token_backtrace.trace --output outputs/token_backtrace_20260930_controls --keys 11907 12015 12045 12219 9022 7305
python -m experiments.token_backtrace.validate --trace outputs/token_backtrace_20260930_pilot
python -m experiments.token_backtrace.contrast --output outputs/token_backtrace_contrast_20260930_v1
python -m experiments.token_backtrace.benchmark --stage fit --output outputs/token_backtrace_readout_20260930_v1
python -m experiments.token_backtrace.benchmark --stage pilot --output outputs/token_backtrace_readout_20260930_v1
python -m experiments.token_backtrace.diagnose --output outputs/token_backtrace_readout_20260930_v1
python -m experiments.token_backtrace.logic_benchmark --stage pilot --previous outputs/token_backtrace_readout_20260930_v1 --output outputs/token_backtrace_readout_20260930_v2
python -m experiments.token_backtrace.diagnose --output outputs/token_backtrace_readout_20260930_v2 --logic
python -m experiments.token_backtrace.benchmark --stage score-test --output outputs/token_backtrace_readout_20260930_v1
python -m experiments.token_backtrace.logic_benchmark --stage score-test --previous outputs/token_backtrace_readout_20260930_v1 --output outputs/token_backtrace_readout_20260930_v2
python -m experiments.token_backtrace.benchmark --stage evaluate-test --output outputs/token_backtrace_readout_20260930_v1
python -m experiments.token_backtrace.logic_benchmark --stage evaluate-test --previous outputs/token_backtrace_readout_20260930_v1 --output outputs/token_backtrace_readout_20260930_v2
python -m pytest -q experiments/token_backtrace/test_readout.py experiments/token_backtrace/test_logic.py
```

`pilot.html` and `pilot_tokens.csv` inspect every original token. The raw
`trace.npz`, numerical checks, counterfactual records, constraint witnesses,
frozen test NPZs, and `test_results.json` retain separate measurement scopes.
The shared handoff is `lys/codex/research/refine-logs/token_iteration_20260930/`.

## Whole-answer graph anomaly pilot, 2026-09-30

The existing `benchmark.py` now supports `graph-score` and `graph-evaluate`.
These reuse all eight answers' independent root gradients and saved final
normalized 4096-dimensional hidden states. There are no new LLM forwards.
Every original token receives a score. This is a cached-observer graph
anomaly experiment, not the full native-path/finite-intervention model.

The primary model is signed, directed conditional ridge reconstruction.
Node targets comprise a fixed seed-17 Gaussian projection of hidden states
to 32 dimensions plus NLL, entropy, negative margin, log root-gradient norm,
and positive/negative source fractions. Source-excluded standardization is
followed by equal weighting of hidden and scalar reconstruction losses.
Predictors include position, lexical format, task, signed history mass,
and separate positive/negative weighted ancestor attributes. The weights
are total root sensitivities; they are used for one-hop feature context,
never recursively multiplied as if they were direct neural edges. Feature
aggregation does not broadcast neighbouring risk scores.

Controls omit endpoint identities: conditional ridge retains signed history
mass covariates, and Isolation Forest uses node attributes only. Further
controls use expected endpoints within log2-lag/repeated-token-ID strata and three randomized
endpoint assignments (seeds 17/29/43). Nulls preserve receiver-stratum signed
mass; randomization also preserves weight multisets, but not every sender's
outgoing degree. They test endpoint identity beyond the retained lag/repetition
structure, not whether all graph structure is useless.

Fitting holds out the entire source. Inner source-held predictions from the
remaining sources calibrate an equal-source empirical percentile. Alarm is
percentile > .95; this is not a calibrated error probability or a guaranteed
normal FPR. Inner fits have six sources and final outer fits seven, so their
score distributions can differ. Only eight historically exposed sources
are available, including just one Summary source; task shift and small-data
effects remain. Natural labels are opened only in `graph-evaluate`.

```bash
python -m experiments.token_backtrace.benchmark --stage graph-score --output outputs/token_backtrace_readout_20260930_v2/graph_nodes_v1
python -m experiments.token_backtrace.benchmark --stage graph-evaluate --output outputs/token_backtrace_readout_20260930_v2/graph_nodes_v1
python -m pytest experiments/token_backtrace/test_readout.py -q
```

Run commands in the existing research environment with four BLAS/OpenMP
threads. Scoring refuses to overwrite its output. `graph.npz` preserves
full history/prompt root responses, attributes and predictor matrices;
`tokens.csv` and `TOKEN_GRAPH.html` expose every token, method score, raw
error, official span membership and top history/prompt roots. The common
79-token alarm budget is a ranking diagnostic separate from frozen alarms.
Shared protocol, execution, results and audit are in the existing
`lys/codex/research/refine-logs/gsm8k_states_20260929/TOKEN_GRAPH_*` files.

The exposed pilot contains 1487 valid tokens (134 official span members):
`graph_ridge` AUROC/AP = 0.516395/0.098217, TP7/FP60; `node_ridge`
0.520921/0.101166, TP8/FP58; historical `base` 0.808717/0.279095,
TP19/FP58. Neither a useful detector nor an endpoint-structure advantage
was established. All graph-vs-node/temporal/shuffled source-bootstrap
intervals include zero. Official span membership is not independently
annotated key-token truth.

One explicitly posthoc ablation preserves raw signed edge mass instead of
normalizing each channel. All five graph/null variants are reported, with
no label-based selection. `graph_mass` AUROC 0.541365 (TP9/FP61), expected
endpoints 0.535317, and randomized endpoints 0.538816/0.540507/0.539338 do
not establish an endpoint advantage either. Original files remain frozen.
The original seven methods' scores reproduce bit-exact after this optional
parameter refactor; see `refactor_verification.json`.

```bash
python -m experiments.token_backtrace.benchmark --stage graph-mass-score --output outputs/token_backtrace_readout_20260930_v2/graph_nodes_v1
python -m experiments.token_backtrace.benchmark --stage graph-mass-evaluate --output outputs/token_backtrace_readout_20260930_v2/graph_nodes_v1
```

Supplemental outputs use `mass_` prefixes in the same directory.
`graph_structure_diagnostics` computes exchangeable edge mass and rank
correlations; its tiny-mass cutoff is descriptive and never used to score.
Only five QA, one Summary and two Data2txt sources are available (four
official train and four official test). Source-held cross-fitting here is
an exploratory protocol, not the official held-out benchmark. The baseline
uses a larger historical calibration set; its comparison is same-token,
not equal-training-budget. No new full-test run or default replacement.

## 2026-10-06: source grounding and selective witnesses

`main.py token relations` now supports `grounding-capture`,
`grounding-evaluate`, and `witness-capture`. `--grounding-version pointer`
keeps the entire original response intact and addresses each original token
by character range. The Llama observer completes an A/B JSON decision;
its conditional A/B score and full-vocabulary choice mass are saved separately.
Use batch size 1: the BF16 batch-4 sanity changed choice log probabilities.
This is an external verifier with the full answer visible, not an internal
graph score, calibrated factual probability, or original-generator mechanism.

```bash
python main.py token relations --stage grounding-capture --grounding-version pointer --batch-size 1 --output outputs/token_grounding_20261006_pointer_v1
python main.py token relations --stage grounding-evaluate --output outputs/token_grounding_20261006_pointer_v1
python main.py token relations --stage witness-capture --previous outputs/token_grounding_20261006_pointer_v1 --output outputs/token_grounding_20261006_witness_v1
python main.py token relations --stage grounding-evaluate --score-field witness_score --output outputs/token_grounding_20261006_witness_v1
python main.py token relations --stage grounding-evaluate --score-field strict_score --output outputs/token_grounding_20261006_witness_v1
python main.py token relations --stage grounding-evaluate --score-field hybrid_score --output outputs/token_grounding_20261006_witness_v1
```

Captures require new directories. Evaluation reads official token/span
annotations only after a complete score freeze; repeated evaluation writes
the same deterministic metric files. `direct` and `audit` preserve the failed
historical prompt variants. Audit memos stop the capture if they reach the
generation cap without EOS. Reuse only copies complete, aligned answers
under an identical prompt/manifest, preserving the original raw scores and
recording zero new forwards for copied answers.

Selective witnesses extend the existing polarity/duration constraints with
restricted single-business attributes and explicit day/hour ranges, plus a
numbered-passage procedure denial check. Source fields, values, or exact
source quotes are saved in `claims.json`. Malformed/missing evidence and
unrecognized scope abstain. `witness_score` additionally alarms on numbered
steps explicitly self-disclaimed as unspecified in passages; this is a
heuristic, not an independent proof of missing source evidence.
`strict_score` removes that heuristic. `hybrid_score` overrides the frozen
pointer score only where a witness is recognized.

These schemas were informed by exposed development annotations. Rule
assertion spans are intersected with original token offsets; this is a
symbolic scope readout. Unknown tokens do not alarm in witness-only mode
and must not be described as verified correct. The preselected six-source
extension is an expanded regression on previously exposed data. Shared
plans, failures, reviews, costs, and results live under
`lys/codex/research/refine-logs/token_controls_20261006/`.
