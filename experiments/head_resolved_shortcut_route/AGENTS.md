# Head-resolved shortcut-route project rules

Read `../grounded_route/iclr/ENGINEERING_RULES.md` and `METHOD.md` before
changing this directory.  The rules below define the current audit.

## Computation being measured

- A response token at position `p` is evaluated only from its causal predictor
  `q = p - 1`.  Persist both coordinates; the embedding at `p` must never enter
  its own route graph.
- Use one frozen, native teacher-forced forward.  The primary audit does not use
  knockout branches, generated alternatives, a learned encoder, an
  autoencoder, a GNN, or a fitted feature combiner.
- Every non-self edge is the actual residual write

      A[l,h,q,s] W_O[l,h] W_V[l,g(h)] RMS(x[l,s])

  with source, query, prediction, layer, query head, and GQA mapping intact.
  Never average heads or layers before per-edge support/veto and dispersion are
  computed.

## Root and carrier are different coordinates

- Maintain the additive observed-gate roots `E/Q/R/N`: evidence, remaining
  prompt/question, prior response embeddings, and numeric closure.  `N` is not
  unresolved semantics, MLP state, parametric knowledge, or a hallucination
  class.
- Independently retain physical carriers `evidence_prompt`, `other_prompt`, and
  `response_history`.  A sparse tail is not a fourth carrier.
- The named projections are:
  - `D = E × evidence_prompt`;
  - `P_E = E × other_prompt`;
  - `G = E × response_history`;
  - `B = R × response_history`;
  - `Q` retains its carrier subtype;
  - `I` is predictor-input injection through the same-position suffix;
  - `N` is numerical variation only.
- Persist the complete root×carrier table.  Do not force `P_E` or other valid
  cells into `D/G/B/N` merely to obtain an exhaustive class label.

## Three fixed mechanism objects

- `carrier_drift`: support and veto maps over `[token, layer, head]`, plus the
  preregistered prompt-versus-response depth-centroid summary.
- `prompt_source_dispersion`: normalized per-head source entropy computed from
  signed physical functional atoms; aggregate only after the per-head
  nonlinearity.
- `response_born_takeover`: within response carriers, the `R`-root share of
  `E/Q/R` support or veto.  High response mass with low takeover is a grounded
  relay, not a shortcut.
- Keep all three separate.  Do not tune a weighted sum, flip directions after
  labels, or rename response-born support as parametric bias.

## Sparse persistence and validity

- Select physical edges independently within each `(token, layer, head)` using
  target-independent post-`W_O` message norm.  Keep the smallest prefix reaching
  `rho=0.95`, capped by `K=64`, with source index as the stable tie break.
- Store four root contributions as columns of one physical edge.  Never count
  its attention, value energy, or message norm once per root.
- Tail statistics retain carrier×root positive/negative mass and physical
  `sum(x log x)` so the three axes close exactly.  Do not invent tail endpoints.
- Undefined values are `NaN` with an explicit false mask.  Absolute `N`
  variation and closure error enter the resolution threshold but never a
  scientific numerator or denominator.

## Labels, outputs, and claims

- Capture, sparse selection, axes, directions, and validity masks are
  label-free.  Collection must not retain, expose, or consult hallucination
  labels; only task-specific final evaluation requests them from the dataset
  interface.
- Save one NPZ per sample.  Report QA, Summary, and Data2txt separately with
  token-micro AUROC, sklearn AP, source-cluster bootstrap intervals, and
  position/length/confidence controls.
- Treat target log-probability as observer surprisal, not generator confidence
  when observer and generator differ.  Keep all non-graph controls out of the
  three mechanism axes and report each axis on its own validity mask.
- This is an observed-computation attribution audit.  It is not a complete
  causal graph and does not identify parameter knowledge.  Causal necessity or
  sufficiency requires a separate re-forward intervention with recomputed
  gates.
- The retired four-branch registers, Gram score, prompt-collapse statistics,
  and SFAC conflict are historical controls only.  They must not enter the
  three mechanism objects.

## Code and tests

- `route_capture.py` owns the native one-pass hook capture.  The older
  four-branch audit remains isolated in the sibling
  `../attention_mechanism_audit/` package and must not be imported here.
- `route_suffix.py` owns the same-position observed suffix adjoint.
- `route_shortcut.py` owns true-message atoms, the three axes, and sparse tails.
- `route_pipeline.py` assembles captured operators into one artifact.
- `route_artifact.py` owns label-free NPZ persistence.
- `collect.py` owns label-free traversal, atomic journaling, and exact resume.
- `evaluate.py` is the only phase that requests labels; canonical token
  rebinding and `frozen_axes.npz` must complete before embedded labels are
  enabled.
- `run.py`/`run_all.sh` are the only public execution entry.  A limited capture
  is always marked as a partial smoke run.
- Tests must include GQA AVWO oracles, attention/root/margin closure,
  `q -> q+1`, self/MLP suffix, head-before-nonlinearity/entropy/sign-split
  ordering, D/G/B relay counterexamples, NaN masks, target-independent
  selection, exact tails, roundtrip persistence, label isolation, per-axis
  validity, independent controls, and partial/full collection integrity.
