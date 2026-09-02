# Evidence-conditioned route state

This project asks a narrower question than generic hallucination detection:

> When a response route becomes narrow, is it still carrying prompt evidence,
> or has it become a persistent feedback loop rooted only in response history?

The distinction matters because narrow routing is often correct. A summary may
copy one document span, a QA answer may extract one entity, and Data2txt may
verbalize one field. Route contraction becomes the proposed risk mechanism only
when prompt-carried evidence and evidence-rooted response relay give way to
unrooted response feedback.

`METHOD.md` is the frozen mathematical specification. Code follows the same
order as the research argument:

1. `data.py` reconstructs the exact historical prompt and assigns prompt tokens
   to evidence units. It never opens hallucination labels.
2. `capture.py` runs one chunked teacher-forced forward pass. It captures every
   predictor query, including prompt queries needed for multi-hop ancestry.
3. `messages.py` reconstructs each head/source attention write
   `A * W_O^h V` in FP32 derived arithmetic and checks it against the native
   attention update.
4. `graph.py` stores a compact response graph with explicit layer, query, head,
   and source endpoints. Sparse tails enter one endpoint-free `unknown` account.
5. `lineage.py` uses every dense source/head account—not the sparse graph—to
   propagate boundary ancestry over the causal computation DAG.
   Response-history messages are split into evidence-rooted relay and unrooted
   feedback; predictor self remains separate.
6. `state.py` retains the earlier route-collapse equation and calibrates
   lineage route volume against task-specific position and length. The two
   state coordinates are calibrated contraction and unrooted takeover.
7. `detector.py` fits a label-free sticky three-state model and emits the online
   filtered posterior of `captured` state.
8. `evaluate.py` is the only module allowed to open labels. It compares the
   frozen score with confidence, the earlier route-collapse control, and graph
   controls using source-cluster bootstrap intervals.
9. `run.py` only orders these operations. It contains no scientific formula.

## What is exact, and what is operational

For one layer and query, the transient AVWO vectors exactly reconstruct the
model's attention residual write, conditional on the observed attention gate.
The artifact saves their scalar capacity/support accounts and exact endpoints,
not full hidden-size vectors. Head and source identities are preserved.
Multi-layer lineage is an operational propagation rule over positive
constructive support; it is not a Shapley value or a claim that the replay
observer reproduces the generator's original internal mechanism.

MLP writes are recorded only as same-token diagnostics. They are not called
"parameter knowledge": a single forward pass cannot uniquely separate stored
knowledge from input-triggered computation.

## Primary result and controls

The primary token score is

```text
P(captured at t | observations through t)
```

where observations are position/length-calibrated route contraction and
unrooted-history takeover. The absolute attainable-capacity deficit saved at
capture time is only a raw diagnostic. No hallucination label chooses the
coordinates' direction, weights, state count, or model parameters. The first
two answer tokens are excluded because strict history feedback is not yet
identifiable.

The earlier QA route-collapse score remains an equation-locked control: it
keeps the f7344e2 feature equation, lower-volume direction, source-equal
nuisance fit, robust scale, and position-wise ECDF. It is not a numerical
reproduction of the old five-fold run. The current physical train/test split
is inviolable, so sorted sources on each training side use a fixed modulo-four
nuisance-fit/calibration partition (approximately 3:1) before the opposite
physical side is scored. A useful result requires the ancestry-conditioned
state to improve not only QA detection, but also rejection of correct
narrow-focus Summary and Data2txt tokens. The independent-token, one-hop,
endpoint-rewiring, and weight-shuffling controls are refitted separately and
compared on paired held-out tokens. Learned transition probabilities and dwell
times are saved, so a sticky prior cannot stand in for measured persistence.
Route collapse alone is neither necessary nor sufficient for hallucination.

## Run

The one-click entry remains foreground-only:

```bash
bash experiments/evidence_route_state/run_all.sh
```

The shell file contains only:

```bash
python -m experiments.evidence_route_state.run all
```

Captures are written per sample under `outputs/<observer>/captures/`; fitted
fold models, contraction calibrations, and persistence diagnostics under
`models/`; task reports and frozen token scores under `reports/`.
`run_metadata.json` records plain-text
observer, generator, dtype, cache, chunk, and graph-storage provenance without
hashing files or scanning model directories.

The persisted `K=2` response graph is an inspection artifact and can occupy
tens of gigabytes over the full corpus. The dense detector is evaluated online
before sparsification and is independent of `K`; check server disk space before
a full run.

Use `--limit 1` through the Python entry for a pipeline smoke test only; its
single source necessarily shares nuisance-fit and ECDF-calibration data, so its
metrics are not scientific results. A completed capture is resumed by the
presence of its per-sample NPZ; the project intentionally has no file-hash
identity chain or schema migration layer.

## Scientific gates

Before a full run, tests must establish:

- query `q` predicts token `q + 1`;
- per-head/source messages reconstruct the native attention write;
- BF16 replay never mixes FP32 operands with BF16 projection weights;
- GQA and exact head/source endpoints survive capture;
- a `prompt -> response1 -> response2` path is grounded relay;
- the same response feedback without prompt ancestry is unrooted;
- narrow but evidence-rooted focus is not named captured;
- omitted sparse mass stays unknown;
- graph construction, state fitting, and scoring are invariant to labels.

Passing these gates proves that code implements the specification. It does not
by itself prove that the proposed mechanism detects hallucinations.
