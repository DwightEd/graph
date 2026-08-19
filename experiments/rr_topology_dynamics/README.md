# Prompt-grounded routing-attractor audit

This experiment tests one focused mechanism hypothesis: hallucination is
associated with a simple, stable response-history attractor that loses prompt
grounding. It is a post-hoc mechanism audit, not a detector selected after
reading test labels.

## Method

The cache stores response queries with exact retained prompt and earlier-response
source identities. For every generated token, the extractor first builds:

```text
prompt_source_mass[response_token, prompt_token]
response_source_mass[response_token, earlier_response_token]
```

All layer/head edges are accumulated by exact source identity. Missing cache
entries remain censored below `attention_floor`; they are not treated as exact
zeros. Prompt query rows are unavailable, but prompt source identities for every
response query are available and retained.

The core feature matrix has eight predeclared columns:

| Group | Feature | Hypothesized hallucination direction |
|---|---|---|
| Prompt support | `prompt_attention_share` | lower |
| Prompt concentration | `prompt_source_effective_fraction` | lower |
| Prompt concentration | `prompt_source_top1_share` | higher |
| Response concentration | `response_source_effective_fraction` | lower |
| Response concentration | `response_source_top1_share` | higher |
| Local response feedback | `recent_response_share` | higher |
| Routing stability | `source_stability` | higher |
| Prompt provenance | `prompt_groundedness` | lower |

`prompt_source_effective_fraction` divides the entropy-effective source count by
prompt length. The response version divides by the number of legal earlier
response sources. `source_stability` is one minus the normalized Jensen-Shannon
distance between consecutive exact-source distributions.

Prompt groundedness is propagated causally:

```text
direct_t   = prompt_mass_t / retained_mass_t
relay_t    = sum_j p(response_source=j | t) * grounded_j
grounded_t = direct_t + (1 - direct_t) * relay_t
```

This distinguishes a token that has no direct prompt edge but follows a grounded
response relay from a token inside an unsupported response-only feedback loop.

No arbitrary product of the eight signals is declared as a detector. Evaluation
reports each predeclared direction at matched task/position and at the first
hallucination onset.

## Controls and diagnostics

Two cache-quality controls are stored separately and never standardized as core
features:

```text
retained_attention_mass
retained_edge_count
```

The previous PCA reconstruction family is diagnostic-only. It lives in
`spectral_diagnostics.py` and produces:

```text
spectral_residual_energy
layer_residual_energy
spectral_rank_residual_energy
rr_embedding
```

These fields do not enter the eight-column attractor feature matrix.

## Code structure

```text
routing_state.py
  sparse attention blocks -> exact prompt/RR source mass

attractor.py
  concentration, stability, and causal prompt grounding

spectral_diagnostics.py
  optional frozen-PCA residual diagnostics

extractor.py
  core TopologyDynamicsExtractor.extract(sample) -> TopologyExtraction

experiment.py
  label-blind fit; score explicitly adds spectral diagnostics

evaluation.py
  labels opened only after the score artifact is frozen

artifacts.py
  strict v4 artifact validation and provenance
```

The core extraction cost is linear in retained sparse edges plus the exact-source
state size `R * (P + R)`, where `P` and `R` are prompt and response lengths.

## Protocol

```text
train attention + frozen spectral reference
    -> fit task/position scales for eight core features (labels sealed)

test attention
    -> freeze core features, controls, and spectral diagnostics (labels sealed)

frozen artifact
    -> feature, within-response, phase, and first-onset analysis (labels opened)
```

The artifact contracts are:

```text
rr-topology-dynamics-reference-v4
rr-topology-dynamics-features-v4
rr-topology-dynamics-evaluation-v4
```

## Run

The RR spectral reference is used only for diagnostics and provenance. The run
script builds it automatically when it does not exist and reuses it on later runs.

Local Windows full run from PowerShell:

```powershell
cd D:\projects\python_projects\research\graph
& "C:\Program Files\Git\bin\bash.exe" experiments/rr_topology_dynamics/run.sh
```

The script detects the local RAGTruth directory, the checked Python environment,
and whether CUDA is available. Each topology run gets a timestamped directory;
every stage writes a log below its `logs/` subdirectory. A failed stage exits
non-zero and prints the exact log path.

Smoke test:

```bash
LIMIT=5 bash experiments/rr_topology_dynamics/run.sh
```

Full audit:

```bash
bash experiments/rr_topology_dynamics/run.sh
```

Important environment variables:

```text
ROOT                  dataset root containing train/ and test/
SPECTRAL_REFERENCE    frozen RR spectral reference.npz
OUT                   output directory
RECENT_LAG_MAX        recent-response lag threshold, default 4
RUN_TESTS             run focused preflight tests, default 1
```

Outputs:

```text
reference.npz
test_features.npz
evaluation/report.json
evaluation/feature_metrics.csv
evaluation/control_metrics.csv
evaluation/within_sample_effects.csv
evaluation/onset_effects.csv
evaluation/phase_curves.csv
evaluation/layer_metrics.csv
evaluation/spectral_rank_metrics.csv
evaluation/residual_correlations.csv
```

## Claim boundary

The method observes retained attention routing, not factual truth, confidence,
or the complete sub-floor attention distribution. A convincing error-attractor
claim requires the joint directional pattern to repeat across tasks and sources:

```text
less and more concentrated prompt support
+ more concentrated, local response reuse
+ higher routing stability
+ lower prompt groundedness
```

Correct responses may also converge. Concentration or stability alone is not
evidence of hallucination.
