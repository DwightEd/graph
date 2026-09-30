# Automatic evidence-span experiment

Inputs are full original prompts/responses, source-format masks, cached per-target full-history embedding-gradient summaries (positionwise norm and gradient·embedding, not full Jacobian tensors or gradient vectors) and full current-query message derivatives. No manual evidence, entities, semantic constraints, gold answer boundaries or correctness labels enter preparation/capture/scoring. These eight answers are previously exposed development diagnostics, not new blind evaluation.

Source candidates cover every source token and are split only by text punctuation. Source root norm and centered per-head message effects propose associations. Every candidate is then excluded as keys at all layers in a new native Llama-3.1-8B forward, retaining positions and teacher-forced answer text. Save complete token × source-span full-vocabulary JS, signed log-probability changes, final-hidden changes and layer/head message changes. Attribution is influence under a specified intervention, not entailment or the original generator's mechanism.

Output regimes are variable-length groups detected from adjacent source-profile JS and hidden/message changes. They only organize provenance: no risk averaging or score broadcasting. Fixed token scores are frozen before evaluation. Localized probability contrast is a testable proxy, not a solution to polarity/role binding. Full-vocabulary JS and automatic source ablation have direct precedent in ARC-JSD; this is a reuse/combination experiment, not a claim of a new attribution primitive.

Run from graph, with the existing environment; no installs:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m pytest experiments/automatic_evidence/test_evidence.py -q
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.automatic_evidence.prepare --output outputs/automatic_evidence_20260930_v2
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.automatic_evidence.capture --output outputs/automatic_evidence_20260930_v2
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.automatic_evidence.score --output outputs/automatic_evidence_20260930_v2
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.automatic_evidence.evaluate --output outputs/automatic_evidence_20260930_v2
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.automatic_evidence.report --output outputs/automatic_evidence_20260930_v2
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.automatic_evidence.verify --output outputs/automatic_evidence_20260930_v2
```

Preparation requires a fresh output directory. Capture refuses completed answers. Run as attached tool-managed processes, not detached jobs. Raw results remain under outputs; canonical plan, literature, execution and results are under `/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/automatic_evidence_20260930/`.

v1 contains preparation only. A token-boundary lookahead bug could split numbers before decimal points; it was corrected and tested before model capture. v1 is preserved, v2 is the corrected frozen experiment. Neither variant used gold for preparation.

The completed v2 has 304 source spans, 1487 targets and 352 native forwards. Primary detection failed: TP8/FP71, AUROC .552876; historical fixed on the same answers is .808717. Joint attribution did not outperform attention. The first robust state-boundary rule yielded only nine output regimes.

A separate second iteration uses source addresses as latent states, verified profiles as observation factors and hidden/message changes as a time-varying refresh hazard. Offline Viterbi creates variable-length evidence regimes, retaining each token's independent contrast. It never pools risk. Run only after v2:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.automatic_evidence.state_path --input outputs/automatic_evidence_20260930_v2 --output outputs/automatic_evidence_state_20260930_v1
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.automatic_evidence.evaluate_path --input outputs/automatic_evidence_20260930_v2 --output outputs/automatic_evidence_state_20260930_v1
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.automatic_evidence.report --output outputs/automatic_evidence_20260930_v2 --state outputs/automatic_evidence_state_20260930_v1
```

This second experiment produced 172 regimes but also failed detection (TP7/FP72, AUROC .594150). Regimes are not validated semantic boundaries. Both outputs and negative results are preserved; no default replacement or new full-test run. See [actual results](RESULTS.md).

State-input provenance: hidden adjacency uses this run's `effects.npz:original_hidden`. Message adjacency reuses `outputs/span_maintenance_20260929_v2/<key>/payload_changes.npz:head_value_change[:,0,...]`: source-group, pre-W_O adjacent message-vector changes, aggregated across physical layers/heads by RMS. The producer is `experiments/span_maintenance/payload.py`, reading `outputs/route_complement_20260928/<key>/head_vectors.npy`. Newly captured per-source intervention `effects.npz:message_change` is diagnostic-only and does not enter scoring or transitions. The state output includes a post-audit `state_input_provenance.json` inventory and SHA256 hashes for all eight reused payloads; this is not a pre-run freeze.

## Continue in this module: automatic relation controls

The existing prepare/capture/score/evaluate/report entry points now accept a relation run. The runner reuses the old source capture and native measurement functions; it does not copy the old experiment into another code package. Original outputs stay immutable; new raw arrays go in a run-specific child directory. Generic boolean/negation/bounded-duration edits and repeated numeric field-value sets use no manual evidence positions. Controls are assumed paraphrases/orthographic equivalents, not certified semantic labels; text edits can change prompt length. Uncovered selected sources cause explicit abstention, not a correctness verdict.

From graph, use the existing environment:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.automatic_evidence.prepare --mode relation --input outputs/automatic_evidence_20260930_v2 --output outputs/automatic_evidence_20260930_v2/relation_v1
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.automatic_evidence.capture --output outputs/automatic_evidence_20260930_v2/relation_v1
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.automatic_evidence.score --output outputs/automatic_evidence_20260930_v2/relation_v1
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.automatic_evidence.evaluate --output outputs/automatic_evidence_20260930_v2/relation_v1
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.automatic_evidence.report --output outputs/automatic_evidence_20260930_v2/relation_v1
```

Preparation requires a new destination. Capture refuses completed answers. Reruns must use a fresh result destination; code stays in this module. Canonical continuation protocol: shared `refine-logs/automatic_evidence_20260930/RELATION_PLAN.md`.

### Control-quality iteration with exact measurement reuse

The archived `relation_v1/executed_python.zip` preserves the first executed implementation. Current code uses ordinary case controls and adds an automatic exact-boundary alternative for every bounded duration. The primary takes the maximum corrected relation gain within the selected source; the mean remains an ablation. This design followed exposed development diagnostics. No output-token risk averaging or label fitting occurs.

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.automatic_evidence.prepare --mode relation --input outputs/automatic_evidence_20260930_v2 --reuse-effects outputs/automatic_evidence_20260930_v2/relation_v1 --output outputs/automatic_evidence_20260930_v2/relation_v2
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.automatic_evidence.capture --output outputs/automatic_evidence_20260930_v2/relation_v2
```

Then run the existing score/evaluate/report/verify commands with `--output outputs/automatic_evidence_20260930_v2/relation_v2`. Reuse requires exactly matching original prompt/answer, source addresses and replacement token IDs; changed replacements alone trigger new forwards. Metadata separates actual new forwards, reused conditions and reused shams. No source/evidence arrays or historical scores are overwritten.

### CPU comparison without new capture or a new directory

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.automatic_evidence.score --output outputs/automatic_evidence_20260930_v2/relation_v2 --readout all
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.automatic_evidence.evaluate --output outputs/automatic_evidence_20260930_v2/relation_v2 --readout all
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false /share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python -m experiments.automatic_evidence.report --output outputs/automatic_evidence_20260930_v2/relation_v2 --readout all
```

This creates separate `all_*` outputs beside the original scores; frozen all-candidate scoring refuses reruns. The exploratory primary uses raw log-probability contrast over all candidates; odds and message-gated variants are separate. It followed the exposed v2 result and is not a blind confirmation. Current code/source archives distinguish execution versions: v1/v2 main `input_freeze.json` describes their captured source archives, while `all_readout_input_freeze.json` describes the later optional CPU extension. Historical hash verification requires its archived implementation; do not expect evolving code to match an old freeze.

Actual continuation: 116 + 44 new 8B forwards, 74 condition forwards reused in the second run, 19 final tests. Main v2 AUROC .514989; exploratory all-candidate raw-logp .474507; historical fixed .808717. Local key-token recovery did not produce a better full detector. See the continuation section in [RESULTS.md](RESULTS.md). Future iterations should extend these functions rather than create parallel experiment packages.
