# Automatic evidence-span experiment

Inputs are full original prompts/responses, source-format masks, cached independent-token root gradients and full current-query message derivatives. No manual evidence, entities, semantic constraints, gold answer boundaries or correctness labels enter preparation/capture/scoring. These eight answers are previously exposed development diagnostics, not new blind evaluation.

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
