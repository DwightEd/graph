# Ordered source transport research

This is a source-derived self-supervised candidate readout. It does not use
natural hallucination labels for fitting and is not a calibrated unsupervised
natural hallucination detector. Inputs end before the current candidate.
Unverified draft cues are crossed with source swaps and role queries. These are
controlled prefix choices, not self-generated clean-first-error trajectories.

`observe.py` retains eight prediction positions at all layers, with 4096
coordinates at each site: residual before the layer, attention write, MLP write,
and native source-only attention write after W_O. Unprojected source messages
retain all 32 physical heads with 128 coordinates each. The entire prefix is
forwarded; the eight-position horizon applies only to the readout. Head vectors
and source attention mass are diagnostics, not additional fitted inputs.

`readout.py` fits a restricted shared-coordinate readout with position/site
weights and coordinatewise products of neighboring states. These products are
statistical interactions, not a native Jacobian or physical transport operator.
No PCA, rank truncation, token rarity, density scoring, or span-wide score
broadcasting is used. The candidate direction is a normalized frozen lm_head
row. Fit-only coordinate normalization is shared over the time axis.

Single/mean views collapse lag weights despite sharing the nominal parameter
structure. A wider current-only linear/square baseline has 8449 parameters,
versus ordered6017. Shuffling retains the current row and the past-row multiset,
but relocates the draft cue. Dropping source_write is a feature ablation:
residual/attention/MLP still contain source information. None of these controls
alone establishes native causal transport.

Boolean targets come from official source JSON. QA targets come from literal
quote ownership and containment in the official passages, not factual entailment.
Source swaps, two roles, two task families, and both independent draft cues are
crossed. Fit/dev sources are disjoint and exclude the two exposed natural
regression sources. Control metrics are source-program metrics. Risk AUROC/AP
are null when the group has no positive/negative native errors.

Run from the repository root with the existing research environment:

```bash
PYTHONPATH=.:teaching/state_audit/src OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
-m experiments.flow_latent.ordered_source_transport.run \
--stage collect --cohort full --domain mixed \
--reuse outputs/ordered_source_transport_full_20261008 \
--output outputs/ordered_source_transport_mixed_v2_20261008

PYTHONPATH=.:teaching/state_audit/src OMP_NUM_THREADS=4 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
-m experiments.flow_latent.ordered_source_transport.run \
--stage fit --cache outputs/ordered_source_transport_mixed_v2_20261008 \
--output outputs/ordered_source_transport_mixed_fits_20261008 \
--steps 400 --seeds 42 123 2026 \
--variants single mean ordered shuffled drop_source linear wide_single

PYTHONPATH=.:teaching/state_audit/src OMP_NUM_THREADS=4 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
-m experiments.flow_latent.ordered_source_transport.natural \
--stage transfer --cache outputs/ordered_source_transport_natural_20261008 \
--fits outputs/ordered_source_transport_mixed_fits_20261008 \
--output outputs/ordered_source_transport_mixed_natural_transfer_20261008

PYTHONPATH=.:teaching/state_audit/src OMP_NUM_THREADS=4 \
/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python \
-m experiments.flow_latent.ordered_source_transport.audit \
--cache outputs/ordered_source_transport_mixed_v2_20261008 \
--fits outputs/ordered_source_transport_mixed_fits_20261008
```

Outputs must be new directories. Collection and fitting protocols hash input
and implementation files; executed-code archives bind the code used. Scores and
weights are frozen before evaluation. CPU NumPy metric/score checks and a fresh
same-family provisional scientific audit are recorded separately. Two natural
pairs are manually chosen, previously exposed diagnostics, not automatic
candidate proposal or an independent natural test.

Canonical design, tracker, negative results, and claim decisions:
`/share/home/tm902089733300000/a903202310/lys/codex/research/refine-logs/ordered_source_transport_20261008/`.
