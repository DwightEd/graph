# Fresh agent follows documentation: adoption smoke, 2026-09-12

The fresh smoke agent read `run-experiment/SKILL.md`, the shared compute
environment contract, `graph/.aris/compute/local.md`, and the adoption protocol's
local invocation. It executed the following command verbatim, once, from
`/share/home/tm902089733300000/a903202310/lys/research/reanchor`:

```bash
bash scripts/run_adoption_probe.sh --queries 128 --output outputs/adoption_smoke_20260912
```

Exit status: **0**. No code edits, installations, fixes, reruns, deletions, or git
actions were performed. The output directory did not exist before invocation.
Preflight reported GPU 0, NVIDIA GeForce RTX 4090, 1 MiB used of 24564 MiB. The
canonical environment spec hash was `8d044d57`, matching the latest ledger block.
No `AGENTS.md` or `CLAUDE.md` was found in the working directories or ancestors.

## Summary and numerical checks

Artifact directory: `reanchor/outputs/adoption_smoke_20260912`.

| Check | Observed value |
|---|---:|
| `summary.json`: cases / expensive message queries | 1 / 1 |
| Query | 128 |
| Elapsed seconds reported by runner | 50.88707685470581 |
| Peak GPU memory bytes | 17738558976 (16.5203204 GiB) |
| Sham maximum logit error | 0.0 |
| `numerical_checks_only` | true |
| Scientific review | REVIEW_UNAVAILABLE |
| Token groups / candidate token IDs | 53 / 145 |
| Message reconstruction maximum absolute error | 0.010499000549316406 |
| Message reconstruction relative L2 error | 0.0017028589500114322 |
| Suffix surrogate maximum logit error | 0.08088302612304688 |
| Native / surrogate selected margin | 1.875 / 1.8643150329589844 |
| Surrogate top-1 matches native | true |
| Attenuation rho=0.1: mean absolute / relative L1 error | 0.00012061004963470623 / 0.04402468726038933 |
| Attenuation rho=1: mean absolute / relative L1 error | 0.004069026559591293 / 0.1345495581626892 |

All three conditions contain 196 prediction entries. For both cuts, the maximum
logit error before activation is exactly zero. Before prediction 134, saved-token
log probability changes and argmax flips are zero; JS entries equal the sham's
entries exactly. The sham has no saved-token log probability changes or argmax
flips anywhere. The JS computation reports roundoff despite exactly equal sham
logits: maximum sham JS is `2.8978185184769245e-08` nats.

| Condition | Mean JS for predictions >=134 (nats) | Argmax flips >=134 | Mean saved-token log probability change >=134 |
|---|---:|---:|---:|
| Sham | 3.9571484948866355e-09 | 0 | 0.0 |
| History cut [119,133) | 0.013999681907294483 | 3 | -0.13363271345243602 |
| Earlier cut [85,99) | 0.004563790189790878 | 3 | -0.013416353246582495 |

All five manifest hashes match their files. Every numerical JSON value is finite.
NPZ arrays load with `allow_pickle=False`; all values are finite:
`group_messages (53,4096)`, `candidate_group_head_contributions (144,53,32)`,
`candidate_logits (145,)`, and `native_candidate_logits (145,)`.

## Documentation versus reality

The invocation, saved case, single requested message query, model/environment,
bf16/eager native inference, fp32 suffix surrogate, and both history interventions
match the documentation and output metadata. No execution-blocking discrepancy
was observed. The protocol explicitly anticipates reconstruction, surrogate, and
finite attenuation mismatch; these nonzero values are measurements, not a claim
that those approximations are exact. No numerical acceptance threshold is supplied
by the invocation section, so this report does not invent one.

The existing environment spec's `run_commands` and ledger validation entries still
refer to the earlier binding-validation runner. They substantiate reuse of the
same package and weight environment, but do not yet record this adoption smoke.
The parent can append the new validation evidence to the ledger. This smoke is
restricted to one previously inspected diagnostic case and one message query;
it does not establish the full pilot, factual detection quality, or independent
scientific approval.
