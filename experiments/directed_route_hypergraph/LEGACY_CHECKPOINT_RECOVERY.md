# Legacy directed-route checkpoint recovery

The first `directed_route_hypergraph` prototype used a learned cross-head
transition tensor. Its 32-layer, 32-head encoder has **137,888 parameters** and
stores `head_transition_identity_bias` in `model_config`.

Later revisions replaced that architecture with deterministic ordered flow and
then the ordered endpoint-layout objective. These checkpoints are not
interchangeable. A process that fits with the prototype source and starts the
next Python process after the checkout changes fails with:

```text
TypeError: ModelConfig.__init__() got an unexpected keyword argument
'head_transition_identity_bias'
```

Do not delete that field and do not load the state dictionary into the current
encoder. The state dictionary belongs to a materially different model.

## Finish the existing 137,888-parameter run without fitting again

Pull the fix, then run the recovery entry point from the repository root:

```bash
git pull
conda run -n research bash \
  experiments/directed_route_hypergraph/resume_legacy.sh
```

The script validates the checkpoint, creates a temporary detached Git worktree
at commit `80acf557180132c95f4daac4417aa17219426a90`, and executes stages 2-5
with the exact source that can strictly load the legacy state dictionary. It
does not modify the current checkout and removes the temporary worktree on
exit.

Defaults match the QA path that produced
`outputs/qa/real_seed20260827/model.pt`. Override any path explicitly when
needed:

```bash
CHECKPOINT=/path/to/model.pt \
TRAIN_SPLIT=/path/to/train \
TEST_SPLIT=/path/to/test \
OUT=/path/to/recovered_outputs \
CUDA_VISIBLE_DEVICES=0 \
conda run -n research bash \
  experiments/directed_route_hypergraph/resume_legacy.sh
```

Recovered artifacts are written to `<checkpoint-directory>/legacy_recovery`
by default. `legacy_source_commit.txt` records the exact historical source.
Set `EVALUATE=0` to keep labels closed or `TEST_LIMIT=<N>` for a smoke test.

## Run the active ordered-layout method

The legacy result is a historical baseline, not a result for the active
ordered-layout model. Start the active method in a new output directory under a
stable checkout:

```bash
conda run -n research bash \
  experiments/directed_route_hypergraph/run_qa.sh
```

The current runner stores a source fingerprint before fitting and checks it
before and after every stage. A compatible interrupted run can resume with:

```bash
START_STAGE=2 conda run -n research bash \
  experiments/directed_route_hypergraph/run_qa.sh
```

Resumption is rejected when the source differs from the source that began the
run, preventing another cross-version checkpoint load.
