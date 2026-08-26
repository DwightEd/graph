# GroundedRoute saved-embedding effectiveness audit

This subpackage evaluates the final node representations already written by
GroundedRoute. Edge/neighbour aggregation remains inside GroundedRouteEncoder.
All representation detectors and representation probes consume only
`node_embedding`; the explicitly named offline position baselines consume only
`token_index` and `response_length` as nuisance controls.

The current method uses PCA-whitened kNN. This audit adds a fixed suite of
node-only unsupervised detectors and an isolated source-grouped supervised
readability ceiling. See [METHOD.md](METHOD.md) for the evidence rules.

## Run one saved real-graph output

```bash
REPO=/share/home/tm902089733300000/a903202310/lys/research/graph
GROUND_OUT=${REPO}/experiments/grounded_route/outputs/qa
TEST_SPLIT=/share/home/tm902089733300000/a903202310/lys/data/RAGTruth/attention/llama31_8b/test
OUT=${GROUND_OUT}/graph_effectiveness

cd "${REPO}" || exit 1

CUDA_VISIBLE_DEVICES=0 \
PYTHON=python \
CALIBRATION_INDEX="${GROUND_OUT}/calibration/index.npz" \
GRAPH_INDEX="${GROUND_OUT}/test/index.npz" \
UNSUPERVISED_SCORES="${GROUND_OUT}/scores.npz" \
TEST_SPLIT="${TEST_SPLIT}" \
OUT="${OUT}" \
DEVICE=cuda \
FOLDS=5 \
EPOCHS=20 \
SEEDS="20260825 20260826 20260827" \
bash experiments/grounded_route/graph_effectiveness/run.sh
```

The repository also includes `run_qa.sh` with the cluster paths above, so the
same run can be started directly after adjusting `GROUND_OUT` if necessary.

`UNSUPERVISED_SCORES` is optional. It includes the already frozen PCA-kNN
score in the report without using it for training.

## Add full-pipeline construction controls

First run GroundedRoute separately with the registered `no_message`,
`endpoint_rewire` and `weight_shuffle` controls. Then append their already
encoded bundles:

```bash
CONTROL_ARGUMENTS="\
--control no_message /path/to/no_message/calibration/index.npz /path/to/no_message/test/index.npz \
--control endpoint_rewire /path/to/rewire/calibration/index.npz /path/to/rewire/test/index.npz \
--control weight_shuffle /path/to/shuffle/calibration/index.npz /path/to/shuffle/test/index.npz" \
bash experiments/grounded_route/graph_effectiveness/run.sh
```

These comparisons occur after each construction has passed through its own
trained encoder. Rewiring a sidecar beside a fixed real embedding is not used
as evidence that the construction works.

`run_controls_qa.sh` contains the complete four-pipeline QA run followed by
the aligned audit. It uses separate output directories, so the variants cannot
overwrite one another. It also keeps the canonical labels closed during all
four encoder/detector runs; labels are opened once, after every score artifact
has been frozen. `ENCODER_SEED` and `BASE` can be overridden to repeat the
complete matched experiment under independent encoder seeds; one run alone is
reported as diagnostic rather than paper-ready evidence.

## Explicit commands

```bash
python -m experiments.grounded_route.graph_effectiveness.run verify \
  --index /path/to/real/test/index.npz \
  --output /path/to/audit/integrity.json \
  --topology-output /path/to/audit/label_free_topology.npz

python -m experiments.grounded_route.graph_effectiveness.run audit \
  --calibration /path/to/real/calibration/index.npz \
  --index /path/to/real/test/index.npz \
  --test /path/to/canonical/test \
  --scores /path/to/real/scores.npz \
  --output /path/to/audit \
  --device cuda --folds 5 --epochs 20 \
  --seeds 20260825 20260826 20260827
```

## Outputs

```text
integrity.json             label-free artifact/conservation audit
label_free_topology.npz    mechanism-only layer/head alignment tensors
unsupervised_scores.npz    frozen representation and position-control scores, without labels
oof_predictions.npz        source-disjoint probe predictions, without labels
report.json                metrics, paired deltas and interpretation scope
```

The supervised ceiling must not replace the unsupervised AUROC/AUPRC in the
main experiment table.
