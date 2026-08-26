# GCN node-embedding bundle

This bundle contains the frozen 64-dimensional node representations produced
by the label-free first-order GCN experiment on RAGTruth QA.

## Files

```text
calibration/index.npz  train-split response nodes for fitting a detector
test/index.npz         test-split response nodes for frozen scoring
reference/             optional scores and reports from the repository run
calibration/graphs/    optional per-sample graph sidecars
test/graphs/           optional per-sample graph sidecars
```

The two `index.npz` files have the same row contract:

```text
embedding          [node, 64] float32
sample_id          [node]
source_id          [node]
task_type          [node]
token_index        [node]
response_length    [node]
response_token_id  [node]
```

Only `embedding` is a model feature. The remaining arrays identify the sample
and response token represented by each row.

The package contains no hallucination labels. Fit every unsupervised detector
on `calibration/index.npz` only, freeze it, and then score
`test/index.npz`. Test labels may be opened separately after scores have been
returned or frozen.

```python
import numpy as np

calibration = np.load("calibration/index.npz")
test = np.load("test/index.npz")

x_calibration = calibration["embedding"]
x_test = test["embedding"]

print(x_calibration.shape)
print(x_test.shape)
```

When the optional `graphs/` directories are included, every `.pt` sidecar also
contains the typed attention edges and the same GCN `node_embedding` aligned to
all prompt-response tokens. They are not needed for node-only detection.
