# Compact GCN node data

`package_gcn_embeddings.sh` creates one compressed file containing only the
frozen node representations and their held-out test labels:

```text
gcn_node_data_qa.npz
├── calibration_embeddings  [calibration_node, 64] float32
├── test_embeddings         [test_node, 64] float32
└── test_labels             [test_node] int8
```

The label in row `i` belongs to `test_embeddings[i]`. Fit an unsupervised
detector using only `calibration_embeddings`, freeze it, score
`test_embeddings`, and open `test_labels` only for evaluation.

```python
import numpy as np

data = np.load("gcn_node_data_qa.npz")

x_calibration = data["calibration_embeddings"]
x_test = data["test_embeddings"]
y_test = data["test_labels"]

print(x_calibration.shape)
print(x_test.shape)
print(y_test.shape)
```
