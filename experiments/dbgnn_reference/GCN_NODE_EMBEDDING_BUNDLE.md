# Compact GCN node data

`package_gcn_embeddings.sh` creates one compressed file containing only the
frozen node representations and their row-aligned binary node labels:

```text
gcn_node_data_qa.npz
├── node_embeddings  [node, 64] float32
└── node_labels      [node] int8
```

Each label is the binary hallucination annotation of the response-token node
in the same row: `0` is negative and `1` is positive.

```python
import numpy as np

data = np.load("gcn_node_data_qa.npz")
x = data["node_embeddings"]
y = data["node_labels"]

print(x.shape)
print(y.shape)
```
