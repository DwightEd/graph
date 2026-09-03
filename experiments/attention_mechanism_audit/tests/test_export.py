import json

import numpy as np
import torch

from experiments.attention_mechanism_audit.export import export_nodes


def test_export_concatenates_nodes_without_labels(tmp_path):
    root = tmp_path / "state"
    (root / "samples").mkdir(parents=True)
    graph = {
        "node_embedding": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "response_start": 2,
        "token_ids": torch.tensor([7, 8, 9, 10]),
    }
    torch.save(graph, root / "samples" / "a.pt")
    (root / "index.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "a",
                "source_id": "s",
                "task_type": "QA",
                "path": "samples/a.pt",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    path = export_nodes(root, tmp_path / "nodes.npz", "QA")
    arrays = np.load(path)
    np.testing.assert_allclose(arrays["embedding"], [[1, 2], [3, 4]])
    assert arrays["sample_id"].tolist() == ["a", "a"]
    assert arrays["target_token_id"].tolist() == [9, 10]
