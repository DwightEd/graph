import json

import numpy as np

from experiments.unsupervised_token_graph.reanchor_run import run


def test_reanchor_run_freezes_label_free_artifacts(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    attention = np.zeros((1, 1, 6, 8), dtype=np.float32)
    for t in range(6):
        q = 3 + t - 1
        attention[0, 0, t, :q + 1] = 1 / (q + 1)
    np.savez(cache / "1.npz", attention=attention, prompt_length=3, source_id="s1")
    rows = run(cache, tmp_path / "out")
    assert rows[0]["events"] >= 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["config"]["labels_read"] is False
    with np.load(tmp_path / "out" / "1.npz", allow_pickle=False) as saved:
        assert "reanchor_ratio" in saved.files
        assert saved["reanchor_ratio"].shape == (6,)


def test_reanchor_evaluation_reads_renamed_artifacts(tmp_path):
    import hashlib

    from experiments.unsupervised_token_graph.reanchor_evaluate import evaluate

    cache = tmp_path / "cache"
    cache.mkdir()
    attention = np.zeros((1, 1, 6, 8), dtype=np.float32)
    for t in range(6):
        q = 3 + t - 1
        attention[0, 0, t, :q + 1] = 1 / (q + 1)
    np.savez_compressed(cache / "1.npz", attention=attention, prompt_length=3,
                        source_id="s1")
    response = "abcdef"
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(json.dumps({"trace": "1.npz", "source_id": "s1",
        "response_sha256": hashlib.sha256(response.encode()).hexdigest()}) + "\n")
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(json.dumps({"id": "1", "source_id": "s1", "response": response,
        "offsets": [[i, i + 1] for i in range(6)],
        "labels": [{"start": 2, "end": 4}]}) + "\n")
    run(cache, tmp_path / "out", metadata=metadata)
    result = evaluate(tmp_path / "out", metadata, annotations)
    assert result["tokens"] == 6
    assert result["positives"] == 2
    assert "reanchor_ratio" in result["metrics"]
