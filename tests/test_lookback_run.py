import json

import numpy as np

from experiments.unsupervised_token_graph.lookback_run import run


def test_lookback_run_freezes_label_free_artifacts(tmp_path):
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