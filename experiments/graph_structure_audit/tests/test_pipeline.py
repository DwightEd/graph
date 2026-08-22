import json

import numpy as np

from experiments.graph_structure_audit import evaluate, extract
from experiments.graph_structure_audit.artifacts import load_npz
from experiments.graph_structure_audit.config import GraphAuditConfig

from .helpers import SyntheticDataset, SyntheticSample


def test_extract_then_evaluate_keeps_labels_out(tmp_path, monkeypatch):
    dataset = SyntheticDataset(
        [
            SyntheticSample("sample-a", [0, 0, 1, 0]),
            SyntheticSample("sample-b", [0, 1, 0, 0]),
        ]
    )
    monkeypatch.setattr(extract, "_open_dataset", lambda split_root: dataset)
    monkeypatch.setattr(evaluate, "_open_dataset", lambda split_root: dataset)

    token_path = extract.extract_graph_audit(
        split_root="test",
        output_dir=tmp_path / "audit",
        config=GraphAuditConfig(
            prompt_bins=4,
            minimum_sources_for_recovery=2,
            minimum_channels_for_recovery=2,
            show_progress=False,
        ),
    )
    arrays = load_npz(token_path)
    assert not bool(arrays["labels_included"].item())
    assert arrays["structural"].shape[0] == 8
    assert len(list((tmp_path / "audit" / "graphs").glob("*.npz"))) == 2

    evaluate.evaluate_graph_audit(
        split_root="test",
        token_path=token_path,
        output_dir=tmp_path / "evaluation",
        bootstrap_replicates=10,
        seed=3,
    )
    report = json.loads((tmp_path / "evaluation" / "evaluation.json").read_text())
    assert report["labels_read"] is True
    assert report["tokens"] == 8
    assert (tmp_path / "evaluation" / "feature_metrics.csv").is_file()
    assert (tmp_path / "evaluation" / "recoverability_hypotheses.csv").is_file()
    assert np.isfinite(arrays["structural"]).all()
