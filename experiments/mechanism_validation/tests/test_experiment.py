import json

import torch

from cache import AttentionSample
from experiments.mechanism_validation import experiment


class _Sample:
    def __init__(self, sample_id, source_id, attention):
        self.sample_id = sample_id
        self.source_id = source_id
        self.task_type = "QA"
        self.data_source = "MARCO"
        self._attention = attention

    def attention(self):
        return self._attention

    def release_attention(self):
        self._attention = None


class _Dataset:
    def __init__(self, sample):
        self.sample = sample

    def __iter__(self):
        yield self.sample


def _attention():
    return AttentionSample(
        "sample", "source", 2, torch.arange(4, dtype=torch.int32),
        torch.zeros((1, 1, 4), dtype=torch.float16),
        torch.tensor([0, 1, 2], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([.4, .3], dtype=torch.float16), .01,
    )


def _mechanism_features(path):
    path.mkdir()
    torch.save({
        "sample_id": "sample", "source_id": "source",
        "task_type": "QA", "data_source": "MARCO",
        "values": torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]]),
        "valid": torch.ones((2, 4), dtype=torch.bool),
    }, path / "sample.pt")
    (path / "metadata.json").write_text(json.dumps({
        "schema": "mechanism_features.v2", "ema_decay": .9, "attention_floor": .01,
        "cache_bound_invalid_rows": 0, "cache_bound_total_rows": 2, "cache_bound_invalid_fraction": 0.0,
        "labels_included": False,
        "feature_names": [
            "routing_a:global_mean", "mass_a:global_mean",
            "concentration_a:global_mean", "locality_a:global_mean",
        ],
        "family_slices": {
            "routing": [0, 1], "mass": [1, 2],
            "concentration": [2, 3], "locality": [3, 4],
        },
    }))
    (path / "index.json").write_text(json.dumps({"samples": [
        {"sample_id": "sample", "source_id": "source", "tokens": 2}
    ]}))


def test_evaluation_loads_all_label_free_artifacts_before_opening_labels(tmp_path, monkeypatch):
    train, test, output = tmp_path / "train", tmp_path / "test", tmp_path / "out"
    _mechanism_features(train)
    _mechanism_features(test)
    events = []
    real_load = experiment.load_feature_split

    def tracked_load(path, **kwargs):
        events.append(f"features:{path.name}")
        return real_load(path, **kwargs)

    class LabelDataset:
        def __getitem__(self, sample_id):
            return _Sample("sample", "source", _attention())

        def __iter__(self):
            return iter([_Sample("sample", "source", _attention())])

        def labels(self):
            events.append("labels")
            return type("Labels", (), {"response_labels": lambda _, sample: torch.tensor([0, 1])})()

    monkeypatch.setattr(experiment, "load_feature_split", tracked_load)
    monkeypatch.setattr(experiment, "open_research_dataset", lambda *args, **kwargs: LabelDataset())

    experiment.evaluate_mechanisms("train-split", train, "test-split", test, output, bootstrap=2)

    assert events.index("labels") > events.index("features:train")
    assert events.index("labels") > events.index("features:test")
    result = json.loads((output / "results.json").read_text())
    assert result["probe_uses_labels"] is True
    assert result["analysis_status"] == "post_hoc_exploratory"
    assert result["adjusted_global_mean"]


def test_build_graph_writes_identical_node_features_for_every_variant(tmp_path, monkeypatch):
    features, output = tmp_path / "features", tmp_path / "graphs"
    _mechanism_features(features)
    dataset = _Dataset(_Sample("sample", "source", _attention()))
    monkeypatch.setattr(experiment, "open_research_dataset", lambda *args, **kwargs: dataset)

    experiment.build_graphs(
        "split", features, output, device="cpu", variants=["exact", "no_edges", "source_free"],
    )

    base = torch.load(output / "base" / "sample.pt", weights_only=True)
    exact = torch.load(output / "exact" / "sample.pt", weights_only=True)
    no_edges = torch.load(output / "no_edges" / "sample.pt", weights_only=True)
    source_free = torch.load(output / "source_free" / "sample.pt", weights_only=True)
    assert base["values"].shape == (2, 4)
    assert base["values"].dtype == torch.float16
    assert exact["values"].dtype == torch.float16
    assert exact["values"].shape == no_edges["values"].shape == source_free["values"].shape
    assert set(exact) == {"sample_id", "source_id", "values"}
    assert base["valid"].all()
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["labels_included"] is False


def test_evaluate_graphs_end_to_end_and_requires_every_variant_file(tmp_path, monkeypatch):
    def write_graph(root):
        root.mkdir(); (root / "base").mkdir()
        variants = ["exact", "no_edges", "rp_only", "rr_only"]
        for variant in variants: (root / variant).mkdir()
        samples = []
        for number, sample_id in enumerate(("a", "b")):
            samples.append({"sample_id": sample_id, "source_id": sample_id, "tokens": 2})
            base = {"sample_id": sample_id, "source_id": sample_id, "prompt_length": 2, "task_type": "q", "data_source": "d", "values": torch.tensor([[.1], [.2]]), "valid": torch.ones((2, 1), dtype=torch.bool)}
            torch.save(base, root / "base" / f"{sample_id}.pt")
            for index, variant in enumerate(variants):
                torch.save({"sample_id": sample_id, "source_id": sample_id, "values": torch.tensor([[number + index, 1.], [1., number + index]], dtype=torch.float)}, root / variant / f"{sample_id}.pt")
        (root / "metadata.json").write_text(json.dumps({"schema":"graph_features.v2", "labels_included":False, "seed":0, "variants":variants, "feature_names":["g0", "g1"], "node_feature_names":["n"], "source_aware":[False, True], "mechanism_fingerprint": {}}))
        (root / "index.json").write_text(json.dumps({"samples":samples}))
    train, test, output = tmp_path / "train", tmp_path / "test", tmp_path / "out"
    write_graph(train); write_graph(test)
    monkeypatch.setattr(experiment, "_load_labels", lambda root, features: (features.positions % 2).astype(int))
    result = experiment.evaluate_graphs("x", train, "y", test, output, bootstrap=3)
    assert set(result["representation_sufficiency"]) == set(result["decoder_sensitivity"]) == {"exact", "no_edges", "rp_only", "rr_only"}
    assert len(result["paired_cluster_intervals"]) == 3
    for deltas in (result["representation_point_deltas"], result["decoder_point_deltas"]):
        assert {"auroc", "auprc"} <= set(deltas["exact_minus_no_edges"]["full"])
    for interval in result["paired_cluster_intervals"].values():
        for metric in interval.values():
            assert {"point", "ci_low", "ci_high", "valid_replicates"} <= set(metric)
    assert (output / "predictions.npz").exists() and (output / "results.json").exists()
    (train / "no_edges" / "a.pt").unlink()
    import pytest
    with pytest.raises(ValueError): experiment._graph_split(train, "no_edges")
