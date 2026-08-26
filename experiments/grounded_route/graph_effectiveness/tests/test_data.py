from pathlib import Path

import pytest

from experiments.grounded_route.graph_effectiveness.data import verify_bundle
from experiments.grounded_route.graph_effectiveness.tests.helpers import write_bundle


def test_saved_graph_bundle_is_verified_and_aligned(tmp_path):
    index_path = write_bundle(tmp_path / "real")
    bundle, report = verify_bundle(index_path)

    assert report["labels_read"] is False
    assert report["graphs"] == 5
    assert report["response_nodes"] == 25
    assert report["embedding_dimension"] == 8
    assert report["maximum_row_mass_error"] < 1e-2
    assert report["maximum_lineage_mass_error"] == 0.0
    assert len(bundle.records) == 5


def test_saved_graph_hash_change_is_rejected(tmp_path):
    index_path = write_bundle(tmp_path / "real")
    sidecar = next((index_path.parent / "graphs").glob("*.pt"))
    sidecar.write_bytes(sidecar.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="SHA-256"):
        verify_bundle(index_path)


def test_downstream_modules_are_strictly_node_only():
    package = Path(__file__).parents[1]
    for name in ("detectors.py", "model.py", "upper_bound.py"):
        source = (package / name).read_text(encoding="utf-8")
        assert "edge_index" not in source
        assert "edge_weight" not in source
        assert "message_passing" not in source
