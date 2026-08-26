from types import SimpleNamespace

import numpy as np

from experiments.grounded_route.artifacts import load_npz
from experiments.grounded_route.graph_effectiveness.audit import (
    AuditConfig,
    _position_features,
    _save_score_artifact,
)
from experiments.grounded_route.graph_effectiveness.tests.helpers import write_bundle
from experiments.grounded_route.graph_effectiveness.views import load_embedding_views


def test_frozen_detector_scores_carry_full_dataset_and_variant_binding(tmp_path):
    calibration = load_embedding_views(
        {"real": write_bundle(tmp_path / "calibration", split="train")}
    )
    test = load_embedding_views(
        {"real": write_bundle(tmp_path / "test", split="test")}
    )
    path = tmp_path / "scores.npz"
    _save_score_artifact(
        path,
        calibration,
        test,
        {
            "pca_knn__real": np.arange(25, dtype=np.float32),
            "position_pca_knn": np.arange(25, dtype=np.float32),
        },
        AuditConfig(),
    )
    artifact = load_npz(path)

    assert not bool(artifact["labels_included"].item())
    assert not bool(artifact["labels_read"].item())
    assert artifact["dataset_manifest_sha256"].item() == "a" * 64
    assert artifact["audit_scope"].item() == "selected_samples"
    assert artifact["calibration_index_sha256"].shape == (1,)
    assert artifact["test_index_sha256"].shape == (1,)
    assert artifact["graph_variant"].tolist() == ["real"]
    assert artifact["message_mode"].tolist() == ["neighbor"]
    assert artifact["nuisance_score_names"].tolist() == ["position_pca_knn"]
    assert artifact["pca_knn__real"].shape == (25,)


def test_position_nuisance_features_do_not_read_node_embeddings():
    index = SimpleNamespace(
        token_index=np.asarray((0, 1, 2)),
        response_length=np.asarray((3, 3, 3)),
    )
    features = _position_features(index)

    assert np.allclose(features[:, 0], (0.0, 0.5, 1.0))
    assert np.allclose(features[:, 1], np.log(4.0))
