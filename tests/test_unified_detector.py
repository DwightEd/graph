"""Source-scale integrity, constrained matrix inference, and immutable CPU cache scoring."""

from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pytest
from scipy import sparse

from state_audit.storage import read_arrays, read_json, write_json
from experiments.native_support.choice_cache import CaptureReader
from experiments.native_support.evidence_contrast.unit_budget import span_availability
from experiments.native_support.message_carriers.run import main as carriers_main
from experiments.native_support.unified.calibration import fit_distribution, transform
from experiments.native_support.unified.data import edge_block, read_dataset
from experiments.native_support.unified.graph import dependence_laplacian, solve_residual, unit_operator
from experiments.native_support.unified.run import main, reference_records
from test_evidence_contrast import tiny_model
from test_message_carriers import make_contrast_input


def test_calibration_balances_sources_and_does_not_turn_ties_into_different_scores():
    fitted = fit_distribution([0., 0., 1.], ["long", "long", "short"])
    repeated = fit_distribution([0.] * 20 + [1.], ["long"] * 20 + ["short"])
    query = np.array([-.1, 0., .5, 1., 1.1])
    np.testing.assert_allclose(transform(query, fitted), [0., .25, .5, .75, 1.])
    np.testing.assert_allclose(transform(query, fitted), transform(query, repeated))


def test_no_graph_and_zero_regularization_recover_exact_within_unit_routing():
    units = [dict(start=0, stop=3), dict(start=3, stop=4)]
    route = np.array([.1, .4, .7, .9])
    zero = sparse.csr_matrix((4, 4))
    residual, centered, diagnostic = solve_residual(route, units, zero, 1)
    np.testing.assert_allclose(residual, [-.3, 0., .3, 0.], atol=1e-15)
    np.testing.assert_allclose(residual, centered, atol=1e-15)
    assert diagnostic["constraint_max_error"] < 1e-15
    arbitrary = sparse.diags([1., 2., 3., 4.])
    np.testing.assert_allclose(solve_residual(route, units, arbitrary, 0)[0], residual, atol=1e-15)


def test_graph_changes_token_residuals_preserves_unit_means_and_keeps_parallel_head_effects():
    edges = dict(target=np.array([2, 2, 3]), key=np.array([0, 0, 1]), sham_key=np.array([1, 1, 0]),
        effect_with_source=np.array([2., -2., 1.]), effect_without_source=np.zeros(3))
    laplacian, adjacency = dependence_laplacian(edges, 4)
    assert adjacency[2, 0] > 0  # Opposing physical heads do not cancel before abs.
    np.testing.assert_allclose(laplacian.toarray(), laplacian.T.toarray())
    assert np.linalg.eigvalsh(laplacian.toarray()).min() > -1e-12
    units = [dict(start=0, stop=2), dict(start=2, stop=4)]
    residual, centered, diagnostic = solve_residual(np.array([0., 1., .3, .8]), units, laplacian, 1)
    assert np.linalg.norm(residual - centered) > .01
    np.testing.assert_allclose(unit_operator(units, 4) @ residual, 0, atol=1e-14)
    assert diagnostic["stationarity_max_error"] < 1e-14
    shuffled, _ = dependence_laplacian(edges, 4, True)
    assert not np.allclose(shuffled.toarray(), laplacian.toarray())
    tiny = {**edges, "effect_with_source": edges["effect_with_source"] * 1e-8}
    assert dependence_laplacian(tiny, 4)[1].sum() < 1e-7


def test_v1_bundle_and_v2_exact_query_edge_axes_remain_distinct():
    keep = np.array([[-2., -3.], [-4., -5.]])
    effects = np.array([[[1., 2.], [3., 4.]], [[.1, .2], [.3, .4]]])
    v1 = dict(keep=keep, single=keep[:, None, :] - effects,
              edges=np.array([[0, 2, 0], [1, 3, 1]]), sham_edges=np.array([[0, 2, 1], [1, 3, 0]]))
    old = edge_block(v1, np.array([2, 3]), False)
    np.testing.assert_array_equal(old["target"], [2, 3, 2, 3])
    np.testing.assert_array_equal(old["effect_with_source"], [1, 2, 3, 4])
    np.testing.assert_array_equal(old["receiver"], [-1] * 4)
    v2 = dict(keep=keep, single=keep[:, None, :] - effects,
              edges=np.array([[0, 2, 2, 0], [1, 3, 1, 1]]), sham_edges=np.array([[0, 2, 2, 1], [1, 3, 1, 0]]))
    new = edge_block(v2, np.array([3]), True)
    np.testing.assert_array_equal(new["effect_with_source"], [1, 3])
    np.testing.assert_array_equal(new["receiver"], [2, 1])
    np.testing.assert_array_equal(new["layer"], [0, 1])


def test_whole_answer_graph_is_not_reported_as_an_online_onset_alarm():
    record = dict(id="a", valid=np.ones(4, dtype=bool), labels=np.array([1, 1, 0, 0]),
                  onsets=np.array([1, 0, 0, 0], dtype=bool))
    units = [dict(start=0, stop=1), dict(start=1, stop=4)]
    row = span_availability(record, units, [True, False], "unified", offline=True)[0]
    assert row["onset_unit_flagged"]
    assert row["earliest_score_delay"] is None
    assert not row["score_available_before_span_end"]
    assert row["score_scope"] == "completed_answer_and_reference"


@pytest.mark.parametrize("mode", ["unit", "token"])
def test_cpu_pipeline_both_native_cache_versions_label_isolation_and_complete_ablations(tmp_path, mode):
    source, capture, output = tmp_path / "input", tmp_path / "capture", tmp_path / "joint"
    pieces, annotations = make_contrast_input(source)
    class Tokenizer:
        def decode(self, ids):
            return pieces.get(ids[0], str(ids[0]))
    with patch("state_audit.model.load_model", return_value=(tiny_model(), Tokenizer())):
        carriers_main(["--mode", mode, "--input", str(source), "--output", str(capture),
                       "--device", "cpu", "--dtype", "float32", "--top-k", "2"])
    archive = capture.with_name("capture_review.zip")
    original_archive = archive.read_bytes()
    original_bytes, original_json = CaptureReader.bytes, CaptureReader.json
    def read_bytes(reader, name):
        if name == "annotations.json":
            assert (output / "coverage.json").exists()
            assert all((output / f"responses/{i:04d}/scores.npz").exists() for i in range(2))
        return original_bytes(reader, name)
    def read_json_without_truth(reader, name):
        assert name != "annotations.json"
        return original_json(reader, name)
    args = ["--input", str(archive), "--output", str(output)]
    with patch("state_audit.model.load_model", side_effect=AssertionError("CPU cache stage loaded model")), \
            patch.object(CaptureReader, "bytes", read_bytes), patch.object(CaptureReader, "json", read_json_without_truth):
        main(args)
    first = read_arrays(output / "responses/0000/scores.npz")
    original = read_arrays(capture / "responses/0000/scores.npz")
    for name, values in original.items():
        np.testing.assert_array_equal(first[name], values)
    components = read_arrays(output / "responses/0000/components.npz")
    views = read_json(output / "responses/0000/views.json")
    operator = unit_operator(views["units"], len(first["target"]))
    np.testing.assert_allclose(operator @ (first["unified"] - components["source_anchor"]), 0, atol=1e-14)
    np.testing.assert_array_equal(first["unified_unit_mean"], components["source_anchor"])
    metrics = read_json(output / "evaluation.json")["methods"]
    assert {"unified", "unified_no_graph", "unified_random_graph", "unified_local_anchor",
            "source_consensus", "flat_fusion", "raw_route", "source_local_unit_mean"} <= set(metrics)
    for annotation in annotations.values():
        annotation["labels"] = [1 - value for value in annotation["labels"]]
    changed = tmp_path / "flipped.json"
    write_json(changed, annotations)
    main([*args, "--resume", "--annotations", str(changed)])
    repeated = read_arrays(output / "responses/0000/scores.npz")
    for name, values in first.items():
        np.testing.assert_array_equal(values, repeated[name])
    assert archive.read_bytes() == original_archive
    with ZipFile(output.with_name("joint_review.zip")) as bundle:
        assert {"ranking_scope.json", "calibration/route.npz", "responses/0000/edges.npz",
                "responses/0000/components.npz", "unit_budget.json"} <= set(bundle.namelist())
    with pytest.raises(ValueError, match="settings changed"):
        main([*args, "--resume", "--graph-strength", "2"])
    _, protocol, records = read_dataset(archive)
    with pytest.raises(ValueError, match="overlaps target sources"):
        reference_records(archive, records, protocol)
