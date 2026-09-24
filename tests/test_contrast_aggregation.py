"""CPU-only scientific checks for aggregation, ties, availability, and label isolation."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from state_audit.storage import read_arrays, read_json, write_arrays, write_json

from experiments.native_support.choice_cache import CaptureReader
from experiments.native_support.evidence_contrast.aggregation import (
    TOKEN_METHODS, WINDOW_CONTROLS, aggregate_scores, select_whole_units,
)
from experiments.native_support.evidence_contrast.run import main
from experiments.native_support.evidence_contrast.unit_budget import span_availability, unit_budgets
from experiments.native_support.evidence_contrast.unit_report import annotation_alignment, localization_metrics


def observations():
    saved = {name: np.array([0., 2., 9., 3.]) for name in (*TOKEN_METHODS, *WINDOW_CONTROLS)}
    saved.update(target=np.arange(4), token_id=np.arange(4), risk=np.arange(4.))
    views = dict(answer_ids=list(range(4)), units=[dict(start=0, stop=2), dict(start=2, stop=4)])
    return saved, views


class AggregationTests(unittest.TestCase):
    def test_means_preserve_original_scores_and_reject_misalignment(self):
        saved, views = observations()
        scored = aggregate_scores(saved, views)
        for name in TOKEN_METHODS:
            np.testing.assert_array_equal(scored[name + "_unit_mean"], [1, 1, 6, 6])
            np.testing.assert_array_equal(scored[name], saved[name])
        np.testing.assert_array_equal(scored["risk"], saved["risk"])
        np.testing.assert_array_equal(scored["unit_end_target"], [1, 1, 3, 3])
        with self.assertRaisesRegex(ValueError, "partition"):
            aggregate_scores(saved, {**views, "units": [dict(start=0, stop=3), dict(start=2, stop=4)]})
        with self.assertRaisesRegex(ValueError, "target"):
            aggregate_scores({**saved, "target": np.arange(1, 5)}, views)

    def test_within_unit_metric_detects_loss_of_localization(self):
        saved, views = observations()
        record = dict(units=views["units"], labels=np.array([0, 1, 1, 0]), valid=np.ones(4, dtype=bool))
        measured = localization_metrics([record], [aggregate_scores(saved, views)])["methods"]
        self.assertEqual(measured["source_local"]["pair_weighted_auroc"], 1.)
        self.assertEqual(measured["source_local_unit_mean"]["pair_weighted_auroc"], .5)
        self.assertEqual(measured["source_local"]["mixed_units"], 2)

    def test_boundary_ties_never_split_units(self):
        np.testing.assert_array_equal(select_whole_units([3, 3, 1, 0]), [True, True, False, False])
        self.assertTrue(select_whole_units([0, 0, 0]).all())

    def test_homogeneous_units_are_unavailable_not_a_localization_success(self):
        saved, views = observations()
        record = dict(units=views["units"], labels=np.array([0, 0, 1, 1]), valid=np.ones(4, dtype=bool))
        measured = localization_metrics([record], [aggregate_scores(saved, views)])
        self.assertEqual(measured["status"], "unavailable_no_mixed_units")
        self.assertIsNone(measured["methods"]["source_local_unit_mean"]["pair_weighted_auroc"])
        rows = [dict(fully_annotated=True, error_tokens=0, length=2),
                dict(fully_annotated=True, error_tokens=2, length=2),
                dict(fully_annotated=False, error_tokens=1, length=2)]
        alignment = annotation_alignment(rows)
        self.assertEqual(alignment["fully_annotated_units"], 2)
        self.assertFalse(alignment["within_unit_localization_identifiable"])

    def test_retrospective_onset_has_nonzero_availability_delay(self):
        record = dict(id="a", valid=np.ones(4, dtype=bool), labels=np.array([1, 1, 0, 0]),
                      onsets=np.array([1, 0, 0, 0], dtype=bool))
        units = [dict(start=0, stop=4)]
        row = span_availability(record, units, [True], "source_local")[0]
        self.assertTrue(row["onset_unit_flagged"])
        self.assertEqual(row["earliest_score_delay"], 3)
        self.assertFalse(row["score_available_before_span_end"])

    def test_missing_truth_does_not_remove_a_unit_from_the_ranking_budget(self):
        _, views = observations()
        record = dict(id="a", response_length=4, units=views["units"],
                      valid=np.array([1, 0, 1, 1], dtype=bool), labels=np.zeros(4, dtype=int),
                      onsets=np.zeros(4, dtype=bool))
        rows = [dict(response_id="a", unit_id=i, fully_annotated=bool(i), error_tokens=0,
                     **{name: 2-i for name in TOKEN_METHODS}) for i in range(2)]
        budget, _, alarms = unit_budgets([record], rows)
        measured = budget["methods"]["source_local"]
        self.assertEqual(measured["selected_incomplete_units"], 1)
        self.assertEqual(measured["false_positive_units"], 0)
        self.assertAlmostEqual(measured["normal_token_fpr"], 1/3)
        self.assertTrue(next(row for row in alarms if row["method"] == "source_local")["selected"])

    def test_zip_cpu_pipeline_writes_scores_before_reading_labels_and_keeps_input(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "input", root / "analysis"
            saved, views = observations()
            response = dict(id="a", source_id="s", prompt_length=1,
                            token_ids=[9, 0, 1, 2, 3], token_text=["Q", "A", ".", "B", "."])
            write_json(source / "settings.json", dict(responses=[response], model="never_loaded"))
            write_json(source / "protocol.json", dict(version="evidence-contrast-v1"))
            write_json(source / "responses/0000/views.json", views)
            write_arrays(source / "responses/0000/scores.npz", **saved)
            truth = dict(a=dict(source_id="s", token_ids=[0, 1, 2, 3], labels=[0, 1, 1, 0]))
            write_json(source / "annotations.json", truth)
            archive = root / "input.zip"
            with ZipFile(archive, "w") as bundle:
                for path in source.rglob("*"):
                    if path.is_file():
                        bundle.write(path, path.relative_to(source))
            original_archive = archive.read_bytes()
            original_read = CaptureReader.bytes

            def check_label_order(reader, name):
                if name == "annotations.json":
                    self.assertTrue((output / "responses/0000/scores.npz").is_file())
                return original_read(reader, name)

            args = ["--stage", "analyze", "--input", str(archive), "--output", str(output)]
            with patch.object(CaptureReader, "bytes", check_label_order):
                main(args)
            self.assertEqual(archive.read_bytes(), original_archive)
            first = read_arrays(output / "responses/0000/scores.npz")
            truth["a"]["labels"] = [1, 0, 0, 1]
            write_json(root / "flipped.json", truth)
            main([*args, "--resume", "--annotations", str(root / "flipped.json")])
            repeated = read_arrays(output / "responses/0000/scores.npz")
            for name in first:
                np.testing.assert_array_equal(first[name], repeated[name])
            self.assertEqual(read_json(output / "protocol.json")["new_model_forwards"], 0)
            with ZipFile(root / "analysis_review.zip") as bundle:
                self.assertTrue({"units.csv", "within_unit.json", "span_availability.csv",
                                 "unit_budget.json", "aggregation.png"} <= set(bundle.namelist()))


if __name__ == "__main__":
    unittest.main()
