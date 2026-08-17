from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.conditioned_benchmark.conditions import (
    aggregate_responses,
    prevalence_weights,
)
from experiments.conditioned_benchmark.runner import BenchmarkConfig, run_benchmark
from experiments.conditioned_benchmark.metrics import evaluate_metrics
from experiments.conditioned_benchmark.types import BenchmarkFrame, MethodScore


def frame():
    return BenchmarkFrame(
        sample_id=np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"]),
        token_index=np.tile(np.arange(2), 4),
        methods={
            "perfect": MethodScore(
                "perfect",
                np.asarray([0.0, 0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6]),
                protocol="label_free_frozen_score",
            )
        },
        source_id=np.asarray(["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"]),
        task_type=np.asarray(["QA", "QA", "QA", "QA", "Summary", "Summary", "Summary", "Summary"]),
        data_source=np.asarray(["MARCO"] * 8),
        generator_model=np.asarray(["llama"] * 8),
        response_length=np.asarray([2] * 8),
        relative_position=np.tile(np.asarray([0.0, 1.0]), 4),
        labels=np.asarray([0, 1, 0, 1, 0, 1, 0, 1]),
    ).validate()


class ConditionedRunnerTests(unittest.TestCase):
    def test_prevalence_reweighting_hits_target(self):
        labels = np.asarray([0, 0, 0, 1])
        weights = prevalence_weights(labels, 0.5)
        self.assertAlmostEqual(float(np.average(labels, weights=weights)), 0.5)

    def test_operational_alert_metrics_are_registered(self):
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.0, 0.1, 0.8, 0.9])
        values = evaluate_metrics(
            labels,
            scores,
            np.ones(4),
            ("precision_at_alert_10", "recall_at_alert_10", "f1_at_alert_10"),
        )
        self.assertGreater(values["precision_at_alert_10"], 0.0)
        self.assertGreater(values["recall_at_alert_10"], 0.0)
        self.assertGreater(values["f1_at_alert_10"], 0.0)

    def test_response_aggregation(self):
        result = aggregate_responses(frame(), aggregation="max")
        self.assertEqual(len(result.labels), 4)
        np.testing.assert_array_equal(result.labels, np.ones(4, dtype=np.int8))
        np.testing.assert_allclose(
            result.methods["perfect"].values,
            np.asarray([0.9, 0.8, 0.7, 0.6]),
        )

    def test_task_and_ratio_grid_writes_tidy_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            config = BenchmarkConfig(
                task_types=("all", "each"),
                positive_rates=("native", 0.25),
                metrics=("auroc", "auprc", "auprc_lift"),
                bootstrap_replicates=5,
            )
            report = run_benchmark(frame(), directory, config=config)
            complete = [
                condition
                for condition in report["conditions"]
                if condition["state"] == "complete"
            ]
            self.assertEqual(len(complete), 6)
            self.assertTrue((Path(directory) / "metrics_long.csv").is_file())
            self.assertTrue((Path(directory) / "metrics_wide.csv").is_file())
            payload = json.loads((Path(directory) / "results.json").read_text())
            self.assertFalse(payload["score_fitting_repeated_per_condition"])
            overall_native = next(
                value
                for value in complete
                if value["condition"]["task_type"] is None
                and value["condition"]["target_positive_rate"] is None
            )
            self.assertAlmostEqual(
                overall_native["methods"]["perfect"]["metrics"]["auroc"]["value"],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
