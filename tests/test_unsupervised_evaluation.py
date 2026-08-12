import csv
import json
import tempfile
import unittest
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score

from unsupervised_evaluation import compare_variants, evaluate_records


class UnsupervisedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"sample_id": "a1", "source_id": "source-a", "task_type": "QA", "data_source": "MARCO", "generator_model": "model-1", "token_index": 0, "score": 0.10, "label": 0, "embedding": [1, 2]},
            {"sample_id": "a1", "source_id": "source-a", "task_type": "QA", "data_source": "MARCO", "generator_model": "model-1", "token_index": 1, "score": 0.90, "label": 1, "embedding": [3, 4]},
            {"sample_id": "a2", "source_id": "source-a", "task_type": "QA", "data_source": "MARCO", "generator_model": "model-1", "token_index": 0, "score": 0.80, "label": 0, "embedding": [5, 6]},
            {"sample_id": "a2", "source_id": "source-a", "task_type": "QA", "data_source": "MARCO", "generator_model": "model-1", "token_index": 1, "score": 0.70, "label": 0, "embedding": [7, 8]},
            {"sample_id": "b1", "source_id": "source-b", "task_type": "Summary", "data_source": "CNN/DM", "generator_model": "model-2", "token_index": 0, "score": 0.20, "label": 0, "embedding": [9, 10]},
            {"sample_id": "b1", "source_id": "source-b", "task_type": "Summary", "data_source": "CNN/DM", "generator_model": "model-2", "token_index": 1, "score": 0.95, "label": 1, "embedding": [11, 12]},
        ]

    def test_compare_variants_uses_paired_oof_tokens_and_source_bootstrap(self):
        ablated = [{**row, "score": -row["score"]} for row in self.records]

        comparison = compare_variants(
            {"full": self.records, "no_message": ablated}, bootstraps=50, seed=3
        )["no_message"]

        self.assertGreater(comparison["delta_auroc"], 0.0)
        self.assertGreater(comparison["delta_auprc"], 0.0)
        self.assertEqual(len(comparison["delta_auroc_ci"]), 2)

    def test_evaluates_all_tokens_and_answer_top_twenty_percent_with_strata(self):
        report = evaluate_records(self.records, bootstraps=80, seed=19)

        token = report.metrics["token"]
        self.assertEqual(token["overall"]["n"], 6)
        self.assertEqual(token["overall"]["positives"], 2)
        self.assertAlmostEqual(token["overall"]["prevalence"], 2 / 6)
        self.assertAlmostEqual(token["overall"]["auroc"], roc_auc_score([row["label"] for row in self.records], [row["score"] for row in self.records]))
        self.assertAlmostEqual(token["overall"]["auprc"], average_precision_score([row["label"] for row in self.records], [row["score"] for row in self.records]))
        self.assertEqual(token["by_task_type"]["QA"]["n"], 4)
        self.assertEqual(token["by_data_source"]["CNN/DM"]["positives"], 1)
        self.assertEqual(token["by_generator_model"]["model-2"]["n"], 2)
        self.assertGreater(token["overall"]["mean_score_difference"], 0.0)
        self.assertAlmostEqual(token["overall"]["correct_score_median"], 0.45)
        self.assertEqual(token["overall"]["hallucination_score_median"], 0.925)

        answer = report.metrics["answer"]
        self.assertEqual(answer["overall"]["n"], 3)
        self.assertEqual(answer["overall"]["positives"], 2)
        self.assertAlmostEqual(answer["overall"]["prevalence"], 2 / 3)
        self.assertAlmostEqual(answer["overall"]["auroc"], 1.0)
        self.assertAlmostEqual(answer["overall"]["auprc"], 1.0)

    def test_source_cluster_bootstrap_is_seeded_and_includes_confidence_intervals(self):
        first = evaluate_records(self.records, bootstraps=80, seed=19).metrics
        second = evaluate_records(self.records, bootstraps=80, seed=19).metrics

        for level in ("token", "answer"):
            metric = first[level]["overall"]
            self.assertEqual(metric["auroc_ci_low"], second[level]["overall"]["auroc_ci_low"])
            self.assertEqual(metric["auprc_ci_high"], second[level]["overall"]["auprc_ci_high"])
            self.assertLessEqual(metric["auroc_ci_low"], metric["auroc"])
            self.assertGreaterEqual(metric["auroc_ci_high"], metric["auroc"])
            self.assertLessEqual(metric["auprc_ci_low"], metric["auprc"])
            self.assertGreaterEqual(metric["auprc_ci_high"], metric["auprc"])

    def test_save_writes_json_and_token_csv_without_embeddings(self):
        report = evaluate_records(self.records, bootstraps=20, seed=3)
        with tempfile.TemporaryDirectory() as directory:
            report.save(directory)
            output = Path(directory)
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            with (output / "token_scores.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(metrics, report.metrics)
        self.assertEqual(len(rows), len(self.records))
        self.assertNotIn("embedding", rows[0])
        self.assertEqual(rows[1]["label"], "1")


if __name__ == "__main__":
    unittest.main()
