import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cache import (
    AttentionSample,
    index_row,
    save_attention_sample,
    sha256,
    write_split_index,
)
from onset_experiment import (
    OnsetValidation,
    ValidationConfig,
    holm_adjust,
    map_pseudo_onset,
    onset_delta,
    paired_statistics,
)
from research_dataset import STRUCTURAL_FEATURE_NAMES, ResearchDataset

PRIMARY_FEATURES = (
    "prompt_mass_share",
    "normalized_entropy",
    "history_lag",
    "in_density",
    "history_edge_share",
)


def make_attention_sample(
    sample_id: str,
    source_id: str,
    offset: float,
    *,
    prompt_tokens: int = 2,
    response_tokens: int = 6,
) -> AttentionSample:
    """Create a small, valid response with enough history for onset effects."""
    response_idx = prompt_tokens
    token_ids = torch.arange(prompt_tokens + response_tokens, dtype=torch.int32)
    columns, values, pointers = [], [], [0]
    for target in range(response_idx, len(token_ids)):
        for source in range(target):
            columns.append(source)
            values.append(0.02 + offset + 0.005 * source + 0.002 * target)
        pointers.append(len(columns))
    return AttentionSample(
        sample_id,
        source_id,
        response_idx,
        token_ids,
        torch.linspace(1.0, 0.1, len(token_ids), dtype=torch.float16).reshape(1, 1, -1),
        torch.tensor(pointers, dtype=torch.int32),
        torch.tensor(columns, dtype=torch.int32),
        torch.tensor(values, dtype=torch.float16),
        0.01,
    )


def write_canonical_split(root: Path, *, include_e1_match: bool = True) -> Path:
    """Write matched pairs plus length-matched prompt-length distractors."""
    root.mkdir(parents=True)
    rows = []
    stratum = {
        "split": "test",
        "task_type": "QA",
        "data_source": "MARCO",
        "generator_model": "llama-2-7b-chat",
        "temperature": 0.7,
    }
    samples = [
        ("e1", "source-error-1", 2, 6, "error-quality"),
        ("e2", "source-error-2", 3, 7, "error-quality"),
        ("c3", "source-control-3", 3, 7, "control-quality"),
    ]
    if include_e1_match:
        samples.extend((
            ("c0", "source-control-0", 4, 6, "control-quality"),
            ("c1", "source-control-1", 2, 6, "control-quality"),
            ("c2", "source-control-2", 2, 7, "control-quality"),
        ))
    for index, (sample_id, source_id, prompt_tokens, response_tokens, quality) in enumerate(samples):
        sample = make_attention_sample(
            sample_id, source_id, 0.001 * index,
            prompt_tokens=prompt_tokens, response_tokens=response_tokens,
        )
        path = root / "attention" / f"{sample_id}.npz"
        save_attention_sample(sample, path)
        rows.append(index_row(root, sample, path, metadata={**stratum, "quality": quality}))
    write_split_index(
        root,
        rows,
        attention_floor=0.01,
        num_layers=1,
        num_heads=1,
        alignment="post_token_query_at_same_position",
    )
    labels = root / "labels.jsonl"
    positive_runs = {
        "e1": [[2, 3], [4, 5]],
        "e2": [[2, 3], [5, 6]],
    }
    labels.write_text(
        "".join(
            json.dumps({"sample_id": sample_id, "positive_runs": positive_runs.get(sample_id, [])}) + "\n"
            for sample_id, *_ in samples
        ),
        encoding="utf-8",
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["labels_sha256"] = sha256(labels)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


def write_sparse_canonical_pair(root: Path) -> Path:
    """Write one pair whose sparse causal topology admits source swaps."""
    root.mkdir(parents=True)
    row_ptr = torch.tensor([0, 1, 2, 3, 5, 7, 9], dtype=torch.int32)
    columns = torch.tensor([0, 1, 2, 0, 3, 1, 4, 2, 5], dtype=torch.int32)
    rows = []
    metadata = {
        "split": "test",
        "task_type": "QA",
        "data_source": "MARCO",
        "generator_model": "llama-2-7b-chat",
        "temperature": 0.7,
        "quality": "good",
    }
    for sample_id, source_id, offset in (
        ("error", "source-error", 0.00),
        ("control", "source-control", 0.01),
    ):
        sample = AttentionSample(
            sample_id,
            source_id,
            3,
            torch.arange(9, dtype=torch.int32),
            torch.ones((1, 1, 9), dtype=torch.float16),
            row_ptr,
            columns,
            torch.linspace(0.10 + offset, 0.18 + offset, len(columns), dtype=torch.float16),
            0.01,
        )
        path = root / "attention" / f"{sample_id}.npz"
        save_attention_sample(sample, path)
        rows.append(index_row(root, sample, path, metadata=metadata))
    write_split_index(
        root,
        rows,
        attention_floor=0.01,
        num_layers=1,
        num_heads=1,
        alignment="post_token_query_at_same_position",
    )
    labels = root / "labels.jsonl"
    labels.write_text(
        json.dumps({"sample_id": "error", "positive_runs": [[2, 3]]}) + "\n"
        + json.dumps({"sample_id": "control", "positive_runs": []}) + "\n",
        encoding="utf-8",
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["labels_sha256"] = sha256(labels)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


def rewrite_index_metadata(root: Path, update) -> None:
    index_path = root / "index.jsonl"
    rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        update(row)
    index_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["index_sha256"] = sha256(index_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class OnsetExperimentUnitTests(unittest.TestCase):
    def test_onset_delta_uses_the_first_k_error_tokens_and_the_adjacent_k_pre_tokens(self):
        features = np.array(
            [[0.0, 0.0], [1.0, 10.0], [2.0, 20.0], [5.0, 50.0],
             [8.0, 80.0], [99.0, 990.0], [100.0, 1000.0]],
            dtype=np.float64,
        )

        actual = onset_delta(features, start=3, end=6, effect_width=2)

        np.testing.assert_allclose(actual, np.array([5.0, 50.0]))

    def test_map_pseudo_onset_preserves_normalized_response_position(self):
        self.assertEqual(map_pseudo_onset(start=6, error_tokens=11, control_tokens=21), 12)
        self.assertEqual(map_pseudo_onset(start=10, error_tokens=11, control_tokens=21), 20)

    def test_paired_statistics_is_seeded_and_reports_a_positive_effect(self):
        effects = np.arange(1.0, 9.0)

        first = paired_statistics(effects, bootstraps=400, permutations=2000, seed=17)
        second = paired_statistics(effects, bootstraps=400, permutations=2000, seed=17)

        self.assertEqual(first, second)
        self.assertGreater(first["mean_effect"], 0.0)
        self.assertGreater(first["ci_low"], 0.0)
        self.assertGreater(first["ci_high"], first["ci_low"])
        self.assertLess(first["sign_flip_p"], 0.02)

    def test_paired_statistics_does_not_report_dz_without_effect_variance(self):
        for effects in (np.array([2.0]), np.array([2.0, 2.0, 2.0])):
            with self.subTest(effects=effects.tolist()):
                statistics = paired_statistics(effects, bootstraps=20, permutations=40, seed=3)
                self.assertIsNone(statistics["dz"])

    def test_holm_adjustment_is_monotone_and_never_smaller_than_its_input(self):
        p_values = np.array([0.60, 0.01, 0.03, 0.02])

        adjusted = np.asarray(holm_adjust(p_values), dtype=float)

        self.assertTrue(np.all(adjusted >= p_values))
        order = np.argsort(p_values)
        self.assertTrue(np.all(np.diff(adjusted[order]) >= 0.0))
        self.assertTrue(np.all(adjusted <= 1.0))
        np.testing.assert_allclose(adjusted, np.array([0.60, 0.04, 0.06, 0.06]))


class OnsetExperimentIntegrationTests(unittest.TestCase):
    def test_run_verifies_each_attention_payload_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            split = write_sparse_canonical_pair(Path(directory) / "canonical" / "test")
            payload_path = split / "attention" / "error.npz"
            payload = bytearray(payload_path.read_bytes())
            payload[-20] ^= 1
            payload_path.write_bytes(payload)

            with self.assertRaisesRegex(ValueError, "attention sample SHA256"):
                OnsetValidation(ValidationConfig(
                    canonical_split=split,
                    output_dir=Path(directory) / "output",
                    effect_width=1,
                    bootstraps=10,
                    permutations=10,
                    rewires=1,
                )).run()

    def test_requires_every_exact_stratum_key_but_accepts_null_temperature(self):
        for missing in ("task_type", "data_source", "generator_model", "temperature"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                split = write_sparse_canonical_pair(Path(directory) / "missing" / "test")
                rewrite_index_metadata(split, lambda row, key=missing: row.pop(key))
                with self.assertRaisesRegex(ValueError, missing):
                    OnsetValidation(ValidationConfig(
                        canonical_split=split,
                        output_dir=Path(directory) / "output",
                        bootstraps=10,
                        permutations=10,
                        rewires=1,
                    )).run()

        with tempfile.TemporaryDirectory() as directory:
            split = write_sparse_canonical_pair(Path(directory) / "null-temperature" / "test")
            rewrite_index_metadata(split, lambda row: row.__setitem__("temperature", None))
            result = OnsetValidation(ValidationConfig(
                canonical_split=split,
                output_dir=Path(directory) / "output",
                effect_width=1,
                bootstraps=10,
                permutations=10,
                rewires=1,
                seed=9,
            )).run()
            self.assertEqual(result["pairs"], 1)

    def test_run_matches_without_replacement_merges_events_and_writes_prespecified_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            split = write_canonical_split(Path(directory) / "canonical" / "test")
            output = Path(directory) / "output"
            config = ValidationConfig(
                canonical_split=split,
                output_dir=output,
                effect_width=2,
                bootstraps=100,
                permutations=200,
                rewires=2,
                seed=13,
            )

            result = OnsetValidation(config).run()

            self.assertEqual(result["pairs"], 2)
            self.assertEqual(result["events"], 3)
            for name in (
                "matches.csv",
                "pair_effects.csv",
                "event_study.csv",
                "primary_effects.csv",
                "rewire_null.csv",
                "event_study.png",
                "metadata.json",
            ):
                self.assertTrue((output / name).is_file(), name)

            with (output / "matches.csv").open(encoding="utf-8", newline="") as handle:
                matches = list(csv.DictReader(handle))
            self.assertEqual({row["error_sample_id"] for row in matches}, {"e1", "e2"})
            self.assertEqual(
                {row["error_sample_id"]: row["control_sample_id"] for row in matches},
                {"e1": "c1", "e2": "c3"},
            )
            self.assertEqual(len({row["control_sample_id"] for row in matches}), len(matches))
            for row in matches:
                self.assertEqual(row["match_stratum"], "exact_metadata_nearest_lengths")
                self.assertEqual(row["error_prompt_tokens"], row["control_prompt_tokens"])
                self.assertEqual(row["error_response_tokens"], row["control_response_tokens"])
            merged = next(row for row in matches if row["error_sample_id"] == "e1")
            self.assertEqual((int(merged["run_start"]), int(merged["run_end"])), (2, 5))

            with (output / "pair_effects.csv").open(encoding="utf-8", newline="") as handle:
                pair_effects = list(csv.DictReader(handle))
            error_two_effects = [row for row in pair_effects if row["error_sample_id"] == "e2"]
            self.assertEqual(len(error_two_effects), len(STRUCTURAL_FEATURE_NAMES))
            self.assertTrue(all(int(row["event_count"]) == 2 for row in error_two_effects))

            dataset = ResearchDataset(split)
            error_features = dataset["e2"].structural_features().numpy()
            control_features = dataset["c3"].structural_features().numpy()
            expected = np.mean(
                (
                    onset_delta(error_features, start=2, end=3, effect_width=2)
                    - onset_delta(control_features, start=2, end=3, effect_width=2),
                    onset_delta(error_features, start=5, end=6, effect_width=2)
                    - onset_delta(control_features, start=5, end=6, effect_width=2),
                ),
                axis=0,
            )
            reported = {row["feature"]: float(row["effect"]) for row in error_two_effects}
            for name in PRIMARY_FEATURES:
                index = STRUCTURAL_FEATURE_NAMES.index(name)
                self.assertAlmostEqual(reported[name], expected[index], places=5)

            with (output / "primary_effects.csv").open(encoding="utf-8", newline="") as handle:
                primary = list(csv.DictReader(handle))
            self.assertEqual({row["feature"] for row in primary}, set(PRIMARY_FEATURES))
            self.assertEqual(len(primary), 5)

            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["pairs"], 2)
            self.assertEqual(metadata["events"], 3)
            self.assertEqual(set(metadata["primary_features"]), set(PRIMARY_FEATURES))
            self.assertEqual(metadata["alignment"], {
                "coordinate_system": "response_relative",
                "effect_width": 2,
                "pseudo_onset": "normalized_start_position",
            })
            self.assertEqual(metadata["matching"], {
                "stratum_fields": ["task_type", "data_source", "generator_model", "temperature"],
                "without_replacement": True,
            })
            self.assertEqual(metadata["rewire"]["draws"], 2)
            self.assertIn("accepted_swaps", metadata["rewire"])
            provenance = metadata["input_provenance"]
            self.assertEqual(provenance, {
                "canonical_split": str(split.resolve()),
                "manifest_sha256": sha256(split / "manifest.json"),
                "index_sha256": sha256(split / "index.jsonl"),
                "labels_sha256": sha256(split / "labels.jsonl"),
                "attention_floor": 0.01,
                "num_layers": 1,
                "num_heads": 1,
            })
            self.assertIsInstance(metadata["method_version"], str)
            self.assertTrue(metadata["method_version"])
            self.assertEqual(
                metadata["length_cost"],
                "abs(log1p(error_prompt_tokens) - log1p(control_prompt_tokens)) + "
                "abs(log1p(error_response_tokens) - log1p(control_response_tokens))",
            )

    def test_run_excludes_unmatched_errors_and_reports_them_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            split = write_canonical_split(Path(directory) / "canonical" / "test", include_e1_match=False)
            output = Path(directory) / "output"
            result = OnsetValidation(ValidationConfig(
                canonical_split=split,
                output_dir=output,
                effect_width=2,
                bootstraps=20,
                permutations=40,
                rewires=1,
                seed=5,
            )).run()

            self.assertEqual(result["pairs"], 1)
            self.assertEqual(result["events"], 2)
            with (output / "matches.csv").open(encoding="utf-8", newline="") as handle:
                matches = list(csv.DictReader(handle))
            self.assertEqual([(row["error_sample_id"], row["control_sample_id"]) for row in matches], [("e2", "c3")])
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["unmatched_error_samples"], ["e1"])

    def test_sparse_pair_produces_an_estimable_rewire_null(self):
        with tempfile.TemporaryDirectory() as directory:
            split = write_sparse_canonical_pair(Path(directory) / "canonical" / "test")
            output = Path(directory) / "output"
            OnsetValidation(ValidationConfig(
                canonical_split=split,
                output_dir=output,
                effect_width=1,
                bootstraps=20,
                permutations=40,
                rewires=3,
                seed=9,
            )).run()

            with (output / "rewire_null.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["draw"]) for row in rows], [0, 1, 2])
            self.assertTrue(any(int(row["accepted_swaps"]) > 0 for row in rows))
            self.assertTrue(any(int(row["changed_pairs"]) > 0 for row in rows))
            metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["topology_test"]["status"], "estimable")
            self.assertIn("approximate_randomization_p", metadata["topology_test"])
            self.assertNotIn("randomization_p", metadata["topology_test"])
            self.assertIn("null_q025", metadata["topology_test"])
            self.assertIn("null_q975", metadata["topology_test"])
            self.assertLessEqual(
                metadata["topology_test"]["null_q025"],
                metadata["topology_test"]["null_q975"],
            )
            self.assertEqual(metadata["rewire"]["chain"], "continuous")
            self.assertGreaterEqual(metadata["rewire"]["burn_in"], 0)
            self.assertGreaterEqual(metadata["rewire"]["thinning"], 1)


if __name__ == "__main__":
    unittest.main()
