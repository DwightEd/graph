import json
import unittest
from types import SimpleNamespace
from pathlib import Path
import tempfile

import numpy as np

from experiment_protocol import (
    FrozenFile,
    FrozenEvaluation,
    HeldOutSourceAudit,
    dataset_manifest_sha256,
    file_sha256,
    partition_source_groups,
    validate_complete_token_rows,
    validate_source_audit,
)


def _load_npz(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


class _Sample:
    def __init__(self, dataset, sample_id, source_id, labels):
        self.dataset = dataset
        self.sample_id = sample_id
        self.source_id = source_id
        self._labels = np.asarray(labels, dtype=np.int8)

    def attention(self):
        self.dataset.opened.add(self.sample_id)
        return SimpleNamespace(num_response_tokens=len(self._labels))

    def release_attention(self):
        pass


class _Labels:
    def response_labels(self, sample):
        return sample._labels


class _LabelLockedDataset:
    def __init__(self, *, manifest_marker="primary"):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.sample_ids = ["a", "b"]
        self.manifest = {"split": "test", "marker": manifest_marker}
        (self.root / "manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )
        self.opened = set()
        self.labels_called = False
        self.samples = {
            "a": _Sample(self, "a", "source-a", [0, 1, 1]),
            "b": _Sample(self, "b", np.nan, [0, 0]),
        }

    def __getitem__(self, sample_id):
        return self.samples[str(sample_id)]

    def labels(self):
        self.labels_called = True
        if self.opened != set(self.sample_ids):
            raise RuntimeError(
                "formal labels become available only after every attention sample "
                "has been processed"
            )
        return _Labels()


def _complete_evaluation_rows(dataset):
    return {
        "dataset_manifest_sha256": np.asarray(
            dataset_manifest_sha256(dataset)
        ),
        "sample_id": np.asarray(["b", "a", "a", "b", "a"]),
        "source_id": np.asarray(
            ["nan", "source-a", "source-a", "nan", "source-a"]
        ),
        "token_index": np.asarray([1, 2, 0, 0, 1], dtype=np.int32),
        "response_length": np.asarray([2, 3, 3, 2, 3], dtype=np.int32),
    }


class SourceGroupAuditTests(unittest.TestCase):
    def test_audit_derives_test_groups_and_records_a_partial_scope(self):
        dataset = _LabelLockedDataset()

        audit = HeldOutSourceAudit(
            dataset,
            selected_sample_ids=["b"],
            reserved_source_ids=["source-a", "source-c"],
            require_complete_split=False,
        )
        sample = dataset["b"]
        sample.attention()
        audit.observe(sample)
        result = audit.finish()

        self.assertEqual(result.test_source_ids, ("b",))
        self.assertEqual(result.test_sample_ids, ("b",))
        self.assertEqual(result.test_scope, "selected_samples")

    def test_audit_rejects_a_partial_scope_by_default(self):
        with self.assertRaisesRegex(ValueError, "complete split"):
            HeldOutSourceAudit(
                _LabelLockedDataset(),
                selected_sample_ids=["a"],
                reserved_source_ids=["fit", "calibration"],
            )

    def test_audit_rejects_a_missing_frozen_reference_group(self):
        with self.assertRaisesRegex(ValueError, "valid source IDs"):
            HeldOutSourceAudit(
                _LabelLockedDataset(),
                selected_sample_ids=["a", "b"],
                reserved_source_ids=[np.nan],
            )

    def test_audit_rejects_an_overlapping_or_duplicate_observation(self):
        dataset = _LabelLockedDataset()
        overlap = HeldOutSourceAudit(
            dataset,
            selected_sample_ids=["a"],
            reserved_source_ids=["source-a", "calibration"],
            require_complete_split=False,
        )
        sample = dataset["a"]
        sample.attention()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            overlap.observe(sample)

        audit = HeldOutSourceAudit(
            dataset,
            selected_sample_ids=["b"],
            reserved_source_ids=["fit", "calibration"],
            require_complete_split=False,
        )
        sample = dataset["b"]
        sample.attention()
        audit.observe(sample)
        with self.assertRaisesRegex(ValueError, "more than once"):
            audit.observe(sample)

    def test_audit_rejects_an_omitted_selected_sample(self):
        dataset = _LabelLockedDataset()
        audit = HeldOutSourceAudit(
            dataset,
            selected_sample_ids=["a", "b"],
            reserved_source_ids=["fit", "calibration"],
        )
        sample = dataset["a"]
        sample.attention()
        audit.observe(sample)

        with self.assertRaisesRegex(ValueError, "not observed"):
            audit.finish()


class FrozenSourceAuditTests(unittest.TestCase):
    def test_validates_declared_groups_against_frozen_rows(self):
        validate_source_audit(
            reserved_source_ids=["fit-a", "cal-b"],
            test_source_ids=["held-c", "sample-b"],
            test_sample_ids=["sample-a", "sample-b"],
            row_sample_ids=["sample-a", "sample-a", "sample-b"],
            row_source_ids=["held-c", "held-c", "nan"],
            audit_scope="complete_split",
        )

    def test_rejects_corrupt_frozen_source_audits(self):
        valid = {
            "reserved_source_ids": ["fit-a", "cal-b"],
            "test_source_ids": ["held-c", "sample-b"],
            "test_sample_ids": ["sample-a", "sample-b"],
            "row_sample_ids": ["sample-a", "sample-a", "sample-b"],
            "row_source_ids": ["held-c", "held-c", "nan"],
            "audit_scope": "selected_samples",
        }
        corruptions = {
            "overlap": {"test_source_ids": ["fit-a", "sample-b"]},
            "duplicate": {"test_sample_ids": ["sample-a", "sample-a"]},
            "omitted sample": {"test_sample_ids": ["sample-a"]},
            "inconsistent mapping": {
                "row_source_ids": ["held-c", "changed", "nan"]
            },
            "undeclared group": {
                "test_source_ids": ["held-c", "different"]
            },
            "non-text row source": {
                "test_source_ids": ["1", "2"],
                "row_source_ids": np.asarray([1, 1, 2]),
            },
            "scope": {"audit_scope": "unknown"},
        }
        for reason, change in corruptions.items():
            with self.subTest(reason=reason), self.assertRaisesRegex(
                ValueError, "source-group audit"
            ):
                validate_source_audit(**(valid | change))


class CompleteTokenRowsTests(unittest.TestCase):
    def test_validates_complete_per_response_rows(self):
        validate_complete_token_rows(
            sample_id=["a", "b", "a", "b", "a"],
            source_id=["source-a", "nan", "source-a", "nan", "source-a"],
            token_index=np.asarray([2, 1, 0, 0, 1], dtype=np.int32),
            response_length=np.asarray([3, 2, 3, 2, 3], dtype=np.int32),
        )

    def test_rejects_inconsistent_or_incomplete_response_rows(self):
        valid = {
            "sample_id": ["a", "a", "a"],
            "source_id": ["source-a", "source-a", "source-a"],
            "token_index": np.asarray([0, 1, 2], dtype=np.int32),
            "response_length": np.asarray([3, 3, 3], dtype=np.int32),
        }
        corruptions = {
            "source": {"source_id": ["source-a", "changed", "source-a"]},
            "length": {
                "response_length": np.asarray([3, 2, 3], dtype=np.int32)
            },
            "missing": {
                "token_index": np.asarray([0, 2], dtype=np.int32),
                "sample_id": ["a", "a"],
                "source_id": ["source-a", "source-a"],
                "response_length": np.asarray([3, 3], dtype=np.int32),
            },
            "duplicate": {"token_index": np.asarray([0, 1, 1], dtype=np.int32)},
        }
        for reason, change in corruptions.items():
            with self.subTest(reason=reason), self.assertRaisesRegex(
                ValueError, "complete token rows"
            ):
                validate_complete_token_rows(**(valid | change))


class SourceGroupPartitionTests(unittest.TestCase):
    def test_partitions_exact_group_count_deterministically(self):
        dataset = _LabelLockedDataset()
        dataset.sample_ids = ["a", "b", "c", "d", "e"]
        dataset.samples.update(
            {
                "c": _Sample(dataset, "c", "source-c", [0]),
                "d": _Sample(dataset, "d", "source-d", [0]),
                "e": _Sample(dataset, "e", "source-d", [0]),
            }
        )

        first = partition_source_groups(
            dataset,
            dataset.sample_ids,
            calibration_fraction=0.5,
            seed=17,
        )
        second = partition_source_groups(
            dataset,
            list(reversed(dataset.sample_ids)),
            calibration_fraction=0.5,
            seed=17,
        )

        self.assertEqual(first["fit_group_ids"], second["fit_group_ids"])
        self.assertEqual(
            first["calibration_group_ids"], second["calibration_group_ids"]
        )
        self.assertEqual(len(first["calibration_group_ids"]), 2)
        self.assertTrue(first["fit_group_ids"])
        self.assertFalse(
            set(first["fit_group_ids"]) & set(first["calibration_group_ids"])
        )
        self.assertEqual(
            set(first["fit_sample_ids"]) | set(first["calibration_sample_ids"]),
            set(dataset.sample_ids),
        )

    def test_rejects_an_invalid_fraction_or_single_group(self):
        dataset = _LabelLockedDataset()
        with self.assertRaisesRegex(ValueError, "calibration_fraction"):
            partition_source_groups(
                dataset,
                dataset.sample_ids,
                calibration_fraction=1.0,
                seed=0,
            )
        dataset.samples["b"].source_id = "source-a"
        with self.assertRaisesRegex(ValueError, "two.*source groups"):
            partition_source_groups(
                dataset,
                dataset.sample_ids,
                calibration_fraction=0.5,
                seed=0,
            )


class FrozenEvaluationTests(unittest.TestCase):
    def test_unlocks_and_aligns_only_a_frozen_test_artifact(self):
        dataset = _LabelLockedDataset()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "scores.npz"
            np.savez_compressed(artifact, **_complete_evaluation_rows(dataset))
            evaluation = FrozenEvaluation.capture(artifact, expected_split="test")
            loaded_paths = []

            def load_captured(path):
                loaded_paths.append(Path(path))
                return _load_npz(path)

            rows, labels = evaluation.load_and_align(dataset, load_captured)

        self.assertEqual(loaded_paths, [artifact.resolve()])
        np.testing.assert_array_equal(
            rows["sample_id"], ["b", "a", "a", "b", "a"]
        )
        np.testing.assert_array_equal(labels.token_label, [0, 1, 0, 0, 1])
        np.testing.assert_array_equal(labels.response_positive, [0, 1, 1, 0, 1])
        np.testing.assert_array_equal(
            labels.source_id, ["b", "source-a", "source-a", "b", "source-a"]
        )
        np.testing.assert_array_equal(
            labels.response_length, [2, 3, 3, 2, 3]
        )

    def test_rejects_another_manifest_before_opening_labels(self):
        scored_dataset = _LabelLockedDataset(manifest_marker="scored")
        evaluated_dataset = _LabelLockedDataset(manifest_marker="different")
        evaluated_dataset.samples["a"].source_id = "other-source"
        evaluated_dataset.samples["a"]._labels = np.asarray([1, 1, 1])
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "scores.npz"
            np.savez_compressed(
                artifact, **_complete_evaluation_rows(scored_dataset)
            )
            with self.assertRaisesRegex(ValueError, "dataset manifest"):
                FrozenEvaluation.capture(artifact).load_and_align(
                    evaluated_dataset, _load_npz
                )
        self.assertFalse(evaluated_dataset.labels_called)

    def test_rejects_source_mismatch_before_opening_labels(self):
        dataset = _LabelLockedDataset()
        rows = _complete_evaluation_rows(dataset)
        rows["source_id"] = np.asarray(
            ["nan", "wrong-source", "wrong-source", "nan", "wrong-source"]
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "scores.npz"
            np.savez_compressed(artifact, **rows)
            with self.assertRaisesRegex(ValueError, "canonical source"):
                FrozenEvaluation.capture(artifact).load_and_align(
                    dataset, _load_npz
                )
        self.assertFalse(dataset.labels_called)

    def test_rejects_response_length_mismatch_before_opening_labels(self):
        dataset = _LabelLockedDataset()
        rows = {
            "dataset_manifest_sha256": np.asarray(
                dataset_manifest_sha256(dataset)
            ),
            "sample_id": np.asarray(["a", "a"]),
            "source_id": np.asarray(["source-a", "source-a"]),
            "token_index": np.asarray([0, 1], dtype=np.int32),
            "response_length": np.asarray([2, 2], dtype=np.int32),
        }
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "scores.npz"
            np.savez_compressed(artifact, **rows)
            with self.assertRaisesRegex(ValueError, "response length"):
                FrozenEvaluation.capture(artifact).load_and_align(
                    dataset, _load_npz
                )
        self.assertFalse(dataset.labels_called)

    def test_does_not_accept_rows_from_another_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "scores.npz"
            other = Path(directory) / "other.npz"
            np.savez_compressed(
                artifact,
                sample_id=np.asarray(["a"]),
                token_index=np.asarray([1]),
            )
            np.savez_compressed(
                other,
                sample_id=np.asarray(["b"]),
                token_index=np.asarray([0]),
            )
            evaluation = FrozenEvaluation.capture(artifact)

            with self.assertRaises(TypeError):
                evaluation.load_and_align(
                    _LabelLockedDataset(),
                    _load_npz,
                    rows=_load_npz(other),
                )
            self.assertFalse(hasattr(evaluation, "align_labels"))

    def test_rejects_token_rows_outside_the_declared_response(self):
        dataset = _LabelLockedDataset()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "scores.npz"
            np.savez_compressed(
                artifact,
                dataset_manifest_sha256=np.asarray(
                    dataset_manifest_sha256(dataset)
                ),
                sample_id=np.asarray(["a", "a", "a"]),
                source_id=np.asarray(["source-a"] * 3),
                token_index=np.asarray([0, 1, 3]),
                response_length=np.asarray([3, 3, 3]),
            )
            with self.assertRaisesRegex(ValueError, "full response"):
                FrozenEvaluation.capture(artifact).load_and_align(
                    dataset, _load_npz
                )

    def test_rejects_a_non_integral_external_token_index(self):
        dataset = _LabelLockedDataset()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "scores.npz"
            rows = _complete_evaluation_rows(dataset)
            rows["token_index"] = rows["token_index"].astype(np.float32)
            np.savez_compressed(artifact, **rows)
            with self.assertRaisesRegex(ValueError, "integer"):
                FrozenEvaluation.capture(artifact).load_and_align(
                    dataset, _load_npz
                )

    def test_rejects_a_digest_change_or_unexpected_dataset_split(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "scores.npz"
            np.savez_compressed(
                artifact,
                sample_id=np.asarray(["a"]),
                token_index=np.asarray([1]),
            )
            evaluation = FrozenEvaluation.capture(artifact)
            np.savez_compressed(
                artifact,
                sample_id=np.asarray(["b"]),
                token_index=np.asarray([0]),
            )
            with self.assertRaisesRegex(ValueError, "digest"):
                evaluation.load_and_align(_LabelLockedDataset(), _load_npz)

            np.savez_compressed(
                artifact,
                sample_id=np.asarray(["a"]),
                token_index=np.asarray([1]),
            )
            other = FrozenEvaluation.capture(artifact)
            dataset = _LabelLockedDataset()
            dataset.manifest["split"] = "train"
            with self.assertRaisesRegex(ValueError, "split"):
                other.load_and_align(dataset, _load_npz)


class FrozenFileTests(unittest.TestCase):
    def test_shared_file_digest_matches_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "artifact.bin"
            artifact.write_bytes(b"abc")

            self.assertEqual(
                file_sha256(artifact),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_frozen_file_rejects_a_changed_digest_or_different_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "scores.npz"
            other = Path(directory) / "other.npz"
            artifact.write_bytes(b"frozen-score")
            other.write_bytes(b"frozen-score")
            frozen = FrozenFile.capture(artifact)

            frozen.verify(artifact)
            with self.assertRaisesRegex(ValueError, "identity"):
                frozen.verify(other)

            artifact.write_bytes(b"changed-score")
            with self.assertRaisesRegex(ValueError, "digest"):
                frozen.verify(artifact)


if __name__ == "__main__":
    unittest.main()
