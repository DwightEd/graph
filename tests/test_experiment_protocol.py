import unittest
from types import SimpleNamespace
from pathlib import Path
import tempfile

import numpy as np

from experiment_protocol import (
    FrozenFile,
    FrozenEvaluation,
    HeldOutSourceAudit,
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
    def __init__(self):
        self.sample_ids = ["a", "b"]
        self.manifest = {"split": "test"}
        self.opened = set()
        self.samples = {
            "a": _Sample(self, "a", "source-a", [0, 1, 1]),
            "b": _Sample(self, "b", np.nan, [0, 0]),
        }

    def __getitem__(self, sample_id):
        return self.samples[str(sample_id)]

    def labels(self):
        if self.opened != set(self.sample_ids):
            raise RuntimeError(
                "formal labels become available only after every attention sample "
                "has been processed"
            )
        return _Labels()


class SourceGroupAuditTests(unittest.TestCase):
    def test_audit_derives_test_groups_and_records_a_partial_scope(self):
        dataset = _LabelLockedDataset()

        audit = HeldOutSourceAudit(
            dataset,
            selected_sample_ids=["b"],
            fit_source_ids=["source-a"],
            calibration_source_ids=["source-c"],
            require_complete_split=False,
        )
        sample = dataset["b"]
        sample.attention()
        audit.observe(sample)
        result = audit.finish()

        self.assertEqual(result.fit_source_ids, ("source-a",))
        self.assertEqual(result.calibration_source_ids, ("source-c",))
        self.assertEqual(result.test_source_ids, ("b",))
        self.assertEqual(result.test_sample_ids, ("b",))
        self.assertEqual(result.test_scope, "selected_samples")

    def test_audit_rejects_a_partial_scope_by_default(self):
        with self.assertRaisesRegex(ValueError, "complete split"):
            HeldOutSourceAudit(
                _LabelLockedDataset(),
                selected_sample_ids=["a"],
                fit_source_ids=["fit"],
                calibration_source_ids=["calibration"],
            )

    def test_audit_rejects_a_missing_frozen_reference_group(self):
        with self.assertRaisesRegex(ValueError, "valid source IDs"):
            HeldOutSourceAudit(
                _LabelLockedDataset(),
                selected_sample_ids=["a", "b"],
                fit_source_ids=[np.nan],
                calibration_source_ids=["calibration"],
            )

    def test_audit_rejects_an_overlapping_or_duplicate_observation(self):
        dataset = _LabelLockedDataset()
        overlap = HeldOutSourceAudit(
            dataset,
            selected_sample_ids=["a"],
            fit_source_ids=["source-a"],
            calibration_source_ids=["calibration"],
            require_complete_split=False,
        )
        sample = dataset["a"]
        sample.attention()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            overlap.observe(sample)

        audit = HeldOutSourceAudit(
            dataset,
            selected_sample_ids=["b"],
            fit_source_ids=["fit"],
            calibration_source_ids=["calibration"],
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
            fit_source_ids=["fit"],
            calibration_source_ids=["calibration"],
        )
        sample = dataset["a"]
        sample.attention()
        audit.observe(sample)

        with self.assertRaisesRegex(ValueError, "not observed"):
            audit.finish()


class FrozenEvaluationTests(unittest.TestCase):
    def test_unlocks_and_aligns_only_a_frozen_test_artifact(self):
        dataset = _LabelLockedDataset()
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "scores.npz"
            np.savez_compressed(
                artifact,
                sample_id=np.asarray(["b", "a", "a"]),
                token_index=np.asarray([1, 2, 1]),
            )
            evaluation = FrozenEvaluation.capture(artifact, expected_split="test")
            loaded_paths = []

            def load_captured(path):
                loaded_paths.append(Path(path))
                return _load_npz(path)

            rows, labels = evaluation.load_and_align(dataset, load_captured)

        self.assertEqual(loaded_paths, [artifact.resolve()])
        np.testing.assert_array_equal(rows["sample_id"], ["b", "a", "a"])
        np.testing.assert_array_equal(labels.token_label, [0, 1, 1])
        np.testing.assert_array_equal(labels.response_positive, [0, 1, 1])
        np.testing.assert_array_equal(
            labels.source_id, ["b", "source-a", "source-a"]
        )
        np.testing.assert_array_equal(labels.response_length, [2, 3, 3])

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

    def test_rejects_a_token_index_outside_the_canonical_response(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "scores.npz"
            np.savez_compressed(
                artifact,
                sample_id=np.asarray(["a"]),
                token_index=np.asarray([3]),
            )
            with self.assertRaisesRegex(ValueError, "outside canonical response"):
                FrozenEvaluation.capture(artifact).load_and_align(
                    _LabelLockedDataset(), _load_npz
                )

    def test_rejects_a_non_integral_external_token_index(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "scores.npz"
            np.savez_compressed(
                artifact,
                sample_id=np.asarray(["a"]),
                token_index=np.asarray([1.5]),
            )
            with self.assertRaisesRegex(ValueError, "integer"):
                FrozenEvaluation.capture(artifact).load_and_align(
                    _LabelLockedDataset(), _load_npz
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
