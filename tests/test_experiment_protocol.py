import unittest
from types import SimpleNamespace
from pathlib import Path
import tempfile

import numpy as np

from experiment_protocol import (
    FrozenFile,
    align_evaluation_labels,
    audit_source_groups,
)


class SourceGroupAuditTests(unittest.TestCase):
    def test_audit_records_disjoint_fit_calibration_and_test_groups(self):
        audit = audit_source_groups(
            fit_source_ids=["source-a", "source-b"],
            calibration_source_ids=["source-c"],
            test_source_ids=["source-d", "source-e"],
        )

        self.assertEqual(audit.fit_source_ids, ("source-a", "source-b"))
        self.assertEqual(audit.calibration_source_ids, ("source-c",))
        self.assertEqual(audit.test_source_ids, ("source-d", "source-e"))

    def test_audit_rejects_a_source_group_reused_for_calibration_and_test(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            audit_source_groups(
                fit_source_ids=["source-a"],
                calibration_source_ids=["source-b"],
                test_source_ids=["source-b"],
            )

    def test_audit_rejects_an_empty_protocol_role(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            audit_source_groups(
                fit_source_ids=["source-a"],
                calibration_source_ids=[],
                test_source_ids=["source-b"],
            )


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
        self.opened = set()
        self.samples = {
            "a": _Sample(self, "a", "source-a", [0, 1, 1]),
            "b": _Sample(self, "b", "source-b", [0, 0]),
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


class EvaluationLabelAlignmentTests(unittest.TestCase):
    def test_unlocks_and_aligns_canonical_response_labels_for_filtered_rows(self):
        dataset = _LabelLockedDataset()

        labels = align_evaluation_labels(
            dataset,
            sample_ids=np.asarray(["b", "a", "a"]),
            token_indices=np.asarray([1, 2, 1]),
        )

        np.testing.assert_array_equal(labels.token_label, [0, 1, 1])
        np.testing.assert_array_equal(labels.response_positive, [0, 1, 1])
        np.testing.assert_array_equal(
            labels.source_id, ["source-b", "source-a", "source-a"]
        )
        np.testing.assert_array_equal(labels.response_length, [2, 3, 3])

    def test_rejects_a_token_index_outside_the_canonical_response(self):
        with self.assertRaisesRegex(ValueError, "outside canonical response"):
            align_evaluation_labels(
                _LabelLockedDataset(),
                sample_ids=np.asarray(["a"]),
                token_indices=np.asarray([3]),
            )


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
