"""A review archive must preserve scores, signed evidence and sample identity."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from numpy.testing import assert_array_equal

from test_choice_state import row_for
from experiments.native_support.choice_cache import CaptureReader
from experiments.native_support.choice_state_pack import pack_results
from experiments.native_support.choice_state_run import BASELINES, main


def create_input(root):
    response = dict(id="sample", source_id="source", prompt_length=3,
                    token_ids=[1, 2, 3, 10, 11], token_text=["a", "b", "c", "d", "e"])
    settings = dict(model="fixture", responses=[response])
    capture = root / "value_transport/capture/0000"
    capture.mkdir(parents=True)
    for name in ("settings.json", "value_transport/capture_settings.json"):
        (root / name).write_text(json.dumps(settings))
    sources = dict(blocks=[[0]], group_ids=[0, 2, 3, 1, 1])
    (capture / "sources.json").write_text(json.dumps(sources))
    for target, row in enumerate([row_for([-1, 0, 0]), row_for([0, 0, 0, 1])]):
        row["attention"] = np.full((1, 2, target + 3), 1 / (target + 3))
        row["edge_value_energy"] = row["attention"] ** 2
        row["head_positive"] = np.broadcast_to(row["root_positive"], (1, 2, 4, 1)).copy()
        row["head_negative"] = np.broadcast_to(row["root_negative"], (1, 2, 4, 1)).copy()
        row["ffn_choice"] = np.array([[-.2]])
        np.savez(capture / f"token_{target:06d}.npz", **row)
    baseline = {name: np.array([.1, .2]) for name in BASELINES}
    baseline.update(target=np.arange(2), query=np.array([2, 3]), token_id=np.array([10, 11]))
    directory = root / "value_transport/responses/0000"
    directory.mkdir(parents=True)
    np.savez(directory / "scores.npz", **baseline)
    return {"sample": {"source_id": "source", "token_ids": [10, 11], "labels": [0, 1]}}


class ChoicePackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "input"
        self.output = self.root / "result"
        self.annotations = create_input(self.source)

    def call(self, *args):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            main(["--input", str(self.source), "--output", str(self.output), *args])

    def test_public_run_automatically_packages_external_annotations_and_signed_heads(self):
        external = self.root / "labels.json"
        external.write_text(json.dumps(self.annotations))
        self.call("--annotations", str(external))
        archive = self.root / "result_review.zip"
        self.assertTrue(archive.is_file())
        self.assertEqual(json.loads((self.output / "annotations.json").read_text()), self.annotations)
        with ZipFile(archive) as packed:
            manifest = json.loads(packed.read("review_manifest.json"))
            self.assertTrue(manifest["annotations_included"])
            self.assertEqual(manifest["missing_result_files"], [])
            self.assertFalse(manifest["model_run"])
            self.assertEqual(packed.read("choice_state/responses/0000/scores.npz"),
                             (self.output / "responses/0000/scores.npz").read_bytes())
            name = "value_transport/capture/0000/token_000001.npz"
            with np.load(io.BytesIO(packed.read(name))) as compact:
                with np.load(self.source / name) as original:
                    for field in ("head_positive", "head_negative", "ffn_choice", "root_token_choice"):
                        assert_array_equal(compact[field], original[field])
                self.assertNotIn("attention", compact.files)
                self.assertIn("head_group_attention", compact.files)
                self.assertIn("history_total_attention", compact.files)
        self.assertNotIn("torch", sys.modules)

    def test_pack_stage_never_rescores_and_compact_input_remains_complete(self):
        self.call()
        before = {p: p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        with patch("experiments.native_support.choice_state_run.run", side_effect=AssertionError("rescored")):
            self.call("--stage", "pack", "--archive", str(self.root / "second.zip"))
        self.assertTrue(all(p.read_bytes() == data for p, data in before.items()))
        compact = self.root / "result_review.zip"
        with contextlib.redirect_stdout(io.StringIO()):
            pack_results(compact, self.output, self.root / "third.zip")
        with contextlib.closing(CaptureReader(compact)) as left:
            with contextlib.closing(CaptureReader(self.root / "third.zip")) as right:
                name = "value_transport/capture/0000/token_000001.npz"
                for field, value in left.arrays(name).items():
                    assert_array_equal(value, right.arrays(name)[field])

    def test_existing_archive_is_not_overwritten(self):
        self.call()
        archive = self.root / "result_review.zip"
        before = archive.read_bytes()
        with self.assertRaises(FileExistsError):
            pack_results(self.source, self.output)
        self.assertEqual(archive.read_bytes(), before)

    def test_protocol_mismatch_does_not_publish_partial_archive(self):
        self.call()
        path = self.source / "value_transport/capture_settings.json"
        data = json.loads(path.read_text())
        data["choices"] = 99
        path.write_text(json.dumps(data))
        archive = self.root / "mismatch.zip"
        with self.assertRaisesRegex(ValueError, "capture protocols differ"):
            pack_results(self.source, self.output, archive)
        self.assertFalse(archive.exists())
        self.assertEqual(list(self.root.glob("*.partial.zip")), [])

    def test_pack_cannot_replace_evaluated_annotation_snapshot(self):
        external = self.root / "labels.json"
        external.write_text(json.dumps(self.annotations))
        self.call("--annotations", str(external))
        self.annotations["sample"]["labels"] = [1, 0]
        external.write_text(json.dumps(self.annotations))
        archive = self.root / "wrong_labels.zip"
        with self.assertRaisesRegex(ValueError, "evaluated annotation snapshot"):
            pack_results(self.source, self.output, archive, external)
        self.assertFalse(archive.exists())


if __name__ == "__main__":
    unittest.main()
