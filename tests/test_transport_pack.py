"""CPU export checks: source totals, exact retained effects and key alignment."""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from experiments.native_support.transport_pack import grouped_reads, history_addresses


class TransportPackTests(unittest.TestCase):
    def test_group_totals_keep_every_head_and_separate_predictor_self(self):
        attention = np.arange(1, 31, dtype=np.float32).reshape(2, 3, 5)
        attention /= attention.sum(-1, keepdims=True)
        norm = 2 * attention
        groups = np.array([0, 0, 1, 2, 3])
        measured = grouped_reads(attention, norm ** 2, groups, 4)
        self.assertEqual(measured["head_group_attention"].shape, (2, 3, 5))
        np.testing.assert_allclose(measured["head_group_attention"].sum(-1), 1, atol=1e-7)
        np.testing.assert_allclose(measured["head_group_message_norm"].sum(-1), 2, atol=1e-7)
        np.testing.assert_allclose(measured["head_group_attention"][..., -1], attention[..., -1])
        np.testing.assert_array_equal(groups, [0, 0, 1, 2, 3])

    def test_history_top_eight_preserves_total_and_reports_discarded_mass(self):
        attention = np.arange(1, 41, dtype=np.float32).reshape(2, 20)
        attention /= attention.sum(-1, keepdims=True)
        result = history_addresses(attention, prompt=3, query=19)
        expected_keys = np.broadcast_to(np.arange(18, 10, -1), (2, 8))
        np.testing.assert_array_equal(result["history_top_key"], expected_keys)
        np.testing.assert_allclose(result["history_top_attention"], attention[:, 11:19][:, ::-1])
        np.testing.assert_allclose(result["history_total_attention"], attention[:, 3:19].sum(-1))
        self.assertTrue((result["history_top_retained_attention"] < result["history_total_attention"]).all())
        empty = history_addresses(attention[..., :3], prompt=3, query=2)
        self.assertEqual(empty["history_top_key"].shape, (2, 0))

    def test_main_packs_saved_data_without_loading_model_or_changing_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            capture = output / "value_transport/capture/0000"
            capture.mkdir(parents=True)
            response = {"id": "fixture", "prompt_length": 2, "token_ids": [1, 2, 3, 4, 5]}
            (output / "settings.json").write_text(json.dumps({"responses": [response]}))
            (output / "annotations.json").write_text(json.dumps({"fixture": {"labels": [0, 1, 1]}}))
            (capture / "sources.json").write_text(json.dumps({"blocks": ["source"]}))
            effects = np.arange(16, dtype=np.float32).reshape(1, 2, 4, 2)
            for target in range(3):
                query = target + 1
                attention = np.full((1, 2, query + 1), 1 / (query + 1), dtype=np.float32)
                np.savez(capture / f"token_{target:06d}.npz", attention=attention,
                    edge_value_energy=attention ** 2, group_ids=np.array([0, 2, 1, 1])[:query + 1],
                    root_positive=np.ones((4, 2)), head_positive=effects, query=query,
                    candidate_ids=[response["token_ids"][query + 1], 6, 7], ledger_error=[.1, -.2])
            before = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
            # Fail if the public command tries to import any model implementation.
            (root / "torch.py").write_text("raise RuntimeError('packing imported torch')\n")
            (root / "transformers.py").write_text("raise RuntimeError('packing imported transformers')\n")
            launcher = "import sys; sys.path.insert(0, sys.argv.pop(1)); import main; main.main()"
            archive = root / "audit.zip"
            run = subprocess.run([sys.executable, "-c", launcher, str(root), "transport-pack",
                                  "--output", str(output), "--archive", str(archive)],
                                 capture_output=True, text=True, check=True)
            self.assertIn("3 tokens", run.stdout)
            with ZipFile(archive) as packed:
                schema = json.loads(packed.read("audit_pack_schema.json"))
                self.assertEqual(schema["captured_tokens"], 3)
                self.assertFalse(schema["heads_filtered"])
                path = "value_transport/capture/0000/token_000002.npz"
                with np.load(io.BytesIO(packed.read(path)), allow_pickle=False) as saved:
                    self.assertNotIn("attention", saved.files)
                    np.testing.assert_array_equal(saved["head_positive"], effects)
                    np.testing.assert_array_equal(saved["candidate_ids"], [5, 6, 7])
                    np.testing.assert_array_equal(saved["ledger_error"], [.1, -.2])
                    np.testing.assert_array_equal(saved["history_top_key"], [[[2], [2]]])
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)


if __name__ == "__main__":
    unittest.main()
