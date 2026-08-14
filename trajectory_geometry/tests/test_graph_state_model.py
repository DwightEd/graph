from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from trajectory_geometry.data import load_attention_sample
from trajectory_geometry.hidden import (
    discover_hidden_files,
    load_hidden_sample,
    pair_attention_hidden,
)
from trajectory_geometry.state_model import (
    StateModelConfig,
    build_state_features,
    causal_rewire_sources,
    encode_graph_state,
    fit_graph_state_model,
    load_graph_state_model,
    save_graph_state_model,
)
from trajectory_geometry.state_pipeline import run_graph_state_pipeline


def _arrays(sample_id: str, seed: int, *, layers: int = 2, heads: int = 2):
    prompt, response, dimension = 8, 20, 4
    tokens = prompt + response
    diagonal = np.full((layers, heads, tokens), 0.1, dtype=np.float32)
    columns = []
    values = []
    row_ptr = [0]
    messages = np.zeros((layers, heads, response, dimension), dtype=np.float32)
    rng = np.random.default_rng(seed)
    states = np.zeros((layers + 1, tokens, dimension), dtype=np.float32)
    states[0] = rng.normal(size=(tokens, dimension))

    edge_rows = []
    for layer in range(layers):
        for head in range(heads):
            for query in range(response):
                target = prompt + query
                prompt_source = (3 * query + 2 * head + layer) % prompt
                edge = [(prompt_source, 0.55)]
                if query:
                    edge.append((target - 1, 0.2))
                edge_rows.append(edge)
                for source, weight in sorted(edge):
                    columns.append(source)
                    values.append(weight)
                row_ptr.append(len(columns))

    for layer in range(layers):
        states[layer + 1, :prompt] = states[layer, :prompt]
        row = 0
        for current_layer in range(layers):
            for head in range(heads):
                for query in range(response):
                    if current_layer == layer:
                        target = prompt + query
                        message = diagonal[layer, head, target] * states[layer, target]
                        for source, weight in edge_rows[row]:
                            message += weight * states[layer, source]
                        messages[layer, head, query] = message
                    row += 1
        update = 0.15 * states[layer, prompt:] + 0.7 * messages[layer].sum(axis=0)
        states[layer + 1, prompt:] = states[layer, prompt:] + update

    attention = {
        "response_id": np.asarray(sample_id),
        "response_idx": np.asarray(prompt, dtype=np.int32),
        "token_ids": np.arange(tokens, dtype=np.int64),
        "attention_diagonal": diagonal,
        "response_row_ptr": np.asarray(row_ptr, dtype=np.int64),
        "response_column_indices": np.asarray(columns, dtype=np.int64),
        "response_values": np.asarray(values, dtype=np.float32),
        "attention_floor": np.asarray(0.01, dtype=np.float32),
    }
    hidden = {
        "sample_id": np.asarray(sample_id),
        "token_ids": np.arange(tokens, dtype=np.int64),
        "hidden_states": states,
    }
    return attention, hidden


def _write_split(root: Path, split: str, count: int, seed: int):
    attention = root / "attention" / split
    hidden = root / "hidden" / split
    attention.mkdir(parents=True)
    hidden.mkdir(parents=True)
    for index in range(count):
        sample_id = f"{split}_{index}"
        attention_payload, hidden_payload = _arrays(sample_id, seed + index)
        np.savez_compressed(attention / f"attention_{sample_id}.npz", **attention_payload)
        np.savez_compressed(hidden / f"hidden_{sample_id}.npz", **hidden_payload)
    return attention, hidden


class GraphStateModelTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch is unavailable in this test environment")
    def test_pt_hidden_sidecar_uses_the_same_validated_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attention_payload, hidden_payload = _arrays("sample", 1)
            attention_path = root / "attention_sample.pt"
            hidden_path = root / "hidden_sample.pt"
            def tensor_payload(payload):
                return {
                    key: (
                        torch.as_tensor(value)
                        if isinstance(value, np.ndarray)
                        and (np.issubdtype(value.dtype, np.number) or value.dtype == np.bool_)
                        else value.item()
                        if isinstance(value, np.ndarray) and value.ndim == 0
                        else value
                    )
                    for key, value in payload.items()
                }

            torch.save(tensor_payload(attention_payload), attention_path)
            torch.save(tensor_payload(hidden_payload), hidden_path)
            attention = load_attention_sample(attention_path)
            states, offset = load_hidden_sample(hidden_path).align(attention)
            self.assertEqual(states.shape, (3, 28, 4))
            self.assertEqual(offset, 0)

    def test_hidden_alignment_accepts_both_layer_conventions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attention_payload, hidden_payload = _arrays("sample", 1)
            attention_path = root / "attention_sample.npz"
            hidden_path = root / "hidden_sample.npz"
            np.savez_compressed(attention_path, **attention_payload)
            np.savez_compressed(hidden_path, **hidden_payload)
            attention = load_attention_sample(attention_path)
            hidden = load_hidden_sample(hidden_path)
            states, offset = hidden.align(attention)
            self.assertEqual(states.shape, (3, 28, 4))
            self.assertEqual(offset, 0)

            hidden_payload["hidden_states"] = states[1:].transpose(1, 0, 2)
            np.savez_compressed(hidden_path, **hidden_payload)
            states, offset = load_hidden_sample(hidden_path).align(attention)
            self.assertEqual(states.shape, (2, 28, 4))
            self.assertEqual(offset, 1)

    def test_causal_rewire_preserves_relation_and_distance_bucket(self):
        source = np.asarray([0, 3, 8, 9, 10, 12], dtype=np.int64)
        target = np.asarray([12, 12, 12, 13, 15, 20], dtype=np.int64)
        layer = np.zeros_like(source)
        head = np.arange(source.size) % 2
        changed = causal_rewire_sources(
            source,
            target,
            layer,
            head,
            prompt_count=8,
            prompt_bins=2,
            seed=9,
        )
        self.assertTrue(np.all(changed < target))
        np.testing.assert_array_equal(source < 8, changed < 8)
        original_lag = target[source >= 8] - source[source >= 8]
        changed_lag = target[source >= 8] - changed[source >= 8]
        edges = np.asarray([1, 2, 4, 8, 16, 32])
        np.testing.assert_array_equal(
            np.searchsorted(edges, original_lag, side="left"),
            np.searchsorted(edges, changed_lag, side="left"),
        )

    def test_length_normalized_control_and_messages_are_not_renormalized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attention_payload, hidden_payload = _arrays("sample", 2)
            attention_path = root / "attention_sample.npz"
            hidden_path = root / "hidden_sample.npz"
            np.savez_compressed(attention_path, **attention_payload)
            np.savez_compressed(hidden_path, **hidden_payload)
            attention = load_attention_sample(attention_path)
            states, offset = load_hidden_sample(hidden_path).align(attention)
            config = StateModelConfig(
                projection_dim=2,
                projection_reference_rows=32,
                head_components=2,
                fit_tokens_per_layer=16,
                prompt_rewire_bins=2,
            )
            projected = states[..., :2]
            result = build_state_features(
                attention,
                projected,
                attention_layer_offset=offset,
                head_bucket=np.asarray([0, 1]),
                head_sign_scale=np.ones(2, dtype=np.float32),
                config=config,
            )
            # First response token has prompt mass .55, self .1, and no RR edge.
            self.assertAlmostEqual(float(result.controls[0, 0, 1]), 0.55 / 0.65, places=5)
            self.assertAlmostEqual(float(result.controls[0, 0, 2]), 1.0, places=5)
            self.assertAlmostEqual(float(result.controls[0, 0, 4]), 0.35, places=5)
            self.assertFalse(np.allclose(result.true_message, result.rewired_message))

    def test_end_to_end_true_topology_beats_rewired_and_serializes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_attention, train_hidden = _write_split(root, "train", 10, 10)
            test_attention, test_hidden = _write_split(root, "test", 2, 100)
            train_pairs = pair_attention_hidden(
                sorted(train_attention.glob("*.npz")), discover_hidden_files(train_hidden)
            )
            test_pairs = pair_attention_hidden(
                sorted(test_attention.glob("*.npz")), discover_hidden_files(test_hidden)
            )
            config = StateModelConfig(
                projection_dim=2,
                projection_reference_rows=180,
                head_components=2,
                fit_tokens_per_layer=180,
                fit_fraction=0.75,
                trim_fraction=0.95,
                ridge=1e-4,
                dct_components=2,
                prompt_rewire_bins=2,
                seed=17,
            )
            model = fit_graph_state_model(train_pairs, config)
            self.assertLess(
                model.calibration["prediction_mse"]["true_graph"],
                model.calibration["prediction_mse"]["rewired_graph"],
            )
            model_path = root / "model.npz"
            save_graph_state_model(model, model_path)
            restored = load_graph_state_model(model_path)
            attention = load_attention_sample(test_pairs[0].attention_path)
            states, offset = load_hidden_sample(test_pairs[0].hidden_path).align(attention)
            first = encode_graph_state(attention, states, offset, model)
            second = encode_graph_state(attention, states, offset, restored)
            np.testing.assert_allclose(
                first.embeddings["true_graph"], second.embeddings["true_graph"], atol=1e-6
            )
            self.assertEqual(first.embeddings["true_graph"].shape, (20, 4))

            output = root / "output"
            manifest = run_graph_state_pipeline(
                train_pairs,
                test_pairs,
                output,
                config=config,
                save_train_embeddings=False,
            )
            self.assertFalse(manifest["labels_read"])
            self.assertEqual(manifest["splits"]["test"]["response_tokens"], 40)
            sample_file = output / "test" / "state_test_0.npz"
            with np.load(sample_file, allow_pickle=False) as payload:
                self.assertEqual(payload["true_graph_embedding"].shape, (20, 4))
                self.assertFalse(bool(payload["labels_included"]))

    def test_graph_gate_does_not_reward_a_self_only_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attention_root = root / "attention"
            hidden_root = root / "hidden"
            attention_root.mkdir()
            hidden_root.mkdir()
            for index in range(8):
                sample_id = f"null_{index}"
                attention_payload, hidden_payload = _arrays(sample_id, 200 + index)
                states = hidden_payload["hidden_states"]
                states[1] = 1.2 * states[0]
                states[2] = 1.2 * states[1]
                hidden_payload["hidden_states"] = states
                np.savez_compressed(
                    attention_root / f"attention_{sample_id}.npz", **attention_payload
                )
                np.savez_compressed(hidden_root / f"hidden_{sample_id}.npz", **hidden_payload)
            pairs = pair_attention_hidden(
                sorted(attention_root.glob("*.npz")), sorted(hidden_root.glob("*.npz"))
            )
            model = fit_graph_state_model(
                pairs,
                StateModelConfig(
                    projection_dim=2,
                    projection_reference_rows=128,
                    head_components=2,
                    fit_tokens_per_layer=120,
                    fit_fraction=0.75,
                    ridge=1e-3,
                    prompt_rewire_bins=2,
                    seed=23,
                ),
            )
            mse = model.calibration["prediction_mse"]
            self.assertLess(mse["node_control"], 1e-7)
            self.assertLess(mse["node_control"] - mse["true_graph"], 1e-7)
            self.assertFalse(model.calibration["gate_passed"])


if __name__ == "__main__":
    unittest.main()
