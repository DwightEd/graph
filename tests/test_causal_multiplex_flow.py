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
from experiments.causal_multiplex_flow.artifacts import (
    load_reference,
    load_score_artifact,
)
from experiments.causal_multiplex_flow.controls import (
    lag_preserving_rewired_source,
    source_candidates,
)
from experiments.causal_multiplex_flow.calibration import topology_gate_summary
from experiments.causal_multiplex_flow.events import (
    CausalEventSample,
    EventConfig,
    RP,
    RR,
    extract_causal_events,
    log_lag_bin,
)
from experiments.causal_multiplex_flow.experiment import (
    TrainConfig,
    evaluate_cmrp,
    fit_cmrp,
    score_cmrp,
)
from experiments.causal_multiplex_flow.model import (
    CausalMultiplexRouter,
    ModelConfig,
)
from research_dataset import ResearchDataset


def _sample(sample_id: str, source_id: str, multiplier: float = 1.0):
    diagonal = torch.tensor(
        [[[0.8, 0.7, 0.4, 0.3, 0.2], [0.6, 0.5, 0.2, 0.4, 0.1]]],
        dtype=torch.float16,
    )
    columns = torch.tensor(
        [0, 2, 1, 2, 3, 1, 2, 0, 2, 3],
        dtype=torch.int32,
    )
    values = (
        torch.tensor(
            [0.2, 0.2, 0.15, 0.1, 0.3, 0.25, 0.05, 0.05, 0.4, 0.1],
            dtype=torch.float32,
        )
        * float(multiplier)
    ).to(torch.float16)
    return AttentionSample(
        sample_id,
        source_id,
        2,
        torch.tensor([10, 11, 12, 13, 14]),
        diagonal,
        torch.tensor([0, 1, 2, 5, 6, 7, 10]),
        columns,
        values,
        0.01,
    )


def _write_dataset(
    root: Path,
    multipliers,
    *,
    positive_sample: int | None = None,
    source_prefix: str = "source",
):
    (root / "attention").mkdir(parents=True)
    rows = []
    label_rows = []
    for index, multiplier in enumerate(multipliers):
        sample = _sample(f"r{index}", f"{source_prefix}-{index}", float(multiplier))
        path = root / "attention" / f"r{index}.npz"
        save_attention_sample(sample, path)
        rows.append(
            index_row(
                root,
                sample,
                path,
                metadata={
                    "split": "test" if positive_sample is not None else "train",
                    "task_type": "QA",
                    "data_source": "synthetic",
                    "generator_model": "generator",
                },
            )
        )
        label_rows.append(
            {
                "sample_id": sample.sample_id,
                "positive_runs": [[1, 2]] if positive_sample == index else [],
            }
        )
    labels = root / "labels.jsonl"
    labels.write_text(
        "".join(json.dumps(row) + "\n" for row in label_rows),
        encoding="utf-8",
    )
    write_split_index(
        root,
        rows,
        attention_floor=0.01,
        num_layers=1,
        num_heads=2,
        alignment="post_token_query_at_same_position",
        extra={
            "split": "test" if positive_sample is not None else "train",
            "labels_sha256": sha256(labels),
        },
    )
    return ResearchDataset(root)


class CausalMultiplexFlowTests(unittest.TestCase):
    def test_event_extraction_preserves_rr_source_channel_and_role_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _write_dataset(Path(directory), [1.0])
            sample = dataset["r0"]
            events = extract_causal_events(
                sample,
                config=EventConfig(
                    max_prompt_events_per_token=1,
                    max_rr_events_per_token=2,
                ),
            )
            self.assertEqual(events.response_count, 3)
            self.assertEqual(events.num_channels, 2)
            self.assertFalse(any("centroid" in name for name in events.__dataclass_fields__))
            np.testing.assert_allclose(
                events.role_summary[:, 2:].cpu().numpy(),
                [[2.0, 0.0], [0.0, 2.0], [2.0, 4.0]],
            )
            token_two = events.target_slice(2)
            relation = events.relation[token_two].cpu().numpy()
            source = events.source[token_two].cpu().numpy()
            channel = events.channel[token_two].cpu().numpy()
            lag = events.lag[token_two].cpu().numpy()
            rr = relation == RR
            self.assertEqual(set(source[rr].tolist()), {0, 1})
            self.assertEqual(set(channel[rr].tolist()), {0, 1})
            self.assertEqual(set(lag[rr].tolist()), {1, 2})
            self.assertTrue(np.all(source[relation == RP] == -1))
            sample.release_attention()

    def test_lag_preserving_control_uses_exact_source_when_available(self):
        events = CausalEventSample(
            sample_id="manual",
            response_count=5,
            num_layers=1,
            num_heads=1,
            attention_floor=0.01,
            target_ptr=torch.tensor([0, 0, 0, 0, 0, 1]),
            relation=torch.tensor([RR]),
            source=torch.tensor([1]),
            channel=torch.tensor([0]),
            weight=torch.tensor([0.4]),
            lag=torch.tensor([3]),
            role_summary=torch.tensor(
                [[0.0, 0.0, 0.0, 0.0]] * 4 + [[0.0, 0.4, 0.0, 1.0]]
            ),
        ).validate()
        rewired = lag_preserving_rewired_source(events, 0, seed=7)
        self.assertNotEqual(rewired, 1)
        self.assertLess(rewired, 4)
        self.assertEqual(log_lag_bin(4 - rewired), log_lag_bin(3))
        first = source_candidates(events, 0, negatives=3, seed=7)
        second = source_candidates(events, 0, negatives=3, seed=7)
        self.assertEqual(int(first[0]), 1)
        self.assertTrue(torch.equal(first, second))
        self.assertIn(rewired, first[1:].tolist())

    def test_router_forward_is_finite_and_backpropagates(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _write_dataset(Path(directory), [1.0])
            sample = dataset["r0"]
            events = extract_causal_events(
                sample,
                config=EventConfig(
                    max_prompt_events_per_token=2,
                    max_rr_events_per_token=4,
                ),
            )
            model = CausalMultiplexRouter(
                num_layers=1,
                num_heads=2,
                config=ModelConfig(
                    hidden_dim=12,
                    channel_embedding_dim=4,
                    relation_embedding_dim=3,
                    lag_frequencies=2,
                    negatives_per_edge=2,
                    dropout=0.0,
                    seed=3,
                ),
            )
            output = model(events)
            self.assertEqual(output.raw_route_surprise.shape, (3,))
            self.assertEqual(output.state.shape, (3, 12))
            self.assertTrue(bool(torch.isfinite(output.loss)))
            self.assertTrue(bool(torch.isfinite(output.raw_route_surprise).all()))
            output.loss.backward()
            self.assertIsNotNone(model.event_mlp.network[0].weight.grad)
            self.assertIsNotNone(model.query_mlp.network[0].weight.grad)
            sample.release_attention()

    def test_router_reports_raw_source_nll_and_edge_level_rewire_gaps(self):
        events = CausalEventSample(
            sample_id="manual",
            response_count=5,
            num_layers=1,
            num_heads=1,
            attention_floor=0.01,
            target_ptr=torch.tensor([0, 0, 0, 0, 0, 1]),
            relation=torch.tensor([RR]),
            source=torch.tensor([1]),
            channel=torch.tensor([0]),
            weight=torch.tensor([0.4]),
            lag=torch.tensor([3]),
            role_summary=torch.tensor(
                [[0.0, 0.0, 0.0, 0.0]] * 4 + [[0.0, 0.4, 0.0, 1.0]]
            ),
        ).validate()
        model = CausalMultiplexRouter(
            num_layers=1,
            num_heads=1,
            config=ModelConfig(
                hidden_dim=8,
                channel_embedding_dim=2,
                relation_embedding_dim=2,
                lag_frequencies=1,
                negatives_per_edge=1,
                dropout=0.0,
                seed=7,
            ),
        )
        for parameter in model.parameters():
            parameter.data.zero_()

        output = model(events)

        self.assertAlmostEqual(
            float(output.source_nll[4].detach()), float(np.log(2))
        )
        np.testing.assert_allclose(output.rewire_edge_gap.detach().numpy(), [0.0])

    def test_topology_gate_fails_without_a_finite_rewired_edge(self):
        gate = topology_gate_summary([], selected_edge_count=3)

        self.assertEqual(gate["evaluated_edge_count"], 0)
        self.assertEqual(gate["selected_edge_count"], 3)
        self.assertEqual(gate["coverage"], 0.0)
        self.assertIsNone(gate["mean_gap"])
        self.assertIsNone(gate["positive_fraction"])
        self.assertFalse(gate["pass"])

    def test_label_free_fit_score_then_posthoc_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _write_dataset(
                root / "train",
                [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
            )
            test = _write_dataset(
                root / "test",
                [1.0, 1.2],
                positive_sample=1,
                source_prefix="test-source",
            )
            output_dir = root / "cmrp"
            score_path = output_dir / "test_scores.npz"
            report_path = output_dir / "evaluation.json"

            fit = fit_cmrp(
                train,
                output_dir,
                event_config=EventConfig(
                    max_prompt_events_per_token=2,
                    max_rr_events_per_token=4,
                ),
                model_config=ModelConfig(
                    hidden_dim=12,
                    channel_embedding_dim=4,
                    relation_embedding_dim=3,
                    lag_frequencies=2,
                    negatives_per_edge=2,
                    dropout=0.0,
                    seed=11,
                ),
                train_config=TrainConfig(
                    epochs=1,
                    learning_rate=1e-3,
                    weight_decay=0.0,
                    gradient_clip=1.0,
                    calibration_fraction=0.25,
                    seed=11,
                ),
            )
            self.assertFalse(fit["labels_read"])
            reference = load_reference(output_dir / "reference.npz")
            self.assertTrue(
                set(reference["fit_group_id"].tolist()).isdisjoint(
                    reference["calibration_group_id"].tolist()
                )
            )
            with np.load(output_dir / "reference.npz", allow_pickle=False) as arrays:
                self.assertNotIn("label", arrays.files)
                self.assertNotIn("y_token", arrays.files)
                self.assertIn("calibration_raw_route_surprise", arrays.files)
                self.assertIn("topology_gate_evaluated_edge_count", arrays.files)
                self.assertIn("topology_gate_selected_edge_count", arrays.files)
                self.assertIn("topology_gate_coverage", arrays.files)
                self.assertIn("topology_gate_positive_fraction", arrays.files)
                evaluated = int(arrays["topology_gate_evaluated_edge_count"])
                selected = int(arrays["topology_gate_selected_edge_count"])
                self.assertLessEqual(evaluated, selected)
                self.assertAlmostEqual(
                    float(arrays["topology_gate_coverage"]),
                    evaluated / selected,
                )
                positive_fraction = float(arrays["topology_gate_positive_fraction"])
                if evaluated:
                    self.assertGreaterEqual(positive_fraction, 0.0)
                    self.assertLessEqual(positive_fraction, 1.0)
                else:
                    self.assertTrue(np.isnan(positive_fraction))

            scored = score_cmrp(test, output_dir / "reference.npz", score_path)
            self.assertFalse(scored["labels_read"])
            self.assertEqual(scored["tokens"], 6)
            artifact = load_score_artifact(score_path)
            self.assertEqual(artifact["score"].shape, (6,))
            with np.load(score_path, allow_pickle=False) as arrays:
                self.assertNotIn("label", arrays.files)
                self.assertNotIn("y_token", arrays.files)
                self.assertIn("raw_route_surprise", arrays.files)
                self.assertIn("test_group_id", arrays.files)
                self.assertEqual(arrays["test_sample_id"].tolist(), ["r0", "r1"])
                self.assertEqual(str(arrays["audit_scope"].item()), "complete_split")
                self.assertNotIn("score_source_nll", arrays.files)

            report = evaluate_cmrp(test, score_path, report_path)
            self.assertTrue(report["labels_read"])
            self.assertEqual(report["metrics"]["tokens"], 6)
            self.assertEqual(report["metrics"]["positive_tokens"], 1)
            self.assertIn("source_nll", report["components"])
            self.assertTrue(report_path.is_file())

    def test_fit_reproduces_model_initialization_and_dropout_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _write_dataset(root / "train", [0.7, 0.8, 0.9, 1.0])
            event_config = EventConfig(
                max_prompt_events_per_token=2,
                max_rr_events_per_token=4,
            )
            model_config = ModelConfig(
                hidden_dim=12,
                channel_embedding_dim=4,
                relation_embedding_dim=3,
                lag_frequencies=2,
                negatives_per_edge=2,
                dropout=0.2,
                seed=11,
            )
            train_config = TrainConfig(
                epochs=1,
                learning_rate=1e-3,
                weight_decay=0.0,
                gradient_clip=1.0,
                calibration_fraction=0.25,
                seed=11,
            )
            first = root / "first"
            second = root / "second"

            fit_cmrp(
                train,
                first,
                event_config=event_config,
                model_config=model_config,
                train_config=train_config,
            )
            fit_cmrp(
                train,
                second,
                event_config=event_config,
                model_config=model_config,
                train_config=train_config,
            )

            first_state = torch.load(first / "model.pt", weights_only=False)[
                "state_dict"
            ]
            second_state = torch.load(second / "model.pt", weights_only=False)[
                "state_dict"
            ]
            for name, value in first_state.items():
                self.assertTrue(torch.equal(value, second_state[name]), name)

    def test_score_rejects_test_source_overlap_with_frozen_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _write_dataset(root / "train", [0.7, 0.8, 0.9, 1.0])
            test = _write_dataset(root / "test", [1.0], positive_sample=0)
            output_dir = root / "cmrp"
            reference_path = output_dir / "reference.npz"
            score_path = output_dir / "test_scores.npz"

            fit_cmrp(
                train,
                output_dir,
                event_config=EventConfig(
                    max_prompt_events_per_token=2,
                    max_rr_events_per_token=4,
                ),
                model_config=ModelConfig(
                    hidden_dim=12,
                    channel_embedding_dim=4,
                    relation_embedding_dim=3,
                    lag_frequencies=2,
                    negatives_per_edge=2,
                    dropout=0.0,
                    seed=11,
                ),
                train_config=TrainConfig(
                    epochs=1,
                    learning_rate=1e-3,
                    weight_decay=0.0,
                    gradient_clip=1.0,
                    calibration_fraction=0.25,
                    seed=11,
                ),
            )

            with self.assertRaisesRegex(ValueError, "source groups must be disjoint"):
                score_cmrp(test, reference_path, score_path)
            self.assertFalse(score_path.exists())


if __name__ == "__main__":
    unittest.main()
