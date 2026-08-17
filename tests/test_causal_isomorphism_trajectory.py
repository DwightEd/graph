import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import attention_graph.causal_events as causal_events

from attention_graph.causal_events import (
    CausalMultiplexEvents,
    MultiplexEventConfig,
    RR,
    SUMMARY_NAMES,
    extract_causal_multiplex_events,
    log_lag_bin,
)
from attention_graph.topology_controls import (
    event_target,
    rewire_causal_sources,
)
from cache import (
    AttentionSample,
    index_row,
    save_attention_sample,
    sha256,
    write_split_index,
)
from experiments.causal_isomorphism_trajectory.artifacts import (
    load_reference,
    load_score_artifact,
)
from experiments.causal_isomorphism_trajectory.experiment import (
    evaluate_citg,
    fit_citg,
    score_citg,
)
from experiments.causal_isomorphism_trajectory.geometry import (
    GeometryConfig,
)
from experiments.causal_isomorphism_trajectory.signatures import (
    SignatureConfig,
    extract_trajectory_features,
)
from research_dataset import ResearchDataset


def _sample(
    sample_id: str,
    source_id: str,
    multiplier: float = 1.0,
):
    layers = 2
    heads = 2
    prompt = 2
    response = 4
    tokens = prompt + response
    diagonal = torch.full(
        (layers, heads, tokens), 0.2, dtype=torch.float16
    )
    row_ptr = [0]
    columns = []
    values = []
    base_rows = (
        ((0,), (0.20,)),
        ((1, 2), (0.12, 0.22)),
        ((0, 2, 3), (0.10, 0.16, 0.24)),
        ((1, 3, 4), (0.11, 0.18, 0.26)),
    )
    for layer in range(layers):
        for head in range(heads):
            factor = float(multiplier) * (
                1.0 + 0.05 * layer + 0.03 * head
            )
            for source_row, weight_row in base_rows:
                columns.extend(source_row)
                values.extend([factor * value for value in weight_row])
                row_ptr.append(len(columns))
    return AttentionSample(
        sample_id,
        source_id,
        prompt,
        torch.arange(10, 10 + tokens),
        diagonal,
        torch.tensor(row_ptr, dtype=torch.int64),
        torch.tensor(columns, dtype=torch.int32),
        torch.tensor(values, dtype=torch.float16),
        0.01,
    )


def _write_dataset(
    root: Path,
    multipliers,
    *,
    split: str,
    source_prefix: str,
    positive_sample: int | None = None,
):
    (root / "attention").mkdir(parents=True)
    rows = []
    label_rows = []
    for index, multiplier in enumerate(multipliers):
        sample = _sample(
            f"{split}-{index}",
            f"{source_prefix}-{index}",
            float(multiplier),
        )
        path = root / "attention" / f"{sample.sample_id}.npz"
        save_attention_sample(sample, path)
        rows.append(
            index_row(
                root,
                sample,
                path,
                metadata={
                    "split": split,
                    "task_type": "QA",
                    "data_source": "synthetic",
                    "generator_model": "generator",
                },
            )
        )
        label_rows.append(
            {
                "sample_id": sample.sample_id,
                "positive_runs": (
                    [[2, 4]] if positive_sample == index else []
                ),
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
        num_layers=2,
        num_heads=2,
        alignment="post_token_query_at_same_position",
        extra={
            "split": split,
            "labels_sha256": sha256(labels),
        },
    )
    return ResearchDataset(root)


def _manual_events(
    *,
    sample_id: str,
    response_count: int,
    edges,
):
    """Edges are (source, target, layer, head, weight)."""
    edges = sorted(edges, key=lambda value: value[1])
    target_count = np.zeros(response_count, dtype=np.int64)
    for _, target, _, _, _ in edges:
        target_count[target] += 1
    target_ptr = np.zeros(response_count + 1, dtype=np.int64)
    target_ptr[1:] = np.cumsum(target_count)
    summary = torch.zeros(
        (response_count, len(SUMMARY_NAMES)), dtype=torch.float32
    )
    band_summary = torch.zeros(
        (response_count, 1, len(SUMMARY_NAMES)),
        dtype=torch.float32,
    )
    for _, target, _, _, weight in edges:
        summary[target, 1] += weight
        summary[target, 3] += 1
        summary[target, 5] += weight * np.log(weight)
        summary[target, 7] = max(
            float(summary[target, 7]), weight
        )
    summary[:, 8] = summary[:, 5]
    summary[:, 9] = summary[:, 7]
    band_summary[:, 0] = summary
    return CausalMultiplexEvents(
        sample_id=sample_id,
        response_count=response_count,
        num_layers=1,
        num_heads=1,
        attention_floor=0.01,
        layer_bands=1,
        target_ptr=torch.tensor(target_ptr, dtype=torch.long),
        relation=torch.full((len(edges),), RR, dtype=torch.long),
        source=torch.tensor([edge[0] for edge in edges]),
        layer=torch.tensor([edge[2] for edge in edges]),
        head=torch.tensor([edge[3] for edge in edges]),
        weight=torch.tensor(
            [edge[4] for edge in edges], dtype=torch.float32
        ),
        lag=torch.tensor(
            [edge[1] - edge[0] for edge in edges],
            dtype=torch.long,
        ),
        role_summary=summary,
        band_summary=band_summary,
    ).validate()


class CausalIsomorphismTrajectoryTests(unittest.TestCase):
    def test_event_extraction_is_band_balanced_and_has_no_centroid(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _write_dataset(
                Path(directory),
                [1.0],
                split="train",
                source_prefix="train-source",
            )
            sample = dataset["train-0"]
            events = extract_causal_multiplex_events(
                sample,
                config=MultiplexEventConfig(
                    layer_bands=2,
                    max_prompt_events_per_band=1,
                    max_rr_events_per_band=1,
                ),
            )
            self.assertEqual(events.num_layers, 2)
            self.assertEqual(events.layer_bands, 2)
            self.assertFalse(
                any(
                    "centroid" in name
                    for name in events.__dataclass_fields__
                )
            )
            token = events.target_slice(3)
            self.assertEqual(
                set(events.band[token].tolist()), {0, 1}
            )
            self.assertTrue(
                bool((events.source[events.relation == RR] >= 0).all())
            )
            sample.release_attention()

    def test_lag_preserving_rewire_changes_source_and_keeps_channel(self):
        events = _manual_events(
            sample_id="manual",
            response_count=5,
            edges=[
                (0, 1, 0, 0, 0.2),
                (1, 2, 0, 0, 0.3),
                (1, 4, 0, 0, 0.4),
            ],
        )
        rewired, changed = rewire_causal_sources(events, seed=7)
        candidates = torch.nonzero(changed, as_tuple=False).flatten()
        self.assertGreater(len(candidates), 0)
        edge = int(candidates[-1])
        self.assertNotEqual(
            int(events.source[edge]), int(rewired.source[edge])
        )
        self.assertEqual(
            log_lag_bin(int(events.lag[edge])),
            log_lag_bin(int(rewired.lag[edge])),
        )
        self.assertEqual(
            int(events.channel[edge]), int(rewired.channel[edge])
        )
        self.assertEqual(
            event_target(events, edge),
            event_target(rewired, edge),
        )

    def test_shifted_rooted_motifs_have_equal_topology_signature(self):
        left = _manual_events(
            sample_id="left",
            response_count=4,
            edges=[
                (1, 2, 0, 0, 0.3),
                (2, 3, 0, 0, 0.4),
            ],
        )
        right = _manual_events(
            sample_id="right",
            response_count=6,
            edges=[
                (3, 4, 0, 0, 0.3),
                (4, 5, 0, 0, 0.4),
            ],
        )
        config = SignatureConfig(
            hash_dim=16,
            position_buckets=4,
            max_parent_events=4,
        )
        left_features = extract_trajectory_features(left, config=config)
        right_features = extract_trajectory_features(right, config=config)
        np.testing.assert_allclose(
            left_features.global_signature[3],
            right_features.global_signature[5],
        )

    def test_rewire_changes_rooted_signature_when_available(self):
        events = _manual_events(
            sample_id="rewire",
            response_count=5,
            edges=[
                (0, 1, 0, 0, 0.2),
                (1, 2, 0, 0, 0.3),
                (0, 3, 0, 0, 0.25),
                (1, 4, 0, 0, 0.4),
            ],
        )
        rewired, changed = rewire_causal_sources(events, seed=3)
        self.assertTrue(bool(changed.any()))
        config = SignatureConfig(hash_dim=32, max_parent_events=4)
        original = extract_trajectory_features(events, config=config)
        counterfactual = extract_trajectory_features(
            rewired, config=config
        )
        self.assertGreater(
            float(
                np.abs(
                    original.global_signature
                    - counterfactual.global_signature
                ).sum()
            ),
            0.0,
        )

    def test_label_free_fit_score_then_posthoc_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _write_dataset(
                root / "train",
                [
                    0.70,
                    0.75,
                    0.80,
                    0.85,
                    0.90,
                    0.95,
                    1.00,
                    1.05,
                    1.10,
                    1.15,
                    1.20,
                    1.25,
                ],
                split="train",
                source_prefix="train-source",
            )
            test = _write_dataset(
                root / "test",
                [1.0, 1.2],
                split="test",
                source_prefix="test-source",
                positive_sample=1,
            )
            output_dir = root / "citg"
            reference_path = output_dir / "reference.npz"
            score_path = output_dir / "test_scores.npz"
            report_path = output_dir / "evaluation.json"

            fit = fit_citg(
                train,
                output_dir,
                event_config=MultiplexEventConfig(
                    layer_bands=2,
                    max_prompt_events_per_band=1,
                    max_rr_events_per_band=2,
                ),
                signature_config=SignatureConfig(
                    hash_dim=16,
                    lag_bins=3,
                    position_buckets=3,
                    max_parent_events=3,
                ),
                geometry_config=GeometryConfig(
                    pca_dim=2,
                    reference_per_sample=3,
                    min_condition_rows=2,
                    trim_fraction=1.0,
                    calibration_fraction=0.25,
                    bootstrap_replicates=10,
                    topology_gate_min_coverage=0.0,
                    seed=11,
                ),
            )
            self.assertFalse(fit["labels_read"])
            reference = load_reference(reference_path)
            self.assertTrue(
                set(reference["fit_group_id"].tolist()).isdisjoint(
                    reference["calibration_group_id"].tolist()
                )
            )
            with np.load(reference_path, allow_pickle=False) as arrays:
                self.assertNotIn("label", arrays.files)
                self.assertNotIn("y_token", arrays.files)
                self.assertIn("calibration_energy_full", arrays.files)

            scored = score_citg(test, reference_path, score_path)
            self.assertFalse(scored["labels_read"])
            self.assertEqual(scored["tokens"], 8)
            artifact = load_score_artifact(score_path)
            self.assertEqual(artifact["score"].shape, (8,))
            with np.load(score_path, allow_pickle=False) as arrays:
                self.assertNotIn("label", arrays.files)
                self.assertNotIn("y_token", arrays.files)
                self.assertIn("score_static", arrays.files)
                self.assertIn("score_topology", arrays.files)
                self.assertIn("score_mass", arrays.files)

            report = evaluate_citg(test, score_path, report_path)
            self.assertTrue(report["labels_read"])
            self.assertEqual(report["metrics"]["tokens"], 8)
            self.assertEqual(report["metrics"]["positive_tokens"], 2)
            self.assertIn(
                "static_state_ablation", report["components"]
            )
            self.assertTrue(report_path.is_file())


if __name__ == "__main__":
    unittest.main()
