import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from cache import (
    AttentionSample,
    index_row,
    save_attention_sample,
    sha256,
    write_split_index,
)
from experiments.causal_typed_path_debruijn.artifacts import (
    load_reference,
    load_score_artifact,
)
from experiments.causal_typed_path_debruijn.calibration import (
    build_calibration,
    score_channels,
)
from experiments.causal_typed_path_debruijn.change_lockin import (
    change_lockin_score,
    fit_robust_change_stats,
    prompt_lineage_drop,
)
from experiments.causal_typed_path_debruijn.config import (
    CalibrationConfig,
    ChangeConfig,
    DeBruijnConfig,
    GraphConfig,
    PathConfig,
)
from experiments.causal_typed_path_debruijn.debruijn import DeBruijnAccumulator
from experiments.causal_typed_path_debruijn.evaluation import evaluate_scores
from experiments.causal_typed_path_debruijn.experiment import (
    ExperimentConfig,
    fit_reference,
    score_split,
    visualize_scored_sample,
)
from experiments.causal_typed_path_debruijn import main as method_main
from experiments.causal_typed_path_debruijn.graph_builder import (
    build_causal_routing_graph,
)
from experiments.causal_typed_path_debruijn.layered_automaton import (
    R_PLUS,
    layered_attention_automaton,
)
from experiments.causal_typed_path_debruijn.nulls import (
    causal_endpoint_rewire,
    offline_noncausal_bucket_time_shuffle,
)
from experiments.causal_typed_path_debruijn.spectral_bridge import (
    ALLOWED_RR_SCORE,
    build_rr_hybrid,
    load_rr_hybrid,
)
from experiments.causal_typed_path_debruijn.typed_path_dp import typed_path_dp
from research_dataset import ResearchDataset


def _sample(sample_id: str, source_id: str, multiplier: float = 1.0):
    layers, heads, response_idx, tokens = 2, 2, 2, 6
    response_count = tokens - response_idx
    diagonal = torch.full((layers, heads, tokens), 0.1, dtype=torch.float16)
    diagonal[:, :, response_idx:] = torch.tensor(
        [0.24, 0.22, 0.20, 0.18], dtype=torch.float16
    )
    rows = (
        ([0, 1], [0.12, 0.16]),
        ([1, 2], [0.10, 0.24]),
        ([0, 2, 3], [0.08, 0.16, 0.22]),
        ([1, 2, 3, 4], [0.07, 0.11, 0.13, 0.19]),
    )
    row_ptr = [0]
    columns = []
    values = []
    for channel in range(layers * heads):
        channel_scale = 1.0 + 0.04 * channel
        for source, weight in rows:
            columns.extend(source)
            values.extend(
                float(multiplier) * channel_scale * value for value in weight
            )
            row_ptr.append(len(columns))
    return AttentionSample(
        sample_id,
        source_id,
        response_idx,
        torch.arange(10, 10 + tokens, dtype=torch.long),
        diagonal,
        torch.tensor(row_ptr, dtype=torch.int64),
        torch.tensor(columns, dtype=torch.int32),
        torch.tensor(values, dtype=torch.float16),
        0.01,
    )


def _prefix_sample(sample: AttentionSample, response_count: int) -> AttentionSample:
    response_count = int(response_count)
    channels = sample.num_channels
    full_response_count = sample.num_response_tokens
    token_count = sample.response_idx + response_count
    row_ptr = [0]
    columns = []
    values = []
    for channel in range(channels):
        for query in range(response_count):
            row = channel * full_response_count + query
            start = int(sample.response_row_ptr[row])
            stop = int(sample.response_row_ptr[row + 1])
            columns.extend(sample.response_column_indices[start:stop].tolist())
            values.extend(sample.response_values[start:stop].tolist())
            row_ptr.append(len(columns))
    return AttentionSample(
        f"{sample.sample_id}-prefix-{response_count}",
        sample.source_id,
        sample.response_idx,
        sample.token_ids[:token_count].clone(),
        sample.attention_diagonal[:, :, :token_count].clone(),
        torch.tensor(row_ptr, dtype=torch.int64),
        torch.tensor(columns, dtype=torch.int32),
        torch.tensor(values, dtype=torch.float16),
        sample.attention_floor,
    )


class _PublicSample:
    def __init__(self, attention):
        self.sample_id = attention.sample_id
        self._attention = attention

    def attention(self):
        return self._attention

    def iter_sparse_attention_blocks(self, block_rows=4096):
        response_count = self._attention.num_response_tokens
        rows_per_layer = self._attention.num_heads * response_count
        total_rows = self._attention.num_channels * response_count
        row_ptr = self._attention.response_row_ptr.long()
        for row_start in range(0, total_rows, block_rows):
            row_stop = min(row_start + block_rows, total_rows)
            pointer = row_ptr[row_start : row_stop + 1]
            lengths = pointer[1:] - pointer[:-1]
            rows = torch.repeat_interleave(
                torch.arange(row_start, row_stop), lengths
            )
            value_start, value_stop = int(pointer[0]), int(pointer[-1])
            query = rows.remainder(response_count)
            from research_dataset import SparseAttentionBlock

            yield SparseAttentionBlock(
                row=rows,
                layer=torch.div(rows, rows_per_layer, rounding_mode="floor"),
                head=torch.div(
                    rows.remainder(rows_per_layer),
                    response_count,
                    rounding_mode="floor",
                ),
                query=query,
                target=self._attention.response_idx + query,
                source=self._attention.response_column_indices[
                    value_start:value_stop
                ].long(),
                weight=self._attention.response_values[value_start:value_stop],
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
    labels = []
    for index, multiplier in enumerate(multipliers):
        sample = _sample(f"r{index}", f"{source_prefix}{index}", multiplier)
        path = root / "attention" / f"r{index}.npz"
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
        labels.append(
            {
                "sample_id": sample.sample_id,
                "positive_runs": [[2, 3]] if positive_sample == index else [],
            }
        )
    label_path = root / "labels.jsonl"
    label_path.write_text(
        "".join(json.dumps(row) + "\n" for row in labels), encoding="utf-8"
    )
    write_split_index(
        root,
        rows,
        attention_floor=0.01,
        num_layers=2,
        num_heads=2,
        alignment="post_token_query_at_same_position",
        extra={"split": split, "labels_sha256": sha256(label_path)},
    )
    return ResearchDataset(root)


def _hybrid_mock_artifacts(root: Path):
    """Minimal already-validated component rows for bridge contract tests."""

    rows = {
        "sample_id": np.asarray(["r0", "r0"]),
        "source_id": np.asarray(["test-source", "test-source"]),
        "token_index": np.asarray([0, 1], dtype=np.int32),
        "response_length": np.asarray([2, 2], dtype=np.int32),
        "task_type": np.asarray(["QA", "QA"]),
        "data_source": np.asarray(["synthetic", "synthetic"]),
        "generator_model": np.asarray(["generator", "generator"]),
    }
    manifest = "a" * 64
    path = {
        **rows,
        "score": np.asarray([0.2, 0.7], dtype=np.float32),
        "dataset_manifest_sha256": np.asarray(manifest),
        "audit_scope": np.asarray("selected_samples"),
        "test_group_id": np.asarray(["test-source"]),
        "test_sample_id": np.asarray(["r0"]),
        "fit_group_id": np.asarray(["path-fit"]),
        "channel_calibration_group_id": np.asarray(["path-channel"]),
        "fusion_calibration_group_id": np.asarray(["path-fusion"]),
        "calibration_group_id": np.asarray(["path-channel", "path-fusion"]),
        "reference_path": np.asarray(str((root / "path_reference.npz").resolve())),
        "reference_sha256": np.asarray("b" * 64),
    }
    rr = {
        **rows,
        "scores": np.asarray([[0.4], [0.1]], dtype=np.float32),
        "score_names": np.asarray([ALLOWED_RR_SCORE]),
        "dataset_manifest_sha256": np.asarray(manifest),
        "audit_scope": np.asarray("selected_samples"),
        "test_group_id": np.asarray(["test-source"]),
        "test_sample_id": np.asarray(["r0"]),
        "fit_group_id": np.asarray(["rr-fit"]),
        "calibration_group_id": np.asarray(["rr-calibration"]),
        "reference_path": np.asarray(str((root / "rr_reference.npz").resolve())),
        "reference_sha256": np.asarray("c" * 64),
    }
    path_reference = {
        "train_dataset_manifest_sha256": np.asarray("d" * 64),
        "topology_gate_mean_gap": np.asarray(0.1, dtype=np.float32),
        "topology_gate_pass": np.asarray(True, dtype=np.bool_),
    }
    rr_reference = {
        "train_dataset_manifest_sha256": np.asarray("e" * 64)
    }
    return path, rr, path_reference, rr_reference


class CausalTypedPathDeBruijnTests(unittest.TestCase):
    def test_llama_geometry_remains_32_by_32_until_final_calibration(self):
        layers = heads = 32
        channels = layers * heads
        attention = AttentionSample(
            "llama-shape",
            "source",
            2,
            torch.tensor([10, 11, 12]),
            torch.full((layers, heads, 3), 0.2, dtype=torch.float16),
            torch.arange(channels + 1, dtype=torch.int64),
            torch.zeros(channels, dtype=torch.int32),
            torch.full((channels,), 0.3, dtype=torch.float16),
            0.01,
        )
        graph = build_causal_routing_graph(_PublicSample(attention))
        self.assertEqual(graph.prompt_mass.shape, (1, 32, 32))
        route = layered_attention_automaton(graph)
        self.assertEqual(route.route_distribution.shape, (1, 32, 32, 5))
        self.assertEqual(route.flat_route_distribution.shape, (1, 1024, 5))

    def test_graph_automaton_and_token_path_conserve_every_channel(self):
        graph = build_causal_routing_graph(
            _PublicSample(_sample("one", "source")),
            config=GraphConfig(block_rows=3, recent_lag=4),
        )
        self.assertEqual(graph.role_mass.shape, (4, 2, 2, 4))
        torch.testing.assert_close(
            graph.role_mass.sum(dim=-1),
            torch.ones((4, 2, 2)),
        )
        automaton = layered_attention_automaton(graph)
        self.assertEqual(automaton.route_distribution.shape, (4, 2, 2, 5))
        torch.testing.assert_close(
            automaton.route_distribution.sum(dim=-1),
            torch.ones((4, 2, 2)),
        )
        path = typed_path_dp(graph, config=PathConfig(max_hops=3))
        self.assertEqual(path.exit_mass.shape, (4, 4, 3, 3))
        self.assertEqual(path.survival_pattern.shape, (4, 4, 8))
        torch.testing.assert_close(
            path.route_distribution.sum(dim=-1),
            torch.ones((4, 4)),
            atol=3e-5,
            rtol=3e-5,
        )

    def test_graph_rejects_non_numerical_attention_mass_overshoot(self):
        corrupted = _sample("overshoot", "source")
        corrupted.response_values.mul_(4.0)
        with self.assertRaisesRegex(ValueError, "overshoot"):
            build_causal_routing_graph(_PublicSample(corrupted))

    def test_prefix_rows_cannot_change_earlier_lineage(self):
        full_attention = _sample("full", "source")
        prefix_attention = _prefix_sample(full_attention, 3)
        full = layered_attention_automaton(
            build_causal_routing_graph(_PublicSample(full_attention))
        )
        prefix = layered_attention_automaton(
            build_causal_routing_graph(_PublicSample(prefix_attention))
        )
        torch.testing.assert_close(
            full.route_distribution[:3], prefix.route_distribution
        )

    def test_debruijn_phase_score_is_finite_and_first_token_has_no_lockin(self):
        graph = build_causal_routing_graph(_PublicSample(_sample("one", "source")))
        route = layered_attention_automaton(graph)
        q = route.flat_route_distribution
        accumulator = DeBruijnAccumulator(
            num_channels=4,
            num_states=5,
            config=DeBruijnConfig(order=2, soft_top_k=2, alpha=0.5),
        )
        accumulator.update(q)
        accumulator.update(q.roll(1, dims=0))
        model = accumulator.freeze()
        torch.testing.assert_close(
            model.transition.sum(dim=-1), torch.ones_like(model.transition[..., 0])
        )
        surprisal = model.score(q).float()
        predicted = model.predict_distribution(q).float()
        values, indices = torch.topk(q, k=2, dim=-1, sorted=False)
        observed = torch.zeros_like(q).scatter(-1, indices, values)
        observed = observed / observed.sum(dim=-1, keepdim=True)
        expected_cross_entropy = -(
            observed * torch.log(predicted.clamp_min(1e-30))
        ).sum(dim=-1)
        torch.testing.assert_close(surprisal, expected_cross_entropy)
        drop = prompt_lineage_drop(route.flat_prompt_lineage)
        stats = fit_robust_change_stats(
            torch.cat((surprisal, surprisal + 0.1), dim=0),
            torch.cat((drop, drop + 0.01), dim=0),
            config=ChangeConfig(prompt_lineage_drop_weight=0.0),
        )
        phase = change_lockin_score(
            q,
            surprisal,
            route.flat_prompt_lineage,
            stats=stats,
            detached_indices=R_PLUS,
            predicted_route_distribution=predicted,
            config=ChangeConfig(prompt_lineage_drop_weight=0.0),
        )
        self.assertTrue(bool(torch.isfinite(phase.raw_channel_score).all()))
        torch.testing.assert_close(phase.lockin[0], torch.zeros(4))

        prefix_route = layered_attention_automaton(
            build_causal_routing_graph(
                _PublicSample(_prefix_sample(_sample("one", "source"), 3))
            )
        )
        prefix_q = prefix_route.flat_route_distribution
        prefix_phase = change_lockin_score(
            prefix_q,
            model.score(prefix_q).float(),
            prefix_route.flat_prompt_lineage,
            stats=stats,
            detached_indices=R_PLUS,
            predicted_route_distribution=model.predict_distribution(prefix_q).float(),
            config=ChangeConfig(prompt_lineage_drop_weight=0.0),
        )
        torch.testing.assert_close(
            phase.raw_channel_score[:3],
            prefix_phase.raw_channel_score,
        )

    def test_nulls_preserve_first_order_rows_and_time_shuffle_preserves_values(self):
        graph = build_causal_routing_graph(_PublicSample(_sample("one", "source")))
        rewired = causal_endpoint_rewire(graph, seed=7)
        torch.testing.assert_close(rewired.graph.role_mass, graph.role_mass)
        torch.testing.assert_close(rewired.graph.weight, graph.weight)
        self.assertGreater(rewired.changed_fraction, 0.0)
        q = layered_attention_automaton(graph).flat_route_distribution
        shuffled = offline_noncausal_bucket_time_shuffle(
            q, bucket_size=4, seed=7, sample_id="one"
        )
        self.assertFalse(torch.equal(q, shuffled))
        np.testing.assert_allclose(
            np.sort(q.reshape(4, -1).numpy(), axis=0),
            np.sort(shuffled.reshape(4, -1).numpy(), axis=0),
        )

    def test_two_stage_calibration_uses_independent_fusion_rows(self):
        channel = np.asarray(
            [[0.0, 1.0], [0.2, 0.9], [0.4, 0.7], [0.6, 0.5]],
            dtype=np.float32,
        )
        fusion = channel + 0.05
        reference = build_calibration(channel, fusion)
        scored = score_channels(reference, np.asarray([[1.0, 1.0]], dtype=np.float32))
        self.assertTrue(reference.independent_fusion_reference)
        self.assertEqual(scored.channel_p_value.shape, (1, 2))
        self.assertTrue(np.isfinite(scored.score).all())

    def test_rr_bridge_freezes_manifest_alignment_and_writes_strict_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path_file = root / "path_scores.npz"
            rr_file = root / "rr_scores.npz"
            path_file.write_bytes(b"frozen path score")
            rr_file.write_bytes(b"frozen rr score")
            path, rr, path_reference, rr_reference = _hybrid_mock_artifacts(root)
            output = root / "hybrid.npz"
            module = "experiments.causal_typed_path_debruijn.spectral_bridge"
            with (
                patch(f"{module}.load_path_score_artifact", return_value=path),
                patch(f"{module}.load_rr_score_artifact", return_value=rr),
                patch(
                    f"{module}.verify_path_score_provenance",
                    return_value=path_reference,
                ),
                patch(
                    f"{module}._verify_rr_reference_frozen",
                    return_value=rr_reference,
                ),
                patch(f"{module}.verify_rr_hybrid_provenance"),
            ):
                result = build_rr_hybrid(path_file, rr_file, output)
            self.assertFalse(result["labels_read"])
            hybrid = load_rr_hybrid(output)
            self.assertEqual(hybrid["score"].shape, (2,))
            np.testing.assert_array_equal(hybrid["sample_id"], path["sample_id"])
            with (
                patch(
                    f"{module}.verify_rr_hybrid_provenance",
                    return_value=(path_reference, rr_reference),
                ),
                patch(
                    "experiments.causal_typed_path_debruijn.evaluation."
                    "FrozenEvaluation.validate_loaded"
                ),
                patch(
                    "experiments.causal_typed_path_debruijn.evaluation."
                    "FrozenEvaluation.align_loaded",
                    return_value=SimpleNamespace(
                        token_label=np.asarray([0, 1], dtype=np.int8)
                    ),
                ),
            ):
                report = evaluate_scores(
                    object(),
                    output,
                    root / "hybrid_evaluation.json",
                    bootstrap_replicates=2,
                )
            self.assertEqual(
                report["schema"],
                "causal-typed-path-rr-hybrid-evaluation-v1",
            )

            mismatched = dict(rr)
            mismatched["dataset_manifest_sha256"] = np.asarray("f" * 64)
            with (
                patch(f"{module}.load_path_score_artifact", return_value=path),
                patch(
                    f"{module}.load_rr_score_artifact",
                    return_value=mismatched,
                ),
                patch(
                    f"{module}.verify_path_score_provenance",
                    return_value=path_reference,
                ),
                patch(
                    f"{module}._verify_rr_reference_frozen",
                    return_value=rr_reference,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "different dataset manifests"):
                    build_rr_hybrid(path_file, rr_file, root / "bad_hybrid.npz")

    def test_label_free_fit_score_then_frozen_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _write_dataset(
                root / "train",
                [0.80, 0.88, 0.96, 1.04, 1.12, 1.20],
                split="train",
                source_prefix="train-",
            )
            test = _write_dataset(
                root / "test",
                [0.92, 1.00, 1.08],
                split="test",
                source_prefix="test-",
                positive_sample=1,
            )
            reference_path = root / "ctpdb" / "reference.npz"
            score_path = root / "ctpdb" / "test_scores.npz"
            report_path = root / "ctpdb" / "evaluation.json"
            config = ExperimentConfig(
                graph=GraphConfig(block_rows=3, recent_lag=4),
                debruijn=DeBruijnConfig(order=2, soft_top_k=2, alpha=0.5),
                change=ChangeConfig(
                    cusum_slack=0.0,
                    prompt_lineage_drop_weight=0.0,
                    scale_floor=1e-3,
                ),
                calibration=CalibrationConfig(
                    channel_fraction=0.2,
                    fusion_fraction=0.2,
                    reference_size=32,
                    top_channels=2,
                    seed=9,
                ),
            )
            fitted = fit_reference(train, reference_path, config=config)
            self.assertFalse(fitted["labels_read"])
            reference = load_reference(reference_path)
            self.assertNotIn("label", reference)
            with self.assertRaisesRegex(ValueError, "requires manifest split 'train'"):
                fit_reference(test, root / "ctpdb" / "wrong_reference.npz")
            malicious_reference = root / "ctpdb" / "malicious_reference.npz"
            np.savez_compressed(
                malicious_reference,
                **reference,
                gold=np.asarray([1], dtype=np.int8),
            )
            with self.assertRaisesRegex(ValueError, "unsupported fields"):
                load_reference(malicious_reference)
            sidecar_dir = root / "ctpdb" / "sidecars"
            with self.assertRaisesRegex(ValueError, "requires manifest split 'test'"):
                score_split(
                    train,
                    reference_path,
                    root / "ctpdb" / "wrong_scores.npz",
                )
            scored = score_split(
                test,
                reference_path,
                score_path,
                save_channel_sidecars=True,
                sidecar_dir=sidecar_dir,
            )
            self.assertFalse(scored["labels_read"])
            self.assertEqual(len(list(sidecar_dir.glob("*.npz"))), 3)
            artifact = load_score_artifact(score_path)
            self.assertEqual(artifact["top_channel_index"].shape, (12, 2))
            self.assertNotIn("label", artifact)
            malicious_score = root / "ctpdb" / "malicious_scores.npz"
            np.savez_compressed(
                malicious_score,
                **artifact,
                is_hallucination=np.zeros(12, dtype=np.int8),
            )
            with self.assertRaisesRegex(ValueError, "unsupported fields"):
                load_score_artifact(malicious_score)
            report = evaluate_scores(
                test,
                score_path,
                report_path,
                bootstrap_replicates=20,
                seed=9,
            )
            self.assertTrue(report["labels_read"])
            self.assertEqual(report["token_metrics"]["tokens"], 12)
            self.assertIn("QA", report["task_reports"])
            figure = root / "ctpdb" / "sample_r0.png"
            rendered = visualize_scored_sample(
                test,
                reference_path,
                score_path,
                sample_id="r0",
                output_path=figure,
            )
            self.assertFalse(rendered["labels_read"])
            self.assertTrue(figure.is_file())

    def test_cli_keeps_labels_sealed_until_evaluate(self):
        commands = (
            (
                ["fit", "--train-split", "train", "--reference", "reference"],
                "fit_reference",
                False,
            ),
            (
                [
                    "score",
                    "--test-split",
                    "test",
                    "--reference",
                    "reference",
                    "--output",
                    "scores",
                ],
                "score_split",
                False,
            ),
            (
                [
                    "evaluate",
                    "--test-split",
                    "test",
                    "--scores",
                    "scores",
                    "--output",
                    "report",
                ],
                "evaluate_scores",
                True,
            ),
        )
        for arguments, target, retain_labels in commands:
            with self.subTest(command=arguments[0]):
                with (
                    patch.object(
                        method_main,
                        "open_research_dataset",
                        return_value=object(),
                    ) as opened,
                    patch.object(
                        method_main,
                        target,
                        return_value={"labels_read": retain_labels},
                    ),
                ):
                    method_main.main(arguments)
                self.assertTrue(opened.call_args.kwargs["verify_hashes"])
                self.assertEqual(
                    opened.call_args.kwargs["retain_embedded_labels"],
                    retain_labels,
                )


if __name__ == "__main__":
    unittest.main()
