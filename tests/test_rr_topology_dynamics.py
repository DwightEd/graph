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
from experiments.rr_topology_dynamics.experiment import (
    TopologyAuditConfig,
    evaluate_topology_artifact,
    fit_topology_reference,
    score_topology_dataset,
)
from experiments.rr_topology_dynamics.features import (
    TopologyDynamicsConfig,
    _batched_route_spectrum,
    _mean_pairwise_cosine,
    _prompt_groundedness,
    extract_sample_topology_dynamics,
    load_rr_reference,
)
from experiments.spectral_feasibility.experiment import fit_spectral_reference
from experiments.spectral_feasibility.representations import (
    SpectralConfig,
    prefix_laplacian_modes,
)
from research_dataset import ResearchDataset


def _sample(sample_id: str, source_id: str, multiplier: float = 1.0):
    diagonal = torch.tensor(
        [[[0.8, 0.7, 0.4, 0.3, 0.2], [0.6, 0.5, 0.2, 0.4, 0.1]]],
        dtype=torch.float16,
    )
    columns = torch.tensor(
        [0, 2, 1, 2, 3, 1, 2, 0, 2, 3], dtype=torch.int32
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


def _write_dataset(root: Path, multipliers, *, positive_sample: int | None = None):
    (root / "attention").mkdir(parents=True)
    rows = []
    labels = []
    for index, multiplier in enumerate(multipliers):
        sample = _sample(f"r{index}", f"s{index}", multiplier)
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
        labels.append(
            {
                "sample_id": sample.sample_id,
                "positive_runs": [[1, 2]] if positive_sample == index else [],
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
        num_layers=1,
        num_heads=2,
        alignment="post_token_query_at_same_position",
        extra={
            "split": "test" if positive_sample is not None else "train",
            "labels_sha256": sha256(label_path),
        },
    )
    return ResearchDataset(root)


class RRTopologyDynamicsTests(unittest.TestCase):
    def test_prefix_modes_preserve_source_identity_and_lag(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _write_dataset(Path(directory), [1.0])
            sample = dataset["r0"]
            modes = prefix_laplacian_modes(
                sample,
                positions=[1, 2],
                config=SpectralConfig(top_k=2),
            )
            self.assertEqual(modes.values.shape, (2, 2, 2))
            self.assertEqual(int(modes.source_index[0, 0, 0]), 0)
            self.assertEqual(int(modes.lag[0, 0, 0]), 1)
            self.assertEqual(int(modes.source_index[1, 0, 0]), 0)
            self.assertEqual(int(modes.lag[1, 0, 0]), 2)
            self.assertEqual(int(modes.source_index[1, 1, 0]), 1)
            self.assertEqual(int(modes.lag[1, 1, 0]), 1)
            self.assertEqual(int(modes.source_index[1, 1, 1]), 0)
            self.assertEqual(int(modes.lag[1, 1, 1]), 2)
            sample.release_attention()

    def test_route_rank_and_consensus_distinguish_collapsed_routes(self):
        collapsed = torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0]]], dtype=torch.float32
        )
        diverse = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32
        )
        collapsed_rank = _batched_route_spectrum(collapsed, 1e-8)[0]
        diverse_rank = _batched_route_spectrum(diverse, 1e-8)[0]
        self.assertGreater(float(diverse_rank[0]), float(collapsed_rank[0]))
        active = torch.ones((1, 2), dtype=torch.bool)
        collapsed_consensus = _mean_pairwise_cosine(collapsed, active, 1e-8)
        diverse_consensus = _mean_pairwise_cosine(diverse, active, 1e-8)
        self.assertGreater(
            float(collapsed_consensus[0]), float(diverse_consensus[0])
        )

    def test_prompt_grounding_separates_relay_from_feedback(self):
        prompt = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        grounded_chain = np.zeros((3, 3), dtype=np.float32)
        grounded_chain[1, 0] = 1.0
        grounded_chain[2, 1] = 1.0
        _, grounded, relay, feedback = _prompt_groundedness(
            prompt, grounded_chain, 1e-8
        )
        self.assertGreater(float(grounded[2]), 0.99)
        self.assertGreater(float(relay[2]), 0.99)
        self.assertLess(float(feedback[2]), 0.01)

        no_prompt = np.zeros(3, dtype=np.float32)
        _, ungrounded, _, ungrounded_feedback = _prompt_groundedness(
            no_prompt, grounded_chain, 1e-8
        )
        self.assertLess(float(ungrounded[2]), 0.01)
        self.assertGreater(float(ungrounded_feedback[2]), 0.99)

    def test_label_free_fit_score_then_posthoc_topology_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = _write_dataset(root / "train", [0.8, 1.0, 1.2, 1.1])
            test = _write_dataset(
                root / "test", [1.0, 1.1], positive_sample=1
            )
            spectral_path = root / "spectral_reference.npz"
            topology_path = root / "topology_reference.npz"
            feature_path = root / "test_features.npz"
            evaluation_dir = root / "evaluation"

            spectral_config = SpectralConfig(
                top_k=2,
                position_bins=2,
                pca_dim=2,
                reference_per_sample=3,
                trim_fraction=0.9,
                channel_tail_fraction=0.5,
                attribution_topk=2,
            )
            spectral_fit = fit_spectral_reference(
                train, spectral_path, config=spectral_config
            )
            self.assertFalse(spectral_fit["labels_read"])

            topology_config = TopologyDynamicsConfig(
                lag_bins=3,
                spectral_top_k=2,
                position_bins=2,
                top_source_count=2,
                recent_lag_max=1,
                mid_lag_max=2,
            )
            audit_config = TopologyAuditConfig(
                reference_per_sample=3,
                min_task_bin_rows=2,
                phase_bins=2,
                onset_window=1,
                bootstrap_replicates=10,
                seed=7,
            )
            fitted = fit_topology_reference(
                train,
                spectral_path,
                topology_path,
                topology_config=topology_config,
                audit_config=audit_config,
            )
            self.assertFalse(fitted["labels_read"])
            self.assertEqual(fitted["feature_dim"], 37)
            with np.load(topology_path, allow_pickle=False) as arrays:
                self.assertNotIn("label", arrays.files)
                self.assertNotIn("y_token", arrays.files)
                self.assertIn("task_center", arrays.files)
                self.assertIn("feature_names", arrays.files)

            scored = score_topology_dataset(
                test,
                spectral_path,
                topology_path,
                feature_path,
            )
            self.assertFalse(scored["labels_read"])
            self.assertEqual(scored["tokens"], 6)
            with np.load(feature_path, allow_pickle=False) as arrays:
                self.assertNotIn("label", arrays.files)
                self.assertNotIn("y_token", arrays.files)
                self.assertEqual(arrays["features_raw"].shape, (6, 37))
                self.assertEqual(arrays["layer_residual_energy"].shape, (6, 1))
                self.assertEqual(
                    arrays["spectral_rank_residual_energy"].shape, (6, 2)
                )

            spectral_reference = load_rr_reference(spectral_path)
            sample = test["r0"]
            extracted = extract_sample_topology_dynamics(
                sample, spectral_reference, config=topology_config
            )
            self.assertEqual(extracted["features"].shape, (3, 37))
            sample.release_attention()

            report = evaluate_topology_artifact(
                test,
                feature_path,
                evaluation_dir,
                bootstrap_replicates=10,
                onset_window=1,
                phase_bins=2,
                seed=7,
            )
            self.assertEqual(report["overall"]["tokens"], 6)
            self.assertEqual(report["overall"]["positive_tokens"], 1)
            self.assertFalse(
                report["claim_boundaries"]["confidence_available"]
            )
            self.assertIn(
                "route_effective_rank", report["feature_metrics_raw"]
            )
            self.assertTrue((evaluation_dir / "report.json").is_file())
            self.assertTrue((evaluation_dir / "onset_effects.csv").is_file())


if __name__ == "__main__":
    unittest.main()
