from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from attention_graph.causal_topology import CausalTopologyConfig, TopologyEncoding
from attention_graph.one_class import OneClassConfig
from attention_graph.topology_experiment import (
    TopologyExperiment,
    TopologyExperimentConfig,
)


def _encoding(seed: int) -> TopologyEncoding:
    generator = torch.Generator().manual_seed(seed)

    def values(width: int) -> torch.Tensor:
        return torch.randn(4, 1, 1, width, generator=generator)

    return TopologyEncoding(
        balance_log_scale=values(2),
        attention_marginals=values(3),
        retained_support=values(2),
        prompt_provenance=values(4),
        rr_one_hop=values(6),
        rr_two_hop=values(2),
        rewired_rr_one_hop=values(6),
        rewired_rr_two_hop=values(2),
    )


class _Attention:
    def __init__(self, sample_id: str):
        self.sample_id = sample_id
        self.response_idx = 2
        self.num_response_tokens = 4
        self.token_ids = torch.arange(6, dtype=torch.int32)


class _Sample:
    task_type = "QA"
    data_source = "fixture"
    generator_model = "generator"

    def __init__(self, dataset, sample_id: str, index: int):
        self.dataset = dataset
        self.sample_id = sample_id
        self.source_id = f"source-{index}"
        self.index = index
        self._attention = _Attention(sample_id)

    def attention(self):
        return self._attention

    def release_attention(self):
        pass


class _Labels:
    def response_labels(self, sample: _Sample):
        positive = sample.index % 2
        return torch.tensor([0, positive, 0, positive], dtype=torch.long)


class _Dataset:
    def __init__(self, prefix: str, count: int, *, required_files=()):
        self.sample_ids = [f"{prefix}-{index}" for index in range(count)]
        self.samples = {
            sample_id: _Sample(self, sample_id, index)
            for index, sample_id in enumerate(self.sample_ids)
        }
        self.manifest = {
            "schema": "test-attention-v1", "split": prefix,
            "count": count, "num_layers": 1, "num_heads": 1,
            "attention_floor": .01,
        }
        self.required_files = tuple(Path(path) for path in required_files)
        self.labels_opened = 0

    def __getitem__(self, sample_id: str):
        return self.samples[sample_id]

    def labels(self):
        for path in self.required_files:
            if not path.is_file():
                raise AssertionError(f"labels opened before {path.name} was frozen")
        self.labels_opened += 1
        return _Labels()


class _CountingEncoder:
    def __init__(self, fail_on=None):
        self.config = CausalTopologyConfig(fourier_frequencies=2, rewire_seed=9)
        self.calls = []
        self.fail_on = fail_on

    def encode(self, attention: _Attention) -> TopologyEncoding:
        if attention.sample_id == self.fail_on:
            raise KeyboardInterrupt
        self.calls.append(attention.sample_id)
        prefix, index = attention.sample_id.rsplit("-", 1)
        return _encoding(int(index) + (100 if prefix == "test" else 0))


class TopologyExperimentTests(unittest.TestCase):
    @staticmethod
    def _config(**changes):
        values = dict(
            reference_size=8, checkpoint_interval=2,
            bootstrap_replicates=10, seed=17,
            topology=CausalTopologyConfig(fourier_frequencies=2, rewire_seed=9),
            one_class=OneClassConfig(
                position_bins=1, subspace_components=1,
                tail_fraction=.5, seed=17,
            ),
        )
        values.update(changes)
        return TopologyExperimentConfig(**values)

    def test_run_freezes_label_free_scores_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            required = (
                output / "topology_one_class_model.npz",
                output / "topology_label_free.npz",
                output / "topology_label_free_report.json",
            )
            train = _Dataset("train", 12)
            test = _Dataset("test", 4)
            evaluation = _Dataset("test", 4, required_files=required)
            encoder = _CountingEncoder()
            config = self._config()

            result = TopologyExperiment(
                train, test, evaluation, output_dir=output,
                config=config, encoder=encoder,
            ).run()

            self.assertEqual(result["primary_score"], "full_signal")
            self.assertEqual(
                encoder.calls,
                train.sample_ids + test.sample_ids,
            )
            self.assertEqual(evaluation.labels_opened, 1)
            self.assertFalse((output / "train_reference_checkpoint.npz").exists())
            with np.load(required[0], allow_pickle=False) as model:
                self.assertFalse(bool(model["labels_included"]))
                self.assertIn("signature", model.files)
                self.assertTrue(any(name.startswith("atomic/") for name in model.files))
            with np.load(required[1], allow_pickle=False) as artifact:
                self.assertFalse(bool(artifact["labels_included"]))
                self.assertEqual(str(artifact["primary_score"]), "full_signal")
                self.assertEqual(artifact["score_coordinates"].shape, (16, 2))
                self.assertEqual(
                    artifact["score_coordinates"][:, 0].tolist(),
                    artifact["attention_marginals_score"].tolist(),
                )
                self.assertEqual(
                    artifact["score_coordinates"][:, 1].tolist(),
                    artifact["causal_topology_exact_score"].tolist(),
                )
                self.assertFalse(any(
                    "representation" in name or "node" in name
                    for name in artifact.files
                ))
                self.assertEqual(len(str(artifact["train_dataset_fingerprint"])), 64)
                self.assertEqual(len(str(artifact["test_dataset_fingerprint"])), 64)
                self.assertNotEqual(
                    str(artifact["train_dataset_fingerprint"]),
                    str(artifact["test_dataset_fingerprint"]),
                )
                for score_name in artifact["score_names"].tolist():
                    self.assertEqual(artifact[f"{score_name}_score"].ndim, 1)
            label_free_report = json.loads(required[2].read_text(encoding="utf-8"))
            self.assertFalse(label_free_report["labels_used"])
            self.assertEqual(label_free_report["config"], asdict(config))
            self.assertEqual(label_free_report["reference_split"]["per_group_budget"], 4)
            self.assertTrue(label_free_report["reference_split"]["mutually_exclusive"])
            self.assertEqual(
                set(label_free_report["dataset_fingerprints"]), {"train", "test"}
            )
            self.assertEqual(
                set(result["paired_bootstrap"]),
                {
                    "full_signal_vs_attention_marginals",
                    "causal_topology_exact_vs_attention_marginals",
                    "rr_multihop_exact_vs_lag_rewired",
                    "rr_multihop_exact_vs_one_hop_exact",
                },
            )
            self.assertEqual(set(result["score_evaluation"]), set(
                label_free_report["score_names"]
            ))

    def test_atomic_checkpoint_rejects_changed_signature_and_resumes_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interrupted_output = root / "resumed"
            baseline_output = root / "baseline"
            train = _Dataset("train", 12)
            test = _Dataset("test", 4)
            failing = _CountingEncoder(fail_on="train-2")
            config = self._config(checkpoint_interval=1)

            with self.assertRaises(KeyboardInterrupt):
                TopologyExperiment(
                    train, test, _Dataset("test", 4),
                    output_dir=interrupted_output, config=config, encoder=failing,
                ).run()

            checkpoint = interrupted_output / "train_reference_checkpoint.npz"
            self.assertTrue(checkpoint.is_file())
            self.assertFalse(any(interrupted_output.glob("*.tmp.npz")))
            with np.load(checkpoint, allow_pickle=False) as saved:
                self.assertEqual(int(saved["next_sample_index"]), 2)
                self.assertEqual(str(saved["schema"]), "causal-topology-checkpoint-v1")
                self.assertIn("signature", saved.files)

            changed = self._config(
                checkpoint_interval=1,
                topology=CausalTopologyConfig(fourier_frequencies=3, rewire_seed=9),
            )
            with self.assertRaisesRegex(ValueError, "signature"):
                TopologyExperiment(
                    train, test, _Dataset("test", 4),
                    output_dir=interrupted_output, config=changed,
                    encoder=_CountingEncoder(),
                ).run()

            resumed = _CountingEncoder()
            TopologyExperiment(
                train, test, _Dataset("test", 4),
                output_dir=interrupted_output, config=config, encoder=resumed,
            ).run()
            self.assertEqual(resumed.calls, train.sample_ids[2:] + test.sample_ids)
            self.assertFalse(checkpoint.exists())

            TopologyExperiment(
                _Dataset("train", 12), _Dataset("test", 4), _Dataset("test", 4),
                output_dir=baseline_output, config=config, encoder=_CountingEncoder(),
            ).run()
            with np.load(
                interrupted_output / "topology_label_free.npz", allow_pickle=False
            ) as resumed_artifact, np.load(
                baseline_output / "topology_label_free.npz", allow_pickle=False
            ) as baseline_artifact:
                for score_name in resumed_artifact["score_names"].tolist():
                    np.testing.assert_array_equal(
                        resumed_artifact[f"{score_name}_score"],
                        baseline_artifact[f"{score_name}_score"],
                    )


if __name__ == "__main__":
    unittest.main()
