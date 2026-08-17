"""Scientific CLIs must verify every raw trace file before use."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from experiments.causal_multiplex_flow import main as cmrp_main
from experiments.rr_topology_dynamics import main as topology_main
from experiments.spectral_feasibility import main as spectral_main


class CliDatasetIntegrityTests(unittest.TestCase):
    def test_cmrp_fit_opens_a_hash_verified_dataset(self):
        with (
            patch.object(cmrp_main, "open_research_dataset", return_value=object()) as open_dataset,
            patch.object(cmrp_main, "fit_cmrp", return_value={}),
        ):
            cmrp_main.main(["fit", "--train-split", "train", "--output-dir", "out"])

        self.assertTrue(open_dataset.call_args.kwargs["verify_hashes"])

    def test_cmrp_score_opens_a_hash_verified_dataset(self):
        with (
            patch.object(cmrp_main, "open_research_dataset", return_value=object()) as open_dataset,
            patch.object(cmrp_main, "score_cmrp", return_value={}),
        ):
            cmrp_main.main(
                ["score", "--split-root", "test", "--reference", "ref", "--output", "out"]
            )

        self.assertTrue(open_dataset.call_args.kwargs["verify_hashes"])

    def test_cmrp_evaluate_opens_a_hash_verified_dataset(self):
        with (
            patch.object(cmrp_main, "open_research_dataset", return_value=object()) as open_dataset,
            patch.object(cmrp_main, "evaluate_cmrp", return_value={"metrics": {}}),
        ):
            cmrp_main.main(
                ["evaluate", "--split-root", "test", "--scores", "scores", "--output", "out"]
            )

        self.assertTrue(open_dataset.call_args.kwargs["verify_hashes"])

    def test_spectral_cli_opens_a_hash_verified_dataset_for_each_command(self):
        commands = (
            (
                ["fit", "--train-split", "train", "--output", "out"],
                "fit_spectral_reference",
                {},
            ),
            (
                ["score", "--split-root", "test", "--reference", "ref", "--output", "out"],
                "score_spectral_dataset",
                {},
            ),
            (
                ["evaluate", "--split-root", "test", "--scores", "scores", "--output", "out"],
                "evaluate_score_artifact",
                {"metrics": {}, "components": {}},
            ),
        )
        for arguments, target, result in commands:
            with self.subTest(command=arguments[0]):
                with (
                    patch.object(spectral_main, "open_research_dataset", return_value=object()) as open_dataset,
                    patch.object(spectral_main, target, return_value=result),
                ):
                    spectral_main.main(arguments)
                self.assertTrue(open_dataset.call_args.kwargs["verify_hashes"])

    def test_topology_cli_opens_a_hash_verified_dataset_for_each_command(self):
        commands = (
            (
                [
                    "fit", "--train-split", "train", "--spectral-reference", "spectral",
                    "--output", "out",
                ],
                "fit_topology_reference",
                {},
            ),
            (
                [
                    "score", "--split-root", "test", "--spectral-reference", "spectral",
                    "--topology-reference", "topology", "--output", "out",
                ],
                "score_topology_dataset",
                {},
            ),
            (
                [
                    "evaluate", "--split-root", "test", "--features", "features",
                    "--output-dir", "out",
                ],
                "evaluate_topology_artifact",
                {"overall": {}},
            ),
        )
        for arguments, target, result in commands:
            with self.subTest(command=arguments[0]):
                with (
                    patch.object(topology_main, "open_research_dataset", return_value=object()) as open_dataset,
                    patch.object(topology_main, target, return_value=result),
                ):
                    topology_main.main(arguments)
                self.assertTrue(open_dataset.call_args.kwargs["verify_hashes"])


if __name__ == "__main__":
    unittest.main()
