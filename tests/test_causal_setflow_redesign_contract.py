import importlib
import inspect
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments" / "causal_setflow"


class CausalSetFlowRedesignContractTests(unittest.TestCase):
    def test_core_modules_import(self):
        for name in (
            "config",
            "corruptions",
            "model",
            "losses",
            "trainer",
            "calibration",
            "artifacts",
            "experiment",
            "main",
        ):
            importlib.import_module(f"experiments.causal_setflow.{name}")

    def test_method_is_mechanism_guided_not_scalar_imputation(self):
        model = (PACKAGE / "model.py").read_text(encoding="utf-8")
        config = (PACKAGE / "config.py").read_text(encoding="utf-8")
        corruptions = (PACKAGE / "corruptions.py").read_text(encoding="utf-8")
        experiment = (PACKAGE / "experiment.py").read_text(encoding="utf-8")
        calibration = (PACKAGE / "calibration.py").read_text(encoding="utf-8")

        self.assertIn("Mechanism-Guided", model)
        self.assertIn("CORRUPTION_NAMES", config)
        self.assertIn("apply_corruption", corruptions)
        self.assertIn("channel_state", model)
        self.assertTrue("teacher" in model.lower() and "ema" in model.lower())
        self.assertNotIn("empirically_recalibrated_fisher_setflow", experiment)
        self.assertNotIn("calibration_fisher", calibration)

    def test_corruptions_are_domain_specific_and_causal(self):
        config = importlib.import_module("experiments.causal_setflow.config")
        names = tuple(config.CORRUPTION_NAMES)
        self.assertGreaterEqual(len(names), 4)
        joined = " ".join(names).lower()
        for mechanism in ("collapse", "rewire", "freeze", "head"):
            self.assertIn(mechanism, joined)

    def test_model_exposes_structured_channel_field(self):
        module = importlib.import_module("experiments.causal_setflow.model")
        encoder_outputs = [
            value
            for _, value in inspect.getmembers(module, inspect.isclass)
            if value.__module__ == module.__name__
            and "Output" in value.__name__
        ]
        annotations = {
            name
            for value in encoder_outputs
            for name in getattr(value, "__annotations__", {})
        }
        self.assertIn("token_embedding", annotations)
        self.assertIn("channel_state", annotations)
        self.assertIn("channel_active", annotations)

    def test_primary_score_is_oriented_by_synthetic_anomaly_energy(self):
        experiment = (PACKAGE / "experiment.py").read_text(encoding="utf-8").lower()
        calibration = (PACKAGE / "calibration.py").read_text(encoding="utf-8").lower()
        combined = experiment + "\n" + calibration
        self.assertIn("energy", combined)
        self.assertNotIn("orientation_free", combined)
        self.assertNotIn("1.0 - metrics['auroc']", combined)


if __name__ == "__main__":
    unittest.main()
