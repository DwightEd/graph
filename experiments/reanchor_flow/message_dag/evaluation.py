"""Offline evaluation entry point for explicitly identified derived artifacts."""

from pathlib import Path

from .artifacts import (
    COUNTERFACTUAL_MEDIATION,
    LOOKBACK_TRANSPORT,
    SOURCE_ALLOCATION,
    read_descriptor,
)


class OfflineEvaluator:
    """Evaluate committed artifacts without native captures or model weights."""

    def __init__(self, output: Path, *, bootstrap: int = 200):
        self.output = Path(output)
        self.bootstrap = bootstrap

    def run(self):
        descriptor, manifest = read_descriptor(self.output)
        if descriptor.method_id == LOOKBACK_TRANSPORT:
            from .event_report import evaluate

            return evaluate(self.output, bootstrap=self.bootstrap)
        if descriptor.method_id == SOURCE_ALLOCATION:
            from .report import evaluate

            return evaluate(self.output, manifest, bootstrap=self.bootstrap)
        if descriptor.method_id == COUNTERFACTUAL_MEDIATION:
            from .counterfactual_report import evaluate

            return evaluate(self.output)
        raise AssertionError(f"unhandled method identity: {descriptor.method_id}")
