# Code architecture

The package follows the same separation principle used by compact graph
research repositories: one file owns one research object.

| File | Owns | Main public interface |
|---|---|---|
| `graph.py` | event graph and higher-order relations | `build_graph` |
| `model.py` | neural message passing | `HoloRoute.forward` |
| `learning.py` | graph views, losses, optimizer loop | `self_supervised_loss`, `train_model` |
| `detection.py` | token residuals and one-class reference | `score_graph`, `ConditionalReference` |
| `baseline.py` | all-layer no-topology control | `Flat1024`, `build_pairs` |
| `pipeline.py` | dataset-level orchestration | `train_holoroute`, `score_holoroute`, `train_flat`, `score_flat` |
| `evaluate.py` | evaluation-only labels and metrics | `evaluate` |
| `run.py` | CLI | `main` |

Design rules:

1. Public functions have descriptive names; implementation functions are not
   hidden behind leading underscores.
2. Structural tensors are grouped in data classes instead of returned as long
   tuples or dictionaries.
3. A matrix carries mechanism residuals through the pipeline; feature names are
   attached only when saving or reporting.
4. Validation is concentrated at data/artifact boundaries. The model forward
   path contains no repeated schema checks.
5. Training tasks have separate graph views, so each loss corresponds to one
   identifiable structural operation.
