import numpy as np
import torch

from experiments.holoroute.config import HoloRouteConfig, ModelConfig, MaskConfig
from experiments.holoroute.density import ConditionalDensity
from experiments.holoroute.model import HoloRouteEncoder
from experiments.holoroute.objectives import score_graph, self_supervised_loss
from experiments.holoroute.tests.helpers import synthetic_graph


def test_minimal_train_score_density_pipeline():
    graph = synthetic_graph()
    config = HoloRouteConfig(
        model=ModelConfig(hidden_dim=32, head_encoder_heads=4, transport_rank=4),
        masking=MaskConfig(event_fraction=0.4, relay_fraction=0.25, score_rounds=3),
    )
    model = HoloRouteEncoder(graph.num_layers, graph.num_heads, config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = self_supervised_loss(
            model,
            graph,
            config,
            generator=torch.Generator().manual_seed(100 + step),
        )
        loss.total.backward()
        optimizer.step()

    feature, coverage = score_graph(model, graph, config, seed=31)
    assert coverage[:, 0].sum() >= 3
    nuisance = np.column_stack(
        (
            np.arange(graph.num_response_tokens),
            np.linspace(0, 1, graph.num_response_tokens),
        )
    )
    task = np.repeat("QA", graph.num_response_tokens)
    calibration_feature = np.concatenate((feature, feature + 0.05, feature + 0.1), axis=0)
    calibration_nuisance = np.tile(nuisance, (3, 1))
    calibration_task = np.tile(task, 3)
    density = ConditionalDensity.fit(
        calibration_feature,
        calibration_nuisance,
        calibration_task,
        ridge_alpha=1e-3,
        covariance_shrinkage=0.2,
        scale_floor=1e-3,
    )
    score, standardized = density.score(feature, nuisance, task)
    assert score.shape == (graph.num_response_tokens,)
    assert standardized.shape == feature.shape
    assert np.isfinite(score).all()
