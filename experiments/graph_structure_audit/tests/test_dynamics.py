import json

import numpy as np
import torch

from experiments.graph_structure_audit import dynamics_evaluate, dynamics_experiment
from experiments.graph_structure_audit.dynamics_config import DynamicsConfig
from experiments.graph_structure_audit.dynamics_model import CrossOriginRoutingDynamics
from experiments.graph_structure_audit.graph_data import build_multiplex_graph

from .helpers import Dataset, Sample, dataset, raw_graph


def tiny_dynamics_config(**changes):
    values = dict(
        hidden_dim=16,
        role_dim=4,
        position_dim=4,
        lag_bins=8,
        dropout=0.0,
        input_dropout=0.0,
        epochs=1,
        validation_fraction=0.25,
        patience=1,
        score_rounds=1,
        show_progress=False,
    )
    values.update(changes)
    return DynamicsConfig(**values)


def test_cross_origin_dynamics_keeps_layer_head_maps_and_gradients():
    raw, _ = raw_graph()
    graph = build_multiplex_graph(raw)
    model = CrossOriginRoutingDynamics(
        num_layers=graph.num_layers,
        num_heads=graph.num_heads,
        config=tiny_dynamics_config(),
    )
    output = model(graph, input_dropout=False)
    transitions = graph.num_layers - 1
    assert output.edge_error_map.shape == (
        graph.num_response_tokens,
        transitions,
        graph.num_heads,
    )
    assert output.prompt_edge_error_map.shape == output.edge_error_map.shape
    assert output.response_edge_error_map.shape == output.edge_error_map.shape
    assert output.diagonal_error_map.shape == output.edge_error_map.shape
    assert output.prompt_gate.shape == (graph.num_response_tokens, transitions)
    output.loss.backward()
    gradient = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert np.isfinite(gradient) and gradient > 0


def test_prompt_and_response_messages_are_separate():
    raw, _ = raw_graph()
    graph = build_multiplex_graph(raw)
    model = CrossOriginRoutingDynamics(
        num_layers=graph.num_layers,
        num_heads=graph.num_heads,
        config=tiny_dynamics_config(),
    ).eval()
    full = model(graph, message_mode="full", input_dropout=False)
    prompt = model(graph, message_mode="prompt", input_dropout=False)
    response = model(graph, message_mode="response", input_dropout=False)
    none = model(graph, message_mode="none", input_dropout=False)
    assert not torch.allclose(full.embedding, none.embedding)
    assert not torch.allclose(prompt.embedding, response.embedding)


def test_dynamics_train_score_evaluate_keeps_labels_out(tmp_path, monkeypatch):
    train_dataset = dataset("train", 4)
    test_samples = []
    for index, labels in enumerate(([0, 1, 0, 0], [0, 0, 1, 0])):
        graph, _ = raw_graph(
            sample_id=f"test-{index}",
            source_id=f"test-source-{index}",
            labels=list(labels),
        )
        test_samples.append(Sample(graph, list(labels)))
    test_dataset = Dataset(test_samples)

    monkeypatch.setattr(
        dynamics_experiment,
        "_open_dataset",
        lambda split_root, device: train_dataset
        if str(split_root) == "train"
        else test_dataset,
    )
    monkeypatch.setattr(
        dynamics_evaluate,
        "_open_dataset",
        lambda split_root: test_dataset,
    )

    checkpoint = dynamics_experiment.train_dynamics_model(
        train_split="train",
        output_dir=tmp_path / "train",
        device="cpu",
        config=tiny_dynamics_config(),
    )
    score_path = dynamics_experiment.score_dynamics_split(
        split_root="test",
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "score",
        device="cpu",
    )
    with np.load(score_path, allow_pickle=False) as arrays:
        assert not bool(arrays["labels_included"].item())
        assert arrays["edge_error_map"].shape == (8, 2, 2)
        assert arrays["edge_state_decoupling"].shape == (8,)
        assert arrays["origin_fracture"].shape == (8,)

    dynamics_evaluate.evaluate_dynamics_scores(
        split_root="test",
        score_path=score_path,
        output_dir=tmp_path / "evaluation",
        bootstrap_replicates=10,
    )
    report = json.loads((tmp_path / "evaluation" / "evaluation.json").read_text())
    assert report["labels_read"] is True
    assert (tmp_path / "evaluation" / "layer_metrics.csv").is_file()
    assert (tmp_path / "evaluation" / "layer_head_metrics.csv").is_file()
