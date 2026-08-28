import torch

from ..config import GraphConstructionConfig
from ..encoding import build_node_encoding
from ..graph import build_graph_tensors
from .helpers import synthetic_bundle


def test_full_graph_exposes_every_causal_pair_and_conserves_messages():
    bundle = synthetic_bundle()
    graph = build_graph_tensors(
        bundle.capture,
        bundle.basis,
        GraphConstructionConfig(
            route_mass_retention=1.0,
            value_energy_retention=1.0,
        ),
    )
    audit = graph.audit
    assert audit["exposed_pair_fraction"] == 1.0
    assert audit["exposed_token_edges"] == audit["causal_token_pairs_considered"]
    assert audit["max_attention_reconstruction_abs_error"] < 1e-6
    assert audit["max_o_proj_input_reconstruction_abs_error"] < 1e-6
    assert audit["max_role_context_abs_error"] < 1e-6
    assert audit["max_quotient_context_abs_error"] < 1e-6
    assert audit["max_route_conservation_abs_error"] < 1e-6
    assert torch.count_nonzero(graph.remainder_features[..., 0]) == 0


def test_exact_quotient_consumes_unexposed_sources_without_mass_loss():
    bundle = synthetic_bundle(seed=11)
    graph = build_graph_tensors(
        bundle.capture,
        bundle.basis,
        GraphConstructionConfig(
            route_mass_retention=0.6,
            value_energy_retention=0.6,
        ),
    )
    audit = graph.audit
    assert 0.0 < audit["exposed_pair_fraction"] < 1.0
    assert torch.count_nonzero(graph.remainder_features[..., 0]) > 0
    assert audit["max_quotient_context_abs_error"] < 1e-6
    assert audit["max_route_conservation_abs_error"] < 1e-6


def test_node_encoding_preserves_all_layer_head_channels_without_learning():
    bundle = synthetic_bundle()
    graph = build_graph_tensors(bundle.capture, bundle.basis)
    encoding = build_node_encoding(graph)
    response = bundle.capture.response_count
    expected = (
        bundle.capture.hidden_size
        + graph.route_features[0].numel()
        + graph.layer_features[0].numel()
        + encoding.temporal_features.shape[1]
    )
    assert encoding.node_embedding.shape == (response, expected)
    assert len(encoding.node_feature_names) == expected
    assert torch.isfinite(encoding.node_embedding).all()


def test_persisted_edge_codes_reconstruct_the_frozen_attention_update():
    bundle = synthetic_bundle(seed=37)
    graph = build_graph_tensors(bundle.capture, bundle.basis)
    for layer_index, layer_capture in enumerate(bundle.capture.layers):
        output_factor = bundle.basis.output_factor[layer_index].float()
        hidden = bundle.capture.hidden_size
        output_weight = output_factor.permute(1, 0, 2).reshape(hidden, hidden)
        for query in range(bundle.capture.response_count):
            target = bundle.capture.response_start + query
            selected = (graph.edge_layer == layer_index) & (
                graph.edge_index[1] == target
            )
            sources = graph.edge_index[0, selected]
            code = graph.edge_attention_code[selected]
            values = layer_capture.value_states[sources][
                :, bundle.capture.q_to_kv
            ]
            context = (code[:, :, None] * values).sum(dim=0)
            update = torch.nn.functional.linear(
                context.reshape(1, hidden),
                output_weight,
                bundle.basis.output_bias[layer_index].float(),
            )[0]
            expected = layer_capture.attention_output[target].float()
            assert torch.allclose(update, expected, atol=1e-6, rtol=1e-6)
