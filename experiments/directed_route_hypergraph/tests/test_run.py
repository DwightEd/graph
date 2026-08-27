from experiments.directed_route_hypergraph.run import command_line


def test_fit_cli_defaults_to_endpoint_recovery_and_deterministic_latent():
    parser = command_line()
    ordered = parser.parse_args(
        ["fit", "--train", "train", "--checkpoint", "model.pt"]
    )
    reverse = parser.parse_args(
        [
            "fit",
            "--train",
            "train",
            "--checkpoint",
            "model.pt",
            "--layout-order",
            "reverse",
        ]
    )

    assert ordered.layout_order == "ordered"
    assert ordered.positive_edges_per_graph == 4096
    assert ordered.holdout_fraction == 0.15
    assert ordered.negative_count == 1
    assert ordered.negative_attempt_factor == 8
    assert ordered.incidence_dropout == 0.0
    assert ordered.head_dropout == 0.0
    assert ordered.flow_weight == 0.0
    assert ordered.layout_weight == 0.0
    assert ordered.variance_weight == 0.05
    assert ordered.slot_dim == 16
    assert ordered.edge_hidden_dim == 64
    assert ordered.latent_mode == "deterministic"
    assert ordered.vae_export == "mean_logvar"
    assert ordered.kl_weight == 1e-3
    assert ordered.kl_free_bits == 1e-2
    assert ordered.kl_warmup_epochs == 4
    assert ordered.layout_rows_per_graph == 32
    assert ordered.layout_max_elements == 8_000_000
    assert ordered.layout_max_work_elements == 250_000_000
    assert reverse.layout_order == "reverse"
