from experiments.directed_route_hypergraph.run import command_line


def test_fit_cli_defaults_to_ordered_layout_and_accepts_reverse_control():
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
    assert ordered.layout_weight == 0.25
    assert ordered.layout_rows_per_graph == 32
    assert ordered.layout_max_elements == 8_000_000
    assert ordered.layout_max_work_elements == 250_000_000
    assert reverse.layout_order == "reverse"
