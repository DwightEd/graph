from experiments.grounded_route.graph_effectiveness.run import command_line


def test_cli_accepts_multiple_independently_encoded_controls():
    arguments = command_line().parse_args(
        [
            "audit",
            "--calibration",
            "real-calibration.npz",
            "--index",
            "real-test.npz",
            "--test",
            "/data/test",
            "--output",
            "/tmp/audit",
            "--control",
            "no_message",
            "no-message-calibration.npz",
            "no-message-test.npz",
            "--control",
            "endpoint_rewire",
            "rewire-calibration.npz",
            "rewire-test.npz",
            "--control",
            "weight_shuffle",
            "shuffle-calibration.npz",
            "shuffle-test.npz",
        ]
    )

    assert len(arguments.control) == 3
    assert arguments.control[0][0] == "no_message"
    assert arguments.control[1][0] == "endpoint_rewire"
    assert arguments.control[2][0] == "weight_shuffle"
