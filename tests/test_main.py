import json

from control_graph.cli import main


def test_build_command_exposes_the_graph_builder(tmp_path, capsys) -> None:
    input_path = tmp_path / "events.jsonl"
    output_path = tmp_path / "graphs.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "schema": "control-graph/factorial-event@1",
                "event_id": "event-1",
                "source_id": "source-1",
                "split": "train",
                "relation": "temporal",
                "margins": {
                    "onset_a": 1.0,
                    "onset_b": -1.0,
                    "world_a_after_a": 1.0,
                    "world_a_after_b": 0.5,
                    "world_b_after_a": -0.5,
                    "world_b_after_b": -1.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    main(["build", "--input", str(input_path), "--output", str(output_path)])

    assert output_path.is_file()
    assert json.loads(capsys.readouterr().out)["graphs"] == 1
