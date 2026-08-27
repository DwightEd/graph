from pathlib import Path
import subprocess

from experiments.directed_route_hypergraph.run import command_line


ROOT = Path(__file__).resolve().parents[3]


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
    assert ordered.layout_max_elements == 8_000_000
    assert ordered.layout_max_work_elements == 250_000_000
    assert reverse.layout_order == "reverse"


def test_shell_entrypoints_are_valid_bash():
    directory = ROOT / "experiments" / "directed_route_hypergraph"
    for name in ("run.sh", "run_qa.sh", "resume_legacy.sh"):
        subprocess.run(["bash", "-n", str(directory / name)], check=True)
