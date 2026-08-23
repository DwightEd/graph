from experiments.non_neural_structure_audit.main import build_parser


def test_cli_has_one_explicit_command_for_each_label_boundary_stage():
    parser = build_parser()

    fit = parser.parse_args(
        ["fit", "--train-split", "train", "--output", "reference.npz"]
    )
    score = parser.parse_args(
        [
            "score",
            "--split-root",
            "test",
            "--reference",
            "reference.npz",
            "--output-dir",
            "scores",
        ]
    )
    plan = parser.parse_args(
        ["plan", "--score-dir", "scores", "--output", "split.json"]
    )
    evaluate = parser.parse_args(
        [
            "evaluate",
            "--split-root",
            "test",
            "--score-dir",
            "scores",
            "--output-dir",
            "evaluation",
        ]
    )

    freeze = parser.parse_args(
        [
            "freeze-confirmation",
            "--split-plan",
            "split.json",
            "--discovery-evaluation",
            "discovery.json",
            "--output",
            "confirmation.json",
            "--tokenizer",
            "tokenizer",
        ]
    )

    assert (
        fit.command,
        score.command,
        plan.command,
        evaluate.command,
        freeze.command,
    ) == (
        "fit",
        "score",
        "plan",
        "evaluate",
        "freeze-confirmation",
    )
    assert fit.task_type == score.task_type == "QA"
