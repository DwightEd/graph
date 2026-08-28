from pathlib import Path

from experiments.attention_mechanism_audit.run import DEFAULT_MODEL, _parser


def test_audit_cli_defaults_to_the_frozen_llama_checkpoint():
    args = _parser().parse_args(
        ["audit", "--pairs", "pairs.jsonl", "--output", "control_chain.npz"]
    )

    assert args.model == Path(DEFAULT_MODEL)
    assert args.torch_dtype == "bfloat16"
    assert args.device == "cuda"


def test_evaluate_cli_has_no_label_or_probe_arguments():
    parser = _parser()
    args = parser.parse_args(
        ["evaluate", "--artifact", "control_chain.npz", "--output", "report.json"]
    )

    assert vars(args).keys().isdisjoint({"labels", "folds", "epochs", "probe"})
