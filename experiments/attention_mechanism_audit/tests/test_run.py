from pathlib import Path

from experiments.attention_mechanism_audit.run import DEFAULT_MODEL, parser


def test_capture_cli_targets_real_cached_samples_and_frozen_llama():
    args = parser().parse_args(
        [
            "capture",
            "--split-root",
            "cache/test",
            "--source-info",
            "source_info.jsonl",
            "--output",
            "traces",
        ]
    )

    assert args.split_root == Path("cache/test")
    assert args.source_info == Path("source_info.jsonl")
    assert args.model == Path(DEFAULT_MODEL)
    assert args.device == "cuda:0"
    assert args.dtype == "bfloat16"
    assert args.predictor_chunk == 64
    assert args.intervention_batch == 3
    assert args.top_k == 8
    assert args.logit_chunk == 64
    assert vars(args).keys().isdisjoint({"pairs", "candidate_a", "candidate_b"})


def test_evaluate_cli_is_posthoc_and_has_no_probe_training_options():
    args = parser().parse_args(
        [
            "evaluate",
            "--traces",
            "traces",
            "--split-root",
            "cache/test",
            "--output",
            "report.json",
        ]
    )

    assert args.traces == Path("traces")
    assert args.split_root == Path("cache/test")
    assert args.output == Path("report.json")
    assert args.position_bin == 16
    assert vars(args).keys().isdisjoint({"pairs", "folds", "epochs", "probe"})
