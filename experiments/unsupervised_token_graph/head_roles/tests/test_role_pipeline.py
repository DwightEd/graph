"""End-to-end head-prior isolation and frozen evaluation."""

import json

import numpy as np

from experiments.unsupervised_token_graph.fixed_graph.tests.test_fixed_graph import write_cache
from experiments.unsupervised_token_graph.head_roles import profile
from experiments.unsupervised_token_graph.head_roles.run import main
from experiments.unsupervised_token_graph.head_roles.tests.test_roles import tiny_llama


def fixture(tmp_path, output):
    dataset = tmp_path / "dataset"
    dataset.mkdir(exist_ok=True)
    rows = []
    for split, identities in (("train", range(1, 7)), ("test", range(10, 12))):
        for identity in identities:
            name = str(identity)
            write_cache(tmp_path / split / f"{name}.npz", identity=name, split=split, seed=identity)
            rows.append(dict(id=name, source_id="source_" + name, split=split, model="fixture",
                             response="abcdefghijkl", labels=[dict(start=2, end=6)]))
    (dataset / "response.jsonl").write_text("\n".join(map(json.dumps, rows)))
    return ["--train-cache", str(tmp_path / "train"), "--test-cache", str(tmp_path / "test"),
        "--dataset", str(dataset), "--output", str(output), "--special-token-ids", "10",
        "--probe-sources", "3", "--probe-queries", "2", "--block-width", "2", "--swaps", "3",
        "--bank-size", "32", "--tokens-per-source", "8", "--bootstrap", "3", "--threads", "1",
        "--dtype", "float32", "--device", "cpu", "--reconstruction-atol", "0.000001", "--no-binding-check"]


def test_full_pipeline_native_probe_score_freeze_evaluate_resume_and_lda(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, "load_model", lambda args: tiny_llama(heads=2))
    output = tmp_path / "run"
    argv = fixture(tmp_path, output)
    report = main(argv)
    assert report["groups"]["ALL"]["views"]["all_error"]["contrast__drop_positional"]["evaluated_tokens"] == 20
    assert (output / "profile/head_roles.png").is_file()
    assert (output / "predictions/metrics.csv").is_file()
    before = np.load(output / "profile/priors.npz")["source_gap"].copy()
    main([*argv, "--resume"])
    after = np.load(output / "profile/priors.npz")["source_gap"].copy()
    np.testing.assert_array_equal(before, after)
    rows = main([*argv, "--phase", "diagnostic"])
    assert all(row["status"] == "supervised_diagnostic_only" for row in rows)
    assert (output / "diagnostic/supervised_lda.csv").is_file()


def test_labels_and_test_features_do_not_change_head_prior_or_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(profile, "load_model", lambda args: tiny_llama(heads=2))
    first, second = tmp_path / "first", tmp_path / "second"
    argv = fixture(tmp_path, first)
    original = np.lib.npyio.NpzFile.__getitem__
    def guard(archive, key):
        if key in ("labels", "hallucination_labels"):
            raise AssertionError("Natural labels reached prior discovery or unsupervised fitting")
        return original(archive, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", guard)
    for phase in ("prepare", "profile", "fit", "score"):
        main([*argv, "--phase", phase, *(["--resume"] if phase != "prepare" else [])])
    annotation = tmp_path / "dataset/response.jsonl"
    rows = [json.loads(line) for line in annotation.read_text().splitlines()]
    for row in rows:
        row["labels"] = []
    annotation.write_text("\n".join(map(json.dumps, rows)))
    write_cache(tmp_path / "test/10.npz", identity="10", split="test", seed=991)
    argv[argv.index("--output") + 1] = str(second)
    for phase in ("prepare", "profile", "fit", "score"):
        main([*argv, "--phase", phase, *(["--resume"] if phase != "prepare" else [])])
    for file in ("profile/priors.npz", "reference/0/bank.npz"):
        with np.load(first / file) as a, np.load(second / file) as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key])
