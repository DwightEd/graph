import hashlib
import json

import numpy as np
import pytest

from experiments.unsupervised_token_graph.reanchor_run import run
from experiments.unsupervised_token_graph.reanchor_evaluate import evaluate


def fixture(tmp_path):
    cache = tmp_path / "cache"; cache.mkdir()
    metadata, gold = [], []
    for index in range(3):
        rid = str(index)
        p, tokens = 2, 5
        np.savez_compressed(cache / f"{rid}.npz", token_ids=np.arange(tokens), response_idx=p,
                            attention_diagonal=np.array([[[0, 0, .2, .2, .2]]]),
                            response_row_ptr=np.array([0, 1, 3, 5]),
                            response_column_indices=np.array([0, 0, 2, 1, 3]),
                            response_values=np.array([.8, .3, .5, .3, .5]))
        response = "abc"
        metadata.append(dict(cache=f"{rid}.npz", id=rid, source_id=f"s{rid}", split="test",
                             task="QA", generator="fixture", offsets=[[0, 1], [1, 2], [2, 3]],
                             response_sha256=hashlib.sha256(response.encode()).hexdigest()))
        gold.append(dict(id=rid, source_id=f"s{rid}", split="test", response=response,
                         labels=[{"start": 1, "end": 2}] if index != 1 else []))
    metadata_path, gold_path = tmp_path / "metadata.jsonl", tmp_path / "response.jsonl"
    metadata_path.write_text("\n".join(map(json.dumps, metadata)))
    gold_path.write_text("\n".join(map(json.dumps, gold)))
    return cache, metadata_path, gold_path


def test_canonical_cache_to_frozen_eval_with_explicit_missing_first_token(tmp_path):
    cache, metadata, gold = fixture(tmp_path)
    output = tmp_path / "result"
    rows = run(cache, output, metadata, root_bins=2, min_history=1)
    assert len(rows) == 3
    with np.load(output / "samples/0.npz", allow_pickle=False) as values:
        assert values["source_mismatch_bits"].shape == (1, 3)
        assert values["query_positions"].tolist() == [2, 3, 4]
        assert values["prediction_positions"].tolist() == [3, 4, 5]
        assert "offline_fai" not in values
        assert not any("lookback" in key for key in values.files)
    report = evaluate(output, gold, tmp_path / "evaluation.json", bootstrap=10)
    all_error = report["groups"]["ALL"]["views"]["all_error"]
    assert all_error["event_strength"]["eligible_tokens"] == 9
    assert all_error["event_strength"]["evaluated_tokens"] == 6
    assert all_error["event_strength"]["coverage"] == pytest.approx(2 / 3)
    assert all_error["source_mismatch_bits"]["coverage"] == pytest.approx(1 / 3)
    assert report["groups"]["ALL"]["sources"] == 3
    assert run(cache, output, metadata, root_bins=2, min_history=1, resume=True) == rows


def test_analysis_needs_no_annotations_but_evaluation_requires_identity(tmp_path):
    cache, _, gold = fixture(tmp_path)
    output = tmp_path / "without_identity"
    run(cache, output, controls=False)
    with pytest.raises(ValueError, match="identity-bound"):
        evaluate(output, gold, bootstrap=0)


def test_changed_cache_cannot_silently_resume(tmp_path):
    cache, metadata, _ = fixture(tmp_path)
    output = tmp_path / "result"
    run(cache, output, metadata, controls=False)
    path = cache / "0.npz"
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="cache changed"):
        run(cache, output, metadata, controls=False, resume=True)
    assert not (output / "complete.json").exists()


def test_wrong_answer_hash_is_rejected_without_rerunning_scoring(tmp_path):
    cache, metadata, gold = fixture(tmp_path)
    output = tmp_path / "result"
    run(cache, output, metadata)
    rows = [json.loads(line) for line in gold.read_text().splitlines()]
    rows[0]["response"] = "abd"
    gold.write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="identity mismatch"):
        evaluate(output, gold, bootstrap=0)


def test_main_entry_is_source_flow_not_autoencoder(tmp_path):
    from experiments.unsupervised_token_graph.run import main
    cache, _, _ = fixture(tmp_path)
    out = tmp_path / "default"
    main(["--cache", str(cache), "--output", str(out), "--no-controls"])
    assert (out / "complete.json").exists()
    assert not (out / "model.pt").exists()
