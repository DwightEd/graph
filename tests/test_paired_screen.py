"""Cached confidence controls must not block native audits on older samples."""

import json

import numpy as np
import pandas as pd

from experiments.path_conflict.paired import cli
from experiments.path_conflict.paired_screen import confidence_controls, saved_entropy_nats


def sample_trace():
    rng = np.random.default_rng(4)
    prompt, length, layers, heads = 3, 7, 3, 4
    attention = np.zeros((layers, heads, length, prompt + length))
    for position in range(length):
        attention[:, :, position, :prompt + position] = rng.dirichlet(
            np.ones(prompt + position), size=(layers, heads))
    return dict(prompt_length=np.array(prompt), attention=attention,
                logit_entropy=np.linspace(.2, 1., length),
                token_ids=np.arange(prompt + length),
                token_text=np.array(["sys ", "context ", "answer: ",
                                     "A ", "new ", "claim", " words", ".", " Then", "."]),
                special_mask=np.zeros(prompt + length, dtype=bool),
                chosen_logit=np.ones(length), log_normalizer=np.full(length, 3.),
                top_ids=np.tile(np.arange(5), (length, 1)),
                top_logits=np.tile([2., 1., 0., -1., -2.], (length, 1)))


def test_saved_entropy_keeps_units_and_prefix_only_scores():
    trace = sample_trace()
    np.testing.assert_array_equal(saved_entropy_nats(trace), trace["logit_entropy"] * np.log(2.))
    scores, selected, previous = confidence_controls(trace, 41)
    np.testing.assert_allclose(scores[0], -np.log(np.log(41) - trace["logit_entropy"][0] * np.log(2.)))
    np.testing.assert_array_equal(previous[:, :, 2], trace["attention"][:, :, 2, 4].astype(np.float32))
    trace["attention"][:, :, 4:] = 100
    trace["logit_entropy"][4:] = 100
    changed, changed_heads, _ = confidence_controls(trace, 41)
    np.testing.assert_array_equal(scores[:4], changed[:4])
    np.testing.assert_array_equal(selected[:4], changed_heads[:4])


def test_missing_entropy_does_not_use_top5_or_surprisal_as_entropy():
    trace = sample_trace()
    _, heads, previous = confidence_controls(trace, 41)
    del trace["logit_entropy"]
    scores, old_heads, old_previous = confidence_controls(trace, 41)
    assert np.isnan(saved_entropy_nats(trace)).all()
    assert np.isnan(scores).all()
    np.testing.assert_array_equal(old_heads, heads)
    np.testing.assert_array_equal(old_previous, previous)


def test_screen_cli_handles_mixed_cache_versions_and_resumes(tmp_path, capsys):
    samples, legacy_bytes = [], None
    for seed in (0, 1):
        trace = sample_trace()
        if seed == 1:
            del trace["logit_entropy"]
            trace["token_text"][5:7] = ["wrong", " claim"]
        path = tmp_path / f"{seed}.npz"
        np.savez_compressed(path, **trace)
        if seed == 1:
            legacy_bytes = path.read_bytes()
        samples.append(dict(source_id="one", seed=seed, trace=path.name,
                            response="".join(trace["token_text"][3:])))
    case = dict(case_id="claim", source_id="one", supported=dict(seed=0, target="claim words"),
                unsupported=dict(seed=1, target="wrong claim"), evidence=["context"], value_source=["sys"],
                evidence_status="test_fixture", history_status=dict(supported="unknown", unsupported="unknown"))
    (tmp_path / "settings.json").write_text(json.dumps(dict(model="unused-test-model")))
    (tmp_path / "samples.jsonl").write_text("\n".join(json.dumps(row) for row in samples))
    (tmp_path / "prompts.jsonl").write_text(json.dumps(dict(source_id="one", prompt="sys context answer: ")))
    (tmp_path / "cases.json").write_text(json.dumps([case]))
    output = tmp_path / "output"
    arguments = ["--stage", "screen", "--samples", str(tmp_path), "--cases", str(tmp_path / "cases.json"),
                 "--output", str(output), "--vocabulary-size", "41"]
    cli(arguments)
    config = (output / "paired_config.json").read_bytes()
    cli(arguments)
    assert (output / "paired_config.json").read_bytes() == config
    assert (tmp_path / "1.npz").read_bytes() == legacy_bytes
    rows = pd.read_csv(output / "confidence_controls.csv")
    old, new = rows[rows.side == "unsupported"], rows[rows.side == "supported"]
    fields = ["entropy_nats", "local_confidence_score", "ewma_score", "prefix_rauq_score"]
    assert len(old) == len(new) == 4
    assert old[fields].isna().all().all() and np.isfinite(new[fields]).all().all()
    assert old.entropy_status.eq("not_saved").all() and new.entropy_status.eq("saved_bits").all()
    np.testing.assert_array_equal(old.surprisal_nats, 2.)
    with np.load(output / "screen" / "claim.npz", allow_pickle=False) as saved:
        assert str(saved["unsupported_entropy_status"]) == "not_saved"
        assert np.isnan(saved["unsupported_scores"]).all()
        assert np.isfinite(saved["unsupported_previous_token_attention"]).all()
    assert "logit_entropy not saved" in capsys.readouterr().out
