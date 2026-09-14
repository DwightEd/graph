import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import logsumexp

from binding_detector.projection import Factor, cosine_logits, permute_relations, project
from binding_detector.evaluation import compare_blocks, evaluate_records, label_views, ranking
from binding_detector.run import demo_packets, event_projection, metadata, run, score_record


D = {"entity": ["a", "b"], "value": ["a", "b"]}
F = [Factor(("entity", "value"), frozenset({("a", "a"), ("b", "b")}), closed_world=True,
            provenance="constructed test relation")]


def logs(p=(.95, .05), q=(.95, .05)):
    return dict(entity=np.log(p), value=np.log(q))


def args(tmp, **kw):
    return SimpleNamespace(demo=True, cases=None, annotations=None, roster=None, output=str(tmp / "out"),
                           split="all", max_states=200000, permutations=4, seed=9, bootstrap=4, **kw)


def test_equal_confidence_but_different_joint_binding():
    good = project(D, logs(), F)
    bad = project(D, logs(q=(.05, .95)), F)
    assert good["lower_nats"] == pytest.approx(-np.log(.905))
    assert bad["lower_nats"] == pytest.approx(-np.log(.095))
    assert good["node_entropy"] == pytest.approx(bad["node_entropy"])
    assert good["node_confidence"] == pytest.approx(bad["node_confidence"])
    assert good["upper_nats"] == pytest.approx(good["lower_nats"])


def test_information_projection_identity_and_minimum():
    rng = np.random.default_rng(17)
    for _ in range(20):
        a, b = rng.dirichlet([1, 1], size=2)
        q = np.outer(a, b)
        result = project(D, logs(a, b), F)
        legal = np.diag(q)
        posterior = legal / legal.sum()
        arbitrary = rng.dirichlet([1, 1])
        kl = np.dot(arbitrary, np.log(arbitrary / legal))
        residual = np.dot(arbitrary, np.log(arbitrary / posterior))
        assert kl == pytest.approx(result["lower_nats"] + residual)
        assert result["projected_marginals"]["entity"] == pytest.approx(posterior)


def test_explicit_joint_preserves_dependence():
    q = np.array([[.49, .01], [.01, .49]])
    r = project(D, logs((.5, .5), (.5, .5)), F, joint_log_weights=np.log(q))
    assert r["supported_mass"] == pytest.approx(.98)
    assert r["q_assumption"] == "supplied_joint"
    assert project(D, logs((.5, .5), (.5, .5)), F)["supported_mass"] == pytest.approx(.5)


def test_unknown_relation_is_not_false():
    f = Factor(("entity", "value"), frozenset({("a", "a")}), frozenset({("a", "b")}),
               provenance="partial table")
    r = project(D, logs((.5, .5), (.5, .5)), [f])
    assert r["supported_mass"] == .25 and r["unknown_mass"] == .5
    assert r["lower_nats"] == pytest.approx(-np.log(.75))
    assert r["upper_nats"] == pytest.approx(-np.log(.25))
    assert r["status"] == "partial_relations"


def test_no_constraints_does_not_certify_normality():
    r = project(D, logs(), [])
    assert r["status"] == "no_constraints" and r["unknown_mass"] == pytest.approx(1.)
    assert r["lower_nats"] == 0 and np.isposinf(r["upper_nats"])


def test_contradictory_tables_are_invalid_not_a_hallucination():
    f2 = Factor(("entity", "value"), frozenset({("a", "b"), ("b", "a")}),
                closed_world=True, provenance="incompatible table")
    r = project(D, logs(), F + [f2])
    assert r["status"] == "inconsistent_constraints" and not r["projection_defined"]


def test_stable_log_mass_does_not_smooth_underflow():
    w = dict(entity=[0., -2000.], value=[-2000., 0.])
    r = project(D, w, F)
    assert r["lower_nats"] == pytest.approx(2000 - np.log(2))
    assert np.isfinite(r["lower_nats"])


def test_exact_zero_mass_stays_infinite():
    r = project(D, dict(entity=[0., -np.inf], value=[-np.inf, 0.]), F)
    assert np.isinf(r["lower_nats"]) and not r["projection_defined"]


def test_shared_variable_constraints_are_joint_not_pair_averages():
    d = {"a": ["0", "1"], "b": ["0", "1"], "c": ["0", "1"]}
    same = frozenset({("0", "0"), ("1", "1")})
    fs = [Factor(("a", "b"), same, closed_world=True, provenance="test"),
          Factor(("b", "c"), same, closed_world=True, provenance="test")]
    r = project(d, {v: [0, 0] for v in d}, fs)
    assert r["supported_mass"] == pytest.approx(.25)


def test_candidate_order_invariance():
    a = project(D, logs(), F)
    d = {"entity": ["b", "a"], "value": ["a", "b"]}
    b = project(d, logs((.05, .95)), F)
    assert a["lower_nats"] == pytest.approx(b["lower_nats"])


def test_type_preserving_permutation_and_global_maps():
    groups = {"entity": ["x", "y"], "value": ["x", "y"]}
    fs, info = permute_relations(D, F, 5, groups)
    assert info["moved_fraction"] == 0 and fs[0].allowed == F[0].allowed
    fs, _ = permute_relations(D, F * 2, 17)
    assert fs[0].allowed == fs[1].allowed
    assert len(fs[0].allowed) == len(F[0].allowed)


@pytest.mark.parametrize("case", ["cap", "zero", "nan", "duplicate", "bad_tuple", "provenance"])
def test_invalid_inputs_are_not_silently_fixed(case):
    d, w, fs, cap = copy.deepcopy(D), logs(), F, 200000
    if case == "cap": cap = 3
    if case == "zero": w["entity"] = [-np.inf, -np.inf]
    if case == "nan": w["entity"] = [np.nan, 0]
    if case == "duplicate": d["entity"] = ["a", "a"]
    if case == "bad_tuple": fs = [Factor(("entity",), frozenset({("z",)}), provenance="test")]
    if case == "provenance": fs = [Factor(("entity",), frozenset({("a",)}))]
    with pytest.raises(ValueError): project(d, w, fs, max_states=cap)


def test_native_vector_matcher_keeps_direction():
    np.testing.assert_allclose(cosine_logits([1., 0.], [[1., 0.], [-1., 0.]], 1.), [1, -1])
    with pytest.raises(ValueError): cosine_logits([0., 0.], [[1., 0.]])


def test_cached_vector_packet(tmp_path):
    packet = next(demo_packets())
    e = packet["events"][0]
    e["representations"], e["representation_space"] = "vectors.npz", "observer=X layer=16 head=2 post_WO"
    np.savez(tmp_path / "vectors.npz", q=[1., 0.], keys=[[1., 0.], [0., 1.]])
    for v in e["variables"]:
        v.pop("log_weights")
        v.update(query_array="q", candidate_array="keys")
    r = event_projection(e, tmp_path, 100, 2, 9)
    assert r["lower_nats"] < .001


def test_detection_time_and_uncovered_tokens_are_explicit(tmp_path):
    packet = next(demo_packets())
    m = metadata(packet)
    scores, time, _ = score_record(m, packet, tmp_path, 100, 2, 9)
    assert np.isnan(scores["binding_lower_nats"][:2]).all()
    assert time.tolist() == [-1, -1, 11]
    e = packet["events"][0]
    e["available_after_char"] = 6
    with pytest.raises(ValueError): score_record(m, packet, tmp_path, 100, 2, 9)


def test_overlap_does_not_implicitly_propagate_error(tmp_path):
    packet = next(demo_packets()); packet["events"] *= 2
    with pytest.raises(ValueError): score_record(metadata(packet), packet, tmp_path, 100, 2, 9)


def test_first_error_is_not_each_span_onset():
    v = label_views(np.array([[i, i + 1] for i in range(8)]), [{"start": 1, "end": 3}, {"start": 5, "end": 7}], 8)
    assert np.flatnonzero(v["all_error"][0]).tolist() == [1, 2, 5, 6]
    assert np.flatnonzero(v["span_onset_full_stream"][0]).tolist() == [1, 5]
    assert np.flatnonzero(v["first_error_full_stream"][0]).tolist() == [1]
    assert np.flatnonzero(v["first_error_until_first"][1]).tolist() == [0, 1]
    assert np.flatnonzero(v["continuation_vs_normal"][0]).tolist() == [2, 6]
    assert np.flatnonzero(v["continuation_vs_normal"][1]).tolist() == [0, 2, 3, 4, 6, 7]
    assert np.flatnonzero(v["strict_post_first"][1]).tolist() == list(range(2, 8))


def test_no_error_answer_is_not_dropped():
    v = label_views(np.array([[0, 2], [2, 4]]), [], 4)
    assert v["first_error_until_first"][1].all()
    assert not v["strict_post_first"][1].any()


def test_overlapping_gold_spans_and_zero_length_offset():
    v = label_views(np.array([[0, 0], [0, 3], [3, 5]]), [{"start": 0, "end": 2}, {"start": 1, "end": 4}], 5)
    assert v["span_onset_full_stream"][0].sum() == 1
    assert v["all_error"][0].tolist() == [False, True, True]


def test_missingness_and_infinity_do_not_change_denominator():
    r = ranking([0, 1, 1], [0, np.inf, np.nan], ["a", "a", "b"])
    assert r["eligible_tokens"] == 3 and r["covered_tokens"] == 2
    assert r["positive_coverage"] == .5 and r["pooled"]["auroc"] == 1


def test_paired_bootstrap_reproducible_and_common_coverage():
    blocks = [dict(source_id=str(i), y=np.array([0, 1, 1]),
                   scores=dict(a=np.array([0, 2, np.nan]), b=np.array([1, 0, 3]))) for i in range(3)]
    a = compare_blocks(blocks, "a", ["b"], bootstrap=4)
    b = compare_blocks(blocks, "a", ["b"], bootstrap=4)
    assert a == b
    assert a["comparisons"]["b"]["primary"]["covered_tokens"] == 6


def test_demo_end_to_end_and_refuse_overwrite(tmp_path):
    a = args(tmp_path); run(a)
    out = Path(a.output)
    assert json.loads((out / "complete.json").read_text())["trained_parameters"] == 0
    assert (out / "prediction_freeze.json").exists()
    with np.load(out / "demo_cross_bound.npz") as bad, np.load(out / "demo_supported.npz") as good:
        assert bad["binding_lower_nats"][-1] > good["binding_lower_nats"][-1]
    with pytest.raises(FileExistsError): run(a)


def test_scores_freeze_before_annotation_failure(tmp_path):
    packets = list(demo_packets())
    cases = tmp_path / "cases.jsonl"
    cases.write_text("\n".join(json.dumps(p) for p in packets))
    a = args(tmp_path); a.demo = False; a.cases = str(cases); a.annotations = str(tmp_path / "missing.jsonl")
    with pytest.raises(FileNotFoundError): run(a)
    assert (Path(a.output) / "prediction_freeze.json").exists()
    assert not (Path(a.output) / "complete.json").exists()


def test_full_roster_retains_missing_packets_and_label_join(tmp_path):
    packets = list(demo_packets())
    cases, roster, labels = [tmp_path / f"{n}.jsonl" for n in ("cases", "roster", "labels")]
    cases.write_text(json.dumps(packets[0]) + "\n")
    roster.write_text("\n".join(json.dumps(p) for p in packets))
    labels.write_text("\n".join(json.dumps(dict(id=p["id"], source_id=p["source_id"], split="test", response=p["response"],
                                               labels=[] if i == 0 else [dict(start=6, end=11)])) for i, p in enumerate(packets)))
    a = args(tmp_path); a.demo = False; a.cases = str(cases); a.roster = str(roster); a.annotations = str(labels)
    run(a)
    result = json.loads((Path(a.output) / "evaluation.json").read_text())
    metric = result["views"]["all_error"]["metrics"]["binding_lower_nats"]
    assert metric["eligible_tokens"] == 6 and metric["covered_tokens"] == 1
    assert metric["eligible_positives"] == 1 and metric["covered_positives"] == 0


def make_s10_fixture(tmp):
    from structural_detector.audit import sha
    feature, prediction = tmp / "features", tmp / "predictions"
    feature.mkdir(); prediction.mkdir()
    records, index, gold = [], [], []
    for i, text in enumerate(("abcd", "efgh")):
        rid = str(i)
        np.savez(feature / (rid + ".npz"), offsets=[[j, j + 1] for j in range(4)], values=np.ones((4, 5)))
        records.append(dict(id=rid, source_id=rid, official_split="test", tokens=4,
                            response_sha256=hashlib.sha256(text.encode()).hexdigest(), artifact_sha256=sha(feature / (rid + ".npz"))))
        index.append(dict(id=rid, source_id=rid, split="test", start=4 * i, end=4 * i + 4))
        gold.append(dict(id=rid, source_id=rid, split="test", response=text,
                         labels=[dict(start=1, end=3), dict(start=3, end=4)] if i == 0 else []))
    (feature / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    (feature / "manifest.json").write_text(json.dumps(dict(complete=True, records_sha256=sha(feature / "records.jsonl"))))
    (prediction / "protocol_freeze.json").write_text(json.dumps(dict(feature_manifest_sha256=sha(feature / "manifest.json"))))
    (prediction / "prediction_index.json").write_text(json.dumps(index))
    methods = [target + "__" + arm for target in ("error", "onset")
               for arm in ("combined", "instant", "without_entropy", "without_attention")]
    methods += ["raw_entropy", "raw_remote_history_excl16_minus_source", "raw_negative_margin", "raw_position"]
    np.savez(prediction / "predictions.npz", **{name: np.array([.1, .9, .8, .7, .1, .1, .1, .1]) for name in methods})
    (prediction / "prediction_freeze.json").write_text(json.dumps(dict(hashes={name: sha(prediction / name)
        for name in ("predictions.npz", "prediction_index.json", "protocol_freeze.json")})))
    (prediction / "complete.json").write_text(json.dumps(dict(complete=True)))
    annotations = tmp / "annotations.jsonl"
    annotations.write_text("\n".join(json.dumps(g) for g in gold))
    return feature, prediction, annotations


def test_s10_audit_end_to_end(tmp_path):
    from structural_detector.audit import audit
    f, p, g = make_s10_fixture(tmp_path)
    audit(SimpleNamespace(features=str(f), predictions=str(p), annotations=str(g),
                          output=str(tmp_path / "audit"), bootstrap=2))
    result = json.loads((tmp_path / "audit/evaluation.json").read_text())
    views = result["views"]
    assert views["span_onset_full_stream"]["metrics"]["onset__combined"]["eligible_positives"] == 2
    assert views["first_error_full_stream"]["metrics"]["onset__combined"]["eligible_positives"] == 1
    assert views["continuation_vs_normal"]["metrics"]["error__combined"]["eligible_positives"] == 1
    assert result["scope"]["upstream_score_supervision"] == "natural training and calibration labels"


def test_s10_audit_rejects_changed_artifacts(tmp_path):
    from structural_detector.audit import load_frozen
    f, p, _ = make_s10_fixture(tmp_path)
    with (p / "predictions.npz").open("ab") as out:
        out.write(b"changed")
    with pytest.raises(ValueError): load_frozen(p, f)


def test_new_binding_compares_s10_on_complete_test_roster(tmp_path):
    f, p, gold = make_s10_fixture(tmp_path)
    packet = next(demo_packets())
    packet.update(id="0", source_id="0", response="abcd", offsets=[[i, i + 1] for i in range(4)])
    packet["events"][0].update(target_span=[1, 2], available_after_char=2)
    cases = tmp_path / "packets.jsonl"; cases.write_text(json.dumps(packet))
    a = args(tmp_path); a.demo = False; a.cases = str(cases); a.annotations = str(gold)
    a.s10_predictions = str(p); a.s10_features = str(f); a.split = "test"
    run(a)
    r = json.loads((Path(a.output) / "evaluation.json").read_text())["views"]["all_error"]
    assert r["metrics"]["error__combined"]["covered_tokens"] == 8
    assert r["metrics"]["binding_lower_nats"]["eligible_tokens"] == 8
    assert r["comparisons"]["error__combined"]["primary"]["covered_tokens"] == 1


def test_partial_relations_have_bounds_but_no_exact_risk(tmp_path):
    packet = next(demo_packets())
    factor = packet["events"][0]["factors"][0]
    factor["closed_world"] = False
    scores, _, diagnostic = score_record(metadata(packet), packet, tmp_path, 100, 2, 9)
    assert np.isnan(scores["binding_exact_nats"][-1])
    assert scores["binding_lower_nats"][-1] == 0
    assert np.isfinite(scores["node_entropy"][-1])
    assert diagnostic[0]["unknown_mass"] > 0
