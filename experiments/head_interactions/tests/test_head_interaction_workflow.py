"""Native tiny-model workflow checks; never interpreted as natural-data evidence."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from state_audit.demo import build_demo, demo_examples
from state_audit.model import load_model
from state_audit.storage import read_json, write_json

from experiments.head_interactions.inputs import compile_groups, read_cases
from experiments.head_interactions.protocol import random_groups
from experiments.head_interactions.report import report, semantic_differences, source_summary
from experiments.head_interactions.run import execute, freeze_settings
from experiments.head_interactions.trials import Trials


@pytest.fixture
def audit(tmp_path):
    torch.set_num_threads(1)
    _, checkpoint = build_demo(tmp_path / "fixture", "llama")
    model, tokenizer = load_model(str(checkpoint), "main", "cpu", "float32")
    example = asdict(demo_examples()[0])
    example.pop("response")
    rows = [
        dict(
            id=f"case{index}",
            source_id=f"source{index}",
            example=example,
            response_prefix="Mira wears a",
            candidates=[" blue coat.", " red coat."],
            preferred=index,
            dataset="synthetic",
            task="QA",
            metadata=dict(variant="base"),
        )
        for index in (0, 1)
    ]
    rows[0].update(donor="case1", donor_kind="software_identity_control")
    path = tmp_path / "cases.json"
    write_json(path, dict(template="chat", cases=rows))
    cases = read_cases(path, tokenizer)
    plan = dict(
        groups=[
            dict(name=f"layer{layer}", sites=[dict(layer=layer, heads=[0], source="ordinary")])
            for layer in (0, 1)
        ]
    )
    args = SimpleNamespace(
        output=tmp_path / "audit",
        query_offsets=[0],
        random_seeds=[0],
        atol=1e-5,
        gain=1.5,
        resume=False,
        no_compensation=False,
        revision="main",
        dtype="float32",
        minimum_effect=0.05,
    )
    return model, tokenizer, cases, plan, args


def test_full_workflow_donor_random_report_and_resume_without_forward(audit, monkeypatch):
    model, tokenizer, cases, plan, args = audit
    freeze_settings(model, cases, plan, args, "tiny")
    execute(model, tokenizer, cases, plan, args)
    result = report(args.output, draws=10)
    assert result["expected_panels"] == result["completed_panels"] == 4
    assert result["valid_panels"] == 4 and result["failed_controls"] == 0
    assert result["interaction_rows"] == 18  # 4 zero-reference + 2 donor panels, three readouts.
    assert result["adaptation_rows"] == 12
    assert (args.output.parent / "audit_review.tar.gz").is_file()
    base = read_json(args.output / "cases/000000/query_0/selected/trials/0000.json")
    reverse = read_json(args.output / "cases/000001/query_0/selected/trials/0000.json")
    assert reverse["scores"]["sum_margin"] == pytest.approx(-base["scores"]["sum_margin"])
    assert reverse["scores"]["candidate_sum_margin"] == base["scores"]["candidate_sum_margin"]

    def no_forward(*args, **kwargs):
        raise AssertionError("Completed conditions must be reused")

    monkeypatch.setattr(model, "forward", no_forward)
    args.resume = True
    freeze_settings(model, cases, plan, args, "tiny")
    execute(model, tokenizer, cases, plan, args)
    assert report(args.output, draws=10) == result


def test_prepared_cases_keep_prompt_history_and_partial_trial_resume(audit, tmp_path, monkeypatch):
    model, tokenizer, cases, plan, args = audit
    prepared = tmp_path / "prepared.json"
    write_json(prepared, dict(cases=[asdict(case) for case in cases]))
    reread = read_cases(prepared, tokenizer)
    assert reread[0].sources == cases[0].sources
    groups = compile_groups(plan, cases[0], model)
    original = Trials(model, cases[0], groups, args.output).evaluate("baseline")

    def no_forward(*args, **kwargs):
        raise AssertionError("Saved trial must be reused before panel completion")

    monkeypatch.setattr(model, "forward", no_forward)
    resumed = Trials(model, cases[0], groups, args.output).evaluate("baseline")
    assert original[0] == resumed[0]
    np.testing.assert_array_equal(original[1]["layer0:0"]["write"], resumed[1]["layer0:0"]["write"])


def test_random_control_preserves_sites_and_does_not_use_selected_heads(audit):
    model, _, cases, plan, _ = audit
    groups = compile_groups(plan, cases[0], model)
    control = random_groups(groups, model, seed=0)
    for name, sites in control.items():
        for original, changed in zip(groups[name], sites):
            assert original.layer == changed.layer and original.positions == changed.positions
            assert original.keys == changed.keys and len(original.heads) == len(changed.heads)
            assert not set(original.heads) & set(changed.heads)


def test_source_bootstrap_does_not_count_heads_or_repeated_rows_as_sources():
    rows = [dict(source_id="long", value=1.0)] * 20 + [dict(source_id="short", value=3.0)]
    result = source_summary(rows, "value", draws=100)
    assert result["sources"] == 2 and result["rows"] == 21
    assert result["mean"] == 2.0


def test_report_partial_and_failed_panels_are_not_successful(audit):
    model, tokenizer, cases, plan, args = audit
    freeze_settings(model, cases, plan, args, "tiny")
    partial = report(args.output, draws=0)
    assert partial["completed_panels"] == 0 and partial["expected_panels"] == 4
    execute(model, tokenizer, cases, plan, args)
    path = args.output / "cases/000000/query_0/selected/results.json"
    result = read_json(path)
    result["valid"] = False
    result["controls"]["self_restore"]["passed"] = False
    write_json(path, result)
    summary = report(args.output, draws=0)
    assert summary["failed_controls"] == 1 and summary["valid_panels"] == 3


def test_semantic_comparison_keeps_candidate_direction_when_support_flips():
    common = dict(
        source_id="source",
        dataset="synthetic",
        task="QA",
        generator="tiny",
        panel="selected",
        query_offset=0,
        side="unspecified",
        a="A",
        b="B",
        reference="zero",
        valid=True,
        readout="candidate_sum_margin",
    )
    rows = [
        dict(common, case_id="base", variant="base", interaction=0.4),
        dict(common, case_id="swapped", variant="applicability_swap", interaction=0.4),
    ]
    cases = [
        dict(id="base", candidates=[[12], [18]], preferred=0),
        dict(id="swapped", candidates=[[12], [18]], preferred=1),
    ]
    assert semantic_differences(rows, cases)[0]["interaction_change"] == 0.0
    rows[1]["interaction"] = 0.9
    assert semantic_differences(rows, cases)[0]["interaction_change"] == pytest.approx(0.5)
    cases[1]["candidates"].reverse()
    with pytest.raises(ValueError, match="fixed candidate"):
        semantic_differences(rows, cases)


def test_original_prompt_offsets_handle_punctuation_and_exclude_special_tokens(audit, tmp_path):
    _, tokenizer, _, _, _ = audit
    quote = "Mira [EOS] wears a blue coat."
    row = dict(
        id="quoted",
        source_id="source",
        example=dict(id="q", source_id="source", prompt=quote),
        candidates=[" blue", " red"],
        source_quotes=dict(evidence=[quote]),
    )
    path = tmp_path / "quoted.json"
    write_json(path, dict(cases=[row]))
    case = read_cases(path, tokenizer)[0]
    assert tokenizer.eos_token_id in case.prefix_ids
    assert case.sources["evidence"]
    assert all(
        case.prefix_ids[key] not in tokenizer.all_special_ids for key in case.sources["evidence"]
    )
