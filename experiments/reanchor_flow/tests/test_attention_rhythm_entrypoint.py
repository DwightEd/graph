"""Regression coverage for merging the delivered CLI with the canonical runner."""
import argparse
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from experiments.reanchor_flow.audit_attention_rhythm import (
    main, normalize_scan_directory, translate_arguments,
)


@pytest.mark.parametrize("spelling", [["--split", "both"], ["--split=both"]])
def test_both_split_alias(spelling):
    translated = translate_arguments(spelling)
    assert translated[:2] == ["--split", "all"]
    assert translated.count("--split") == 1


def test_original_pilot_command_keeps_sample_and_budget():
    command = ["--scans", "/data/scans", "--split", "test", "--task", "QA",
               "--sample-id", "12693", "--query-chunk", "8", "--plots-per-task", "1",
               "--output", "/data/audit"]
    result = translate_arguments(command)
    assert result == command + ["--samples-per-task", "0"]


def test_full_cohort_defaults_are_not_the_canonical_single_sample_default():
    assert translate_arguments([]) == ["--split", "all", "--task", "all",
                                        "--samples-per-task", "0"]
    result = translate_arguments(["--samples-per-task=2", "--task=QA", "--split=train"])
    assert "--samples-per-task" not in result
    assert result.count("--task") == result.count("--split") == 1


def test_heads_labels_and_multiple_sample_ids():
    result = translate_arguments(["--heads", "10:4", "20:7", "--labels",
                                  "--sample-id", "12693", "12001"])
    assert result[:9] == ["--head", "10:4", "--head", "20:7", "--evaluate",
                          "--sample-id", "12693", "--sample-id", "12001"]


def test_explicit_all_tasks_translate_to_one_canonical_run():
    result = translate_arguments(["--task", "QA", "Summary", "Data2txt"])
    assert result[:2] == ["--task", "all"]


@pytest.mark.parametrize("command", [["--heads"], ["--sample-id"],
                                     ["--task", "QA", "Summary"]])
def test_ambiguous_or_incomplete_commands_fail_before_loading_model(command):
    with pytest.raises(ValueError):
        translate_arguments(command)


def test_existing_canonical_flags_pass_through():
    command = ["--head", "1:2", "--future-lo", "1", "--future-hi", "64", "--paper-groups"]
    assert translate_arguments(command)[:len(command)] == command


def test_split_scan_directory_is_resolved_without_loading_attention(tmp_path):
    folder = tmp_path / "test"
    folder.mkdir()
    (folder / "run_manifest.json").write_text(json.dumps({"config": {"split": "test"}}))
    args = argparse.Namespace(scans=folder, split="all")
    normalize_scan_directory(args, argparse.ArgumentParser())
    assert args.scans == tmp_path and args.split == "test"


def test_mismatched_split_is_rejected(tmp_path):
    folder = tmp_path / "test"
    folder.mkdir()
    (folder / "run_manifest.json").write_text(json.dumps({"config": {"split": "test"}}))
    with pytest.raises(SystemExit):
        normalize_scan_directory(argparse.Namespace(scans=folder, split="train"),
                                 argparse.ArgumentParser())


def test_parent_scan_directory_is_unchanged(tmp_path):
    args = argparse.Namespace(scans=tmp_path, split="all")
    normalize_scan_directory(args, argparse.ArgumentParser())
    assert args.scans == tmp_path and args.split == "all"


def test_entry_dispatches_once_to_the_existing_runner(monkeypatch):
    # Test the adapter only; this deliberately does not pretend to run an LLM.
    module = ModuleType("experiments.reanchor_flow.attention_rhythm_run")
    def parser():
        p = argparse.ArgumentParser()
        p.add_argument("--split", choices=("train", "test", "all"), default="test")
        p.add_argument("--task", choices=("QA", "Summary", "Data2txt", "all"), default="QA")
        p.add_argument("--samples-per-task", type=int, default=1)
        p.add_argument("--scans", type=Path)
        p.add_argument("--head", action="append", default=[])
        p.add_argument("--evaluate", action="store_true")
        return p
    calls = []
    module.parser = parser
    module.run = lambda args: calls.append(args) or "ran existing runner"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert main(["--split", "both", "--heads", "1:2", "--labels"]) == "ran existing runner"
    assert len(calls) == 1
    assert (calls[0].split, calls[0].task, calls[0].samples_per_task) == ("all", "all", 0)
    assert calls[0].head == ["1:2"] and calls[0].evaluate
