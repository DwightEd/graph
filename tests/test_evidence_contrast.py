"""Native prediction alignment, branch isolation, complete coverage, and label isolation."""

from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from transformers import (LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM,
                          Qwen2Config, Qwen2ForCausalLM)

from state_audit.model.adapter import ModelAdapter
from state_audit.model.replay import attention_backend
from state_audit.storage import read_arrays, read_json, write_arrays, write_json
from experiments.native_support.choice_cache import CaptureReader
from experiments.native_support.dual_state.data import head_routes
from experiments.native_support.observable_state import state_features
from experiments.native_support.evidence_contrast.capture import collect_condition
from experiments.native_support.evidence_contrast.data import prepare
from experiments.native_support.evidence_contrast.run import arguments, main
from experiments.native_support.evidence_contrast.scoring import BASELINES, METHODS, score_contrasts
from experiments.native_support.evidence_contrast.views import prepare_views, query_positions, unit_intervals


def tiny_model(family="llama"):
    torch.manual_seed(37)
    torch.set_num_threads(1)
    configs = {"llama": (LlamaConfig, LlamaForCausalLM),
               "mistral": (MistralConfig, MistralForCausalLM),
               "qwen2": (Qwen2Config, Qwen2ForCausalLM)}
    config_type, model_type = configs[family]
    config = config_type(vocab_size=31, hidden_size=16, intermediate_size=24,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        max_position_embeddings=128, sliding_window=4)
    return ModelAdapter(model_type(config))


def independent_logprob(model, prefix, answer):
    with torch.no_grad(), attention_backend(model, "sdpa"):
        hidden = model.forward(prefix + answer[:-1])
        logits = model.native.lm_head(hidden[len(prefix)-1:]).float().log_softmax(-1)
        return logits.gather(-1, model.input_ids(answer)[0, :, None])[:, 0].numpy()


@pytest.mark.parametrize("family", ["llama", "mistral", "qwen2"])
def test_cached_conditions_equal_independent_native_forwards(family):
    model = tiny_model(family)
    prefix, answer = [1, 3, 5, 7, 9, 11], [13, 15, 17, 19, 21, 23, 25]
    units = [dict(start=0, stop=2), dict(start=2, stop=5), dict(start=5, stop=7)]
    values = collect_condition(model, prefix, answer, units, 2, 2, "test")
    np.testing.assert_allclose(values["full"], independent_logprob(model, prefix, answer), atol=1e-6)
    for unit in units:
        start, stop = unit["start"], unit["stop"]
        np.testing.assert_allclose(values["local"][start:stop],
                                   independent_logprob(model, prefix, answer[start:stop]), atol=1e-6)
    reversed_order = collect_condition(model, prefix, answer, units[::-1], 3, 3, "reverse")
    np.testing.assert_allclose(values["local"], reversed_order["local"], atol=1e-6)
    np.testing.assert_allclose(values["full"][:2], values["local"][:2], atol=1e-6)
    assert model.native.config._attn_implementation == "eager"
    assert all(parameter.grad is None for parameter in model.native.parameters())


def test_future_answer_changes_do_not_change_fixed_prefix_likelihood():
    model = tiny_model()
    prefix, answer = [1, 3, 5], [7, 9, 11, 13, 15]
    units = [dict(start=0, stop=3), dict(start=3, stop=5)]
    original = collect_condition(model, prefix, answer, units, 2, 2, "original")
    changed = collect_condition(model, prefix, answer[:3] + [17, 19], units, 2, 2, "changed")
    for name in ("full", "local"):
        np.testing.assert_array_equal(original[name][:3], changed[name][:3])


def test_units_keep_decimals_list_markers_titles_and_all_tokens():
    pieces = ["Title", "\n", "1", ".", " Value", " 3", ".", "14", ".", "\n", "Tail"]
    response = dict(prompt_length=1, token_text=["prompt", *pieces])
    units = unit_intervals(response, 5)
    covered = [target for unit in units for target in range(unit["start"], unit["stop"])]
    assert covered == list(range(len(pieces)))
    assert all(unit["stop"]-unit["start"] <= 5 for unit in units)
    assert {(u["text_start"], u["text_stop"]) for u in units} == {(0, 2), (2, 9), (9, 11)}


def test_source_deletion_and_queries_preserve_original_answer_ids():
    response = dict(id="a", prompt_length=5, token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
                    token_text=["<", "evidence", "Q", "evidence2", ">", "A", ". ", "B"])
    sources = dict(blocks=[[1], [3]], group_ids=[4, 0, 3, 1, 4, 2, 2, 2])
    views = prepare_views(response, sources, 64)
    assert views["prompt_without_source"] == [1, 3, 5]
    assert views["answer_ids"] == [6, 7, 8]
    assert views["removed_prompt_positions"] == [1, 3]
    positions = query_positions(views)
    np.testing.assert_array_equal(positions["full_query_with_source"], [4, 5, 6])
    np.testing.assert_array_equal(positions["local_query_without_source"], [2, 3, 2])


def test_empty_source_effect_is_zero_and_span_scores_use_every_target():
    model = tiny_model()
    response = dict(id="a", prompt_length=2, token_ids=[1, 2, 3, 4, 5, 6],
                    token_text=["P", ":", "A", ". ", "B", "."])
    sources = dict(blocks=[], group_ids=[1, 2, 0, 0, 0, 0])
    views = prepare_views(response, sources, 64)
    args = (model, views["prompt_with_source"], views["answer_ids"], views["units"], 2, 2, "empty")
    observation = collect_condition(*args)
    baseline = {name: np.arange(4, dtype=float) for name in BASELINES}
    scores = score_contrasts(observation, observation, views, baseline, 16)
    for name in ("source_full", "source_local", "source_pair", "source_pair_span"):
        np.testing.assert_array_equal(scores[name], 0)
    np.testing.assert_array_equal(scores["risk"], baseline["raw_route"])
    other = {**observation, "full": observation["full"]-np.array([1., 3., 2., 4.]),
             "local": observation["local"]-np.array([3., 5., 4., 6.])}
    scores = score_contrasts(observation, other, views, baseline, 16)
    np.testing.assert_array_equal(scores["source_pair"], [-2., -4., -3., -5.])
    np.testing.assert_array_equal(scores["source_pair_span"], [-3., -3., -4., -4.])


def test_full_head_energy_recovers_original_observable_baseline():
    generator = np.random.default_rng(37)
    direct = generator.normal(size=(4, 3, 6, 4))
    ffn = generator.normal(size=direct.shape)
    row = dict(response_residual=direct, response_ffn=ffn, response_total=direct+ffn,
               read_mass=np.full((4, 3, 6), 1/6))
    route, energy, _ = head_routes(iter([row]), 2, "all")
    route, energy = route.reshape(4, 3), energy.reshape(4, 3)
    rebuilt = ((route*energy).sum(1)/energy.sum(1)).mean()
    assert rebuilt == pytest.approx(state_features(row, 2)["observable_route"])


def make_input(path):
    pieces = {1: "<", 2: "evidence", 3: " detail", 4: " Q", 5: ">",
              6: "Alpha", 7: ". ", 8: "Next", 9: " text", 10: "."}
    responses, annotations = [], {}
    for index in range(2):
        ids = list(range(1, 11))
        response = dict(id=str(index), source_id=f"source{index}", prompt_length=5,
                        token_ids=ids, token_text=[pieces[token] for token in ids])
        responses.append(response)
        directory = path / "responses" / f"{index:04d}"
        write_json(directory / "sources.json", dict(blocks=[[1, 2]], group_ids=[3, 0, 0, 2, 3, 1, 1, 1, 1, 1]))
        write_arrays(directory / "scores.npz", target=np.arange(5), token_id=np.arange(6, 11),
                     **{name: np.linspace(index/10, .8, 5) for name in BASELINES})
        annotations[str(index)] = dict(source_id=response["source_id"], token_ids=ids[5:],
                                      labels=[0, 0, index, index, 0])
    write_json(path / "settings.json", dict(model="tiny", responses=responses))
    write_json(path / "annotations.json", annotations)
    return pieces, annotations


def test_run_resume_cpu_rescore_annotation_isolation_and_full_pack(tmp_path):
    source, output = tmp_path / "input", tmp_path / "contrast"
    pieces, annotations = make_input(source)
    class Tokenizer:
        def decode(self, ids):
            return pieces[ids[0]]
    args = ["--input", str(source), "--output", str(output), "--device", "cpu", "--dtype", "float32",
            "--prefill-chunk-size", "2", "--query-chunk-size", "2", "--bootstrap", "5"]
    original_read = CaptureReader.json
    def read_observations(reader, name):
        assert name != "annotations.json", "Capture accessed truth annotations"
        return original_read(reader, name)
    with patch("state_audit.model.load_model", return_value=(tiny_model(), Tokenizer())), \
            patch.object(CaptureReader, "json", read_observations):
        main([*args, "--stage", "capture"])
    assert not (output / "annotations.json").exists()
    main(["--output", str(output), "--stage", "score", "--bootstrap", "5"])
    scores = read_arrays(output / "responses/0001/scores.npz")
    assert read_json(output / "coverage.json")["scored_tokens"] == 10
    assert len(read_json(output / "evaluation.json")["methods"]) == len(METHODS)
    assert (output / "summary.png").is_file()
    with ZipFile(output.with_name("contrast_review.zip")) as archive:
        assert {"source_bootstrap.json", "detection_budget.json", "responses/0001/with_source.npz",
                "responses/0001/views.json", "responses/0001/scores.npz"} <= set(archive.namelist())
    for annotation in annotations.values():
        annotation["labels"] = [1-value for value in annotation["labels"]]
    write_json(source / "annotations.json", annotations)
    with patch("state_audit.model.load_model", side_effect=AssertionError("Resume reloaded the model")):
        main([*args, "--resume"])
    repeated = read_arrays(output / "responses/0001/scores.npz")
    for name in scores:
        np.testing.assert_array_equal(scores[name], repeated[name])
    with pytest.raises(ValueError, match="settings changed"):
        main([*args, "--resume", "--window", "8"])


def test_prepare_accepts_light_zip_and_rejects_misaligned_baseline(tmp_path):
    source = tmp_path / "input"
    make_input(source)
    archive = tmp_path / "input.zip"
    with ZipFile(archive, "w") as bundle:
        for file in source.rglob("*"):
            if file.is_file():
                bundle.write(file, file.relative_to(source))
    args = arguments(["--input", str(archive), "--output", str(tmp_path / "result")])
    settings, _ = prepare(args)
    assert len(settings["responses"]) == 2
    file = source / "responses/0000/scores.npz"
    values = read_arrays(file)
    values["token_id"][0] += 1
    write_arrays(file, **values)
    with pytest.raises(ValueError, match="baseline token alignment"):
        prepare(arguments(["--input", str(source), "--output", str(tmp_path / "invalid")]))


def test_interrupted_capture_resumes_only_unfinished_condition(tmp_path):
    source, output = tmp_path / "input", tmp_path / "contrast"
    pieces, _ = make_input(source)
    class Tokenizer:
        def decode(self, ids):
            return pieces[ids[0]]
    args = ["--input", str(source), "--output", str(output), "--stage", "capture", "--device", "cpu"]
    def interrupt(*args):
        if args[-1] == "0 without_source":
            raise RuntimeError("interrupted")
        return collect_condition(*args)
    with patch("state_audit.model.load_model", return_value=(tiny_model(), Tokenizer())), \
            patch("experiments.native_support.evidence_contrast.capture.collect_condition", side_effect=interrupt):
        with pytest.raises(RuntimeError, match="interrupted"):
            main(args)
    completed = output / "responses/0000/with_source.npz"
    original_bytes = completed.read_bytes()
    with patch("state_audit.model.load_model", return_value=(tiny_model(), Tokenizer())), \
            patch("experiments.native_support.evidence_contrast.capture.collect_condition", wraps=collect_condition) as measured:
        main([*args, "--resume"])
    assert completed.read_bytes() == original_bytes
    assert [call.args[-1] for call in measured.call_args_list] == [
        "0 without_source", "1 with_source", "1 without_source"]
