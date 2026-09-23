"""Actual tiny Llama integration, not natural hallucination evidence."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from state_audit.attribution import _projection_gram, frozen_parameters
from state_audit.model.adapter import ModelAdapter
from state_audit.native_response import (
    RESPONSE_SITES,
    iter_response_traces,
    measure_responses,
    state_basis,
)
from state_audit.native_trace import native_hooks
from state_audit.storage import read_arrays, read_json, write_json
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from experiments.native_support.dynamics import main, verify_disjoint
from experiments.native_support.dynamics_fit import SwitchingDynamics, initialize
from experiments.native_support.dynamics_observations import (
    build_observations,
    partition_prompt,
    project_source_covariance,
)


@pytest.fixture
def tiny_model():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    return ModelAdapter(LlamaForCausalLM(LlamaConfig(
        vocab_size=41, hidden_size=24, intermediate_size=36, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, attention_dropout=0.,
    )))


def response(identity, offset=0):
    tokens = [1, 2, 3, 4, 5, 6] + [(i + offset) % 34 + 6 for i in range(10)]
    return {"id": identity, "source_id": identity, "token_ids": tokens,
            "prompt_length": 6, "token_text": [str(i) for i in tokens]}


def test_native_batched_derivatives_match_full_prefix_and_single_direction(tiny_model):
    item = response("reference")
    tokens, prompt, target = item["token_ids"], 6, 3
    groups, blocks = partition_prompt(item, None, [0], 3)
    def captured(batch):
        return next(iter_response_traces(tiny_model, tokens, prompt, [target], groups, len(blocks), [0],
                                         rank=3, choices=3, gradient_batch=batch, prefill_chunk_size=3))
    batched, single = captured(3), captured(1)
    np.testing.assert_allclose(batched["group_effect"], single["group_effect"], atol=3e-6, rtol=2e-4)
    np.testing.assert_allclose(batched["ffn_effect"], single["ffn_effect"], atol=3e-6, rtol=2e-4)
    cache = DynamicCache()
    grams = {layer: _projection_gram(tiny_model, layer) for layer in range(3)}
    with torch.enable_grad(), frozen_parameters(tiny_model), native_hooks(tiny_model, representations=RESPONSE_SITES) as records:
        output = tiny_model.native.model(input_ids=tiny_model.input_ids(tokens[:prompt + target]),
                                         past_key_values=cache, use_cache=True, return_dict=True)
        full = measure_responses(tiny_model, records, cache, output.last_hidden_state[0, -1], tokens[prompt + target],
                                 state_basis(24, 3, 37, "cpu"), torch.as_tensor(batched["group_ids"]),
                                 len(blocks) + 4, grams, 3, 3)
    for name in ("state", "group_effect", "ffn_effect", "edge_response_energy"):
        np.testing.assert_allclose(batched[name], full[name], atol=3e-6, rtol=3e-4)
    assert batched["group_effect"].shape == (3, 4, len(blocks) + 4, 6)
    assert all(parameter.grad is None for parameter in tiny_model.native.parameters())


def test_native_bfloat16_direction_batches_remain_finite(tiny_model):
    tiny_model.native.to(dtype=torch.bfloat16)
    item = response("bf16")
    groups, blocks = partition_prompt(item, None, [0], 3)
    measured = next(iter_response_traces(tiny_model, item["token_ids"], 6, [1], groups, len(blocks), [0],
                                        rank=3, choices=3, gradient_batch=3))
    for field in ("group_effect", "ffn_effect", "edge_response_energy", "state"):
        assert np.isfinite(measured[field]).all()
    assert measured["group_effect"].shape[:2] == (3, 4)


def write_cohort(path, identities):
    settings = {"model": "tiny-native-fixture", "responses": [response(name, i * 5) for i, name in enumerate(identities)]}
    write_json(path / "settings.json", settings)
    annotations = {item["id"]: {"source_id": item["source_id"], "token_ids": item["token_ids"][6:],
                                "labels": [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]} for item in settings["responses"]}
    write_json(path / "annotations.json", annotations)
    return settings


def test_capture_train_score_evaluate_and_resume_without_label_fitting(tmp_path, tiny_model):
    reference, target = tmp_path / "reference", tmp_path / "test"
    write_cohort(reference, ["train-a", "train-b", "train-c"])
    write_cohort(target, ["test-a", "test-b"])
    tokenizer = SimpleNamespace(all_special_ids=[0], decode=lambda ids: str(ids[0]))
    options = ["--output", str(target), "--reference-output", str(reference), "--device", "cpu",
               "--dtype", "float32", "--rank", "3", "--choices", "3", "--profile-rank", "4",
               "--block-tokens", "3", "--gradient-batch", "3", "--epochs", "3", "--cpu-threads", "1"]
    real_read = read_json
    def no_train_labels(path):
        if Path(path) == reference / "annotations.json":
            raise AssertionError("training must not read natural annotations")
        return real_read(path)
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
         patch("state_audit.model.load_model", return_value=(tiny_model, tokenizer)), \
         patch("experiments.native_support.dynamics.read_json", side_effect=no_train_labels):
        main(["--stage", "run", *options])
    directory = target / "state_dynamics"
    summary = read_json(directory / "summary.json")
    assert summary["scored_tokens"] == 20
    assert summary["future_tokens_used"] and not summary["labels_used_for_training"]
    assert summary["evaluation"]["status"] == "evaluated"
    assert (directory / "comparisons.csv").is_file() and (directory / "report.html").is_file()
    scores = read_arrays(directory / "responses/0000/scores.npz")
    assert np.isfinite(scores["risk"]).all()
    assert not np.allclose(scores["state_dynamics"], scores["state_forward"])
    observation = read_arrays(directory / "capture/0000/observations.npz")
    np.testing.assert_array_equal(observation["future_query_count"], [8, 7, 6, 5, 4, 3, 2, 1, 0, 0])
    assert observation["profile"].shape[1:3] == (3, 4)
    labels = read_json(target / "annotations.json")
    labels["test-a"]["labels"] = [1, 1, 0, 0, 0, 0, 1, 0, 0, 1]
    write_json(target / "annotations.json", labels)
    with patch("state_audit.model.load_model", side_effect=AssertionError("scoring loaded LLM")):
        main(["--stage", "score", *options])
    changed = read_arrays(directory / "responses/0000/scores.npz")
    np.testing.assert_array_equal(changed["risk"], scores["risk"])
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
         patch("state_audit.model.load_model", side_effect=AssertionError("resume loaded LLM")):
        main(["--stage", "run", *options, "--resume"])


def test_source_overlap_rejected_before_any_fitting():
    settings = {"model": "same", "responses": [response("shared")]}
    with pytest.raises(ValueError, match="sources overlap"):
        verify_disjoint(settings, settings)


def test_source_covariance_retains_cross_head_terms(tmp_path):
    from state_audit.storage import write_arrays

    # The same source's opposite head responses cancel only AFTER joint projection.
    effects = np.zeros((1, 2, 6, 1))
    effects[0, 0, :2, 0] = [1., -1.]
    effects[0, 1, :2, 0] = [-1., 1.]
    write_arrays(tmp_path / "token_000000.npz", group_effect=effects)
    canceled = project_source_covariance(tmp_path, 1, 2, 1, np.array([[1., 1.]]), np.ones(2))
    contrasted = project_source_covariance(tmp_path, 1, 2, 1, np.array([[1., -1.]]), np.ones(2))
    assert canceled[0, 0, 0] == 0
    assert contrasted[0, 0, 0] > 0


def test_offline_reuse_and_historical_state_use_correct_key_positions(tmp_path):
    from state_audit.storage import write_arrays

    item = response("alignment")
    item["token_ids"] = item["token_ids"][:10]
    item["token_text"] = item["token_text"][:10]
    for target in range(4):
        query = 5 + target
        attention = np.zeros((1, 1, query + 1))
        attention[..., 0] = .3
        attention[..., -1] = .2
        if target >= 2:
            attention[..., 6] = .5
        groups = np.zeros(query + 1, dtype=int)
        groups[6:] = 1
        groups[-1] = 2
        group_mass = attention @ np.eye(5)[groups]
        write_arrays(tmp_path / f"token_{target:06d}.npz", target=target, query=query,
                     token_id=item["token_ids"][6 + target], attention=attention, group_ids=groups,
                     group_attention=group_mass, group_effect=np.ones((1, 1, 5, 3)),
                     edge_value_energy=attention**2, edge_response_energy=attention**2,
                     state=np.array([10. + target]), ffn_effect=np.ones((1, 3)),
                     entropy=1., surprisal=1., candidate_tail_mass=.2)
    observed = build_observations(item, tmp_path, 1, 1, np.ones(6, dtype=bool), [0])
    # key P is represented at query P -> state row 1, never row 0 or current row 2.
    assert observed["history"][2, 0] == pytest.approx(.5 * 11)
    names = list(observed["profile_channels"])
    assert observed["profile"][0, 0, 0, names.index("future_mean_attention")] == .5
    assert observed["future_query_count"][0] == 2
    assert observed["profile"][3, 0, 0, names.index("source_read")] == pytest.approx(.3)
    assert observed["profile"][3, 0, 0, names.index("source_log_distance")] > 0


def test_input_uncertainty_enters_joint_native_state_likelihood():
    rng = np.random.default_rng(8)
    sequence = {"state": rng.normal(size=(20, 2)), "history": rng.normal(size=(20, 4)),
                "input": rng.normal(size=(20, 2)), "profile": rng.normal(size=(20, 3)),
                "covariance": np.zeros((20, 2, 2))}
    parameters = initialize([sequence], [np.linspace(0, 1, 20)])
    parameters["injection"] = np.eye(2)
    model = SwitchingDynamics(parameters)
    data = {name: torch.tensor(value, dtype=torch.float32) for name, value in sequence.items()}
    baseline = model.emissions(data).detach().numpy()
    data["covariance"] = torch.eye(2).expand(20, -1, -1) * 10
    uncertain = model.emissions(data).detach().numpy()
    assert not np.allclose(baseline[:, 0], uncertain[:, 0])
    np.testing.assert_allclose(baseline[:, 1], uncertain[:, 1])
