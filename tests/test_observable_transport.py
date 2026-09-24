"""Native derivatives, sketch identities, prefix alignment and source exclusion."""

from contextlib import closing
from itertools import product
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from state_audit.functional_hooks import functional_hooks
from state_audit.model.adapter import ModelAdapter
from state_audit.model.replay import attention_backend
from state_audit.observable_transport import (
    fisher_seeds, iter_observable_transport, message_groups, output_probes,
)
from state_audit.storage import read_arrays, read_json, write_json
from experiments.native_support.observable_run import join_reference, main
from experiments.native_support.observable_state import matrix_distance, state_features


@pytest.fixture
def model():
    torch.manual_seed(37)
    torch.set_num_threads(1)
    config = LlamaConfig(vocab_size=19, hidden_size=12, intermediate_size=20,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)
    return ModelAdapter(LlamaForCausalLM(config))


def test_fisher_sketch_exact_covariance_and_logit_shift_invariance():
    probability = torch.tensor([.2, .3, .5], dtype=torch.float64)
    signs = torch.tensor(list(product((-1., 1.), repeat=3)), dtype=torch.float64) / np.sqrt(8)
    seeds = fisher_seeds(probability, signs)
    expected = torch.diag(probability) - probability[:, None] * probability[None]
    torch.testing.assert_close(seeds.T @ seeds, expected)
    torch.testing.assert_close(seeds.sum(-1), torch.zeros(8, dtype=torch.float64))


def test_native_message_response_matches_finite_difference(model):
    ids, prompt, groups = [1, 3, 5, 7, 9, 11], 4, [0, 0, 1, 2, 2, 2]
    row = next(iter_observable_transport(model, ids, prompt, [1], groups, 2, rank=4))
    query, layer, head, source = 4, 0, 1, 0
    grouped = torch.tensor(groups[:query + 1])
    grouped[query] = 5
    with torch.no_grad(), attention_backend(model, "sdpa"), functional_hooks(model) as records:
        hidden = model.native.model(input_ids=model.input_ids(ids[:query + 1])).last_hidden_state
        rotary = model.native.model.rotary_emb(hidden, torch.arange(query + 1)[None])
        messages, *_ = message_groups(model, layer, records[layer], rotary, query, grouped, 6)
        message = messages[head, source].clone()
        logits = model.native.lm_head(hidden[0, query]).float()
        seeds = fisher_seeds(logits.softmax(-1), output_probes(19, 4, 37, logits.device))

    def changed_logits(amount):
        def add_message(values):
            result = values.clone()
            result[query, head] += amount * message
            return result
        with torch.no_grad(), attention_backend(model, "sdpa"), model.bind("head_readout", layer, add_message):
            hidden = model.forward(ids[:query + 1])
            return model.native.lm_head(hidden[query]).float()

    derivative = (changed_logits(.005) - changed_logits(-.005)) / .01
    expected = (seeds @ derivative).numpy()
    np.testing.assert_allclose(row["response_total"][layer, head, source], expected, atol=2e-5, rtol=2e-3)
    np.testing.assert_allclose(row["response_total"], row["response_residual"] + row["response_ffn"], atol=1e-7)
    assert row["head_reconstruction_error"].max() < 1e-7


def test_independent_queries_are_prefix_invariant_and_restore_hooks(model):
    ids, groups = [1, 3, 5, 7, 9, 11, 13], [0, 0, 1, 1, 2, 2, 2]
    batch = list(iter_observable_transport(model, ids, 4, [0, 1, 2], groups, 2, rank=2))
    for target, row in enumerate(batch):
        alone = next(iter_observable_transport(model, ids, 4, [target], groups, 2, rank=2))
        np.testing.assert_allclose(row["response_total"], alone["response_total"], atol=2e-7, rtol=1e-5)
        assert int(row["query"]) == 3 + target
        assert int(row["token_id"]) == ids[4 + target]
    assert all(parameter.grad is None for parameter in model.native.parameters())
    assert all(parameter.requires_grad for parameter in model.native.parameters())
    with closing(iter_observable_transport(model, ids, 4, [0, 1], groups, 2, rank=2)) as stream:
        next(stream)
    assert not model.layers[0].post_attention_layernorm._forward_pre_hooks


def test_gram_metric_matches_dense_and_retains_head_identity():
    torch.manual_seed(2)
    left, right = torch.randn(2, 12, 4), torch.randn(2, 12, 4)
    expected = ((left @ left.transpose(-1, -2) - right @ right.transpose(-1, -2)) ** 2).sum((-1, -2)).mean() / 2
    torch.testing.assert_close(matrix_distance(left, right), expected)
    rotation, _ = torch.linalg.qr(torch.randn(4, 4))
    assert matrix_distance(left, left @ rotation) < 1e-4
    assert matrix_distance(left, left.flip(-2)) > 1


def test_joint_channels_keep_cancellation_instead_of_opposition_labels():
    direct = np.ones((2, 2, 6, 4), dtype=np.float32)
    row = dict(response_residual=direct, response_ffn=-direct,
               response_total=np.zeros_like(direct), read_mass=np.ones((2, 2, 6), dtype=np.float32) / 6)
    features = state_features(row, 2)
    gram = features["matrix"] @ features["matrix"].transpose(0, 2, 1)
    assert np.any(gram < 0)
    assert features["observable_route"] == 0
    assert np.isfinite(features["matrix"]).all()


def test_source_exclusion_includes_other_answers_from_same_source():
    def item(source, answer):
        return dict(source=np.repeat(source, 2), response=np.repeat(answer, 2),
                    target=np.arange(2), previous=np.array([-1, 0]))
    rows = [item("a", "a1"), item("a", "a2"), item("b", "b1"), item("c", "c1")]
    joined = join_reference(rows, "a")
    assert joined["source"].tolist() == ["b", "b", "c", "c"]
    assert joined["previous"].tolist() == [-1, 0, -1, 2]


def test_one_command_capture_score_evaluate_and_pack(model, tmp_path):
    source, destination = tmp_path / "input", tmp_path / "observable"
    responses, labels = [], {}
    for index in range(3):
        identity = str(index)
        ids = [1, 3 + index, 5, 7, 9 + index, 13, 15]
        response = dict(id=identity, source_id="source" + identity, token_ids=ids,
                        token_text=[str(token) for token in ids], prompt_length=4)
        responses.append(response)
        labels[identity] = dict(token_ids=ids[4:], labels=[0, 1, 1])
        write_json(source / "value_transport" / "capture" / f"{index:04d}" / "sources.json",
                   dict(blocks=[{}, {}], group_ids=[0, 0, 1, 1, 2, 2, 2]))
    settings = dict(model="tiny", responses=responses)
    write_json(source / "settings.json", settings)
    write_json(source / "value_transport" / "capture_settings.json", {**settings, "source_scope": "available"})
    write_json(source / "annotations.json", labels)
    arguments = ["--input", str(source), "--output", str(destination), "--device", "cpu",
                 "--dtype", "float32", "--rank", "2", "--chunk-tokens", "2", "--neighbors", "2"]
    with patch("state_audit.model.load_model", return_value=(model, None)), \
            patch("experiments.native_support.observable_run.validate_tokenizer"):
        main(arguments)
    summary = read_json(destination / "summary.json")
    assert summary["scored_tokens"] == 9
    assert summary["evaluation"]["status"] == "evaluated"
    assert len(summary["evaluation"]["methods"]) == 7
    scores = read_arrays(destination / "responses" / "0000" / "scores.npz")
    np.testing.assert_array_equal(scores["risk"], scores["raw_route"])
    neighbors = read_json(destination / "responses" / "0000" / "neighbors.json")
    assert all("0" not in row["reference_response"] for row in neighbors)
    with ZipFile(destination.with_name("observable_review.zip")) as archive:
        assert "evaluation.json" in archive.namelist()
        assert "responses/0000/token_000000.npz" in archive.namelist()
    # Changing truth labels cannot change already captured/scored observations.
    for annotation in labels.values():
        annotation["labels"] = [1, 0, 0]
    write_json(source / "annotations.json", labels)
    main([*arguments, "--stage", "score", "--resume"])
    repeated = read_arrays(destination / "responses" / "0000" / "scores.npz")
    np.testing.assert_array_equal(scores["transport_conditional"], repeated["transport_conditional"])
