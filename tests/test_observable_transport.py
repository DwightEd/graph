"""Native derivatives, sketch identities, prefix alignment and source exclusion."""

from contextlib import closing
from itertools import product

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
