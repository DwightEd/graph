"""A native target keeps its contrast when prefix numerics change its ranking."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

from experiments.common import llama_message_intervention as intervention
from experiments.common.llama_message_intervention import baseline_forward
from experiments.reanchor_flow.attribution import (
    GradientObserver,
    contrast_direction,
    native_target_gradients,
)
from experiments.reanchor_flow.flow import margin
from experiments.reanchor_flow.native_flow import native_flow_screen
from experiments.reanchor_flow.native_world import (
    NativeWorld,
    gated_forward_cache,
    source_gate,
)
from experiments.reanchor_flow.subset_data import freeze_target_plan
from experiments.reanchor_flow.tests.etcc_helpers import paired_world, tiny_model


def test_prefix_runner_drift_keeps_frozen_margin_gradients_and_cut(monkeypatch):
    model = tiny_model()
    pair = paired_world()
    query = 4  # The middle target produces a shorter readout batch than discovery.
    initial = baseline_forward(model, pair.clean_token_ids, pair.response_start)
    state = initial.final_hidden[query]
    with torch.no_grad():
        logits = model.lm_head(state).float()
        logits[int(pair.clean_token_ids[query + 1])] = -torch.inf
        first, second = logits.topk(2).indices.tolist()
        bias = torch.zeros(model.config.vocab_size)
        bias[second] = logits[first] - logits[second]
        model.lm_head.bias = torch.nn.Parameter(bias)

    # Deterministically reproduce near-tie rounding at the vocabulary readout.
    # Different full/prefix matrix shapes select different competitors, while
    # the model weights, prompt, causal states, and observed token stay fixed.
    linear = intervention.F.linear

    def shape_sensitive_readout(value, weight, bias=None):
        result = linear(value, weight, bias)
        if weight is model.lm_head.weight:
            candidate = first if value.shape[0] == 3 else second
            result[..., candidate] += 1e-5
        return result

    monkeypatch.setattr(intervention.F, "linear", shape_sensitive_readout)
    targets, _ = freeze_target_plan(
        model,
        pair.clean_token_ids,
        pair.response_start,
        count=3,
        policy="evenly-spaced",
        query_chunk=2,
    )
    target = next(item for item in targets if item.query_position == query)
    assert target.negative_token_id == first
    world = NativeWorld(
        pair.sample_id,
        pair.tokenizer_id,
        pair.clean_token_ids,
        pair.response_start,
        pair.units,
        pair.candidate_unit_id,
        targets,
    ).check()
    prefix = world.prefix(target)
    reranked = baseline_forward(
        model,
        prefix.token_ids,
        prefix.response_start,
        checkpoint_layers=range(len(model.model.layers)),
        attention_query_chunk=2,
    )
    slot = int(torch.nonzero(reranked.query == query).item())
    assert int(reranked.runner[slot]) == second
    with pytest.raises(ValueError, match="frozen native runner"):
        native_target_gradients(model, reranked, target, torch.tensor([query]))

    flow, gradients = native_flow_screen(
        model,
        world,
        target,
        "message",
        carrier_scope="response",
        coverage=1.0,
        query_chunk=2,
    )
    clean = flow.clean_cache
    direction, bias = contrast_direction(model, target)
    assert flow.target == target
    assert int(clean.runner[slot]) == first
    torch.testing.assert_close(clean.readout_direction[slot], direction)
    torch.testing.assert_close(clean.readout_bias[slot], bias)
    assert float(clean.full_margin[slot]) == pytest.approx(flow.clean_margin)

    hidden = clean.layer_input[0].clone()[None].detach().requires_grad_(True)
    observer = GradientObserver()
    final = intervention.forward_layers(
        model, hidden, 0, observer=observer, attention_query_chunk=2
    )
    (torch.dot(final[0, query].float(), direction) + bias).backward()
    expected = observer.gradients(gradients.position, clean.layer_count)
    for name in ("head_output", "layer_input", "attention_write", "mlp_write"):
        torch.testing.assert_close(
            getattr(gradients, name), getattr(expected, name), atol=2e-6, rtol=2e-5
        )

    cut = gated_forward_cache(model, clean, source_gate(prefix, (1,)))
    assert int(cut.runner[slot]) == first
    torch.testing.assert_close(cut.readout_direction, clean.readout_direction)
    assert float(cut.full_margin[slot]) == pytest.approx(margin(model, cut, target))


@pytest.mark.parametrize(
    "fixed_runner, error",
    [
        ({2: 10}, "outside the response predictors"),
        ({5: 41}, "outside the model vocabulary"),
        ({5: 7}, "differ from the observed target"),
    ],
)
def test_frozen_runner_contract(fixed_runner, error):
    pair = paired_world()
    with pytest.raises(ValueError, match=error):
        baseline_forward(
            tiny_model(),
            pair.clean_token_ids,
            pair.response_start,
            fixed_runner=fixed_runner,
        )
