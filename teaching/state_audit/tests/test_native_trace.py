"""Observed-run accounting and local derivatives, not natural-data mechanism claims."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from test_attribution_memory import causal_model as _causal_model

from state_audit.model.replay import attention_backend, prefill_cache
from state_audit.native_ledger import ledger_total
from state_audit.native_trace import iter_native_traces

causal_model = _causal_model


def query_plan(query=12, dependencies=True):
    groups = [0, 0, 1, 1] + [2] * (query - 3)
    return dict(
        query=query,
        candidate_ids=[5, 7],
        observed_id=5,
        group_ids=groups,
        group_names=["applicable", "other", "history"],
        positive_groups=[0],
        negative_groups=[1],
        dependencies=dependencies,
        receiver_budget=2,
    )


def test_native_trace_matches_logits_and_accounts_for_every_source(causal_model):
    tokens = list(range(1, 19))
    with torch.no_grad():
        hidden = causal_model.forward(tokens[:13])[-1]
        logits = causal_model.native.lm_head(hidden).float()
    arrays = next(iter_native_traces(causal_model, tokens, [query_plan()], prefill_chunk_size=4))
    np.testing.assert_allclose(arrays["candidate_logits"], logits[[5, 7]], atol=1e-6)
    assert abs(float(arrays["ledger_error"])) < 1e-6
    assert abs(ledger_total(arrays) - float(arrays["logit_gap"])) < 1e-6
    np.testing.assert_allclose(arrays["group_route_mass"].sum(-1), 1, atol=1e-6)
    np.testing.assert_allclose(
        arrays["group_logit_write"].sum(-1), arrays["edge_logit_write"].sum(-1), atol=1e-6
    )
    assert arrays["head_writes"].shape == (3, 4, 32)
    assert arrays["residual_scores"].shape == (3, 3)
    for index, layer in enumerate(arrays["receiver_layer"]):
        observed = arrays["dependency_head_observed"][index]
        current = arrays["receiver_kind"][index] == "mlp_logit_write"
        assert not observed[layer + int(current) :].any()
    for module in causal_model.native.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks
    assert all(parameter.requires_grad for parameter in causal_model.native.parameters())


def test_bfloat16_roundoff_is_explicit_and_ledger_still_closes(causal_model):
    causal_model.native.to(torch.bfloat16)
    arrays = next(iter_native_traces(causal_model, list(range(1, 19)), [query_plan()]))
    assert abs(float(arrays["ledger_error"])) < 2e-6
    assert np.isfinite(arrays["attention_reconstruction_error"]).all()
    assert np.isfinite(arrays["edge_margin_sensitivity"]).all()
    assert "norm_roundoff" in arrays and "unembedding_roundoff" in arrays


def test_cached_queries_do_not_read_future_or_recompute_past(causal_model):
    tokens = list(range(1, 25))
    plans = [query_plan(6, False), query_plan(9, False), query_plan(12, False)]
    calls = []
    original = causal_model.native.model.forward

    def observe(*args, **kwargs):
        calls.append(kwargs["input_ids"].shape[-1])
        return original(*args, **kwargs)

    with patch.object(causal_model.native.model, "forward", side_effect=observe):
        outputs = list(iter_native_traces(causal_model, tokens, plans, prefill_chunk_size=3))
    assert sum(calls) == 13
    altered = tokens[:10] + [27] * 14
    other = list(iter_native_traces(causal_model, altered, plans[:2]))
    for left, right in zip(outputs, other):
        np.testing.assert_allclose(left["candidate_logits"], right["candidate_logits"], atol=1e-6)
    assert outputs[0]["attention"].shape[-1] == 7


def test_mlp_dependency_matches_small_native_directional_difference(causal_model):
    tokens = list(range(1, 19))
    arrays = next(iter_native_traces(causal_model, tokens, [query_plan()]))
    receiver = np.flatnonzero(
        (arrays["receiver_kind"] == "mlp_logit_write") & (arrays["receiver_layer"] == 1)
    )[0]
    sender = arrays["group_readouts"][0, 0, 0]
    expected = arrays["dependency_head_effect"][receiver, 0, 0, 0]
    direction = torch.tensor(arrays["direction"])

    def measure(dose):
        cache = prefill_cache(causal_model, tokens[:12], 4)
        measured = []

        def move(value):
            changed = value.clone()
            changed[-1, 0] += dose * torch.tensor(sender)
            return changed

        def read(value):
            measured.append(float(value[-1].float() @ direction))
            return value

        with torch.no_grad(), attention_backend(causal_model, "eager"):
            with (
                causal_model.bind("head_readout", 0, move),
                causal_model.bind("mlp_write", 1, read),
            ):
                causal_model.native.model(
                    input_ids=causal_model.input_ids([tokens[12]]),
                    past_key_values=cache,
                    use_cache=True,
                )
        return measured[0]

    numeric = (measure(0.02) - measure(-0.02)) / 0.04
    np.testing.assert_allclose(expected, numeric, atol=3e-6, rtol=0.03)


def test_failure_removes_native_hooks_and_restores_flags(causal_model):
    with patch.object(causal_model.native.lm_head, "forward", side_effect=RuntimeError("readout")):
        with pytest.raises(RuntimeError, match="readout"):
            next(iter_native_traces(causal_model, list(range(1, 19)), [query_plan()]))
    for module in causal_model.native.modules():
        assert not module._forward_hooks and not module._forward_pre_hooks
    assert all(parameter.requires_grad for parameter in causal_model.native.parameters())


def test_candidate_order_changes_readout_only(causal_model):
    tokens = list(range(1, 19))
    plan = query_plan(dependencies=False)
    left = next(iter_native_traces(causal_model, tokens, [plan]))
    reverse = dict(plan, candidate_ids=plan["candidate_ids"][::-1])
    right = next(iter_native_traces(causal_model, tokens, [reverse]))
    for name in ("attention", "residual_after", "mlp_write", "head_writes"):
        np.testing.assert_array_equal(left[name], right[name])
    for name in ("logit_gap", "edge_logit_write", "edge_margin_sensitivity", "mlp_score"):
        np.testing.assert_allclose(left[name], -right[name], atol=1e-6)


def test_route_dependency_matches_native_directional_difference(causal_model):
    tokens = list(range(1, 19))
    plan = query_plan()
    plan["group_ids"] = [2] * 9 + [0, 0, 1, 1]
    arrays = next(iter_native_traces(causal_model, tokens, [plan]))
    receiver = np.flatnonzero(arrays["receiver_kind"] == "route_mass_difference")[0]
    layer = int(arrays["receiver_layer"][receiver])
    head = int(arrays["receiver_head"][receiver])
    direction = arrays["group_readouts"][0, 0, 0]
    expected = arrays["dependency_head_effect"][receiver, 0, 0, 0]

    def measure(dose):
        cache = prefill_cache(causal_model, tokens[:12], 4)
        measured = []

        def move(value):
            changed = value.clone()
            changed[-1, 0] += dose * torch.tensor(direction)
            return changed

        def read(value):
            row = value[head, -1].float()
            measured.append(float(row[9:11].sum() - row[11:13].sum()))
            return value

        with torch.no_grad(), attention_backend(causal_model, "eager"):
            with causal_model.bind("head_readout", 0, move):
                with causal_model.bind("attention", layer, read):
                    causal_model.native.model(
                        input_ids=causal_model.input_ids([tokens[12]]),
                        past_key_values=cache,
                        use_cache=True,
                    )
        return measured[0]

    numeric = (measure(0.02) - measure(-0.02)) / 0.04
    np.testing.assert_allclose(expected, numeric, atol=3e-6, rtol=0.03)
