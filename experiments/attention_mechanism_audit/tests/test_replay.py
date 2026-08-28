import math
from dataclasses import replace
import itertools

import numpy as np
import pytest

from experiments.attention_mechanism_audit.replay import (
    FrozenCausalReplay,
    ReplayResult,
    VARIANT_NAMES,
    VariantScores,
    TOKEN_DIAGONAL_PROBE_SCHEME,
    _chunk_scores,
    _chunked_jsd,
    apply_token_probe_to_vjp,
    build_variant_allowed_attention,
    predictor_indices,
    q_to_kv_mapping,
    rademacher_token_probes,
    replace_evidence_tokens,
    score_dense_logits,
    teacher_forced_alignment,
)


def test_teacher_forcing_uses_predecessor_logits_for_cached_targets():
    token_ids = np.array([101, 102, 201, 202, 203], dtype=np.int64)

    predictors, targets = teacher_forced_alignment(token_ids, prompt_length=2)

    assert predictors.tolist() == [1, 2, 3]
    assert targets.tolist() == [201, 202, 203]
    assert predictor_indices(2, 3, sequence_length=5).tolist() == [1, 2, 3]
    assert predictors[-1] != len(token_ids) - 1


def test_dense_scoring_uses_factual_target_not_argmax():
    logits = np.array(
        [
            [5.0, 1.0, 0.0],
            [0.0, 8.0, 2.0],
        ]
    )
    factual_targets = np.array([2, 0], dtype=np.int64)

    chosen_logprob, margin, jsd = score_dense_logits(logits, factual_targets)

    expected_first = 0.0 - math.log(math.exp(5.0) + math.exp(1.0) + 1.0)
    assert chosen_logprob[0] == pytest.approx(expected_first)
    assert margin.tolist() == pytest.approx([-5.0, -8.0])
    assert np.argmax(logits, axis=-1).tolist() == [0, 1]
    assert not np.array_equal(np.argmax(logits, axis=-1), factual_targets)
    assert np.array_equal(jsd, np.zeros(2, dtype=np.float32))


def test_jsd_is_zero_for_full_and_positive_for_changed_distribution():
    full = np.array([[3.0, 0.0, -2.0], [0.0, 1.0, 2.0]])
    changed = np.array([[-2.0, 0.0, 3.0], [0.0, 1.0, 2.0]])
    targets = np.array([0, 2])

    _, _, same = score_dense_logits(full, targets, reference_logits=full)
    _, _, forward = score_dense_logits(changed, targets, reference_logits=full)
    _, _, reverse = score_dense_logits(full, targets, reference_logits=changed)

    assert np.allclose(same, 0.0, atol=1e-7)
    assert forward[0] > 0.0
    assert forward[1] == pytest.approx(0.0, abs=1e-7)
    assert np.allclose(forward, reverse, atol=1e-7)


def test_variant_masks_are_causal_and_remove_only_registered_routes():
    length, prompt_length = 7, 4
    evidence = np.array([1, 2])
    masks = {
        name: build_variant_allowed_attention(
            length,
            prompt_length,
            evidence,
            name,
        )
        for name in VARIANT_NAMES
    }

    for mask in masks.values():
        assert mask.dtype == bool
        assert np.array_equal(mask, np.tril(mask))
        assert np.diag(mask).all()
        assert mask.any(axis=1).all()

    for name in ("swapped_evidence_0", "swapped_evidence_1", "swapped_evidence_2"):
        assert np.array_equal(masks["full"], masks[name])
    assert masks["no_evidence"][1, 1]
    assert not masks["no_evidence"][3, 1]
    assert not masks["no_evidence"][6, 2]
    assert masks["no_evidence"][6, 4]

    assert masks["no_history"][5, 5]
    assert not masks["no_history"][5, 4]
    assert not masks["no_history"][6, 5]
    assert masks["no_history"][6, 2]

    joint = masks["no_evidence_no_history"]
    assert not joint[6, 1]
    assert not joint[6, 4]
    assert joint[6, 6]


def test_swapped_evidence_preserves_length_and_response_alignment():
    token_ids = np.array([10, 11, 12, 13, 20, 21])
    swapped = replace_evidence_tokens(token_ids, [1, 2], [91, 92])

    original_predictors, original_targets = teacher_forced_alignment(token_ids, 4)
    swapped_predictors, swapped_targets = teacher_forced_alignment(swapped, 4)

    assert swapped.tolist() == [10, 91, 92, 13, 20, 21]
    assert np.array_equal(swapped_predictors, original_predictors)
    assert np.array_equal(swapped_targets, original_targets)
    with pytest.raises(ValueError, match="exactly one token"):
        replace_evidence_tokens(token_ids, [1, 2], [91])


def test_grouped_query_head_geometry_is_explicit():
    assert q_to_kv_mapping(8, 2).tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    with pytest.raises(ValueError, match="divisible"):
        q_to_kv_mapping(7, 2)


def test_rademacher_token_probes_are_seeded_without_global_rng_state():
    np.random.seed(99)
    first = rademacher_token_probes(7, 5, seed=1234)
    np.random.seed(1)
    second = rademacher_token_probes(7, 5, seed=1234)
    changed = rademacher_token_probes(7, 5, seed=1235)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, changed)
    assert set(np.unique(first)) == {-1.0, 1.0}
    with pytest.raises(ValueError, match="at least one"):
        rademacher_token_probes(0, 5, seed=1234)


def test_capture_cli_exposes_probe_count_and_attribution_seed():
    from experiments.attention_mechanism_audit.run import command_line

    arguments = command_line().parse_args(
        [
            "capture",
            "--data",
            "data",
            "--roles",
            "roles.jsonl",
            "--source-info",
            "source.jsonl",
            "--model",
            "model",
            "--output",
            "audit.npz",
            "--gradient-probes",
            "13",
            "--attribution-seed",
            "73",
        ]
    )

    assert arguments.gradient_probes == 13
    assert arguments.attribution_seed == 73


def test_full_rademacher_enumeration_recovers_nondiagonal_jacobian_exactly():
    torch = pytest.importorskip("torch")
    jacobian = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [7.0, -3.0, 0.0],
            [-5.0, 11.0, 4.0],
        ]
    )
    estimates = []
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        state = torch.tensor([0.2, -0.4, 0.7], requires_grad=True)
        output = jacobian @ state
        probe = torch.tensor(signs)
        vjp = torch.autograd.grad((output * probe).sum(), state)[0]
        estimates.append(apply_token_probe_to_vjp(vjp, probe))

    estimate = torch.stack(estimates).mean(dim=0)
    assert torch.allclose(estimate, jacobian.diagonal(), atol=1e-7)


def test_token_diagonal_probes_cancel_later_output_contamination():
    torch = pytest.importorskip("torch")
    # Later y_1 strongly depends on earlier predictor state c_0.  A backward
    # from answer-mean logprob contaminates the c_0 gradient with that future
    # effect, whereas the token-diagonal estimator retains dy_0/dc_0 only.
    jacobian = torch.tensor([[1.0, 0.0], [100.0, 2.0]])
    state = torch.tensor([0.3, -0.2], requires_grad=True)
    output = jacobian @ state
    answer_mean_gradient = torch.autograd.grad(
        output.mean(), state, retain_graph=True
    )[0]
    estimates = []
    for signs in itertools.product((-1.0, 1.0), repeat=2):
        probe = torch.tensor(signs)
        vjp = torch.autograd.grad(
            (output * probe).sum(), state, retain_graph=True
        )[0]
        estimates.append(apply_token_probe_to_vjp(vjp, probe))

    diagonal = torch.stack(estimates).mean(dim=0)
    assert answer_mean_gradient[0] == pytest.approx(50.5)
    assert torch.allclose(diagonal, torch.tensor([1.0, 2.0]), atol=1e-7)


def test_capture_hook_exact_enumeration_recovers_token_diagonal(monkeypatch):
    """Exercise predictor-row hooks, VJPs, and sign correction end to end."""

    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    import experiments.attention_mechanism_audit.replay as replay_module

    class CrossTokenAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.v_proj = torch.nn.Linear(1, 1, bias=False)
            self.o_proj = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.v_proj.weight.fill_(1.0)
                self.o_proj.weight.fill_(1.0)

        def forward(self, hidden, attention_mask):
            del attention_mask
            context = self.v_proj(hidden)
            return self.o_proj(context)

    class CrossTokenLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = CrossTokenAttention()

    class CrossTokenBackbone(torch.nn.Module):
        def __init__(self, embedding):
            super().__init__()
            self.embed_tokens = embedding
            self.layers = torch.nn.ModuleList([CrossTokenLayer()])
            self.last_used_input_ids = False
            self.last_attention_mask_ndim = -1

        def forward(
            self,
            *,
            input_ids=None,
            inputs_embeds=None,
            attention_mask,
            position_ids=None,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        ):
            del position_ids, use_cache, output_attentions, output_hidden_states
            self.last_used_input_ids = input_ids is not None
            self.last_attention_mask_ndim = attention_mask.ndim
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            context = self.layers[0].self_attn(inputs_embeds, attention_mask)
            # For predictor rows 0 and 1, the score Jacobian with respect to the
            # hooked o_proj input is [[1, 0], [100, 2]].  Thus an answer-mean
            # backward would badly contaminate row zero with the future score.
            transition = context.new_tensor(
                [[1.0, 0.0, 0.0], [100.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
            )
            hidden = torch.einsum("qs,bsd->bqd", transition, context)
            if return_dict:
                return SimpleNamespace(last_hidden_state=hidden)
            return (hidden,)

    class CrossTokenCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                num_attention_heads=1,
                num_key_value_heads=1,
                hidden_size=1,
                head_dim=1,
            )
            embedding = torch.nn.Embedding(8, 1)
            self.model = CrossTokenBackbone(embedding)
            self.lm_head = torch.nn.Linear(1, 8, bias=False)

        def get_input_embeddings(self):
            return self.model.embed_tokens

    probes = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=2)), dtype=np.float32
    )

    def complete_probe_enumeration(probe_count, response_length, seed):
        del seed
        assert probe_count == 4
        assert response_length == 2
        return probes.copy()

    def linear_token_scores(hidden, targets, lm_head, *, vocab_chunk_size):
        del targets, lm_head, vocab_chunk_size
        return hidden[:, 0]

    monkeypatch.setattr(
        replay_module, "rademacher_token_probes", complete_probe_enumeration
    )
    monkeypatch.setattr(
        replay_module, "_differentiable_chosen_logprob", linear_token_scores
    )

    replay = FrozenCausalReplay(CrossTokenCausalLM(), checkpoint="cross-token-toy")
    capture = replay.capture_baseline(
        np.asarray([1, 2, 3], dtype=np.int64),
        prompt_length=1,
        gradient_probes=4,
        attribution_seed=0,
    )

    context = torch.tensor([0.25, -0.5], requires_grad=True)

    def known_scores(value):
        return torch.stack((value[0], 100.0 * value[0] + 2.0 * value[1]))

    exact_jacobian = torch.autograd.functional.jacobian(known_scores, context)
    assert torch.equal(
        exact_jacobian, torch.tensor([[1.0, 0.0], [100.0, 2.0]])
    )
    expected_diagonal = exact_jacobian.diagonal().reshape(2, 1, 1)
    assert torch.allclose(
        capture.o_proj_input_gradients[0], expected_diagonal, atol=1e-7
    )
    assert replay.model.model.last_used_input_ids is True
    assert replay.model.model.last_attention_mask_ndim == 2
    assert not torch.allclose(
        capture.o_proj_input_gradients[0, 0],
        exact_jacobian[:, 0].mean().reshape(1, 1),
    )


def test_unavailable_swap_is_explicit_nan_not_a_fabricated_full_replay():
    predictors = np.array([2, 3], dtype=np.int64)
    targets = np.array([7, 8], dtype=np.int64)
    finite = np.array([-0.2, -0.4], dtype=np.float32)
    variants = {
        name: VariantScores(
            name=name,
            token_ids=np.array([1, 2, 3, 7, 8], dtype=np.int64),
            prompt_length=3,
            predictor_indices=predictors.copy(),
            target_ids=targets.copy(),
            chosen_logprob=finite.copy(),
            chosen_vs_best_other_margin=finite.copy(),
            vocab_jsd_from_full=np.zeros(2, dtype=np.float32),
        )
        for name in VARIANT_NAMES
    }
    missing = np.full(2, np.nan, dtype=np.float32)
    variants["swapped_evidence_0"] = VariantScores(
        name="swapped_evidence_0",
        token_ids=None,
        prompt_length=3,
        predictor_indices=predictors.copy(),
        target_ids=targets.copy(),
        chosen_logprob=missing.copy(),
        chosen_vs_best_other_margin=missing.copy(),
        vocab_jsd_from_full=missing.copy(),
        available=False,
        unavailable_reason="no evidence donor was provided",
    )

    result = ReplayResult(checkpoint="toy", variants=variants).validate()

    swap = result.variants["swapped_evidence_0"]
    assert not swap.available
    assert swap.token_ids is None
    assert np.array_equal(swap.predictor_indices, predictors)
    assert np.array_equal(swap.target_ids, targets)
    assert np.isnan(swap.chosen_logprob).all()
    fabricated = dict(variants)
    fabricated["swapped_evidence_0"] = replace(
        swap,
        chosen_logprob=finite.copy(),
    )
    with pytest.raises(ValueError, match="explicit NaN"):
        ReplayResult(checkpoint="toy", variants=fabricated).validate()


def test_chunked_torch_scoring_matches_dense_reference_when_torch_available():
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)
    hidden = torch.randn(4, 5)
    head = torch.nn.Linear(5, 11, bias=True)
    targets = torch.tensor([9, 0, 7, 3])
    reference_hidden = torch.randn(4, 5)

    actual = _chunk_scores(hidden, targets, head, vocab_chunk_size=3)
    reference = _chunk_scores(reference_hidden, targets, head, vocab_chunk_size=4)
    jsd = _chunked_jsd(
        hidden,
        actual,
        reference_hidden,
        reference,
        head,
        vocab_chunk_size=3,
    )

    dense_logits = head(hidden).detach().numpy()
    reference_logits = head(reference_hidden).detach().numpy()
    expected_logprob, expected_margin, expected_jsd = score_dense_logits(
        dense_logits,
        targets.numpy(),
        reference_logits=reference_logits,
    )
    assert np.allclose(
        actual.chosen_logprob.detach().numpy(), expected_logprob, atol=2e-6
    )
    assert np.allclose(actual.margin.detach().numpy(), expected_margin, atol=2e-6)
    assert np.allclose(jsd.detach().numpy(), expected_jsd, atol=2e-6)


def test_toy_llama_replay_is_frozen_and_captures_value_path_when_torch_available():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    class ToyAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.v_proj = torch.nn.Linear(4, 2, bias=False)
            self.o_proj = torch.nn.Linear(4, 4, bias=False)

        def forward(self, hidden, attention_mask):
            batch, sequence, _ = hidden.shape
            value = self.v_proj(hidden).reshape(batch, sequence, 1, 2)
            value = value.expand(batch, sequence, 2, 2)
            weight = torch.softmax(
                attention_mask.expand(batch, 2, sequence, sequence).float(),
                dim=-1,
            ).to(hidden.dtype)
            context = torch.einsum("bhqs,bshd->bqhd", weight, value)
            return self.o_proj(context.reshape(batch, sequence, 4))

    class ToyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = ToyAttention()

        def forward(self, hidden, attention_mask):
            return hidden + self.self_attn(hidden, attention_mask)

    class ToyBackbone(torch.nn.Module):
        def __init__(self, embedding):
            super().__init__()
            self.embed_tokens = embedding
            self.layers = torch.nn.ModuleList([ToyLayer(), ToyLayer()])
            self.forward_calls = 0
            self.backward_calls = 0
            self.last_used_input_ids = False
            self.last_attention_mask_ndim = -1
            self.call_embedding_twice = False

        def forward(
            self,
            *,
            input_ids=None,
            inputs_embeds=None,
            attention_mask,
            position_ids=None,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        ):
            del position_ids, use_cache, output_attentions, output_hidden_states
            self.last_used_input_ids = input_ids is not None
            self.last_attention_mask_ndim = attention_mask.ndim
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
                if self.call_embedding_twice:
                    self.embed_tokens(input_ids)
            self.forward_calls += 1
            if inputs_embeds.requires_grad:
                def count_backward(gradient):
                    self.backward_calls += 1
                    return gradient

                inputs_embeds.register_hook(count_backward)
            hidden = inputs_embeds
            for layer in self.layers:
                hidden = layer(hidden, attention_mask)
            return SimpleNamespace(last_hidden_state=hidden) if return_dict else (hidden,)

    class ToyCausalLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                num_attention_heads=2,
                num_key_value_heads=1,
                hidden_size=4,
                head_dim=2,
            )
            embedding = torch.nn.Embedding(9, 4)
            self.model = ToyBackbone(embedding)
            self.lm_head = torch.nn.Linear(4, 9, bias=False)

        def get_input_embeddings(self):
            return self.model.embed_tokens

    torch.manual_seed(31)
    model = ToyCausalLM()
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    replay = FrozenCausalReplay(model, checkpoint="toy")
    token_ids = np.array([1, 2, 3, 4, 5])

    def module_hook_counts():
        modules = [model.get_input_embeddings()]
        for layer in model.model.layers:
            modules.extend((layer.self_attn.v_proj, layer.self_attn.o_proj))
        return tuple(
            (len(module._forward_hooks), len(module._forward_pre_hooks))
            for module in modules
        )

    hooks_before_capture = module_hook_counts()

    result = replay.replay(
        token_ids,
        prompt_length=3,
        evidence_positions=[1],
        replacement_evidence_token_ids=[6],
        vocab_chunk_size=3,
    )
    unavailable_variants = replay._default_variants(
        token_ids,
        3,
        np.array([1]),
        np.array([6]),
    )
    unavailable_variants["swapped_evidence_0"] = {
        "available": False,
        "unavailable_reason": "limit excluded every donor",
        "token_ids": None,
        "allowed_attention": None,
        "prompt_length": 3,
    }
    unavailable_result = replay.replay(
        token_ids,
        prompt_length=3,
        evidence_positions=[1],
        variants=unavailable_variants,
        vocab_chunk_size=3,
    )
    forward_before_capture = model.model.forward_calls
    backward_before_capture = model.model.backward_calls
    capture = replay.capture_baseline(
        token_ids,
        prompt_length=3,
        vocab_chunk_size=3,
        gradient_probes=3,
        attribution_seed=41,
    )
    assert model.model.last_used_input_ids is True
    assert model.model.last_attention_mask_ndim == 2
    assert model.model.forward_calls - forward_before_capture == 1
    assert model.model.backward_calls - backward_before_capture == 3
    assert module_hook_counts() == hooks_before_capture
    repeated_capture = replay.capture_baseline(
        token_ids,
        prompt_length=3,
        vocab_chunk_size=3,
        gradient_probes=3,
        attribution_seed=41,
    )
    assert torch.equal(
        capture.o_proj_input_gradient_probes,
        repeated_capture.o_proj_input_gradient_probes,
    )
    assert module_hook_counts() == hooks_before_capture

    with pytest.raises(RuntimeError, match="did not return attention weights"):
        replay.capture_baseline(
            token_ids,
            prompt_length=3,
            vocab_chunk_size=3,
            gradient_probes=1,
            attribution_seed=41,
            expected_graph=object(),
        )
    assert module_hook_counts() == hooks_before_capture

    model.model.call_embedding_twice = True
    with pytest.raises(RuntimeError, match="must fire exactly once"):
        replay.capture_baseline(
            token_ids,
            prompt_length=3,
            vocab_chunk_size=3,
            gradient_probes=1,
            attribution_seed=41,
        )
    model.model.call_embedding_twice = False
    assert module_hook_counts() == hooks_before_capture

    calls_before_reuse = model.model.forward_calls
    reused_result = replay.replay(
        token_ids,
        prompt_length=3,
        evidence_positions=[1],
        replacement_evidence_token_ids=[6],
        baseline_capture=capture,
        vocab_chunk_size=3,
    )
    assert model.model.forward_calls - calls_before_reuse == 4
    np.testing.assert_allclose(
        reused_result.variants["full"].chosen_logprob,
        result.variants["full"].chosen_logprob,
        atol=2e-6,
    )
    calls_before_unavailable_reuse = model.model.forward_calls
    unavailable_reused_result = replay.replay(
        token_ids,
        prompt_length=3,
        evidence_positions=[1],
        variants=unavailable_variants,
        baseline_capture=capture,
        vocab_chunk_size=3,
    )
    assert model.model.forward_calls - calls_before_unavailable_reuse == 3
    assert not unavailable_reused_result.variants["swapped_evidence_0"].available

    calls_before_bad_capture = model.model.forward_calls
    with pytest.raises(ValueError, match="checkpoint"):
        replay.replay(
            token_ids,
            prompt_length=3,
            evidence_positions=[1],
            replacement_evidence_token_ids=[6],
            baseline_capture=replace(capture, checkpoint="different-model"),
            vocab_chunk_size=3,
        )
    assert model.model.forward_calls == calls_before_bad_capture

    assert tuple(result.variants) == VARIANT_NAMES
    assert result.variants["full"].target_ids.tolist() == [4, 5]
    assert result.variants["full"].predictor_indices.tolist() == [2, 3]
    unavailable_swap = unavailable_result.variants["swapped_evidence_0"]
    assert not unavailable_swap.available
    assert unavailable_swap.unavailable_reason == "limit excluded every donor"
    assert unavailable_swap.predictor_indices.tolist() == [2, 3]
    assert unavailable_swap.target_ids.tolist() == [4, 5]
    assert np.isnan(unavailable_swap.chosen_logprob).all()
    assert capture.value_states.shape == (2, 5, 1, 2)
    assert capture.o_proj_input_gradients.shape == (2, 2, 2, 2)
    assert capture.o_proj_input_gradient_probes.shape == (3, 2, 2, 2, 2)
    assert torch.allclose(
        capture.o_proj_input_gradients,
        capture.o_proj_input_gradient_probes.mean(dim=0),
    )
    assert capture.gradient_probe_count == 3
    assert capture.gradient_probe_seed == 41
    assert capture.gradient_probe_scheme == TOKEN_DIAGONAL_PROBE_SCHEME
    assert capture.predictor_hidden.shape == (2, 4)
    assert not capture.predictor_hidden.requires_grad
    assert capture.q_to_kv.tolist() == [0, 0]
    assert np.isfinite(capture.chosen_logprob.numpy()).all()
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])
