"""Scientific checks for prediction-grid edge replacement and GQA isolation."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from state_audit.model.adapter import ModelAdapter
from .binding import edge_delta, measure_case, query_gradients, ridge_scores
from .messages import native_trace, patch_message


def tiny_model():
    torch.manual_seed(12)
    config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    return ModelAdapter(LlamaForCausalLM(config).eval().requires_grad_(False))


def test_native_edge_reconstruction_and_uncontextualized_layer0():
    adapter = tiny_model()
    base = dict(prompt=[1, 2, 7, 4], key=2, query=3)
    donor = dict(prompt=[1, 5, 7, 4], key=2, query=3)
    observed, _ = measure_case(adapter, base, 7)
    swapped, _ = measure_case(adapter, donor, 7)
    assert observed["reconstruction"] < 1e-6
    assert torch.equal(observed["value"][0], swapped["value"][0])
    assert torch.equal(observed["value"][:, 0], observed["value"][:, 1])
    delta = edge_delta(observed, swapped)
    assert torch.equal(delta[0], torch.zeros_like(delta[0]))


def test_one_head_patch_preserves_gqa_neighbour_and_identity():
    adapter = tiny_model()
    prompt, answer = [1, 2, 7, 4], [7]
    original = native_trace(adapter.native, prompt, answer, gradients=False)
    delta = torch.arange(8).float() / 10
    with patch_message(adapter.native, 0, 0, len(prompt) - 1, delta):
        changed = native_trace(adapter.native, prompt, answer, gradients=False)
    difference = (changed.messages[0] - original.messages[0])[0].reshape(5, 4, 8)
    assert torch.allclose(difference[3, 0], delta)
    difference[3, 0] = 0
    assert torch.count_nonzero(difference) == 0
    with patch_message(adapter.native, 0, 0, len(prompt) - 1, torch.zeros(8)):
        identity = native_trace(adapter.native, prompt, answer, gradients=False)
    assert torch.equal(identity.logits, original.logits)


def test_edge_direction_gradient_matches_finite_difference():
    adapter = tiny_model()
    case = dict(prompt=[1, 2, 7, 4], key=2, query=3)
    observed, trace = measure_case(adapter, case, 7, gradients=True)
    gradient = query_gradients(trace, 5)
    delta = observed["value"][1, 0] * observed["attention"][1, 0]
    slope = (gradient[1, 0] * delta).sum()
    margins = []
    for alpha in (-.005, .005):
        with patch_message(adapter.native, 1, 0, case["query"], delta, alpha):
            changed = native_trace(adapter.native, case["prompt"], [7], gradients=False)
        margins.append(changed.logits[0, 7] - changed.logits[0, 5])
    finite = (margins[1] - margins[0]) / .01
    assert torch.allclose(slope, finite, atol=2e-5, rtol=.03)


def test_readout_fit_never_uses_heldout_labels():
    torch.manual_seed(9)
    features = torch.randn(12, 2, 4, 8)
    labels = torch.tensor([0, 1] * 6)
    fit = list(range(8))
    original = ridge_scores(features, labels, fit, 1.)
    labels[8:] = 1 - labels[8:]
    assert torch.equal(original, ridge_scores(features, labels, fit, 1.))


def test_appended_answer_cannot_leak_into_prediction():
    adapter = tiny_model()
    original = native_trace(adapter.native, [1, 2, 7, 4], [7], gradients=False)
    different = native_trace(adapter.native, [1, 2, 7, 4], [5], gradients=False)
    assert torch.equal(original.logits, different.logits)


def test_cholesky_readout_matches_dual_ridge_on_paired_query_sized_fit():
    torch.manual_seed(19)
    features = torch.randn(256, 2, 4, 8)
    labels = torch.tensor([0, 1] * 128)
    fit = list(range(192))
    scores = ridge_scores(features, labels, fit, .1)
    column = features[:, 0, 0].double()
    centered = (column - column[fit].mean(0)) / column[fit].std(0).clamp_min(1e-5)
    train = centered[fit]
    gram = train @ train.T + .1 * len(fit) * torch.eye(len(fit), dtype=torch.float64)
    target = 2 * labels[fit].double() - 1
    dual_prediction = centered @ train.T @ torch.linalg.solve(gram, target)
    assert torch.allclose(scores[:, 0].double(), dual_prediction, atol=1e-6, rtol=1e-6)
