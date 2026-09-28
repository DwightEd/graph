"""Scientific contracts only; these tests do not validate hallucination detection."""

import torch

from experiments.role_free_flow.risk_response import (
    CandidateRiskProbe, message_responses, response_summary,
)


def test_sampled_token_can_distinguish_an_identical_query_state():
    probe = CandidateRiskProbe(2, checkpoints=1, rank=1).double()
    with torch.no_grad():
        for parameter in probe.parameters():
            parameter.zero_()
        probe.state_factor.weight[0, 0] = 1
        probe.token_factor.weight[0, 0] = 1
    state = torch.tensor([[[1., 0.]], [[1., 0.]]], dtype=torch.double)
    token = torch.tensor([[1., 0.], [-1., 0.]], dtype=torch.double)
    scores = probe(state, token)
    assert scores[0] > 0 and scores[1] < 0
    assert sum(p.numel() for p in CandidateRiskProbe(4096).parameters()) == 147457


def test_two_gate_responses_match_explicit_autograd_and_finite_dose():
    torch.manual_seed(17)
    heads, keys, head_dim = 2, 3, 2
    hidden = heads * head_dim
    attention = torch.randn(heads, keys, dtype=torch.double).softmax(-1)
    values = torch.randn(heads, keys, head_dim, dtype=torch.double)
    weight = torch.randn(hidden, hidden, dtype=torch.double)
    residual = torch.randn(hidden, dtype=torch.double)
    native_direction = torch.randn(hidden, dtype=torch.double)
    token = torch.randn(hidden, dtype=torch.double)
    probe = CandidateRiskProbe(hidden, checkpoints=1, rank=2).double()

    def outputs(gates):
        joined = torch.einsum('hk,hkd->hd', gates * attention, values).flatten()
        write = weight @ joined
        state = residual + write
        final = state + torch.tanh(state)  # Nonlinear downstream toy block.
        normalized = final / final.square().mean().add(.2).sqrt()
        margin = normalized @ native_direction
        risk = probe(final[None], token)
        return write, margin, risk

    gates = torch.ones(heads, keys, dtype=torch.double, requires_grad=True)
    write, margin, risk = outputs(gates)
    choice_gradient, explicit_choice = torch.autograd.grad(margin, (write, gates), retain_graph=True)
    risk_gradient, explicit_risk = torch.autograd.grad(risk, (write, gates))
    access, choice, risk_response = message_responses(
        attention, values, weight, residual, choice_gradient, risk_gradient)
    torch.testing.assert_close(choice, explicit_choice)
    torch.testing.assert_close(risk_response, explicit_risk)
    explicit_messages = torch.einsum('hkd,odh->hko', values,
        weight.reshape(hidden, heads, head_dim).permute(0, 2, 1)) * attention[..., None]
    expected_access = explicit_messages.norm(dim=-1) / (residual.norm() + 1e-12)
    torch.testing.assert_close(access, expected_access)

    dose = 1e-5
    changed = gates.detach().clone()
    changed[0, 1] -= dose
    _, next_margin, next_risk = outputs(changed)
    torch.testing.assert_close((next_margin-margin)/dose, -choice[0, 1], atol=1e-5, rtol=1e-4)
    torch.testing.assert_close((next_risk-risk)/dose, -risk_response[0, 1], atol=1e-5, rtol=1e-4)


def test_sign_cancellation_and_probe_scale_do_not_erase_response_geometry():
    access = torch.tensor([1., 1.], dtype=torch.double)
    choice = torch.tensor([2., 2.], dtype=torch.double)
    risk = torch.tensor([1., -1.], dtype=torch.double)
    summary = response_summary(access, choice, risk)
    assert summary[5] == 0
    torch.testing.assert_close(summary[6:], torch.tensor([.5, .5], dtype=torch.double))
    scaled = response_summary(access, choice, 17 * risk)
    torch.testing.assert_close(summary[5:], scaled[5:])
    empty = torch.empty(0, dtype=torch.double)
    torch.testing.assert_close(response_summary(empty, empty, empty), torch.zeros(8, dtype=torch.double))
