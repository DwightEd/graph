import torch
import torch.nn.functional as F

from experiments.evidence_route_state.registers import (
    ENDOGENOUS,
    EVIDENCE,
    PROMPT,
    RESPONSE,
    add_attention,
    add_mlp,
    final_readout_contributions,
    initialize_registers,
    project_register_values,
    rmsnorm_registers,
    route_register_values,
)


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(0.5, 1.5, hidden))
        self.variance_epsilon = 1e-6

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        input_dtype = state.dtype
        scale = torch.rsqrt(
            state.float().square().mean(dim=-1, keepdim=True) + self.variance_epsilon
        )
        return self.weight * (state.float() * scale).to(input_dtype)


def test_initial_registers_partition_observed_token_states():
    hidden = torch.arange(1, 13, dtype=torch.float32).reshape(4, 3)
    positions = torch.tensor([1, 2, 3, 4])
    evidence_mask = torch.tensor([False, True, False])

    registers = initialize_registers(hidden, positions, evidence_mask, response_start=3)

    torch.testing.assert_close(registers.sum(1), hidden)
    torch.testing.assert_close(registers[0, EVIDENCE], hidden[0])
    torch.testing.assert_close(registers[1, PROMPT], hidden[1])
    torch.testing.assert_close(registers[2, RESPONSE], hidden[2])
    torch.testing.assert_close(registers[3, RESPONSE], hidden[3])
    torch.testing.assert_close(registers[:, ENDOGENOUS], torch.zeros_like(hidden))


def test_rmsnorm_and_value_projection_preserve_the_register_sum_in_float32():
    hidden = torch.tensor([[1.0, -2.0, 3.0, 0.5], [2.0, 1.0, -1.0, 4.0]])
    registers = torch.zeros(2, 4, 4)
    registers[:, EVIDENCE] = 0.25 * hidden
    registers[:, RESPONSE] = 0.75 * hidden
    norm = RMSNorm(4)

    normalized = rmsnorm_registers(registers, hidden, norm)
    torch.testing.assert_close(normalized.sum(1), norm(hidden))

    projection = torch.nn.Linear(4, 4, bias=False)
    projection.weight.data.copy_(torch.arange(1, 17).reshape(4, 4) / 16)
    values = project_register_values(normalized, projection, kv_heads=2, head_dim=2)
    expected = F.linear(normalized.sum(1), projection.weight).reshape(2, 2, 2)
    torch.testing.assert_close(values.sum(1), expected)
    assert values.dtype == torch.float32


def test_bfloat16_weights_never_mix_with_float32_register_geometry():
    hidden = torch.tensor([[1.0, -2.0, 3.0, 0.5]], dtype=torch.bfloat16)
    registers = torch.zeros(1, 4, 4)
    registers[:, PROMPT] = 0.25 * hidden.float()
    registers[:, RESPONSE] = 0.75 * hidden.float()
    norm = RMSNorm(4).to(torch.bfloat16)
    projection = torch.nn.Linear(4, 2, bias=False).to(torch.bfloat16)

    normalized = rmsnorm_registers(registers, hidden, norm)
    native_value = projection(norm(hidden))
    values = project_register_values(
        normalized,
        projection,
        kv_heads=1,
        head_dim=2,
        native_value=native_value,
    )

    assert normalized.dtype == values.dtype == torch.float32
    torch.testing.assert_close(normalized.sum(1), norm(hidden).float())
    torch.testing.assert_close(values.sum(1), native_value.float().reshape(1, 1, 2))
    assert torch.isfinite(values).all()


def test_gqa_routes_each_origin_through_each_head_without_head_averaging():
    attention = torch.zeros(4, 1, 2)
    attention[0, 0, 0] = 1.0
    attention[1, 0, 1] = 1.0
    attention[2, 0, 0] = 1.0
    attention[3, 0, 1] = 1.0
    values = torch.zeros(2, 4, 2, 1)
    values[0, EVIDENCE, 0, 0] = 1.0
    values[1, RESPONSE, 0, 0] = 2.0
    values[0, EVIDENCE, 1, 0] = 10.0
    values[1, RESPONSE, 1, 0] = 20.0
    output_weight = torch.eye(4)

    write, context = route_register_values(attention, values, output_weight)

    assert context.shape == (1, 4, 4, 1)
    torch.testing.assert_close(
        context[0, :, EVIDENCE, 0], torch.tensor([1.0, 0.0, 10.0, 0.0])
    )
    torch.testing.assert_close(
        context[0, :, RESPONSE, 0], torch.tensor([0.0, 2.0, 0.0, 20.0])
    )
    torch.testing.assert_close(write[0, EVIDENCE], torch.tensor([1.0, 0.0, 10.0, 0.0]))
    torch.testing.assert_close(write[0, RESPONSE], torch.tensor([0.0, 2.0, 0.0, 20.0]))


def test_register_writes_sum_to_the_native_attention_write_with_head_cancellation():
    attention = torch.tensor([[[1.0]], [[1.0]]])
    values = torch.zeros(1, 4, 2, 1)
    values[0, EVIDENCE, :, 0] = 1.0
    output_weight = torch.tensor([[1.0, -1.0]])

    write, context = route_register_values(attention, values, output_weight)

    torch.testing.assert_close(context[0, :, EVIDENCE, 0], torch.ones(2))
    torch.testing.assert_close(write.sum(1), torch.zeros(1, 1))


def test_native_head_context_rounding_is_closed_before_output_projection():
    attention = torch.ones(2, 1, 1)
    values = torch.zeros(1, 4, 2, 1)
    values[0, EVIDENCE, :, 0] = 1.0
    native_context = torch.tensor([[[1.125], [0.875]]])

    write, context = route_register_values(
        attention,
        values,
        torch.eye(2),
        native_context,
    )

    torch.testing.assert_close(context.sum(2), native_context)
    torch.testing.assert_close(
        context[0, :, ENDOGENOUS, 0], torch.tensor([0.125, -0.125])
    )
    torch.testing.assert_close(write.sum(1), native_context.flatten(1))


def test_attention_and_mlp_closure_reconstruct_native_residual_states():
    registers = torch.zeros(2, 4, 3)
    registers[:, PROMPT] = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]])
    writes = torch.zeros_like(registers)
    writes[:, EVIDENCE] = 0.5
    native_mid = registers.sum(1) + writes.sum(1) + 0.125

    middle = add_attention(registers, writes, native_mid)
    torch.testing.assert_close(middle.sum(1), native_mid)
    torch.testing.assert_close(middle[:, ENDOGENOUS], torch.full((2, 3), 0.125))

    mlp_write = torch.tensor([[0.2, -0.1, 0.3], [-0.2, 0.4, 0.1]])
    native_output = native_mid + mlp_write - 0.05
    output = add_mlp(middle, mlp_write, native_output)
    torch.testing.assert_close(output.sum(1), native_output)
    torch.testing.assert_close(
        output[:, ENDOGENOUS], middle[:, ENDOGENOUS] + mlp_write - 0.05
    )


def test_final_readout_decomposes_the_target_competitor_margin():
    registers = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 2.0], [0.5, 0.5], [0.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [0.5, 0.5]],
        ]
    )
    weight = torch.tensor([[1.0, 2.0], [-2.0, 1.0], [0.5, -1.0]])
    target = torch.tensor([0, 2])
    competitor = torch.tensor([1, 0])

    contribution = final_readout_contributions(registers, weight, target, competitor)
    total = registers.sum(1)
    expected = (
        F.linear(total, weight).gather(1, target[:, None])
        - F.linear(total, weight).gather(1, competitor[:, None])
    ).squeeze(1)

    torch.testing.assert_close(contribution.sum(1), expected)
