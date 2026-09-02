import torch
import torch.nn.functional as F

from experiments.evidence_route_state.messages import (
    attention_messages,
    prompt_carriers,
    reconstruct_attention_write,
)


def native_attention_write(
    attention: torch.Tensor,
    values: torch.Tensor,
    output_weight: torch.Tensor,
) -> torch.Tensor:
    query_heads = attention.shape[0]
    kv_heads = values.shape[1]
    head_to_kv = torch.arange(query_heads) // (query_heads // kv_heads)
    values_by_head = values.float()[:, head_to_kv]
    context = torch.einsum("hqs,shd->qhd", attention.float(), values_by_head)
    return F.linear(context.flatten(1), output_weight.float())


def exact_inputs(dtype: torch.dtype = torch.float32):
    attention = torch.tensor(
        [
            [[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]],
            [[0.1, 0.7, 0.2], [0.4, 0.1, 0.5]],
            [[0.3, 0.2, 0.5], [0.1, 0.8, 0.1]],
            [[0.2, 0.5, 0.3], [0.7, 0.2, 0.1]],
        ],
        dtype=torch.float32,
    )
    values = torch.arange(1, 13, dtype=torch.float32).reshape(3, 2, 2) / 8
    output_weight = torch.arange(1, 65, dtype=torch.float32).reshape(8, 8) / 32
    return attention, values.to(dtype), output_weight.to(dtype)


def test_each_avwo_edge_and_its_sum_match_the_native_attention_write():
    attention, values, output_weight = exact_inputs()
    messages = attention_messages(attention, values, output_weight)
    native = native_attention_write(attention, values, output_weight)

    assert messages.shape == (2, 4, 3, 8)
    for query in range(2):
        for head in range(4):
            kv_head = head // 2
            weight_block = output_weight[:, head * 2 : (head + 1) * 2]
            for source in range(3):
                expected = attention[head, query, source] * F.linear(
                    values[source, kv_head], weight_block
                )
                torch.testing.assert_close(messages[query, head, source], expected)

    torch.testing.assert_close(
        reconstruct_attention_write(messages), native, rtol=1e-5, atol=1e-6
    )


def test_derived_bf16_geometry_uses_one_dtype_and_exports_float32():
    attention, values, output_weight = exact_inputs(torch.bfloat16)

    messages = attention_messages(attention, values, output_weight)
    native = native_attention_write(attention, values, output_weight)

    assert messages.dtype == torch.float32
    torch.testing.assert_close(
        reconstruct_attention_write(messages), native, rtol=1e-5, atol=1e-6
    )


def test_gqa_mapping_keeps_each_query_head_and_source_distinct():
    attention = torch.zeros(4, 1, 2)
    attention[0, 0, 0] = 1.0
    attention[1, 0, 1] = 1.0
    attention[2, 0, 0] = 1.0
    attention[3, 0, 1] = 1.0
    values = torch.tensor([[[1.0], [10.0]], [[2.0], [20.0]]])
    output_weight = torch.eye(4)

    messages = attention_messages(attention, values, output_weight)

    torch.testing.assert_close(messages[0, 0, 0], torch.tensor([1.0, 0, 0, 0]))
    torch.testing.assert_close(messages[0, 1, 1], torch.tensor([0.0, 2.0, 0, 0]))
    torch.testing.assert_close(messages[0, 2, 0], torch.tensor([0.0, 0, 10.0, 0]))
    torch.testing.assert_close(messages[0, 3, 1], torch.tensor([0.0, 0, 0, 20.0]))


def test_opposing_head_messages_cancel_only_after_edges_are_retained():
    attention = torch.ones(2, 1, 1)
    values = torch.ones(1, 2, 1)
    output_weight = torch.tensor([[1.0, -1.0]])

    messages = attention_messages(attention, values, output_weight)

    torch.testing.assert_close(messages[0, 0, 0], torch.tensor([1.0]))
    torch.testing.assert_close(messages[0, 1, 0], torch.tensor([-1.0]))
    torch.testing.assert_close(reconstruct_attention_write(messages), torch.zeros(1, 1))
    torch.testing.assert_close(messages.norm(dim=-1).sum(), torch.tensor(2.0))


def test_locked_prompt_carriers_exclude_predictor_self_and_keep_head_anchors():
    mass = torch.tensor(
        [
            [
                [3.0, 1.0, 100.0, 20.0],
                [0.0, 4.0, 100.0, 20.0],
            ]
        ]
    )
    query = torch.tensor([2])

    carriers = prompt_carriers(mass, query, response_start=3)
    head_probability = torch.tensor([[0.75, 0.25], [0.0, 1.0]])
    mixture = head_probability.mean(0)
    expected_sources = (-(mixture * mixture.log()).sum()).exp()
    gram = head_probability @ head_probability.T
    expected_rank = gram.trace().square() / gram.square().sum()

    torch.testing.assert_close(carriers.effective_sources[0], expected_sources)
    torch.testing.assert_close(carriers.effective_rank[0], expected_rank)
    assert carriers.anchor_source.tolist() == [[0, 1]]

    # Attention mass and functional capacity share the same locked formula.
    # Per-head scaling leaves the normalized carrier geometry unchanged.
    functional = mass * torch.tensor([[[2.0], [0.5]]])
    functional_carriers = prompt_carriers(functional, query, response_start=3)
    torch.testing.assert_close(
        functional_carriers.effective_sources,
        carriers.effective_sources,
    )
    torch.testing.assert_close(
        functional_carriers.effective_rank,
        carriers.effective_rank,
    )
    torch.testing.assert_close(
        functional_carriers.anchor_source,
        carriers.anchor_source,
    )
