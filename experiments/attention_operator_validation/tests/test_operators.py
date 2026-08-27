import torch

from experiments.attention_operator_validation.operators import (
    geometry_from_factors,
    operator_gram_from_factors,
)


def test_factorized_gram_matches_explicit_head_operators():
    torch.manual_seed(11)
    heads, hidden, head_dim = 4, 7, 3
    output = torch.randn(heads, hidden, head_dim)
    value = torch.randn(heads, head_dim, hidden)

    actual = operator_gram_from_factors(output, value, block_heads=2)
    operator = torch.stack(
        [output[head] @ value[head] for head in range(heads)]
    ).flatten(1)
    expected = operator @ operator.T

    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_geometry_factor_reconstructs_gram_and_permutation_changes_binding():
    torch.manual_seed(13)
    output = torch.randn(3, 6, 2)
    value = torch.randn(3, 2, 6)
    geometry = geometry_from_factors([output], [value])

    reconstructed = geometry.factor[0] @ geometry.factor[0].T
    assert torch.allclose(reconstructed, geometry.gram[0], atol=2e-5, rtol=2e-5)

    code = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    real = code @ geometry.factor_for("operator_normalized")[0]
    permuted = code @ geometry.factor_for("operator_permuted", seed=17)[0]
    assert not torch.allclose(real, permuted)


def test_cached_factor_application_matches_explicit_operator():
    from experiments.attention_operator_validation.operators import (
        apply_factorized_operator,
    )

    torch.manual_seed(29)
    heads, kv_heads, hidden, head_dim = 4, 2, 6, 3
    output = torch.randn(heads, hidden, head_dim)
    value_unique = torch.randn(kv_heads, head_dim, hidden)
    q_to_kv = torch.tensor([0, 0, 1, 1])
    code = torch.rand(5, heads)
    source = torch.randn(5, hidden)
    basis = {
        "output_factor": output,
        "value_factor": value_unique,
        "q_to_kv": q_to_kv,
        "value_bias": None,
    }

    actual = apply_factorized_operator(code, source, basis)
    operator = torch.stack(
        [output[h] @ value_unique[q_to_kv[h]] for h in range(heads)]
    )
    expected = torch.einsum("bh,hij,bj->bi", code, operator, source)

    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
