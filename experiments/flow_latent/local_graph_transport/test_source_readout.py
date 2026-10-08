"""Candidate identity, score direction and source-compatible reader contracts."""
import pytest
import torch
from torch.nn import functional as F

from .source_readout import SourceCompatibilityReader


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(rows=7, width=6):
    generator = torch.Generator().manual_seed(73)
    nodes = torch.randn(rows, 6, 8, generator=generator, dtype=torch.float64)
    boundaries = torch.randn(rows, 10, 8, generator=generator, dtype=torch.float64)
    values = torch.randn(rows, 2, 8, generator=generator, dtype=torch.float64)
    indices = torch.arange(rows)[:, None] - torch.arange(1, width + 1)[None]
    valid = indices >= 0
    attention = torch.rand(2, rows, 2, width, generator=generator, dtype=torch.float64)
    attention *= valid[None, :, None]
    scalars = torch.zeros(rows, 8, dtype=torch.float64)
    candidates = torch.randn(rows, 8, generator=generator, dtype=torch.float64)
    return [nodes, boundaries, values, attention, indices, valid, scalars, candidates]


def reader(use_bias=True):
    torch.manual_seed(42)
    return SourceCompatibilityReader(model_dim=8, heads=2, head_dim=4,
                                      dropout=0., use_bias=use_bias).double().eval()


def test_same_prechoice_prefix_different_candidates_receive_different_scores():
    model = reader()
    fields = inputs()
    rows = torch.tensor([3, 3])
    candidates = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 1., 0., 0., 0., 0., 0., 0.]], dtype=torch.float64)
    risks, embedding = model(*fields[:-1], candidates, rows=rows, return_embedding=True)
    torch.testing.assert_close(embedding[0], embedding[1], rtol=0, atol=0)
    assert not torch.isclose(risks[0], risks[1], rtol=0, atol=1e-10)


def test_compatible_direction_has_lower_risk_and_fixed_temperature_range():
    model = reader(use_bias=False)
    fields = inputs()
    _, embedding = model(*fields, return_embedding=True)
    aligned = F.normalize(model.semantic_vector(embedding, fields[0]), dim=-1)
    low_risk = model.compatibility(embedding, aligned, fields[0]).neg()
    high_risk = model.compatibility(embedding, -aligned, fields[0]).neg()
    torch.testing.assert_close(low_risk, torch.full((7,), -16., dtype=torch.float64))
    torch.testing.assert_close(high_risk, torch.full((7,), 16., dtype=torch.float64))
    assert not model.temperature.requires_grad


def test_candidate_rescaling_does_not_change_normalized_dot_or_modify_input():
    model = reader()
    fields = inputs()
    original = fields[-1].clone()
    base = model(*fields)
    scaled = model(*fields[:-1], fields[-1] * 37)
    torch.testing.assert_close(base, scaled, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(fields[-1], original, rtol=0, atol=0)


def test_selected_rows_match_full_readout_and_candidates_remain_row_aligned():
    model = reader()
    fields = inputs()
    full, all_embedding = model(*fields, return_embedding=True)
    rows = torch.tensor([5, 1, 5])
    selected, embedding = model(*fields[:-1], fields[-1][rows], rows=rows,
                                 return_embedding=True)
    torch.testing.assert_close(selected, full[rows])
    torch.testing.assert_close(embedding, all_embedding[rows])
    assert embedding.shape == (3, 136)


@pytest.mark.parametrize('variant', ['real', 'self', 'rewired'])
def test_source_reader_has_no_unused_classifier_and_matched_control_parameter_count(variant):
    model = reader()
    parameter_names = dict(model.named_parameters())
    assert not any(name.startswith('classifier.') for name in parameter_names)
    assert model.semantic_projection.out_features == 8
    count = sum(parameter.numel() for parameter in model.parameters())
    result = model(*inputs(), variant=variant)
    assert result.shape == (7,)
    assert sum(parameter.numel() for parameter in model.parameters()) == count


def test_source_program_bce_reaches_all_node_boundary_and_candidate_coordinates():
    model = reader().train()
    fields = inputs()
    for index in (0, 1, 2):
        fields[index].requires_grad_(True)
    risk = model(*fields)
    compatibility_labels = torch.arange(len(risk)).remainder(2).double()
    F.binary_cross_entropy_with_logits(risk, 1 - compatibility_labels).backward()
    for projection in (model.node_projection, model.boundary_projection):
        assert torch.count_nonzero(projection.weight.grad) == projection.weight.numel()
    semantic_gradient = model.semantic_projection.weight.grad
    assert torch.count_nonzero(semantic_gradient[:, :128]) == semantic_gradient[:, :128].numel()
    assert torch.count_nonzero(semantic_gradient[:, 128:]) == 0
    assert torch.count_nonzero(model.head_projection.grad) == model.head_projection.numel()
    assert torch.count_nonzero(model.node_semantic_weights.grad) == model.node_semantic_weights.numel()
    assert torch.count_nonzero(fields[0].grad) == fields[0].numel()
    assert torch.count_nonzero(fields[1].grad) == fields[1].numel()


@pytest.mark.parametrize('variant', ['real', 'self', 'rewired'])
def test_later_nodes_and_candidate_vectors_do_not_change_prefix_candidate_risk(variant):
    model = reader()
    fields = inputs()
    baseline = model(*fields, variant=variant)
    changed = [value.clone() for value in fields]
    for index in (0, 1, 2, 7):
        changed[index][4:] *= -100
    actual = model(*changed, variant=variant)
    torch.testing.assert_close(actual[:4], baseline[:4], rtol=0, atol=0)


def test_paired_same_prefix_candidates_can_be_fit_without_candidate_free_bias_shortcut():
    model = reader().train()
    fields = inputs()
    rows = torch.tensor([3, 3])
    candidates = fields[-1][[1, 5]]
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    targets = torch.tensor([0., 1.], dtype=torch.float64)
    with torch.no_grad():
        before = F.binary_cross_entropy_with_logits(model(*fields[:-1], candidates, rows=rows), targets)
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        logits = model(*fields[:-1], candidates, rows=rows)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        after = F.binary_cross_entropy_with_logits(model(*fields[:-1], candidates, rows=rows), targets)
    assert after < before * .2


def test_zero_initialized_direct_weights_preserve_existing_semantic_output():
    model = reader()
    fields = inputs()
    _, embedding = model(*fields, return_embedding=True)
    assert not model.node_semantic_weights.detach().any()
    torch.testing.assert_close(model.semantic_vector(embedding, fields[0]),
                               model.semantic_projection(embedding), rtol=0, atol=0)


def test_full_coordinate_direct_path_scores_candidates_without_latent_projection():
    model = reader(use_bias=False)
    fields = inputs()
    with torch.no_grad():
        model.semantic_projection.weight.zero_()
        model.semantic_projection.bias.fill_(1.)
        model.node_semantic_weights.zero_()
        model.node_semantic_weights[0, 7] = 2.
    fields[0].zero_()
    fields[0][3, 0, 7] = 1.
    candidates = torch.zeros(2, 8, dtype=torch.float64)
    candidates[0, 7] = 1.
    candidates[1, 6] = 1.
    risks, embedding = model(*fields[:-1], candidates, rows=torch.tensor([3, 3]),
                              return_embedding=True)
    torch.testing.assert_close(embedding[0], embedding[1], rtol=0, atol=0)
    assert risks[0] < risks[1]
    selected_nodes = fields[0][[3, 3]]
    with_direct = model.semantic_vector(embedding, selected_nodes)
    expected = torch.ones(2, 8, dtype=torch.float64)
    expected[:, 7] = 3.
    torch.testing.assert_close(with_direct, expected, rtol=0, atol=0)


def test_direct_path_covers_each_site_coordinate_and_selected_rows_without_future_leak():
    model = reader(use_bias=False)
    fields = inputs()
    with torch.no_grad():
        model.semantic_projection.weight.zero_()
        model.semantic_projection.bias.fill_(1.)
        model.node_semantic_weights.fill_(.5)
    rows = torch.tensor([1, 3])
    candidates = fields[-1][rows]
    base = model(*fields[:-1], candidates, rows=rows)
    changed = [value.clone() for value in fields]
    changed[0][4:] += 1e5
    result = model(*changed[:-1], candidates, rows=rows)
    torch.testing.assert_close(result, base, rtol=0, atol=0)
    _, embedding = model(*fields[:-1], candidates, rows=rows, return_embedding=True)
    semantic = model.semantic_vector(embedding, fields[0][rows])
    expected = torch.ones_like(semantic) + fields[0][rows].sum(1) * .5
    torch.testing.assert_close(semantic, expected)
