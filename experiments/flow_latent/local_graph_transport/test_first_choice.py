"""Scientific invariants for first-only source training and compact graphs."""
import numpy as np
import torch

from .first_choice import compact_graph, first_loss, first_metrics, first_records, source_groups
from .source_choices import build_choice_records
from .source_readout import SourceCompatibilityReader
from .test_source_choices import programs
from .test_source_readout import inputs


def test_first_controls_drop_later_choices_and_group_assignments():
    records = first_records(build_choice_records(programs()))
    assert len(records) == 2
    assert [row['candidate_ids'] for row in records] == [[10, 20], [20, 10]]
    assert all(row['labels'] == [0, 1] for row in records)
    assert len(source_groups(records)) == 1


def test_pair_loss_gradient_lowers_correct_and_raises_wrong_risk():
    risk = torch.tensor([0., 0., 0., 0.], requires_grad=True)
    first_loss(risk, 'bce_pair').backward()
    assert torch.all(risk.grad[::2] > 0)
    assert torch.all(risk.grad[1::2] < 0)
    pairs = risk.reshape(-1, 2)
    shifted = (pairs + torch.tensor([[3.], [-2.]])).flatten()
    base_pair = torch.nn.functional.softplus(risk[::2] - risk[1::2]).mean()
    shifted_pair = torch.nn.functional.softplus(shifted[::2] - shifted[1::2]).mean()
    torch.testing.assert_close(base_pair, shifted_pair)


def test_compact_true_sender_inputs_preserve_first_logit_and_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = SourceCompatibilityReader(model_dim=8, heads=2, head_dim=4, dropout=0.).double().eval()
    raw = inputs(rows=9, width=8)
    names = ('node_fields', 'boundary_fields', 'value_fields', 'local_attention',
             'indices', 'valid', 'scalars')
    numpy_fields = {name: tensor.detach().numpy() for name, tensor in zip(names, raw[:-1])}
    for receiver in (0, 3, 8):
        compact, target = compact_graph(numpy_fields, receiver)
        compact_fields = {name: torch.from_numpy(value) for name, value in compact.items()}
        candidates = raw[-1][[1, 5]]
        full = model(*raw[:-1], candidates, rows=torch.tensor([receiver, receiver]))
        reduced = model(**compact_fields, candidate_vectors=candidates, rows=torch.tensor([target, target]))
        torch.testing.assert_close(full, reduced, rtol=1e-12, atol=1e-12)
        full_gradient = torch.autograd.grad(full.sum(), model.node_projection.weight, retain_graph=True)[0]
        reduced_gradient = torch.autograd.grad(reduced.sum(), model.node_projection.weight)[0]
        torch.testing.assert_close(full_gradient, reduced_gradient, rtol=1e-12, atol=1e-12)
        assert int(compact['valid'][target].sum()) == int(numpy_fields['valid'][receiver].sum())


def test_metrics_separate_candidate_ranking_from_same_word_source_swaps():
    records = first_records(build_choice_records(programs()))
    scores = {records[0]['id']: np.array([-2., 1.]), records[1]['id']: np.array([-1., 2.])}
    result = first_metrics(records, scores)
    assert result['auroc'] == 1.
    assert result['candidate_ranking_rate'] == 1.
    assert result['source_swap']['direction_rate'] == 1.
    # A word-frequency-only score ranks opposite source conditions identically.
    word_only = {records[0]['id']: np.array([-2., 1.]), records[1]['id']: np.array([1., -2.])}
    result = first_metrics(records, word_only)
    assert result['auroc'] == .5
    assert result['candidate_ranking_rate'] == .5
    assert result['source_swap']['direction_rate'] == 0.
