"""Scientific contracts for source-only same-prefix candidate supervision."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from .run_source_choices import choice_metrics
from .source_choices import build_choice_records, source_choice_weights, choice_coverage
from .source_readout import SourceCompatibilityReader
from .test_source_readout import inputs


def programs(source='s', proposals=None, partition='fit'):
    proposals = proposals or [[3, 10, 11, 12, 13], [3, 20, 11, 22]]
    records = []
    for world, correct_index in (('original', 0), ('swapped', 1)):
        for proposal, answer in enumerate(proposals):
            records.append(dict(id=f'{source}_{world}_{proposal}', source_id=source,
                partition=partition, world=world, proposal_index=proposal,
                correct_index=correct_index, tokens=len(answer), answer_ids=answer,
                first_divergence=1, template_id='quote_field'))
    return records


def test_only_correct_capture_ids_and_paired_same_rows_are_used():
    records = build_choice_records(programs())
    assert [row['id'] for row in records] == ['s_original_0', 's_swapped_1']
    for record in records:
        assert record['rows'] == [1, 1, 3, 3]
        assert record['labels'] == [0, 1, 0, 1]
        assert record['first'] == [True, True, False, False]
        assert record['candidate_ids'][::2] == [record['answer_ids'][1], record['answer_ids'][3]]
        assert record['candidate_ids'][1::2] == [record['other_answer_ids'][1], record['other_answer_ids'][3]]


def test_same_candidate_at_first_divergence_flips_with_source_assignment():
    records = build_choice_records(programs())
    assignments = {}
    for record in records:
        for candidate, label in zip(record['candidate_ids'][:2], record['labels'][:2]):
            assignments.setdefault(candidate, []).append(label)
    assert assignments == {10: [0, 1], 20: [1, 0]}


def test_common_later_ids_and_other_quote_missing_positions_are_excluded():
    records = build_choice_records(programs())
    assert all(2 not in row['rows'] and 4 not in row['rows'] for row in records)
    assert choice_coverage(records)['fit']['choice_examples'] == 8


def test_loss_is_equal_source_and_first_later_mass_is_balanced():
    records = build_choice_records(programs('many') +
        programs('first_only', [[3, 10], [3, 20]]))
    weights = source_choice_weights(records)
    for source in ('many', 'first_only'):
        selected = [row for row in records if row['source_id'] == source]
        total = sum(weights[row['id']].sum() for row in selected)
        first_mass = sum(weights[row['id']][row['first']].sum() for row in selected)
        assert np.isclose(total, 1.)
        assert np.isclose(first_mass, .5 if source == 'many' else 1.)


def test_program_metric_does_not_conflate_continuation_with_first_choice():
    records = build_choice_records(programs())
    predictions = {row['id']: np.array([0., 0., -6., 6.]) for row in records}
    result = choice_metrics(records, predictions)
    assert result['first']['auroc'] == .5
    assert result['later']['auroc'] == 1.
    assert result['overall']['auroc'] == .875
    assert result['source_swap_first']['wrong_above_correct'] == 0.


def test_missing_source_swap_counterpart_is_rejected():
    records = build_choice_records(programs())[:1]
    predictions = {records[0]['id']: np.array([-1., 1., -1., 1.])}
    with pytest.raises(ValueError, match='counterpart'):
        choice_metrics(records, predictions)


def test_same_prefix_pair_bce_cannot_learn_through_candidate_free_bias():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = SourceCompatibilityReader(model_dim=8, heads=2, head_dim=4, dropout=0.).double()
    fields = inputs()
    rows = torch.tensor([1, 1, 3, 3])
    candidate_vectors = fields[-1][[1, 5, 2, 6]]
    labels = torch.tensor([0., 1., 0., 1.], dtype=torch.float64)
    risk = model(*fields[:-1], candidate_vectors, rows=rows)
    loss = F.binary_cross_entropy_with_logits(risk, labels)
    loss.backward()
    assert torch.count_nonzero(model.node_semantic_weights.grad) == model.node_semantic_weights.numel()
    assert torch.count_nonzero(model.head_projection.grad) == model.head_projection.numel()
    assert not torch.isclose(risk[0], risk[1])
    assert not torch.isclose(risk[2], risk[3])
    torch.set_num_threads(previous_threads)
