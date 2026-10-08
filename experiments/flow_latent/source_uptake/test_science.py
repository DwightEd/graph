"""CPU contracts for frozen compatibility scoring; no native 8B validation."""
import pytest
import torch
from torch import nn

from ..ordered_source_transport.readout import OrderedCompatibility, row_permutations, transform_view
from .scoring import candidate_vector, prediction_rows, score_windows


def fixture(variant='ordered', token_count=5):
    torch.manual_seed(42)
    window, layers, sites, width = 4, 2, 4, 8
    model = OrderedCompatibility(window, layers, sites, width)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_()
        if variant == 'linear':
            model.transition.zero_()
    checkpoint = dict(state_dict=model.state_dict(), variant=variant,
                      mean=torch.randn(layers, sites, width),
                      scale=torch.rand(layers, sites, width) + .1)
    states = torch.randn(token_count + window - 1, layers, sites, width)
    ids = torch.tensor([[10 + index, 20 + index, 30 + index, 10 + index]
                        for index in range(token_count)])
    candidates = torch.randn(token_count, 4, width)
    candidates[:, -1] = candidates[:, 0]
    candidates = torch.nn.functional.normalize(candidates, dim=-1)
    logp = torch.tensor([[-.2, -1., -2., -.2]]).expand(token_count, 4).clone()
    return model, checkpoint, states, candidates, ids, ids[:, -1], logp


def test_candidate_contraction_equals_existing_forward_for_arbitrary_rows():
    model, checkpoint, states, candidates, _, _, _ = fixture()
    standardized = (states - checkpoint['mean']) / checkpoint['scale']
    windows = standardized.unfold(0, 4, 1).movedim(-1, 1)
    # Arbitrary candidate rows, not only training pair rows or unit embeddings.
    arbitrary = candidates * torch.arange(1, 5)[None, :, None]
    vector = candidate_vector(windows, checkpoint)
    scores = torch.einsum('bd,bcd->bc', vector, arbitrary) + checkpoint['state_dict']['bias']
    torch.testing.assert_close(scores, model(windows, arbitrary), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize('variant', ['ordered', 'mean', 'single', 'shuffled', 'drop_source', 'linear'])
def test_frozen_statistics_and_view_match_existing_readout(variant):
    model, checkpoint, states, candidates, ids, actual, logp = fixture(variant)
    permutation = row_permutations(5, 4, seed=73)
    standardized = (states - checkpoint['mean']) / checkpoint['scale']
    windows = standardized.unfold(0, 4, 1).movedim(-1, 1)
    expected = model(transform_view(windows, variant, permutation), candidates)
    result = score_windows(states, candidates, ids, actual, logp, checkpoint,
                           permutations=permutation)
    torch.testing.assert_close(result['candidate_scores'], expected, rtol=2e-6, atol=2e-6)
    # Different checkpoints retain their own source-fit statistics.
    changed = dict(checkpoint, mean=checkpoint['mean'] + .4)
    changed_result = score_windows(states, candidates, ids, actual, logp, changed,
                                   permutations=permutation)
    assert not torch.allclose(result['candidate_scores'], changed_result['candidate_scores'])


def test_gaps_exclude_every_duplicate_actual_and_greedy_id():
    _, checkpoint, states, candidates, ids, actual, logp = fixture(token_count=2)
    # Actual token outside top-3 in row 1; duplicate actual/greedy in row 0.
    ids[1, -1] = 40
    actual = ids[:, -1]
    candidates[1, -1] = torch.nn.functional.normalize(torch.ones(8), dim=0)
    logp[1, -1] = -7.
    result = score_windows(states, candidates, ids, actual, logp, checkpoint)
    scores = result['candidate_scores']
    for row in range(2):
        observed = ids[row] != actual[row]
        native = ids[row] != ids[row, 0]
        native[-1] = False
        expected_gap = scores[row, observed].max() - scores[row, -1]
        expected_logp = (scores[row] + logp[row])[observed].max() - scores[row, -1] - logp[row, -1]
        expected_native = scores[row, native].max() - scores[row, 0]
        torch.testing.assert_close(result['gap'][row], expected_gap)
        torch.testing.assert_close(result['logp_gap'][row], expected_logp)
        torch.testing.assert_close(result['native_gap'][row], expected_native)
        assert result['rival_ids'][row] != actual[row]
        assert result['logp_rival_ids'][row] != actual[row]
        assert result['native_rival_ids'][row] != ids[row, 0]


def test_native_gap_is_independent_of_appended_actual_candidate():
    _, checkpoint, states, candidates, ids, actual, logp = fixture()
    original = score_windows(states, candidates, ids, actual, logp, checkpoint)
    changed_ids = ids.clone()
    changed_ids[:, -1] = torch.arange(50, 55)
    changed_candidates = candidates.clone()
    changed_candidates[:, -1] = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    changed_logp = logp.clone()
    changed_logp[:, -1] = -8.
    altered = score_windows(states, changed_candidates, changed_ids, changed_ids[:, -1],
                            changed_logp, checkpoint)
    torch.testing.assert_close(original['candidate_scores'][:, :-1],
                               altered['candidate_scores'][:, :-1], rtol=0, atol=0)
    torch.testing.assert_close(original['native_gap'], altered['native_gap'], rtol=0, atol=0)
    torch.testing.assert_close(original['native_rival_ids'], altered['native_rival_ids'],
                               rtol=0, atol=0)
    assert not torch.allclose(original['gap'], altered['gap'])
    assert not torch.allclose(original['logp_gap'], altered['logp_gap'])


def test_chunking_and_shuffling_use_global_per_answer_permutations():
    _, checkpoint, states, candidates, ids, actual, logp = fixture('shuffled', token_count=7)
    permutation = row_permutations(7, 4, seed=73)
    full = score_windows(states, candidates, ids, actual, logp, checkpoint)
    chunks = []
    for start, end in [(0, 2), (2, 6), (6, 7)]:
        part = score_windows(states, candidates[start:end], ids[start:end], actual[start:end],
                             logp[start:end], checkpoint, start=start,
                             permutations=permutation[start:end])
        chunks.append(part)
    for key in full:
        torch.testing.assert_close(torch.cat([part[key] for part in chunks]), full[key],
                                   rtol=2e-6, atol=2e-6)


def test_capture_indexing_and_future_rows_cannot_enter_earlier_windows():
    capture_start, rows = prediction_rows(prompt_length=13, token_count=5, window=4)
    assert capture_start == 9
    assert rows.tolist() == [12, 13, 14, 15, 16]
    absolute = torch.arange(capture_start, int(rows[-1]) + 1)
    windows = absolute.unfold(0, 4, 1)
    torch.testing.assert_close(windows[:, -1], rows)
    # The actual y_t carrier starts at P+t; no current/future carrier appears.
    assert torch.all(windows < torch.arange(13, 18)[:, None])

    _, checkpoint, states, candidates, ids, actual, logp = fixture()
    baseline = score_windows(states, candidates, ids, actual, logp, checkpoint)
    changed = states.clone()
    changed[5:] += 1000  # Future to targets 0 and 1 (last rows 3 and 4).
    altered = score_windows(changed, candidates, ids, actual, logp, checkpoint)
    torch.testing.assert_close(baseline['candidate_scores'][:2], altered['candidate_scores'][:2],
                               rtol=0, atol=0)


def test_tiny_causal_transformer_future_input_independence():
    torch.manual_seed(73)
    block = nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16,
                                      dropout=0., batch_first=True)
    model = nn.TransformerEncoder(block, num_layers=2).eval()
    inputs = torch.randn(1, 8, 8)
    mask = torch.triu(torch.full((8, 8), -torch.inf), diagonal=1)
    changed = inputs.clone()
    changed[:, 5:] += torch.randn_like(changed[:, 5:]) * 100
    with torch.no_grad():
        original = model(inputs, mask=mask)
        altered = model(changed, mask=mask)
    torch.testing.assert_close(original[:, :5], altered[:, :5], rtol=0, atol=0)
    assert not torch.allclose(original[:, 5:], altered[:, 5:])


def test_degenerate_or_nonfinite_scoring_is_rejected():
    _, checkpoint, states, candidates, ids, actual, logp = fixture()
    all_actual = actual[:, None].expand_as(ids)
    with pytest.raises(ValueError, match='distinct'):
        score_windows(states, candidates, all_actual, actual, logp, checkpoint)
    bad_states = states.clone()
    bad_states[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match='finite'):
        score_windows(bad_states, candidates, ids, actual, logp, checkpoint)
    bad_logp = logp.clone()
    bad_logp[0, 0] = -torch.inf
    with pytest.raises(ValueError, match='finite'):
        score_windows(states, candidates, ids, actual, bad_logp, checkpoint)
