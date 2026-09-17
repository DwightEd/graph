"""Theory identities, non-identifying counterexamples, and actual experiment CLIs."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

from test_charm_structure_audit import fixture_files, graph_sample, model
from experiments.charm_structure_audit.main import main
from experiments.charm_structure_audit.continuity_weights import loss_weights, prepare_weights, SCHEMES
from experiments.charm_structure_audit.continuity_train import node_logits, load_input, score_record
from experiments.charm_structure_audit.continuity import transition_tables, growth_rows
from experiments.charm_structure_audit.data import read_json


def weighted_sample():
    labels = np.zeros(14, bool)
    spans = [[1, 4], [6, 13]]
    labels[1:4] = labels[6:13] = True
    return labels, spans


def test_pointwise_permutation_preserves_predictions_loss_and_parameter_gradient():
    torch.manual_seed(5)
    net = model().double().eval()
    other = copy.deepcopy(net)
    x = torch.rand(23, 4, dtype=torch.float64)
    y = torch.tensor([0.]*9+[1.]*9+[0.]*5, dtype=torch.float64)
    permutation = torch.randperm(len(y))
    original = node_logits(net, x)
    changed = node_logits(other, x[permutation])
    torch.testing.assert_close(original[permutation], changed, atol=1e-10, rtol=1e-10)
    a = F.binary_cross_entropy_with_logits(original, y, reduction='sum')
    b = F.binary_cross_entropy_with_logits(changed, y[permutation], reduction='sum')
    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    a.backward()
    b.backward()
    for left, right in zip(net.parameters(), other.parameters()):
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=1e-10, rtol=1e-10)
    assert all(p.grad is None for layer in net.mp_layers for p in layer.msg_mlp.parameters())


@pytest.mark.parametrize('scheme', SCHEMES)
def test_global_positive_mass_preserved_and_normals_unchanged(scheme):
    labels, spans = weighted_sample()
    before = labels.copy()
    w = loss_weights(labels, spans, scheme, 5, seed=5)
    np.testing.assert_allclose(w[labels].sum(), 10, atol=1e-6)
    np.testing.assert_array_equal(w[~labels], 1)
    np.testing.assert_array_equal(labels, before)


def test_span_equal_has_equal_total_not_equal_individual_weight():
    labels, spans = weighted_sample()
    w = loss_weights(labels, spans, 'span_equal', 5)
    np.testing.assert_allclose([w[a:b].sum() for a, b in spans], [5, 5])


def test_onset_control_preserves_each_span_mass_and_exact_weight_multiset():
    labels, spans = weighted_sample()
    onset = loss_weights(labels, spans, 'onset_half', 5)
    random = loss_weights(labels, spans, 'random_onset_half', 5, seed=10)
    for start, end in spans:
        assert onset[start] == (end-start)/2
        assert onset[start+1:end].sum() == pytest.approx((end-start)/2)
        np.testing.assert_array_equal(np.sort(onset[start:end]), np.sort(random[start:end]))
    assert not np.array_equal(onset, random)
    np.testing.assert_array_equal(random, loss_weights(labels, spans, 'random_onset_half', 5, seed=10))


def test_overlaps_merged_adjacent_preserved_and_singletons_not_doubled():
    labels = np.asarray([0, 1, 1, 1, 1, 1, 0], bool)
    spans = [[1, 3], [2, 4], [4, 5], [5, 6]]
    w = loss_weights(labels, spans, 'span_equal', 5/3)
    np.testing.assert_allclose([w[1:4].sum(), w[4], w[5]], 5/3)
    w = loss_weights(labels, spans, 'onset_half', 5/3)
    assert w[4] == 1 and w[5] == 1


def test_weighted_bayes_optimum_and_constant_shift_invariance():
    eta, alpha, m1, m0 = .2, 3., 2., 1.
    q = torch.tensor(alpha*eta*m1/(alpha*eta*m1+(1-eta)*m0), requires_grad=True, dtype=torch.float64)
    risk = -alpha*eta*m1*q.log()-(1-eta)*m0*torch.log1p(-q)
    risk.backward()
    assert abs(q.grad.item()) < 1e-12
    y = np.asarray([0, 0, 1, 1, 0, 1])
    probability = np.asarray([.1, .4, .3, .7, .8, .9])
    shifted = alpha*probability/(alpha*probability+1-probability)
    assert roc_auc_score(y, probability) == roc_auc_score(y, shifted)


def test_sample_dependent_weighting_can_change_order_not_just_intercept():
    eta = np.asarray([.2, .8])
    m1 = np.asarray([10., .1])
    q = eta*m1/(eta*m1+1-eta)
    assert eta[0] < eta[1] and q[0] > q[1]


def test_history_only_oracle_has_no_conditional_discrimination():
    labels = np.asarray([0, 0, 1, 1, 0, 1, 1, 0, 0])
    prev = np.r_[0, labels[:-1]]
    table = pd.DataFrame(dict(id='a', source_id='s', token=np.arange(len(labels)), gold=labels, score=.2+.6*prev))
    _, conditional, _, _ = transition_tables(table, .5, 0)
    np.testing.assert_allclose(conditional.auroc, .5)
    np.testing.assert_allclose(conditional.within_source_mean, .5)


def test_previous_label_never_crosses_answer_boundaries():
    table = pd.DataFrame(dict(id=['a','a','b','b'], source_id=['s','s','t','t'], token=[0,1,0,1],
                             gold=[1,1,0,0], score=[.8,.7,.2,.1]))
    counts, _, _, oracle = transition_tables(table, .5, 0)
    assert counts.tokens.sum() == 2
    assert not ((counts.previous_gold == 1) & (counts.current_gold == 0)).any()
    assert 'GOLD' in oracle['meaning']


def test_shared_position_growth_cancels_in_paired_difference():
    table = pd.DataFrame(dict(id='a', source_id='s', token=np.arange(8), gold=[0]*4+[1]*4,
                             score=[.2,.3,.4,.5,.2,.3,.4,.5]))
    pair = dict(id='a', source_id='s', error_start=4, normal_start=0, length=4, tier='cluster')
    result = growth_rows(table, [pair])
    assert result.gap_growth.iloc[0] == 0
    assert result.auc_growth.iloc[0] == 0


def test_node_input_does_not_read_edge_arrays(tmp_path, monkeypatch):
    root, prepared = fixture_files(tmp_path)
    record = read_json(prepared/'index.json')[0]
    original = np.load
    class Guard:
        def __init__(self, archive): self.archive = archive
        def __enter__(self): return self
        def __exit__(self, *args): self.archive.close()
        def __getitem__(self, key):
            assert key not in ('edge_attr', 'edge_index', 'edge_mark')
            return self.archive[key]
    monkeypatch.setattr(np, 'load', lambda *a, **k: Guard(original(*a, **k)))
    inputs, sample = load_input(record, prepared, 'node_only')
    assert inputs.shape == (32, 4)
    assert sample['gold'].shape == (32,)


def test_fit_weights_do_not_use_test_labels(tmp_path):
    root, prepared = fixture_files(tmp_path)
    records = [r for r in read_json(prepared/'index.json') if r['id'] == 'fit']
    first, _ = prepare_weights(records, prepared, load_input, 'span_equal', 0)
    test = prepared/'graphs/test/test.npz'
    with np.load(test) as saved:
        arrays = {k:saved[k] for k in saved.files}
    arrays['gold'][:] = False
    np.savez_compressed(test, **arrays)
    second, _ = prepare_weights(records, prepared, load_input, 'span_equal', 0)
    np.testing.assert_array_equal(first['fit'], second['fit'])


def test_original_node_and_message_forward_parity():
    graph, _ = graph_sample()
    net = model()
    expected = net(graph, ablation='no_graph')[2:]
    torch.testing.assert_close(expected, score_record(net, graph['x'][2:], 'node_only'))
    torch.testing.assert_close(net(graph)[2:], score_record(net, graph, 'charm_in'))


def test_actual_observation_and_fixed_epoch_training_clis(tmp_path):
    root, prepared = fixture_files(tmp_path)
    shutil.copytree(root/'charm_in', root/'node_only')
    before = {p:(p.read_bytes(),p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    common = ['--mode','continuity','--root',str(root),'--prepared',str(prepared),'--device','cpu',
              '--models','node_only','charm_in','--bootstrap','0']
    main(common+['--continuity-stage','observe'])
    main(common+['--continuity-stage','train','--continuity-seeds','0','--epochs','2'])
    out = root/'audit_continuity_train'
    assert (out/'paired_seed_deltas.csv').exists()
    assert (out/'continuity_review.tar.gz').exists()
    for variant in ('node_only','charm_in'):
        for scheme in SCHEMES:
            run = out/variant/scheme/'seed_0'
            assert read_json(run/'complete.json')['epochs'] == 2
            assert len(read_json(run/'history.json')) == 2
            assert (run/'threshold.json').exists()
            assert (run/'test/tokens.csv').exists()
    for path, expected in before.items():
        assert (path.read_bytes(),path.stat().st_mtime_ns) == expected
    checkpoint = out/'node_only/token/seed_0/checkpoint.pt'
    stamp = checkpoint.stat().st_mtime_ns
    main(common+['--continuity-stage','train','--continuity-seeds','0','--epochs','2'])
    assert checkpoint.stat().st_mtime_ns == stamp
