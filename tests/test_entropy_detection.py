import numpy as np

from experiments.entropy_detection.features import uncertainty_features, matrices


def test_entropy_peak_stops_at_unit_and_missing_token():
    features = uncertainty_features([9., 1., 2., 3., .5], [1.]*5,
        np.array([0, 0, 1, 1, 1]), np.array([0, 1, 2, 4, 5]))
    np.testing.assert_allclose(features[:, 6], [9, 9, 2, 3, 3])
    np.testing.assert_allclose(features[:, 8], [9, 1, 2, 3, .5])
    assert features[0, 4] == 0 and features[3, 4] == 0


def test_features_do_not_use_labels_or_cross_answers():
    pack = dict(target=np.array([0, 1, 0, 1]), unit_index=np.zeros(4),
        observations=np.zeros((4, 11)), context=np.zeros((4, 6)), labels=np.array([1, 1, 0, 0]))
    pack['observations'][:, 6] = [8, 2, 1, .5]
    records = [dict(packed_start=0, packed_stop=2), dict(packed_start=2, packed_stop=4)]
    first = matrices(pack, records)
    pack['labels'][:] = 1-pack['labels']
    second = matrices(pack, records)
    np.testing.assert_array_equal(first['joint'], second['joint'])
    assert first['uncertainty'][2, 6] == 1.


def test_native_layer_alignment_and_final_norm():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from experiments.entropy_detection.layers import layer_statistics
    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=37, hidden_size=16,
        intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2)).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    positions, targets = torch.tensor([2, 3]), torch.tensor([4, 5])
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        values = layer_statistics(model, out.hidden_states, positions, targets, layers=(1, 2))
        lp = out.logits[0, positions].log_softmax(-1)
        np.testing.assert_allclose(values[:, 4], -(lp.exp()*lp).sum(-1), atol=1e-6)
        np.testing.assert_allclose(values[:, 5], -lp[torch.arange(2), targets], atol=1e-6)
        np.testing.assert_allclose(values[:, 6], 0, atol=1e-7)


def test_conservative_tail_ties_and_answer_span_boundaries():
    from experiments.entropy_detection.fusion import tail_score,span_recall
    score=tail_score(np.array([0.,1.,1.,2.]),np.array([1.,3.]))
    np.testing.assert_allclose(score,-np.log([4/5,1/5]))
    labels=np.array([1,1,1,1])
    onset=np.array([1,0,1,0],dtype=bool)
    answer=np.array([0,0,1,1])
    alarm=np.array([0,0,1,0],dtype=bool)
    assert span_recall(labels,onset,answer,alarm,np.ones(4,dtype=bool))==.5
