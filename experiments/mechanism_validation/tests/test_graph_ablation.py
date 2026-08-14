"""Contracts for trace-level graph interventions."""

import pytest
import torch

from cache import AttentionSample
from experiments.mechanism_validation.graph_ablation import (
    RP,
    RR,
    apply_trace_variant,
    extract_traces,
    fixed_graph_descriptors,
)


def sample() -> AttentionSample:
    # Two channels, two response targets.  Rows are channel-major CSR rows.
    diagonal = torch.zeros((1, 2, 5), dtype=torch.float16)
    return AttentionSample(
        "id", "source", 3, torch.arange(5, dtype=torch.int32), diagonal,
        torch.tensor([0, 2, 4, 6, 8], dtype=torch.int32),
        torch.tensor([0, 2, 1, 3, 0, 1, 2, 3], dtype=torch.int32),
        torch.tensor([.2, .3, .4, .1, .5, .1, .2, .3], dtype=torch.float16), .01,
    )


def rows(trace):
    key = (trace.target * 100 + trace.channel * 10 + trace.relation).long()
    return key, torch.bincount(key, weights=trace.value.float(), minlength=20)


def test_extracts_response_relative_targets_and_absolute_sources_from_csr():
    trace = extract_traces(sample())

    assert trace.target.tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
    assert trace.channel.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert trace.relation.tolist() == [RP, RP, RP, RR, RP, RP, RP, RR]
    assert trace.source.tolist() == [0, 2, 1, 3, 0, 1, 2, 3]
    assert torch.allclose(trace.value, torch.tensor([.2, .3, .4, .1, .5, .1, .2, .3]), atol=2e-4)


@pytest.mark.parametrize("variant", ["exact", "unit_mass", "uniform_on_support", "weight_shuffle", "source_rewire", "rp_only", "rr_only", "no_edges"])
def test_variants_are_audited_and_keep_causality(variant):
    trace = extract_traces(sample())
    changed, audit = apply_trace_variant(trace, variant, response_idx=3, seed=11)

    assert 0 <= audit["changed_fraction"] <= 1
    assert torch.all(changed.source < changed.target + 3)
    assert torch.all((changed.relation == RP) == (changed.source < 3))
    if variant == "no_edges":
        assert changed.value.numel() == 0
    if variant == "rp_only":
        assert torch.all(changed.relation == RP)
    if variant == "rr_only":
        assert torch.all(changed.relation == RR)


def test_normalization_and_shuffle_variants_preserve_their_row_invariants():
    trace = extract_traces(sample())
    unit, _ = apply_trace_variant(trace, "unit_mass", response_idx=3)
    uniform, _ = apply_trace_variant(trace, "uniform_on_support", response_idx=3)
    shuffled, _ = apply_trace_variant(trace, "weight_shuffle", response_idx=3, seed=9)
    original_keys, original_mass = rows(trace)
    uniform_keys, uniform_mass = rows(uniform)

    unit_keys, unit_mass = rows(unit)
    assert torch.allclose(unit_mass[unit_keys.unique()], torch.ones_like(unit_keys.unique(), dtype=torch.float))
    assert torch.allclose(uniform_mass, original_mass)
    for key in original_keys.unique():
        original = trace.value[original_keys == key].sort().values
        permuted = shuffled.value[rows(shuffled)[0] == key].sort().values
        assert torch.equal(original, permuted)
        values = uniform.value[uniform_keys == key]
        assert torch.allclose(values, torch.full_like(values, values.sum() / values.numel()))


def test_rewire_shares_mapping_across_channels_and_changes_valid_endpoints():
    trace = extract_traces(sample())
    rewired, audit = apply_trace_variant(trace, "source_rewire", response_idx=3, seed=4)

    for target in trace.target.unique():
        for relation in (RP, RR):
            original = trace[(trace.target == target) & (trace.relation == relation)]
            changed = rewired[(rewired.target == target) & (rewired.relation == relation)]
            if not original.source.numel():
                continue
            mapping = dict(zip(original.source.tolist(), changed.source.tolist()))
            assert len(mapping) == original.source.unique().numel()
            assert len(set(mapping.values())) == len(mapping)
    assert audit["changed_fraction"] > 0


def test_descriptors_have_fixed_shape_and_source_free_only_zeroes_source_columns():
    trace = extract_traces(sample())
    node_features = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [2., 1.], [1., 2.]])
    full = fixed_graph_descriptors(trace, node_features, 3, 1, 2)
    free = fixed_graph_descriptors(trace, node_features, 3, 1, 2, source_free=True)
    empty, _ = apply_trace_variant(trace, "no_edges", response_idx=3)
    no_edges = fixed_graph_descriptors(empty, node_features, 3, 1, 2)

    assert full.features.shape[0] == 2
    assert full.features.shape == free.features.shape == no_edges.features.shape
    assert torch.isfinite(full.features).all() and torch.isfinite(no_edges.features).all()
    assert full.feature_names == free.feature_names
    assert full.source_aware.shape == (full.features.shape[1],)
    assert torch.equal(full.features[:, ~full.source_aware], free.features[:, ~full.source_aware])
    assert torch.equal(free.features[:, full.source_aware], torch.zeros_like(free.features[:, full.source_aware]))


def test_rr_predecessor_aggregate_uses_response_sources_not_prompt_rows():
    trace = extract_traces(sample())
    nodes = torch.tensor([[10.], [20.], [30.], [2.], [8.]])
    descriptors = fixed_graph_descriptors(trace, nodes, 3, 1, 2)
    column = descriptors.feature_names.index("rr:predecessor_weighted_0:global_mean")

    assert torch.allclose(descriptors.features[:, column], torch.tensor([0., 2.]))
