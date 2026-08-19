from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.token_routing_basin.detector import (
    DetectorConfig,
    TokenRoutingDetector,
)
from experiments.token_routing_basin.artifacts import load_reference, save_reference
from experiments.token_routing_basin.routing import (
    CausalRoutingFeatureExtractor,
    RoutingFeatureConfig,
)


class SparseSample:
    def __init__(
        self,
        sample_id,
        source_id,
        rows,
        *,
        prompt_count=4,
        response_count=4,
        layers=2,
        heads=1,
        floor=0.01,
    ):
        self.sample_id = sample_id
        self.source_id = source_id
        self.task_type = "QA"
        self.data_source = "fixture"
        self._attention = SimpleNamespace(
            response_idx=prompt_count,
            num_response_tokens=response_count,
            num_layers=layers,
            num_heads=heads,
            attention_floor=floor,
        )
        self._rows = [row for row in rows if row[2] < response_count]

    def attention(self):
        return self._attention

    def iter_sparse_attention_blocks(self, block_rows=4096):
        del block_rows
        layer, head, query, source, weight = zip(*self._rows)
        yield SimpleNamespace(
            layer=torch.tensor(layer, dtype=torch.long),
            head=torch.tensor(head, dtype=torch.long),
            query=torch.tensor(query, dtype=torch.long),
            source=torch.tensor(source, dtype=torch.long),
            weight=torch.tensor(weight, dtype=torch.float32),
        )

    def release_attention(self):
        return None


class LabelBombDataset:
    def __init__(self, samples, split):
        self._samples = list(samples)
        self.sample_ids = [sample.sample_id for sample in self._samples]
        self.manifest = {
            "split": split,
            "num_layers": 2,
            "num_heads": 1,
            "attention_floor": 0.01,
            "alignment": "post_token_query_at_same_position",
        }

    def __iter__(self):
        return iter(self._samples)

    def prepare_evaluation_labels(self):
        raise AssertionError("fit/score must not open labels")

    def labels(self):
        raise AssertionError("fit/score must not open labels")


def routing_rows(*, concentrated=False, layers=2):
    rows = []
    for layer in range(layers):
        for query in range(4):
            if not concentrated or query == 0:
                for source in range(4):
                    rows.append((layer, 0, query, source, 0.25))
                continue
            rows.extend(
                [
                    (layer, 0, query, 0, 0.70),
                    (layer, 0, query, 1, 0.10),
                    (layer, 0, query, 4 + query - 1, 0.20),
                ]
            )
    return rows


def sample(sample_id, source_id, *, concentrated=False, response_count=4):
    return SparseSample(
        sample_id,
        source_id,
        routing_rows(concentrated=concentrated),
        response_count=response_count,
    )


def column(sequence, name):
    return sequence.values[:, sequence.names.index(name)]


def test_sparse_features_capture_anchor_concentration_and_operator_spectra():
    extractor = CausalRoutingFeatureExtractor(
        RoutingFeatureConfig(window=3, prompt_bins=2, lag_bins=2)
    )
    sequence = extractor.extract(sample("s", "g", concentrated=True))

    assert sequence.names == (
        "prompt_mass_share",
        "prompt_effective_source_fraction",
        "prompt_top1_share",
        "response_effective_source_fraction",
        "response_top1_share",
        "recent_response_share",
        "prompt_anchor_repeat",
        "prompt_anchor_run_fraction",
        "multiplex_route_effective_rank_fraction",
        "multiplex_route_dominant_mode_share",
        "multiplex_prompt_route_effective_rank_fraction",
        "multiplex_prompt_route_dominant_mode_share",
        "multiplex_response_route_effective_rank_fraction",
        "multiplex_response_route_dominant_mode_share",
        "relative_route_effective_rank_fraction",
        "relative_route_dominant_mode_share",
        "relative_route_velocity",
    )
    assert column(sequence, "prompt_top1_share")[1] > column(
        sequence, "prompt_top1_share"
    )[0]
    assert column(sequence, "prompt_effective_source_fraction")[1] < column(
        sequence, "prompt_effective_source_fraction"
    )[0]
    assert column(sequence, "prompt_anchor_repeat")[2] == pytest.approx(1.0)
    assert torch.isfinite(sequence.values).all()
    assert torch.all(
        (column(sequence, "multiplex_route_dominant_mode_share") >= 0)
        & (column(sequence, "multiplex_route_dominant_mode_share") <= 1)
    )


def test_multiplex_spectrum_preserves_head_source_wiring():
    aligned_rows = [
        (0, 0, 0, 0, 0.50),
        (0, 1, 0, 1, 0.50),
        (0, 0, 1, 0, 0.50),
        (0, 1, 1, 1, 0.50),
    ]
    swapped_rows = [
        (0, 0, 0, 0, 0.50),
        (0, 1, 0, 1, 0.50),
        (0, 0, 1, 1, 0.50),
        (0, 1, 1, 0, 0.50),
    ]
    extractor = CausalRoutingFeatureExtractor(RoutingFeatureConfig(window=2))
    aligned = extractor.extract(
        SparseSample(
            "aligned",
            "source-a",
            aligned_rows,
            prompt_count=2,
            response_count=2,
            layers=1,
            heads=2,
        )
    )
    swapped = extractor.extract(
        SparseSample(
            "swapped",
            "source-b",
            swapped_rows,
            prompt_count=2,
            response_count=2,
            layers=1,
            heads=2,
        )
    )

    assert column(aligned, "multiplex_route_effective_rank_fraction")[1] < column(
        swapped, "multiplex_route_effective_rank_fraction"
    )[1]
    assert column(aligned, "multiplex_route_dominant_mode_share")[1] > column(
        swapped, "multiplex_route_dominant_mode_share"
    )[1]
    torch.testing.assert_close(
        column(aligned, "prompt_top1_share"),
        column(swapped, "prompt_top1_share"),
    )


def test_feature_prefix_is_unchanged_when_future_queries_are_appended():
    extractor = CausalRoutingFeatureExtractor(
        RoutingFeatureConfig(window=3, prompt_bins=2, lag_bins=2)
    )
    prefix = extractor.extract(
        sample("same", "new", concentrated=True, response_count=3)
    )
    full = extractor.extract(sample("same", "new", concentrated=True))

    torch.testing.assert_close(prefix.values, full.values[:3])
    torch.testing.assert_close(prefix.controls, full.controls[:3])


def test_floor_only_edges_do_not_create_a_valid_core_route():
    rows = [(0, 0, query, 0, 0.01) for query in range(2)]
    sequence = CausalRoutingFeatureExtractor().extract(
        SparseSample(
            "floor",
            "new",
            rows,
            prompt_count=2,
            response_count=2,
            layers=1,
        )
    )

    assert not sequence.valid.any()
    assert torch.count_nonzero(sequence.values) == 0


def test_invalid_prefix_does_not_reduce_first_valid_route_rank():
    rows = [
        (0, 0, 0, 0, 0.01),
        (0, 0, 1, 0, 0.60),
        (0, 0, 1, 1, 0.40),
    ]
    sequence = CausalRoutingFeatureExtractor(
        RoutingFeatureConfig(window=3)
    ).extract(
        SparseSample(
            "gap",
            "new",
            rows,
            prompt_count=2,
            response_count=2,
            layers=1,
        )
    )

    assert sequence.valid.tolist() == [False, True]
    assert column(sequence, "multiplex_route_effective_rank_fraction")[1] == pytest.approx(
        1.0
    )


def test_detector_fit_and_score_are_label_free_and_prefix_causal():
    train = LabelBombDataset(
        [
            sample("a", "source-a"),
            sample("b", "source-b"),
            sample("c", "source-c"),
            sample("d", "source-d"),
        ],
        "train",
    )
    detector = TokenRoutingDetector(
        DetectorConfig(calibration_fraction=0.5, ridge=1e-3),
        feature_config=RoutingFeatureConfig(
            window=3, prompt_bins=2, lag_bins=2
        ),
    ).fit(train)

    prefix = LabelBombDataset(
        [sample("held-out", "source-z", concentrated=True, response_count=3)],
        "test",
    )
    full = LabelBombDataset(
        [sample("held-out", "source-z", concentrated=True)], "test"
    )
    prefix_scores = detector.score(prefix)
    full_scores = detector.score(full)

    np.testing.assert_allclose(prefix_scores.score, full_scores.score[:3])
    for name in prefix_scores.component_score:
        np.testing.assert_allclose(
            prefix_scores.component_score[name],
            full_scores.component_score[name][:3],
        )
    assert prefix_scores.online_causal_score is True
    assert prefix_scores.alignment == "post_token_query_at_same_position"
    assert np.isnan(prefix_scores.component_score["transition_surprise"][0])
    assert np.isfinite(prefix_scores.score[0])


def test_concentrated_route_raises_commitment_and_its_causal_smoothing():
    train = LabelBombDataset(
        [sample(str(index), f"source-{index}") for index in range(6)], "train"
    )
    detector = TokenRoutingDetector(
        DetectorConfig(calibration_fraction=0.34, smoothing_decay=0.75),
        feature_config=RoutingFeatureConfig(
            window=3, prompt_bins=2, lag_bins=2
        ),
    ).fit(train)
    scores = detector.score(
        LabelBombDataset(
            [sample("shift", "unseen", concentrated=True)], "test"
        )
    )

    assert scores.component_raw["basin_commitment"][-1] > scores.component_raw[
        "basin_commitment"
    ][0]
    assert scores.component_raw["smoothed_commitment"][-1] > scores.component_raw[
        "smoothed_commitment"
    ][0]


def test_reference_round_trip_preserves_token_scores(tmp_path):
    train = LabelBombDataset(
        [sample(str(index), f"source-{index}") for index in range(6)], "train"
    )
    detector = TokenRoutingDetector(
        DetectorConfig(calibration_fraction=0.34),
        feature_config=RoutingFeatureConfig(
            window=3, prompt_bins=2, lag_bins=2
        ),
    ).fit(train)
    test = LabelBombDataset(
        [sample("shift", "unseen", concentrated=True)], "test"
    )
    expected = detector.score(test)

    path = tmp_path / "reference.npz"
    save_reference(detector, path)
    actual = load_reference(path).score(test)

    np.testing.assert_allclose(actual.score, expected.score)
    assert actual.feature_names == expected.feature_names
    assert actual.threshold == pytest.approx(expected.threshold)
