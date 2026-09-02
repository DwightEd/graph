import numpy as np
import torch

from experiments.evidence_route_state.data import read_route_sample
from experiments.evidence_route_state.detector import StickyRouteHMM
from experiments.evidence_route_state.lineage import propagate_lineage
from experiments.evidence_route_state.state import build_route_state, route_observation

from .helpers import route_row


class CharacterTokenizer:
    def apply_chat_template(
        self, messages, *, tokenize: bool, add_generation_prompt: bool
    ):
        assert not tokenize
        assert add_generation_prompt
        return messages[-1]["content"]

    def __call__(self, text, *, add_special_tokens: bool, return_offsets_mapping: bool):
        assert not add_special_tokens
        assert return_offsets_mapping
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


class FakeAttention:
    def __init__(self, prompt_length: int):
        self.token_ids = torch.cat(
            (torch.arange(prompt_length), torch.tensor([101, 102, 103]))
        )
        self.response_idx = prompt_length


class PoisonLabelSample:
    """A cache sample whose label raises if the method tries to inspect it."""

    def __init__(self, hallucinated: bool, prompt_length: int):
        self.sample_id = "sample"
        self.source_id = "source"
        self.split = "test"
        self.generator_model = "generator"
        self._hallucinated = hallucinated
        self._attention = FakeAttention(prompt_length)
        self.release_calls = 0

    @property
    def hallucination_labels(self):
        raise AssertionError("labels were opened before post-hoc evaluation")

    def attention(self):
        return self._attention

    def release_attention(self):
        self.release_calls += 1


def fixed_detector() -> StickyRouteHMM:
    model = StickyRouteHMM()
    model.initial_ = np.full(3, 1.0 / 3.0)
    model.transition_ = np.full((3, 3), 0.05)
    np.fill_diagonal(model.transition_, 0.9)
    model.means_ = np.array([[0.1, 0.2], [0.9, 0.0], [0.9, 1.0]])
    model.variances_ = np.full((3, 2), 0.02)
    return model


def run_label_free_route(sample: PoisonLabelSample):
    prompt = "evidence"
    source = {
        "source_id": "source",
        "task_type": "QA",
        "prompt": prompt,
        "source_info": {"passages": f"{prompt}\n"},
    }
    route_sample = read_route_sample(sample, source, CharacterTokenizer())
    first = route_sample.response_start - 1
    queries = (first, first + 1, first + 2)
    rows = []
    for layer in range(2):
        for query in queries:
            if layer == 0 and query == first + 1:
                rows.append(
                    route_row(
                        layer,
                        query,
                        source=(0,),
                        support=(1.0,),
                        residual_support=0.0,
                    )
                )
            elif layer == 1 and query == first + 2:
                rows.append(
                    route_row(
                        layer,
                        query,
                        source=(first + 1,),
                        support=(1.0,),
                        residual_support=0.0,
                    )
                )
            else:
                rows.append(route_row(layer, query))

    lineage = propagate_lineage(
        rows,
        route_sample.token_root_unit_id,
        route_sample.response_start,
        route_sample.prompt_units.evidence_count,
    )
    state = build_route_state(lineage)
    observation = route_observation(
        state.raw_contraction,
        state.takeover,
        state.valid,
    )
    score = fixed_detector().score(observation, state.valid)
    return route_sample, rows, lineage, state, score


def test_flipping_poison_labels_cannot_change_graph_state_or_score():
    correct = PoisonLabelSample(False, len("evidence"))
    hallucinated = PoisonLabelSample(True, len("evidence"))

    correct_result = run_label_free_route(correct)
    hallucinated_result = run_label_free_route(hallucinated)

    correct_sample, correct_rows, correct_lineage, correct_state, correct_score = (
        correct_result
    )
    (
        hallucinated_sample,
        hallucinated_rows,
        hallucinated_lineage,
        hallucinated_state,
        hallucinated_score,
    ) = hallucinated_result
    torch.testing.assert_close(
        correct_sample.token_root_unit_id,
        hallucinated_sample.token_root_unit_id,
        rtol=0,
        atol=0,
    )
    for left, right in zip(correct_rows, hallucinated_rows, strict=True):
        torch.testing.assert_close(left.message, right.message, rtol=0, atol=0)
        torch.testing.assert_close(left.source, right.source, rtol=0, atol=0)
    torch.testing.assert_close(
        correct_lineage.ancestry,
        hallucinated_lineage.ancestry,
        rtol=0,
        atol=0,
    )
    np.testing.assert_array_equal(
        correct_state.raw_contraction,
        hallucinated_state.raw_contraction,
    )
    np.testing.assert_array_equal(correct_state.takeover, hallucinated_state.takeover)
    np.testing.assert_array_equal(correct_score, hallucinated_score)
    assert correct.release_calls == hallucinated.release_calls == 1
