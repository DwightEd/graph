from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from next_iteration.grounded_graph_erasure import erase_capture, measure


class ToyTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert text == " "
        assert add_special_tokens is False
        return [0]


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.Identity()

    def forward(self, input_ids, use_cache=False, return_dict=True):
        assert use_cache is False
        assert return_dict is True
        values = input_ids.float()[..., None]
        hidden = torch.cat([values, values + 10.0], dim=-1)
        output = self.norm(hidden)
        return SimpleNamespace(last_hidden_state=output)


class ToyModel:
    def __init__(self):
        self.model = ToyBackbone()
        self.device = torch.device("cpu")


class ToyAdapter:
    def __init__(self, pointer_mass):
        self.pointer_mass = pointer_mass

    def __call__(self, **kwargs):
        query = kwargs["query_states"]
        source_count = kwargs["source_states"].shape[0]
        pointer = torch.zeros((query.shape[0], source_count), device=query.device)
        pointer[:, 0] = self.pointer_mass
        if source_count > 1:
            pointer[:, 1:] = (1.0 - self.pointer_mass) / (source_count - 1)
        return {"hidden": query, "pointer": pointer, "gate": torch.zeros(query.shape[0]), "residual": torch.zeros_like(query)}


def _packet():
    return {
        "sha256": "packet-sha",
        "input_ids": [101, 11, 12, 13, 21, 22],
        "prompt_record": {"prompt_length": 4},
        "source_offsets": [[0, 0], [0, 1], [1, 2], [0, 0]],
        "query_positions": [3, 4],
        "target_token_ids": [0, 1],
        "nodes": [
            {"type": 0, "available": True, "prompt_token_indices": [1, 2]},
            {"type": 1, "available": True, "prompt_token_indices": [3]},
        ],
        "edges": [],
        "weak_targets": [{"owner_node_index": 0}],
        "pointer_supervision": [{"query_index": 0, "owner_node_indices": [0]}],
    }


def test_erase_capture_replaces_only_owner_source_prompt_tokens_and_recomputes_final_norm_states():
    packet = _packet()

    arrays, receipt = erase_capture(ToyModel(), ToyTokenizer(), packet)

    assert receipt["replacement_token_id"] == 0
    assert receipt["source_owner_indices"] == [0]
    assert receipt["erased_prompt_token_indices"] == [1, 2]
    assert receipt["executed_input_ids"] == [101, 0, 0, 13, 21, 22]
    assert packet["input_ids"] == [101, 11, 12, 13, 21, 22]
    assert receipt["executed_input_ids"][packet["prompt_record"]["prompt_length"] :] == packet["input_ids"][packet["prompt_record"]["prompt_length"] :]
    assert receipt["actual_final_norm_hook_count"] == 1
    assert receipt["actual_observer_forwards"] == 1
    assert arrays["source"].shape == (2, 2)
    assert arrays["query"].shape == (2, 2)
    assert np.allclose(arrays["source"][0], [0.0, 10.0])
    assert np.allclose(arrays["query"][0], [13.0, 23.0])


def test_measure_scores_only_anchor_queries_and_reports_coordinate_pointer_probability():
    packet = _packet()
    arrays = {
        "source": np.ones((2, 2), dtype=np.float32),
        "query": np.array([[3.0, 0.0], [0.0, 3.0]], dtype=np.float32),
    }
    head = torch.eye(2)
    adapters = {"graph": ToyAdapter(0.75), "no_edges": ToyAdapter(0.25)}

    result = measure(adapters, head, packet, arrays)

    assert result["query_indices"] == [0]
    assert len(result["base_logp"]) == 1
    assert result["graph"]["coordinate_pointer_probability"] == [0.75]
    assert result["no_edges"]["coordinate_pointer_probability"] == [0.25]
    assert result["graph"]["adapter_logp_gain"] == [0.0]
