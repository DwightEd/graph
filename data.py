from pathlib import Path

import torch


def load_feature(path):
    """Load either the old formal cache or the new minimal feature format."""
    raw = torch.load(Path(path), map_location="cpu", weights_only=True)

    if "attention_diagonal" in raw and "response_row_ptr" in raw:
        return {
            "sample_id": str(raw.get("sample_id", raw.get("response_id"))),
            "source_id": str(raw["source_id"]),
            "response_idx": int(raw["response_idx"]),
            "token_ids": raw["token_ids"].long(),
            "attention_diagonal": raw["attention_diagonal"],
            "row_ptr": raw["response_row_ptr"].long(),
            "source_index": raw["response_column_indices"].long(),
            "attention_weight": raw["response_values"],
            "attention_floor": float(raw["attention_floor"]),
            **({"hidden_layers": raw["hidden_layers"].long(),
                "hidden_states": raw["hidden_states"]}
               if "hidden_states" in raw else {}),
        }

    return {
        "sample_id": str(raw["sample_id"]),
        "source_id": str(raw["source_id"]),
        "response_idx": int(raw["response_idx"]),
        "token_ids": raw["token_ids"].long(),
        "attention_diagonal": raw["attention_diagonal"],
        "row_ptr": raw["row_ptr"].long(),
        "source_index": raw["source_index"].long(),
        "attention_weight": raw["attention_weight"],
        "attention_floor": float(raw["attention_floor"]),
        **({"hidden_layers": raw["hidden_layers"].long(),
            "hidden_states": raw["hidden_states"]}
           if "hidden_states" in raw else {}),
    }


def save_feature(sample, path):
    """Save only fields used by graph construction or node features."""
    keys = [
        "sample_id", "source_id", "response_idx", "token_ids",
        "attention_diagonal", "row_ptr", "source_index",
        "attention_weight", "attention_floor",
    ]
    if "hidden_states" in sample:
        keys += ["hidden_layers", "hidden_states"]
    torch.save({key: sample[key] for key in keys}, Path(path))


def attention_entries(sample):
    """Expand CSR to readable retained attention entries."""
    diagonal = sample["attention_diagonal"]
    layers, heads, tokens = diagonal.shape
    response_idx = sample["response_idx"]
    response_tokens = tokens - response_idx

    counts = sample["row_ptr"][1:] - sample["row_ptr"][:-1]
    row = torch.repeat_interleave(torch.arange(len(counts)), counts)
    channel = row // response_tokens

    return {
        "row": row,
        "layer": channel // heads,
        "head": channel % heads,
        "source": sample["source_index"].long(),
        "target": response_idx + row % response_tokens,
        "weight": sample["attention_weight"],
    }


def node_features(sample, kind="diagonal", hidden_layer=-1):
    """Choose node attributes without changing the graph topology."""
    token_count = len(sample["token_ids"])

    if kind == "none":
        return torch.ones((token_count, 1), dtype=torch.float32)

    if kind == "diagonal":
        x = sample["attention_diagonal"]
        return x.permute(2, 0, 1).reshape(token_count, -1).float()

    if kind == "hidden":
        layers = sample["hidden_layers"].tolist()
        requested = layers[-1] if hidden_layer == -1 else hidden_layer
        index = layers.index(requested)
        return sample["hidden_states"][index].float()

    raise ValueError("node feature must be: diagonal, hidden, or none")


def save_graph(graph, path):
    torch.save(graph, Path(path))


def load_graph(path):
    return torch.load(Path(path), map_location="cpu", weights_only=True)
