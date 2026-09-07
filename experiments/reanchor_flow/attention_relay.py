"""Bounded inspection of real, layer-ordered attention relays and native writes.

Selection is descriptive and label-free. A two-edge witness is not a complete
circuit or proof that a particular fact survives the intermediate operations.
"""
from __future__ import annotations

import numpy as np
import torch


PATH_COLUMNS = ("source", "write_layer", "write_head", "carrier",
                "read_layer", "read_head", "query")


def select_relays(trace, audit, limit=2):
    """At most one example per carrier; all heads remain in the cohort audit.

    Match a WAAD peak and a FAI peak at the SAME carrier position, with a
    strictly deeper reading layer. Rank only the display budget by the product
    of the two actual attention coefficients; this is not a detector score.
    """
    heads = trace["waad"].shape[1]
    p = int(trace["response_start"])
    candidates = []
    if not limit:
        return np.empty((0, 7), np.int64), np.empty(0)
    for offset in range(audit["waad_peaks"].shape[-1]):
        carrier, slot = p + offset, offset + 1
        writers = np.flatnonzero(audit["waad_peaks"][..., offset])
        readers = np.flatnonzero(audit["fai_peaks"][..., offset])
        if not len(writers) or not len(readers):
            continue
        source = trace["past_source_position"][..., slot].ravel()[writers]
        query = trace["fai_best_query"][..., slot].ravel()[readers]
        legal = ((writers[:, None] // heads < readers[None] // heads)
                 & (source[:, None] >= 0) & (source[:, None] < carrier)
                 & (query[None] > carrier) & (query[None] < len(trace["token_ids"]) - 1))
        strength = (trace["past_source_attention"][..., slot].ravel()[writers, None]
                    * trace["fai_best_attention"][..., slot].ravel()[readers][None])
        strength = np.where(legal, strength, -np.inf)
        i, j = np.unravel_index(strength.argmax(), strength.shape)
        if not np.isfinite(strength[i, j]) or strength[i, j] <= 0:
            continue
        wl, wh = divmod(int(writers[i]), heads)
        rl, rh = divmod(int(readers[j]), heads)
        candidates.append((float(strength[i, j]),
                           (int(source[i]), wl, wh, carrier, rl, rh, int(query[j]))))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    chosen = candidates[:limit]
    return (np.asarray([row for _, row in chosen], np.int64).reshape(-1, 7),
            np.asarray([score for score, _ in chosen]))


class RelayObserver:
    """Record exact states at selected carriers/receivers during the map pass.

    Store ALL head codes before W_O at these positions, full residual/attention/
    MLP vectors, and the post-W_O vectors on the two displayed edges. Signed
    readout uses each node's own observed-next-token versus frozen native runner.
    It is direct-logit accounting, not evidence attribution or a causal effect.
    """

    def __init__(self, model, ids, raw_final, paths, maps=None):
        from .native_trace import FrozenReadout

        self.model, self.maps = model, maps
        self.nodes = np.unique(paths[:, (3, 6)].ravel())
        self.rows = torch.as_tensor(self.nodes, device=ids.device)
        self.readout = FrozenReadout.capture(model, raw_final, ids, self.rows)
        self.edges = np.asarray([edge for s, wl, wh, b, rl, rh, q in paths
                                 for edge in ((wl, wh, s, b), (rl, rh, b, q))], np.int64)
        layers, heads = len(model.model.layers), model.config.num_attention_heads
        width = model.get_input_embeddings().weight.shape[1]
        dim = model.model.layers[0].self_attn.head_dim
        self.heads, self.dim = heads, dim
        nodes = len(self.nodes)
        self.arrays = {
            "node_position": self.nodes, "paths": paths,
            "path_columns": np.asarray(PATH_COLUMNS), "edge_index": self.edges,
            "edge_columns": np.asarray(("layer", "head", "source", "query")),
            "edge_attention": np.empty(len(self.edges), np.float32),
            "edge_message": np.empty((len(self.edges), width), np.float32),
            "residual": np.empty((layers + 1, nodes, width), np.float32),
            "post_attention": np.empty((layers, nodes, width), np.float32),
            "attention_write": np.empty((layers, nodes, width), np.float32),
            "mlp_write": np.empty((layers, nodes, width), np.float32),
            "head_code": np.empty((layers, heads, nodes, dim), np.float32),
            "head_margin": np.empty((layers, heads, nodes), np.float32),
            "readout_direction": self.readout.direction.cpu().numpy(),
            **self.readout.arrays,
        }

    def observe_layer_input(self, layer, hidden):
        self.input = hidden[0].index_select(0, self.rows)
        self.arrays["residual"][layer] = self.input.float().cpu().numpy()
        weight = self.model.model.layers[layer].self_attn.o_proj.weight
        blocks = weight.reshape(-1, self.heads, self.dim).permute(1, 2, 0)
        self.projected_readout = torch.einsum(
            "hdk,nk->hnd", blocks.float(), self.readout.direction)

    def observe_chunk(self, layer, begin, probability, value, output_weight):
        if self.maps is not None:
            self.maps.observe_chunk(layer, begin, probability, value, output_weight)
        for edge, (ell, head, source, query) in enumerate(self.edges):
            if ell != layer or not begin <= query < begin + probability.shape[2]:
                continue
            a = probability[0, head, query - begin, source].float()
            block = output_weight[:, head * self.dim:(head + 1) * self.dim].float()
            message = block @ (a * value[0, head, source].float())
            self.arrays["edge_attention"][edge] = float(a)
            self.arrays["edge_message"][edge] = message.cpu().numpy()

    def observe_head_output(self, layer, begin, output):
        slots = np.flatnonzero((self.nodes >= begin) & (self.nodes < begin + output.shape[2]))
        if not len(slots):
            return
        index = torch.as_tensor(self.nodes[slots] - begin, device=output.device)
        code = output[0].index_select(1, index).float()
        self.arrays["head_code"][layer][:, slots] = code.cpu().numpy()
        self.arrays["head_margin"][layer][:, slots] = (
            code * self.projected_readout[:, slots]).sum(-1).cpu().numpy()

    def observe_attention_write(self, layer, write):
        selected = write[0].index_select(0, self.rows)
        self.post = self.input + selected  # preserve native dtype's rounding
        self.arrays["attention_write"][layer] = selected.float().cpu().numpy()
        self.arrays["post_attention"][layer] = self.post.float().cpu().numpy()

    def observe_mlp_write(self, layer, write):
        selected = write[0].index_select(0, self.rows)
        self.arrays["mlp_write"][layer] = selected.float().cpu().numpy()
        self.arrays["residual"][layer + 1] = (self.post + selected).float().cpu().numpy()

    def finish(self):
        a = self.arrays
        direction = a["readout_direction"].astype(np.float64)
        for name, field in (("stage_margin", "residual"), ("attention_margin", "attention_write"),
                            ("mlp_margin", "mlp_write")):
            a[name] = np.einsum("lnd,nd->ln", a[field], direction)
        # Expose finite-precision differences; do not assign them to an edge.
        a["attention_rounding_margin"] = a["attention_margin"] - a["head_margin"].sum(1)
        a["residual_rounding_margin"] = (np.diff(a["stage_margin"], axis=0)
                                          - a["attention_margin"] - a["mlp_margin"])
        a["logit_rounding_margin"] = a["final_margin"] - a["readout_bias"] - a["stage_margin"][-1]
        return {"relay_" + key: value for key, value in a.items()}


def relay_examples(trace):
    text = trace["token_text"]
    examples = []
    for path in trace.get("relay_paths", []):
        item = dict(zip(PATH_COLUMNS, map(int, path)))
        item["tokens"] = {name: str(text[item[name]]) for name in ("source", "carrier", "query")}
        item["context"] = {name: "".join(map(str, text[max(0, item[name]-8):item[name]+9]))
                           for name in ("source", "carrier", "query")}
        source = item["source"]
        item["source_region"] = ("response_history" if source >= int(trace["response_start"])
                                 else "annotated_prompt_source" if trace.get("evidence_mask", np.zeros(len(text), bool))[source]
                                 else "other_prompt")
        item["prediction_position"] = item["query"] + 1
        item["predicted_token"] = str(text[item["query"] + 1])
        item["status"] = "two-hop native attention witness; factual mediation untested"
        examples.append(item)
    return examples
