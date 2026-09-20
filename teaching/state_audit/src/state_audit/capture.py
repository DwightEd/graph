"""Replay the saved IDs and stream each layer to disk; keep native special-token flow."""

from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .models import input_tensor, layers, readout
from .storage import read_json, start_stage, write_arrays, write_json


def numpy(value):
    return value.detach().float().cpu().numpy()


class LayerCapture:
    """Short-lived hook state for exactly one layer during one replay."""

    def __init__(self, queries, path: Path):
        self.queries = queries
        self.path = path
        self.arrays = {}
        self.checks = {}

    def before_layer(self, module, inputs):
        self.arrays["residual_before"] = numpy(inputs[0][0, self.queries])

    def before_attention(self, module, inputs, kwargs):
        length = kwargs["hidden_states"].shape[1]
        self.head_dim = module.head_dim
        cosine, sine = kwargs["position_embeddings"]
        self.arrays["cosine"], self.arrays["sine"] = numpy(cosine[0]), numpy(sine[0])
        self.arrays["scale"] = np.asarray(module.scaling)
        mask = kwargs["attention_mask"]
        if mask is None:
            allowed = np.arange(length)[None, :] <= self.queries[:, None]
            self.arrays["attention_bias"] = np.where(allowed, 0.0, -np.inf)
        else:
            self.arrays["attention_bias"] = numpy(mask[0, 0, self.queries, :length])

    def capture_qkv(self, name, module, inputs, output):
        states = output[0].reshape(output.shape[1], -1, self.head_dim).transpose(0, 1)
        self.arrays[name] = numpy(states[:, self.queries] if name == "query" else states)

    def before_projection(self, module, inputs):
        self.arrays["head_readout"] = numpy(inputs[0][0, self.queries])

    def after_attention(self, module, inputs, output):
        attention = numpy(output[1][0, :, self.queries, :])
        heads, count, _ = attention.shape
        readout_values = self.arrays["head_readout"].reshape(count, heads, module.head_dim)
        self.arrays.update(
            attention=attention,
            head_readout=readout_values,
            attention_write=numpy(output[0][0, self.queries]),
        )
        values = np.repeat(self.arrays["value"], heads // len(self.arrays["value"]), axis=0)
        reconstructed = np.einsum("hts,hsd->thd", attention, values)
        np.testing.assert_allclose(
            reconstructed, readout_values, atol=2e-3, rtol=2e-2, equal_nan=False
        )
        self.checks["head_readout_max_error"] = float(np.max(abs(reconstructed - readout_values)))
        weights = numpy(module.o_proj.weight)
        projected = readout_values.reshape(count, -1) @ weights.T
        if module.o_proj.bias is not None:
            projected += numpy(module.o_proj.bias)
        native = self.arrays["attention_write"]
        np.testing.assert_allclose(projected, native, atol=2e-3, rtol=2e-2, equal_nan=False)
        self.checks["attention_write_max_error"] = float(np.max(abs(projected - native)))

    def after_layer(self, module, inputs, output):
        self.arrays["residual_after"] = numpy(output[0, self.queries])
        self.arrays["queries"] = self.queries
        write_arrays(self.path, **self.arrays)
        self.arrays.clear()


@contextmanager
def capture_hooks(model, selected: list[int], queries, directory: Path):
    handles, records = [], {}
    try:
        for index in selected:
            layer = layers(model)[index]
            record = LayerCapture(queries, directory / f"layer_{index:03d}.npz")
            records[index] = record
            handles.append(layer.register_forward_pre_hook(record.before_layer))
            handles.append(
                layer.self_attn.register_forward_pre_hook(record.before_attention, with_kwargs=True)
            )
            for name, projection in (
                ("query", layer.self_attn.q_proj),
                ("key", layer.self_attn.k_proj),
                ("value", layer.self_attn.v_proj),
            ):
                handles.append(projection.register_forward_hook(partial(record.capture_qkv, name)))
            handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(record.before_projection)
            )
            handles.append(layer.self_attn.register_forward_hook(record.after_attention))
            handles.append(layer.register_forward_hook(record.after_layer))
        yield records
    finally:
        for handle in handles:
            handle.remove()


def capture_sample(model, answer: dict, directory: Path, selected: list[int]) -> dict:
    prompt_length = answer["prompt_length"]
    queries = np.arange(prompt_length - 1, len(answer["token_ids"]) - 1)
    ids = input_tensor(model, answer["token_ids"])
    with torch.inference_mode(), capture_hooks(model, selected, queries, directory) as records:
        # The final response token is a target only; it is never an input to its own query.
        output = model.model(input_ids=ids[:, :-1], use_cache=False, return_dict=True)
        hidden = output.last_hidden_state[0, queries]
        logps, entropies = [], []
        for start in range(0, len(queries), 32):
            logp, entropy = readout(
                model,
                hidden[start : start + 32],
                ids[0, prompt_length + start : prompt_length + start + 32],
            )
            logps.append(numpy(logp))
            entropies.append(numpy(entropy))
    write_arrays(
        directory / "readout.npz",
        queries=queries,
        final_hidden=numpy(hidden),
        target_logp=np.concatenate(logps),
        logit_entropy=np.concatenate(entropies),
    )
    result = dict(
        complete=True,
        layers=selected,
        response_tokens=len(queries),
        checks={str(index): record.checks for index, record in records.items()},
    )
    write_json(directory / "complete.json", result)
    return result


def capture_run(model, root: Path, selected: list[int] | None, resume: bool = False):
    selected = list(range(len(layers(model)))) if selected is None else sorted(set(selected))
    if not selected or min(selected) < 0 or max(selected) >= len(layers(model)):
        raise ValueError("Selected layers are outside this model")
    settings = dict(schema_version=1, layers=selected, attention="full", storage_dtype="float32")
    start_stage(root / "capture.json", settings, resume)
    for index in selected:
        projection = layers(model)[index].self_attn.o_proj
        bias = np.zeros(projection.out_features, dtype=np.float32)
        if projection.bias is not None:
            bias = numpy(projection.bias)
        write_arrays(
            root / "weights" / f"layer_{index:03d}.npz",
            output_projection=numpy(projection.weight),
            bias=bias,
        )
    for sample in tqdm(read_json(root / "run.json")["samples"], desc="state replay"):
        directory = root / "samples" / f"{sample['index']:06d}"
        if not (resume and (directory / "trace" / "complete.json").exists()):
            capture_sample(
                model, read_json(directory / "answer.json"), directory / "trace", selected
            )
