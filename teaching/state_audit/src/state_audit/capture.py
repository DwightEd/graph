"""Select representations, observe native execution, stream each layer to disk."""

from collections import defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .model.sites import AXES, FULL_SEQUENCE_SITES, GLOBAL_SITES, LAYER_SITES
from .state import LayerState
from .storage import read_json, start_stage, write_arrays, write_json


@dataclass(frozen=True)
class CaptureSpec:
    layers: tuple[int, ...] | None = None
    representations: tuple[str, ...] = (*LAYER_SITES, *GLOBAL_SITES)
    scope: str = "response"

    def __post_init__(self):
        if self.layers is not None and any(index < 0 for index in self.layers):
            raise ValueError("Capture layer numbers must be nonnegative")
        for name in self.representations:
            AXES[name]

    def selected_layers(self, model):
        selected = tuple(range(len(model.layers))) if self.layers is None else self.layers
        return tuple(dict.fromkeys(selected))


def numpy(value):
    return value.detach().float().cpu().numpy().copy()


@contextmanager
def capture_targets(model, targets):
    """Observe named Targets in memory: result[name][layer], with None for global sites.

    Install intervention hooks before this context to observe the changed execution.
    Only selected coordinates are copied; layer/head/position identity stays in Target.
    """
    captured = {name: {} for name in targets}
    sites = defaultdict(list)
    for name, target in targets.items():
        for layer in target.layers or (None,):
            sites[target.representation, layer].append((name, target))

    def observe(value, layer, selections):
        for name, target in selections:
            captured[name][layer] = numpy(value[target.indices(value)])
        return value

    with ExitStack() as stack:
        for (representation, layer), selections in sites.items():
            observer = partial(observe, layer=layer, selections=selections)
            stack.enter_context(model.bind(representation, layer, observer))
        yield captured


class LayerCapture:
    def __init__(self, layer: int, positions, path: Path):
        self.state = LayerState(layer, positions, {})
        self.path = path

    def observe(self, name, value):
        if name in FULL_SEQUENCE_SITES:
            observed = value
        else:
            axis = AXES[name].index("position")
            positions = torch.as_tensor(self.state.positions, device=value.device)
            observed = value.index_select(axis, positions)
        self.state.tensors[name] = numpy(observed)
        return value

    def attention_metadata(self, module, inputs, kwargs):
        cosine, sine = kwargs["position_embeddings"]
        length = kwargs["hidden_states"].shape[1]
        mask = kwargs["attention_mask"]
        positions = self.state.positions
        if mask is None:
            allowed = np.arange(length)[None, :] <= positions[:, None]
            bias = np.where(allowed, 0.0, -np.inf)
        else:
            bias = numpy(mask[0, 0, positions, :length])
        self.state.tensors.update(
            cosine=numpy(cosine[0]),
            sine=numpy(sine[0]),
            scale=np.asarray(module.scaling),
            attention_bias=bias,
        )

    def finish(self, module, inputs, output):
        self.state.save(self.path)
        self.state.tensors.clear()


@contextmanager
def capture_hooks(model, spec: CaptureSpec, positions, directory: Path):
    global_states = {}

    def observe_global(name, value):
        global_states[name] = numpy(value[positions])
        return value

    with ExitStack() as stack:
        for layer in spec.selected_layers(model):
            record = LayerCapture(layer, positions, directory / f"layer_{layer:03d}.npz")
            for name in spec.representations:
                if name not in GLOBAL_SITES:
                    observer = partial(record.observe, name)
                    stack.enter_context(model.bind(name, layer, observer))
            if set(spec.representations) & {"attention", "query", "key"}:
                handle = model.layers[layer].self_attn.register_forward_pre_hook(
                    record.attention_metadata, with_kwargs=True
                )
                stack.callback(handle.remove)
            handle = model.layers[layer].register_forward_hook(record.finish)
            stack.callback(handle.remove)
        for name in spec.representations:
            if name in GLOBAL_SITES:
                stack.enter_context(model.bind(name, None, partial(observe_global, name)))
        yield global_states


def capture_sample(model, answer: dict, directory: Path, spec: CaptureSpec) -> dict:
    (directory / "complete.json").unlink(missing_ok=True)
    prompt_length = answer["prompt_length"]
    positions = np.arange(prompt_length - 1, len(answer["token_ids"]) - 1)
    scopes = {"response": positions, "all": np.arange(len(answer["token_ids"]) - 1)}
    state_positions = scopes[spec.scope]
    targets = model.input_ids(answer["response_ids"])[0]
    logps, entropies = [], []
    with torch.inference_mode(), capture_hooks(model, spec, state_positions, directory) as arrays:
        hidden = model.forward(answer["token_ids"][:-1])[positions]
        for start in range(0, len(positions), 32):
            logp, entropy = model.score(hidden[start : start + 32], targets[start : start + 32])
            logps.append(numpy(logp))
            entropies.append(numpy(entropy))
    arrays.update(target_logp=np.concatenate(logps), logit_entropy=np.concatenate(entropies))
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("Nonfinite captured states/readout; trace is incomplete")
    write_arrays(
        directory / "readout.npz", queries=positions, state_positions=state_positions, **arrays
    )
    result = dict(
        schema_version=2,
        complete=True,
        layers=list(spec.selected_layers(model)),
        representations=list(spec.representations),
        scope=spec.scope,
        response_tokens=len(positions),
    )
    write_json(directory / "complete.json", result)
    return result


def capture_run(model, root: Path, spec: CaptureSpec, resume: bool = False):
    selected = spec.selected_layers(model)
    settings = dict(schema_version=2, **asdict(spec), storage_dtype="float32")
    settings["layers"] = list(selected)
    start_stage(root / "capture.json", settings, resume)
    for index in selected:
        projection = model.layers[index].self_attn.o_proj
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
        if resume and (directory / "trace" / "complete.json").exists():
            continue
        capture_sample(model, read_json(directory / "answer.json"), directory / "trace", spec)
