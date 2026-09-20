"""Same-layer four-world native message cuts; an actual model rerun in each world."""

from pathlib import Path

import numpy as np
import torch

from .measurements import source_masks
from .models import input_tensor, layers, readout
from .storage import read_arrays, read_json, write_json


def removed_messages(trace: dict, answer: dict, target: int, source: str):
    attention = trace["attention"][:, target]
    masks = source_masks(answer, attention.shape[-1])
    values = np.repeat(trace["value"], len(attention) // len(trace["value"]), axis=0)
    return np.einsum("hs,hsd->hd", attention * masks[source], values)


def run_world(
    model, answer: dict, layer: int, target: int, heads: list[int], messages, dose: float = 1.0
) -> float:
    query = answer["prompt_length"] + target - 1
    ids = input_tensor(model, answer["token_ids"])

    def cut(module, inputs):
        changed = inputs[0].clone()
        view = changed.reshape(*changed.shape[:2], len(messages), messages.shape[-1])
        removed = torch.as_tensor(messages, device=changed.device, dtype=changed.dtype)
        view[0, query, heads] -= dose * removed[heads]
        return (changed,)

    handle = layers(model)[layer].self_attn.o_proj.register_forward_pre_hook(cut)
    try:
        with torch.inference_mode():
            output = model.model(input_ids=ids[:, :-1], use_cache=False, return_dict=True)
            logp, _ = readout(model, output.last_hidden_state[0, query], ids[0, query + 1])
            return float(logp)
    finally:
        handle.remove()


def four_worlds(
    model, root: Path, sample: int, layer: int, target: int, head_a: int, head_b: int, source: str
) -> dict:
    directory = root / "samples" / f"{sample:06d}"
    answer = read_json(directory / "answer.json")
    trace = read_arrays(directory / "trace" / f"layer_{layer:03d}.npz")
    if not 0 <= target < len(answer["response_ids"]):
        raise ValueError("target is outside the response")
    heads = trace["attention"].shape[0]
    if head_a == head_b or not (0 <= head_a < heads and 0 <= head_b < heads):
        raise ValueError("Choose two distinct native query heads")
    messages = removed_messages(trace, answer, target, source)
    selections = dict(
        full=[], without_a=[head_a], without_b=[head_b], without_both=[head_a, head_b]
    )
    worlds = {
        name: run_world(model, answer, layer, target, chosen, messages)
        for name, chosen in selections.items()
    }
    sham = run_world(model, answer, layer, target, [head_a, head_b], messages, dose=0)
    baseline = read_arrays(directory / "trace" / "readout.npz")["target_logp"][target]
    np.testing.assert_allclose([worlds["full"], sham], baseline, atol=2e-3, rtol=2e-3)
    result = interaction_result(worlds)
    result.update(
        sample=sample,
        layer=layer,
        target=target,
        head_a=head_a,
        head_b=head_b,
        source=source,
        worlds=worlds,
        sham_logp=sham,
        baseline_logp=float(baseline),
        metric="native_actual_target_logp_nats",
        purpose="causal_message_audit",
    )
    path = root / "interventions" / f"s{sample}_l{layer}_t{target}_h{head_a}_{head_b}_{source}.json"
    write_json(path, result)
    return result


def interaction_result(worlds: dict) -> dict:
    with_b = worlds["full"] - worlds["without_a"]
    without_b = worlds["without_b"] - worlds["without_both"]
    return dict(
        effect_a_with_b=with_b, effect_a_without_b=without_b, interaction=with_b - without_b
    )
