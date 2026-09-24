"""Freeze one shared edge set, then measure individual, joint and matched random cuts."""

import numpy as np
import torch
from tqdm import tqdm

from state_audit.functional_capture import log_probability
from state_audit.message_gates import message_gates
from state_audit.model.replay import attention_backend

from .selection import measure_candidates, select_carriers


@torch.no_grad()
def deleted_logp(model, prompt, answer, unit, edges=(), strength=1.0):
    """Delete writes, not text; later layers recompute from the changed residual."""
    device = model.native.device
    queries = torch.arange(len(prompt) + unit["start"] - 1,
                           len(prompt) + unit["stop"] - 1, device=device)
    absolute = [(int(layer), int(head), len(prompt) + int(key)) for layer, head, key in edges]
    with attention_backend(model, "sdpa"), message_gates(model, queries, absolute, strength):
        hidden = model.forward(prompt + answer[:unit["stop"] - 1])
        logits = model.native.lm_head(hidden[queries])
    return log_probability(logits, torch.tensor(answer[unit["start"]:unit["stop"]],
                                               device=device)).cpu().numpy()


def measure_deletions(model, prompt, answer, unit, selected, baseline, description):
    edges, sham = selected["edges"], selected["sham_edges"]
    if not len(edges):
        return dict(single=np.empty((0, len(baseline)), dtype=np.float32),
                    joint=baseline.copy(), sham=baseline.copy())
    single = [deleted_logp(model, prompt, answer, unit, [edge])
              for edge in tqdm(edges, desc=description, leave=False)]
    joint = single[0].copy() if len(edges) == 1 else deleted_logp(model, prompt, answer, unit, edges)
    return dict(single=np.stack(single), joint=joint,
                sham=deleted_logp(model, prompt, answer, unit, sham))


def capture_unit(model, views, unit, top_k, seed):
    prompts = [views[f"prompt_{name}"] for name in ("with_source", "without_source")]
    answer = views["answer_ids"]
    if max(map(len, prompts)) + unit["stop"] - 1 > model.native.config.max_position_embeddings:
        raise ValueError("Observed prefix exceeds model context; no truncation")
    conditions = [measure_candidates(model, prompt, answer, unit) for prompt in prompts]
    selected = select_carriers(conditions, top_k, seed)
    cuts = [measure_deletions(model, prompt, answer, unit, selected, condition["logp"],
                             f"unit {unit['start']} condition {index} cuts")
            for index, (prompt, condition) in enumerate(zip(prompts, conditions))]
    return dict(**selected, keep=np.stack([condition["logp"] for condition in conditions]),
                single=np.stack([cut["single"] for cut in cuts]),
                joint=np.stack([cut["joint"] for cut in cuts]),
                sham=np.stack([cut["sham"] for cut in cuts]),
                reconstruction_error=np.stack([condition["reconstruction_error"] for condition in conditions]),
                target=np.arange(unit["start"], unit["stop"]),
                token_id=np.asarray(answer[unit["start"]:unit["stop"]]),
                queries=np.stack([len(prompt) + np.arange(unit["start"], unit["stop"]) - 1
                                  for prompt in prompts]),
                edge_keys=np.stack([len(prompt) + selected["edges"][:, 2] for prompt in prompts]),
                prompt_lengths=np.asarray(list(map(len, prompts))))
