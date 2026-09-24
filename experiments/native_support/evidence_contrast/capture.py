"""Native SDPA teacher forcing with one reusable, bounded-prefix KV cache."""

from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm

from state_audit.model.replay import attention_backend, prefill_cache


@torch.no_grad()
def score_continuation(model, cache, last_prompt_token, targets, chunk_size):
    """Each input query predicts the corresponding target, including the first."""
    queries = [last_prompt_token, *targets[:-1]]
    values = []
    with attention_backend(model, "sdpa"):
        for start in range(0, len(targets), chunk_size):
            stop = min(start + chunk_size, len(targets))
            output = model.native.model(
                input_ids=model.input_ids(queries[start:stop]), past_key_values=cache,
                use_cache=True, return_dict=True,
            )
            logits = model.native.lm_head(output.last_hidden_state[0]).float()
            target_ids = model.input_ids(targets[start:stop])[0]
            observed = logits.gather(-1, target_ids[:, None])[:, 0]
            values.append((observed - logits.logsumexp(-1)).cpu().numpy())
    return np.concatenate(values)


def collect_condition(model, prefix, answer, units, prefill_chunk, query_chunk, description):
    """Crop before every local branch; no branch can inherit another's answer KV."""
    if len(prefix) + len(answer) - 1 > model.native.config.max_position_embeddings:
        raise ValueError(f"{description}: original input exceeds model context; no truncation")
    device = model.native.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    cache = prefill_cache(model, prefix[:-1], prefill_chunk)
    full = score_continuation(model, cache, prefix[-1], answer, query_chunk)
    local = np.empty_like(full)
    for unit in tqdm(units, desc=description, leave=False):
        cache.crop(len(prefix) - 1)
        start, stop = unit["start"], unit["stop"]
        local[start:stop] = score_continuation(
            model, cache, prefix[-1], answer[start:stop], query_chunk,
        )
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return dict(full=full, local=local, token_id=np.asarray(answer),
                seconds=np.asarray(perf_counter() - started), peak_cuda_bytes=np.asarray(peak))
