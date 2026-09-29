"""Same-position source-key exclusion and independent per-target history branches."""
import torch
from transformers.cache_utils import DynamicCache


def token_mask(prompt_mask, count, device):
    return torch.tensor([list(prompt_mask) + [1] * count], device=device)


@torch.no_grad()
def full_states(model, prompt, answer, prompt_mask):
    tokens = torch.tensor([prompt + answer[:-1]], device=model.device)
    output = model.model(input_ids=tokens,
        attention_mask=token_mask(prompt_mask, len(answer) - 1, model.device), use_cache=True)
    states = output.last_hidden_state[0, len(prompt) - 1:].cpu()
    cache = output.past_key_values
    cache.crop(len(prompt) - 1)
    return states, cache


def repeat_cache(cache, batch):
    return DynamicCache.from_legacy_cache(tuple(
        (layer.keys.repeat(batch, 1, 1, 1), layer.values.repeat(batch, 1, 1, 1))
        for layer in cache.layers))


@torch.no_grad()
def reset_states(model, prompt, answer, prompt_mask, cache, targets, window):
    """All targets in a batch have the same suffix length; retain absolute RoPE IDs."""
    length = min(targets[0], window)
    assert all(min(target, window) == length for target in targets)
    queries = [[prompt[-1]] + answer[target - length:target] for target in targets]
    positions = [[len(prompt) - 1] + list(range(len(prompt) + target - length, len(prompt) + target))
                 for target in targets]
    branch = repeat_cache(cache, len(targets))
    mask = token_mask(prompt_mask, length, model.device).repeat(len(targets), 1)
    output = model.model(input_ids=torch.tensor(queries, device=model.device),
        position_ids=torch.tensor(positions, device=model.device), attention_mask=mask,
        past_key_values=branch, use_cache=True)
    return output.last_hidden_state[:, -1]


def batches(count, window, batch_size):
    # Before the window fills, exact suffix lengths vary; avoid padding artifacts.
    yield from ([target] for target in range(min(window, count)))
    for first in range(window, count, batch_size):
        yield list(range(first, min(first + batch_size, count)))
