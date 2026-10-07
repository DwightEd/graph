"""Native key-channel necessity and rescue, conditional on one attention layer."""
import numpy as np
import torch

from .grounded_projection import observed_attention, replay_queries


def channel_messages(attention, values, prompt, positions, source_mask, heads):
    """Source, history and remote history remain full [token, head, D] messages."""
    keys = torch.arange(len(values), device=values.device)
    source = torch.zeros(len(values), dtype=torch.bool, device=values.device)
    source[:prompt] = torch.tensor(source_mask, device=values.device)
    history = (keys[None] >= prompt) & (keys[None] <= positions[:, None])
    remote = history & (keys[None] < positions[:, None])
    masks = dict(source=source[None].expand(len(positions), -1), history=history, remote=remote)
    messages = {name: torch.einsum('thk,khd->thd', attention * mask[:, None], values)
                for name, mask in masks.items()}
    messages['both'] = messages['source'] + messages['history']
    selected = torch.zeros(attention.shape[1], device=values.device)
    selected[heads] = 1
    for name in ('source', 'history'):
        messages['important_' + name] = messages[name] * selected[None, :, None]
    masses = {name: (attention * mask[:, None]).sum(-1).cpu().numpy() for name, mask in masks.items()}
    return messages, masses


def intervention_worlds(messages, seed):
    names, directions = ['identity'], [torch.zeros_like(messages['source'])]
    generator = torch.Generator(device=messages['source'].device).manual_seed(seed)
    for name, message in messages.items():
        for dose in (1., .25):
            names.append(f'drop_{name}_{dose:g}')
            directions.append(-dose * message)
        random = torch.randn(message.shape, generator=generator, device=message.device)
        random *= message.norm(dim=-1, keepdim=True) / random.norm(dim=-1, keepdim=True)
        names.append('random_' + name)
        directions.append(random)
    return names, torch.stack(directions)


@torch.no_grad()
def choice_readout(model, hidden, actual, rival):
    """Same observed token and baseline alternative in every counterfactual world."""
    logits = model.lm_head(hidden).float()
    rows = torch.arange(len(hidden), device=hidden.device)
    margin = logits[rows, actual] - logits[rows, rival]
    logp = logits[rows, actual] - logits.logsumexp(-1)
    return dict(margin=margin.cpu().numpy(), logp=logp.cpu().numpy(),
                top=logits.argmax(-1).cpu().numpy())


@torch.no_grad()
def measure_layer(adapter, records, layer, rows, prompt, cosine, sine, source_mask, heads, actual, rival):
    full_positions = torch.arange(prompt - 1, len(records[0]['key']), device=actual.device)
    attention, values = observed_attention(adapter, layer, records[layer], full_positions, cosine, sine)
    positions = rows + prompt - 1
    messages, masses = channel_messages(attention[rows], values, prompt, positions, source_mask, heads)
    names, delta = intervention_worlds(messages, 73 + layer)
    count, targets = delta.shape[:2]
    repeated_rows = rows.repeat(count)
    hidden = replay_queries(adapter, records, layer, repeated_rows, positions.repeat(count),
                            delta.flatten(0, 1), cosine, sine)
    scores = choice_readout(adapter.native, hidden, actual.repeat(count), rival.repeat(count))
    scores = {name: value.reshape(count, targets) for name, value in scores.items()}
    self_diagonal = attention[rows + 1, :, positions + 1].cpu().numpy()
    return dict(names=names, scores=scores, masses=masses, self_diagonal=self_diagonal,
                message_norms={name: value.norm(dim=-1).cpu().numpy() for name, value in messages.items()}), delta
