"""Full-prefix source-logit intervention: downstream Q/K/V and MLP recompute."""
from contextlib import contextmanager
from types import MethodType

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


def attention_forward(probe, treatment, dose):
    """Single unpadded sequence; change one receiver, keeping softmax competition."""
    def forward(module, hidden_states, position_embeddings, attention_mask, **kwargs):
        shape = (*hidden_states.shape[:-1], -1, module.head_dim)
        query = module.q_proj(hidden_states).view(shape).transpose(1, 2)
        key = module.k_proj(hidden_states).view(shape).transpose(1, 2)
        value = module.v_proj(hidden_states).view(shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, *position_embeddings)
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
        logits = query @ key.transpose(-1, -2) * module.scaling
        count = query.shape[-2]
        future = torch.ones(count, count, dtype=torch.bool, device=query.device).triu(1)
        logits.masked_fill_(future, -torch.inf)
        logits[:, probe['head'], probe['receiver'], treatment['keys']] += dose
        weights = logits.softmax(-1)
        forward.mass = float(weights[0, probe['head'], probe['receiver'], treatment['keys']].sum())
        head = weights @ value
        output = head.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1)
        return module.o_proj(output), None
    return forward


@contextmanager
def intervene(model, probe, treatment, dose):
    module = model.model.layers[probe['layer']].self_attn
    original = module.forward
    patched = attention_forward(probe, treatment, dose)
    module.forward = MethodType(patched, module)
    try:
        yield patched
    finally:
        module.forward = original


@torch.no_grad()
def observe(model, tokens, prompt, answer, alternatives=None):
    """Whole-prefix native pass, clamped answer; normalized final hidden state."""
    inputs = torch.tensor(tokens, device=model.device)[None]
    hidden = model.model(input_ids=inputs, use_cache=False).last_hidden_state[0, prompt - 1:]
    targets = torch.tensor(answer, device=model.device)
    margins, probabilities, rivals = [], [], []
    for start in range(0, len(answer), 32):
        stop = start + 32
        logits = model.lm_head(hidden[start:stop]).float()
        actual = logits.gather(1, targets[start:stop, None])[:, 0]
        if alternatives is None:
            top = logits.topk(2, dim=-1).indices
            rival = torch.where(top[:, 0] == targets[start:stop], top[:, 1], top[:, 0])
        else:
            rival = torch.tensor(alternatives[start:stop], device=model.device)
        margins.append((actual - logits.gather(1, rival[:, None])[:, 0]).cpu())
        probabilities.append((actual - logits.logsumexp(-1)).cpu())
        rivals.append(rival.cpu())
    return dict(margin=torch.cat(margins).numpy(), logp=torch.cat(probabilities).numpy(),
        alternatives=torch.cat(rivals).numpy(), hidden=hidden.cpu().numpy())
