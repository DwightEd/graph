"""GPU readouts for a causal query block; no full residual vectors cross to CPU."""

import torch


def block_readout(model, hidden, residual, observed):
    """One native competitor and final RMS readout direction per query row."""
    logits = model.native.lm_head(hidden).float()
    top = logits.topk(2, dim=-1).indices
    competitor = torch.where(top[:, 0] == observed, top[:, 1], top[:, 0])
    candidates = torch.stack((observed, competitor), -1)
    weights = model.native.lm_head.weight[candidates].float()
    difference = weights[:, 0] - weights[:, 1]
    norm = model.native.model.norm
    scale = (residual.float().square().mean(-1, keepdim=True) + norm.variance_epsilon).sqrt()
    direction = difference * norm.weight.float() / scale

    candidate_logits = logits.gather(-1, candidates)
    gap = candidate_logits[:, 0] - candidate_logits[:, 1]
    bias = model.native.lm_head.bias
    bias_gap = torch.zeros_like(gap) if bias is None else bias[observed] - bias[competitor]
    ideal_hidden = norm.weight.float() * residual.float() / scale
    logp = logits.log_softmax(-1)
    arrays = dict(
        candidate_logits=candidate_logits, logit_gap=gap,
        observed_id=observed, competitor_id=competitor,
        norm_roundoff=((hidden.float() - ideal_hidden) * difference).sum(-1),
        output_bias_score=bias_gap.float(),
        unembedding_roundoff=gap - (hidden.float() * difference).sum(-1) - bias_gap,
        entropy=-(logp.exp() * logp).sum(-1),
        surprisal=-logp.gather(-1, observed[:, None]).squeeze(-1),
    )
    return direction, arrays


def append_value_energy(values, gram, previous):
    """Only newly cached keys need v^T W_O^T W_O v; old values never change."""
    start = 0 if previous is None else previous.shape[-1]
    unseen = values[:, start:]
    energy = ((unseen @ gram) * unseen).sum(-1).clamp_min(0)
    return energy if previous is None else torch.cat((previous, energy), -1)


def block_messages(model, layer, record, values, direction, gram, masks, energies):
    """Return [query, head, source/group] scalars for all rows together."""
    heads, kv_heads, width = model.head_layout(layer)
    values = values.float().repeat_interleave(heads // kv_heads, dim=0)
    attention = record["attention"].float().transpose(0, 1)
    projection = model.layers[layer].self_attn.o_proj.weight.detach().float()
    projected = (direction @ projection).reshape(-1, heads, width).transpose(0, 1)
    edge_write = attention * (projected @ values.transpose(-1, -2)).transpose(0, 1)
    energies[layer] = append_value_energy(values, gram, energies.get(layer))
    energy = attention.square() * energies[layer]

    # Norms of grouped writes need only the small Gram, not [query,head,group,D].
    grouped = (attention[:, :, None, :] * masks.T) @ values
    group_norm = ((grouped @ gram) * grouped).sum(-1).clamp_min(0).sqrt()
    return dict(
        attention=attention, edge_logit_write=edge_write, edge_value_energy=energy,
        group_write_norm=group_norm, group_logit_write=edge_write @ masks,
        group_route_mass=attention @ masks,
        group_positive=edge_write.clamp_min(0) @ masks,
        group_negative=(-edge_write).clamp_min(0) @ masks,
        group_value_energy=energy @ masks,
    )


def block_residual_ledger(model, layer, record, direction, edge_write):
    """Project on GPU, retaining each row's actual residual additions and errors."""
    sites = ("residual_before", "residual_mid", "residual_after", "attention_write", "mlp_write")
    scores = {name: (record[name].float() * direction).sum(-1) for name in sites}
    before = scores["residual_before"]
    middle, after = scores["residual_mid"], scores["residual_after"]
    attention, mlp = scores["attention_write"], scores["mlp_write"]
    bias = model.layers[layer].self_attn.o_proj.bias
    bias_score = torch.zeros_like(before) if bias is None else direction @ bias.float()
    return dict(
        residual_scores=torch.stack((before, middle, after), -1),
        attention_score=attention, mlp_score=mlp, attention_bias_score=bias_score,
        attention_reconstruction_error=attention - edge_write.sum((-1, -2)) - bias_score,
        attention_add_roundoff=middle - before - attention,
        mlp_add_roundoff=after - middle - mlp,
    )


def measure_block(model, records, cache, hidden, observed, masks, grams, energies):
    residual = records[len(records) - 1]["residual_after"]
    direction, readout = block_readout(model, hidden, residual, observed)
    layers = []
    for layer, record in records.items():
        arrays = block_messages(
            model, layer, record, cache.layers[layer].values[0], direction,
            grams[layer], masks, energies,
        )
        arrays.update(
            block_residual_ledger(model, layer, record, direction, arrays["edge_logit_write"])
        )
        layers.append(arrays)
    # One transfer per field/block, rather than per field/layer/token.
    result = {
        name: torch.stack([item[name] for item in layers], dim=1).cpu().numpy()
        for name in layers[0]
    }
    result.update({name: value.cpu().numpy() for name, value in readout.items()})
    return result
