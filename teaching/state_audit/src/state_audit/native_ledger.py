"""An observed-run logit ledger, with floating-point discrepancies kept explicit."""

import numpy as np
import torch

from .capture import numpy


def readout_direction(model, residual, candidate_ids):
    """Freeze this run's final RMS denominator, not any model computation."""
    norm = model.native.model.norm
    scale = (residual.float().square().mean() + norm.variance_epsilon).sqrt()
    weights = model.native.lm_head.weight[list(candidate_ids)].float()
    difference = weights[0] - weights[1]
    direction = difference * norm.weight.float() / scale
    return direction.detach(), difference.detach(), scale.detach()


def projected_messages(model, layer, record, values, direction, gram, masks):
    """Keep all heads/keys; group energy and squared group-write norm are distinct."""
    heads, kv_heads, _ = model.head_layout(layer)
    values = values.float().repeat_interleave(heads // kv_heads, dim=0)
    attention = record["attention"][:, -1].float()
    weight = model.layers[layer].self_attn.o_proj.weight.detach().float()
    blocks = weight.reshape(weight.shape[0], heads, -1).permute(1, 2, 0)
    edge_write = attention * torch.einsum("hkd,hd->hk", values, blocks @ direction)
    energy = attention.square() * ((values @ gram) * values).sum(-1).clamp_min(0)
    grouped = torch.einsum("hk,hkd,kg->hgd", attention, values, masks)
    writes = model.project_heads(layer, record["head_readout"].detach(), range(heads))[0]
    group_writes = grouped @ blocks
    return dict(
        attention=numpy(attention),
        edge_logit_write=numpy(edge_write),
        edge_value_energy=numpy(energy),
        head_writes=numpy(writes),
        group_readouts=numpy(grouped),
        group_write_norm=numpy(group_writes.norm(dim=-1)),
        group_logit_write=numpy(edge_write @ masks),
        group_route_mass=numpy(attention @ masks),
        group_positive=numpy(edge_write.clamp_min(0) @ masks),
        group_negative=numpy((-edge_write).clamp_min(0) @ masks),
        group_value_energy=numpy(energy @ masks),
    )


def residual_ledger(arrays, direction):
    """Each row is an actual sublayer transition, projected on one fixed direction."""
    before = arrays["residual_before"] @ direction
    middle = arrays["residual_mid"] @ direction
    after = arrays["residual_after"] @ direction
    attention = arrays["attention_write"] @ direction
    mlp = arrays["mlp_write"] @ direction
    edge_total = arrays["edge_logit_write"].sum(axis=(1, 2))
    return dict(
        residual_scores=np.stack((before, middle, after), -1),
        attention_score=attention,
        mlp_score=mlp,
        attention_reconstruction_error=attention - edge_total - arrays["attention_bias_score"],
        attention_add_roundoff=middle - before - attention,
        mlp_add_roundoff=after - middle - mlp,
    )


def finish_ledger(
    model, records, arrays, hidden, logits, candidate_ids, direction, difference, scale
):
    """Close the additive ledger without assigning numeric error to any head/FFN."""
    sites = (
        "residual_before",
        "residual_mid",
        "residual_after",
        "attention_write",
        "mlp_write",
        "mlp_activation",
        "head_readout",
    )
    arrays.update(
        {name: np.stack([numpy(r[name][-1]) for r in records.values()]) for name in sites}
    )
    arrays["direction"] = numpy(direction)
    arrays.update(residual_ledger(arrays, arrays["direction"]))
    residual = records[len(records) - 1]["residual_after"][-1].float()
    ideal_hidden = model.native.model.norm.weight.float() * residual / scale
    arrays["norm_roundoff"] = numpy((hidden.float() - ideal_hidden) @ difference)
    bias = model.native.lm_head.bias
    bias_gap = 0.0 if bias is None else float(bias[candidate_ids[0]] - bias[candidate_ids[1]])
    actual_gap = logits[candidate_ids[0]] - logits[candidate_ids[1]]
    arrays["output_bias_score"] = np.asarray(bias_gap)
    arrays["unembedding_roundoff"] = numpy(actual_gap - hidden.float() @ difference - bias_gap)
    arrays["candidate_logits"] = numpy(logits[list(candidate_ids)])
    arrays["logit_gap"] = numpy(actual_gap)
    arrays["initial_score"] = np.asarray(arrays["residual_scores"][0, 0])
    arrays["ledger_total"] = np.asarray(ledger_total(arrays))
    arrays["ledger_error"] = arrays["ledger_total"] - arrays["logit_gap"]
    return arrays


def ledger_total(arrays):
    terms = (
        "attention_bias_score",
        "attention_reconstruction_error",
        "attention_add_roundoff",
        "mlp_add_roundoff",
        "norm_roundoff",
        "unembedding_roundoff",
        "output_bias_score",
    )
    total = float(arrays["initial_score"]) + arrays["edge_logit_write"].astype(float).sum()
    total += arrays["mlp_score"].astype(float).sum()
    return total + sum(np.asarray(arrays[name], dtype=float).sum() for name in terms)
