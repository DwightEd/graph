"""Exact finite-world attention bookkeeping for each receiver and physical head.

The functions preserve every head coordinate and signed term. Their outputs
describe state differences; none is a hallucination score or a semantic label.
"""
import torch
from torch.nn import functional as F


def decompose_messages(attention_plus, attention_minus, values_plus, values_minus):
    """Split each edge's finite difference into content and routing terms.

    Attention: [receiver, head, sender]. Values: [sender, head, head_dim],
    already expanded to physical query heads if the model uses grouped KV.
    Returns two [receiver, head, sender, head_dim] tensors, before W_O.
    This symmetric identity is not a Jacobian or independent causal attribution.
    """
    mean_attention = (attention_plus + attention_minus) / 2
    attention_difference = attention_plus - attention_minus
    mean_values = ((values_plus + values_minus) / 2).permute(1, 0, 2)
    value_difference = (values_plus - values_minus).permute(1, 0, 2)

    content = mean_attention[..., None] * value_difference[None]
    routing = attention_difference[..., None] * mean_values[None]
    return dict(content=content, routing=routing)


def project_edges(edge_messages, output_weight):
    """Apply W_O to each head/edge without combining different receivers.

    edge_messages: [receiver, head, sender, head_dim].
    output_weight: native linear weight [model_dim, head * head_dim].
    Returns [receiver, head, sender, model_dim]. For storage, retain the
    head_dim factors and W_O rather than materializing this larger tensor.
    """
    heads = edge_messages.shape[1]
    head_dim = edge_messages.shape[-1]
    head_weights = output_weight.reshape(output_weight.shape[0], heads, head_dim)
    return torch.einsum('rhsd,ohd->rhso', edge_messages, head_weights)


def aggregate_attention_write(edge_messages, output_weight):
    """Sum sender messages within each head, then apply native W_O.

    Inputs use the same axes as project_edges. Returns [receiver, model_dim].
    The native linear must be bias-free, as in the supported Llama observer.
    This sum is the native attention operation, not a mean across token states.
    """
    head_messages = edge_messages.sum(dim=2)
    return F.linear(head_messages.flatten(1), output_weight)


def partition_senders(source_mask, answer_start, receiver_positions, local_radius):
    """Partition all causal keys; keep self separate from strict-past neighbors.

    source_mask: [sender] bool. receiver_positions: [receiver] absolute token
    positions. Returns five [receiver, sender] bool masks. Source/prompt keys
    outside the local answer neighborhood remain explicit boundary groups.
    The self key has its own group even for a prompt-end prediction row.
    Future keys belong to no group; local_radius is measured in token positions.
    """
    senders = torch.arange(len(source_mask), device=receiver_positions.device)
    receivers = receiver_positions[:, None]
    strict_past = senders[None] < receivers
    source = strict_past & source_mask[None]
    past_answer = strict_past & (senders[None] >= answer_start) & ~source_mask[None]
    local = past_answer & (receivers - senders[None] <= local_radius)
    remote = past_answer & ~local
    self_key = senders[None] == receivers
    other_prompt = strict_past & (senders[None] < answer_start) & ~source_mask[None]
    return dict(source=source, local=local, remote=remote,
                self=self_key, other_prompt=other_prompt)


def aggregate_partitions(edge_messages, partitions):
    """Return exact [receiver, head, head_dim] sums for each key group.

    edge_messages is [receiver, head, sender, head_dim]. Every receiver
    retains a different result. No local renormalization or span mean is used.
    """
    messages = {}
    for name, mask in partitions.items():
        messages[name] = (edge_messages * mask[:, None, :, None]).sum(dim=2)
    return messages


def state_conservation_error(delta_before, delta_attention, delta_mlp, delta_after):
    """Return the residual update error at every node/coordinate.

    All inputs are same-shaped finite world differences [..., model_dim].
    A zero error verifies bookkeeping for arbitrary states; it says nothing
    about factual correctness and must not be used as a detection score.
    """
    return delta_after - delta_before - delta_attention - delta_mlp
