"""Separate direct detector history reads from physical later-token responses."""
import torch


def decompose_response(node_response, readout, sender, receiver):
    """node_response[k,d] is a full native JVP, readout[lag,d] a frozen matrix.

    This contracts the complete vector only at the final score readout. Rows
    before sender have zero causal response. The sender row is the detector's
    direct history shortcut; every later row is an actual model response.
    """
    first = max(0, receiver - len(readout) + 1)
    positions = torch.arange(first, receiver + 1, device=node_response.device)
    terms = (readout[receiver - positions] * node_response[positions]).sum(dim=-1)
    direct = terms[positions == sender].sum()
    transport = terms[positions > sender].sum()
    endpoint = terms[positions == receiver].sum() if receiver > sender else terms.new_zeros(())
    return dict(per_token=terms, positions=positions, direct=direct,
                transport=transport, endpoint=endpoint, total=terms.sum())


def matrix_readout(states, readout, receiver):
    first = max(0, receiver - len(readout) + 1)
    positions = torch.arange(first, receiver + 1, device=states.device)
    return (readout[receiver - positions] * states[positions]).sum()
