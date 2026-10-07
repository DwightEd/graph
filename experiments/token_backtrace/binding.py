"""Readability and causal use of an owner assignment in native A_q7 V_7 messages.

V and edge tensors retain [case, layer, physical head, head dimension]. A diagnostic
probe uses programmatic owner labels, never natural hallucination labels. Replacing
one edge is performed before W_O; GQA-sharing does not expand the patch to other heads.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from state_audit.functional_capture import attention_rows
from .messages import capture_edge_inputs, native_trace, patch_message


def measure_case(adapter, case, candidate, gradients=False):
    """Capture value7, edge weights and query gradients on the prediction grid."""
    query = case["query"]
    layers = range(len(adapter.layers))
    with capture_edge_inputs(adapter, layers) as records:
        trace = native_trace(adapter.native, case["prompt"], [candidate], gradients)
    values, weights, reconstruction_errors = [], [], []
    with torch.no_grad():
        for layer, record in records.items():
            heads, kv_heads, width = adapter.head_layout(layer)
            attention = attention_rows(adapter, layer, record, record["rotary"],
                                       torch.tensor([query], device=adapter.native.device))[0]
            native_values = record["value"][0].reshape(-1, kv_heads, width)
            expanded = native_values.repeat_interleave(heads // kv_heads, dim=1)
            rebuilt = torch.einsum("hk,khd->hd", attention, expanded.float())
            native = trace.messages[layer][0, query].reshape(heads, width).float()
            reconstruction_errors.append((rebuilt - native).abs().max().item())
            values.append(expanded[case["key"]].float().cpu())
            weights.append(attention[:, case["key"]].cpu())
    result = dict(value=torch.stack(values), attention=torch.stack(weights),
                  logits=trace.logits.detach().cpu(), reconstruction=max(reconstruction_errors))
    return result, trace


def query_gradients(trace, other_candidate):
    objective = trace.logits[0, trace.answer[0]] - trace.logits[0, other_candidate]
    gradients = torch.autograd.grad(objective, trace.messages)
    query = len(trace.prompt) - 1
    return torch.stack([gradient[0, query].reshape(trace.heads, trace.head_dim).detach().cpu()
                        for gradient in gradients])


def edge_delta(base, donor):
    """Freeze the base Q/K/A; exchange V for the same source token at the same position."""
    return base["attention"][..., None] * (donor["value"] - base["value"])


def patch_choice(adapter, case, candidate, other_candidate, layer, head, delta, alpha=1.):
    with patch_message(adapter.native, layer, head, case["query"], delta, alpha):
        trace = native_trace(adapter.native, case["prompt"], [candidate], gradients=False)
    logits = trace.logits[0].cpu()
    return dict(margin=float(logits[candidate] - logits[other_candidate]),
                top_id=int(logits.argmax()), candidate_mass=float(
                    logits.softmax(-1)[[candidate, other_candidate]].sum()))


def ridge_scores(features, labels, fit, ridge):
    """Fit per-head linear readouts; features [N,L,H,D], returned scores [N,L,H]."""
    features = features.double().flatten(1, 2).permute(1, 0, 2)
    mean = features[:, fit].mean(dim=1, keepdim=True)
    scale = features[:, fit].std(dim=1, keepdim=True).clamp_min(1e-5)
    centered = (features - mean) / scale
    train = centered[:, fit]
    identity = torch.eye(train.shape[-1], dtype=torch.float64)
    gram = train.transpose(1, 2) @ train + ridge * len(fit) * identity
    target = (2 * labels[fit].double() - 1).expand(features.shape[0], -1)
    right_hand_side = train.transpose(1, 2) @ target[..., None]
    # Positive ridge guarantees SPD; use feature-space Cholesky, without LU pivots.
    weights = torch.cholesky_solve(right_hand_side, torch.linalg.cholesky(gram))
    scores = centered @ weights
    return scores[..., 0].transpose(0, 1).float()


def readout_metrics(scores, labels):
    """Bernoulli scores are held-out predictive evidence, not an MI estimator."""
    labels = labels.float()[:, None]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        scores, labels.expand_as(scores), reduction="none").mean(dim=0)
    accuracy = ((scores > 0) == labels.bool()).float().mean(dim=0)
    auc = [roc_auc_score(labels[:, 0].numpy(), column.numpy()) for column in scores.T]
    return dict(loss=loss.numpy(), accuracy=accuracy.numpy(), auc=np.asarray(auc))


def fit_readability(features, labels, fit, dev, evaluations):
    """Choose a common ridge/temperature by dev mean loss, then freeze all heads."""
    candidates = []
    for ridge in (.1, 1., 10.):
        scores = ridge_scores(features, labels, fit, ridge)
        for temperature in (.25, .5, 1., 2., 4.):
            target = labels[dev].float()[:, None].expand_as(scores[dev])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(scores[dev] / temperature, target)
            candidates.append((float(loss), ridge, temperature))
    _, ridge, temperature = min(candidates)
    scores = ridge_scores(features, labels, fit, ridge) / temperature
    development = readout_metrics(scores[dev], labels[dev])
    ranked = np.argsort(development["loss"], kind="stable")
    results = {name: readout_metrics(scores[index], labels[index])
               for name, index in evaluations.items()}
    return dict(ridge=ridge, temperature=temperature, selected=int(ranked[0]),
                dev=development, evaluations=results, scores=scores)
