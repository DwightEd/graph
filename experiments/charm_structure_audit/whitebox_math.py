"""Decode the frozen node-only ReLU network. No graph tensors or new features."""

import numpy as np
import torch


def linearize(model, attributes):
    """Actual gates give an exact local affine map: logit = beta @ x + intercept.

    Intercepts are propagated from the learned biases, not fitted to the output.
    At a ReLU kink the chosen derivative is zero, matching PyTorch.
    """
    projected = model.in_proj(attributes)
    input_gate = projected > 0
    state = projected.relu()
    constant = input_gate * model.in_proj.bias
    gates = [input_gate]
    updates = []
    for layer in model.mp_layers:
        joined = torch.cat((state, torch.zeros_like(state)), dim=-1)
        hidden = layer.up_mlp[0](joined)
        inner_gate = hidden > 0
        own_weight = layer.up_mlp[0].weight[:, :state.shape[1]]
        hidden_constant = (constant @ own_weight.T + layer.up_mlp[0].bias) * inner_gate
        update = layer.up_mlp[2](hidden.relu())
        update_constant = layer.up_mlp[2](hidden_constant)
        combined = state + update if layer.residual else update
        constant = constant + update_constant if layer.residual else update_constant
        outer_gate = combined > 0
        constant = constant * outer_gate
        state = combined.relu()
        gates.extend((inner_gate, outer_gate))
        updates.append((layer, inner_gate, outer_gate))
    return finish_linearization(model, state, constant, input_gate, updates, gates)


def finish_linearization(model, state, constant, input_gate, updates, gates):
    hidden = model.pred[0](state)
    readout_gate = hidden > 0
    output_weight = model.pred[3].weight[0]
    contributions = hidden.relu() * output_weight
    logits = contributions.sum(dim=-1) + model.pred[3].bias[0]
    bias_hidden = model.pred[0](constant) * readout_gate
    intercept = (bias_hidden * output_weight).sum(dim=-1) + model.pred[3].bias[0]
    beta = (readout_gate * output_weight) @ model.pred[0].weight
    for layer, inner_gate, outer_gate in reversed(updates):
        downstream = beta * outer_gate
        upstream = (downstream @ layer.up_mlp[2].weight) * inner_gate
        own_weight = layer.up_mlp[0].weight[:, :beta.shape[1]]
        beta = upstream @ own_weight
        if layer.residual:
            beta = beta + downstream
    beta = (beta * input_gate) @ model.in_proj.weight
    gates.append(readout_gate)
    return dict(logits=logits, beta=beta, intercept=intercept,
                contributions=contributions, gates=torch.cat(gates, dim=-1))


def evaluate_nodes(model, attributes, batch_size=512):
    """Batch the pointwise computation without executing any message MLP."""
    outputs = {}
    with torch.no_grad():
        for start in range(0, len(attributes), batch_size):
            block = linearize(model, attributes[start:start + batch_size])
            for name, value in block.items():
                outputs.setdefault(name, []).append(value)
    return {name: torch.cat(values) for name, values in outputs.items()}


def path_integral(model, normal, error, points):
    """Gauss-Legendre integration of analytic local coefficients along N -> E."""
    locations, weights = np.polynomial.legendre.leggauss(points)
    locations = torch.as_tensor((locations + 1) / 2, device=normal.device, dtype=normal.dtype)
    weights = torch.as_tensor(weights / 2, device=normal.device, dtype=normal.dtype)
    difference = error - normal
    total = torch.zeros_like(normal)
    # One quadrature location at a time avoids [tokens,points,channels] GPU arrays.
    with torch.no_grad():
        for location, weight in zip(locations, weights):
            total += weight * linearize(model, normal + location * difference)['beta']
    return difference * total


def integrate_difference(model, normal, error, max_points=512, atol=1e-4, rtol=.005):
    """Check BOTH completeness and per-channel allocation stability.

    An unresolved numerical integral remains marked unresolved. No residual is
    redistributed over heads to manufacture an exact explanation.
    """
    endpoint = evaluate_nodes(model, error)['logits'] - evaluate_nodes(model, normal)['logits']
    count = len(error)
    answer = torch.zeros_like(error)
    used = torch.zeros(count, dtype=torch.int64, device=error.device)
    converged = torch.zeros(count, dtype=torch.bool, device=error.device)
    residual = torch.full_like(endpoint, float('nan'))
    allocation_change = torch.full_like(endpoint, float('inf'))
    previous = path_integral(model, normal, error, 16)
    points = 32
    while points <= max_points:
        active = torch.where(~converged)[0]
        current = path_integral(model, normal[active], error[active], points)
        gap = endpoint[active]
        error_sum = current.sum(dim=-1) - gap
        change = (current - previous[active]).abs().sum(dim=-1)
        scale = torch.maximum(current.abs().sum(dim=-1), gap.abs())
        passed = (error_sum.abs() <= atol + rtol * gap.abs()) & (change <= atol + rtol * scale)
        answer[active], residual[active], allocation_change[active] = current, error_sum, change
        previous[active] = current
        used[active] = points
        converged[active] = passed
        if converged.all():
            break
        points *= 2
    return dict(attribution=answer, converged=converged, points=used,
                completeness_residual=residual, allocation_change=allocation_change)


def hybrid_logits(model, error, normal, mask):
    """Four corners with paired natural endpoints; mask can differ by token.

    Error->normal on selected channels removes their difference. Normal->error
    on those channels introduces it. These hybrid inputs are not native LLM runs.
    """
    error_removed = torch.where(mask, normal, error)
    normal_added = torch.where(mask, error, normal)
    removed = evaluate_nodes(model, error_removed)['logits']
    added = evaluate_nodes(model, normal_added)['logits']
    return removed, added


def matched_random_masks(attribution, difference, heads, budget, repeats, seed):
    """Random controls match layer and within-layer input-change magnitude band.

    The random set can overlap the selected set. Exact overlap and changed-input
    norms are reported; singleton strata are not treated as informative controls.
    """
    values = attribution.detach().cpu().numpy()
    change = difference.detach().cpu().numpy()
    count, width = values.shape
    budget = min(budget, width)
    selected = np.zeros((count, width), bool)
    order = np.argsort(-np.abs(values), axis=1, kind='stable')[:, :budget]
    np.put_along_axis(selected, order, True, axis=1)
    random_masks = np.zeros((repeats, count, width), bool)
    exchangeable = np.zeros(count, int)
    rng = np.random.default_rng(seed)
    for row in range(count):
        for start in range(0, width, heads):
            columns = np.arange(start, start + heads)
            magnitude = abs(change[row, columns])
            cuts = np.quantile(magnitude, [.25, .5, .75])
            bands = np.searchsorted(cuts, magnitude, side='right')
            for band in np.unique(bands):
                candidates = columns[bands == band]
                number = int(selected[row, candidates].sum())
                if not number:
                    continue
                if number < len(candidates):
                    exchangeable[row] += number
                for repeat in range(repeats):
                    random_masks[repeat, row, rng.choice(candidates, number, replace=False)] = True
    device = attribution.device
    return torch.as_tensor(selected, device=device), torch.as_tensor(random_masks, device=device), exchangeable
