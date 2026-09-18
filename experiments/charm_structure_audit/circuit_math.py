"""Actual node MLP: weight direction, paired activation effects, exact accounting."""

import torch


def tail(model, state):
    """The trained pointwise network after input projection. No message calls."""
    for layer in model.mp_layers:
        state = layer.update(state, torch.zeros_like(state))
    return model.pred(state).flatten()


def project(model, values):
    return model.in_proj(values).relu()


def dominant_direction(weight):
    """Use weights only. Orient by the largest absolute coefficient, not labels."""
    left, singular, right = torch.linalg.svd(weight, full_matrices=False)
    direction = right[0].clone()
    direction *= torch.sign(direction[direction.abs().argmax()])
    loading = weight @ direction
    return direction, loading, singular


def axis_experiments(model, error, normal, repeats=20, seed=42):
    """Preserve the original bias/tail; change only the first projection."""
    weight = model.in_proj.weight
    direction, loading, singular = dominant_direction(weight)
    rank_one = loading[:, None] * direction[None]
    values = torch.stack((error, normal))
    flat = values.flatten(0, 1)
    bias = model.in_proj.bias
    residual_mean = (weight - rank_one) @ flat.mean(0)
    scores = {}
    for name, matrix, offset in (
        ('full', weight, bias), ('keep_axis', rank_one, bias),
        ('remove_axis', weight - rank_one, bias),
        ('keep_axis_mean_background', rank_one, bias + residual_mean),
    ):
        scores[name] = tail(model, (flat @ matrix.T + offset).relu()).reshape(2, -1)
    pre_error, pre_normal = model.in_proj(error), model.in_proj(normal)
    delta = ((error - normal) @ direction)[:, None] * loading
    scores['replace_error_axis'] = torch.stack((tail(model, (pre_error-delta).relu()), scores['full'][1]))
    scores['insert_axis_into_normal'] = torch.stack((tail(model, (pre_normal+delta).relu()), scores['full'][1]))
    controls = orthogonal_controls(model, pre_error, delta, loading, scores['full'][1], repeats, seed)
    scores.update(controls)
    return scores, direction, loading, singular


def orthogonal_controls(model, pre_error, delta, loading, normal_logits, repeats, seed):
    """Equal L2 preactivation change at EACH token; orthogonal to the axis."""
    generator = torch.Generator(device=delta.device).manual_seed(seed)
    axis = loading / loading.norm()
    amplitude = delta @ axis
    result = {}
    for repeat in range(repeats):
        vector = torch.randn(len(axis), generator=generator, device=axis.device, dtype=axis.dtype)
        vector = vector - (vector @ axis) * axis
        vector = vector / vector.norm()
        change = amplitude[:, None] * vector
        altered = tail(model, (pre_error - change).relu())
        result[f'orthogonal_{repeat}'] = torch.stack((altered, normal_logits))
    return result


def relu_secant(error, normal):
    """Finite-difference multiplier; identical preactivations contribute zero."""
    difference = error - normal
    scale = (error > 0).to(error.dtype)
    changed = difference != 0
    scale[changed] = (error.relu()[changed] - normal.relu()[changed]) / difference[changed]
    return scale


def paired_accounting(model, error, normal):
    """DeepLIFT-style Rescale accounting, NOT the earlier numerical IG path.

    Signed terms sum exactly to f(error)-f(normal). Reference dependent; this
    numerical conservation alone is not a claim about semantic causation.
    """
    first_error, first_normal = model.in_proj(error), model.in_proj(normal)
    error_state, normal_state = first_error.relu(), first_normal.relu()
    steps = []
    for layer in model.mp_layers:
        error_hidden = layer.up_mlp[0](torch.cat((error_state, torch.zeros_like(error_state)), -1))
        normal_hidden = layer.up_mlp[0](torch.cat((normal_state, torch.zeros_like(normal_state)), -1))
        error_update = layer.up_mlp[2](error_hidden.relu())
        normal_update = layer.up_mlp[2](normal_hidden.relu())
        if layer.residual:
            error_update, normal_update = error_update + error_state, normal_update + normal_state
        steps.append((layer, relu_secant(error_hidden, normal_hidden), relu_secant(error_update, normal_update)))
        error_state, normal_state = error_update.relu(), normal_update.relu()
    multiplier = readout_multiplier(model, error_state, normal_state)
    for layer, inner, outer in reversed(steps):
        downstream = multiplier * outer
        own_weight = layer.up_mlp[0].weight[:, :multiplier.shape[1]]
        multiplier = ((downstream @ layer.up_mlp[2].weight) * inner) @ own_weight
        if layer.residual:
            multiplier = multiplier + downstream
    unit_contribution = (first_error.relu() - first_normal.relu()) * multiplier
    head_multiplier = (multiplier * relu_secant(first_error, first_normal)) @ model.in_proj.weight
    return dict(head_contribution=(error-normal)*head_multiplier, head_multiplier=head_multiplier,
                unit_contribution=unit_contribution, downstream_multiplier=multiplier)


def readout_multiplier(model, error_state, normal_state):
    error = model.pred[0](error_state)
    normal = model.pred[0](normal_state)
    return (relu_secant(error, normal) * model.pred[3].weight[0]) @ model.pred[0].weight


def local_gradients(model, values):
    """Independent autograd check of the direction actually used by the function."""
    values = values.detach().requires_grad_(True)
    logits = tail(model, project(model, values))
    return torch.autograd.grad(logits.sum(), values)[0].detach()
