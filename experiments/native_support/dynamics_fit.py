"""Unlabelled multivariate emissions plus input-driven switching dynamics.

The H mode has B=0; E can use source input. Their posterior is a detection
hypothesis, not a calibrated truth probability. No natural labels enter fitting.
"""

from collections import Counter

import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd
from tqdm import trange

from .dynamics_core import mode_posteriors


def fit_projection(values, rank, weights):
    mean = np.average(values, axis=0, weights=weights).astype(np.float32)
    variance = np.average((values - mean) ** 2, axis=0, weights=weights).astype(np.float32)
    scale = np.maximum(np.sqrt(variance), .05)
    centered = (values - mean) / scale
    weighted = centered * np.sqrt(weights / weights.mean()).astype(np.float32)[:, None]
    count = min(rank, values.shape[1], len(values) - 1)
    _left, singular, right = randomized_svd(weighted, n_components=count, random_state=37)
    components = right / np.maximum(singular / np.sqrt(len(values)), .2)[:, None]
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32),
            "components": components.astype(np.float32),
            "retained_variance": float(np.square(singular).sum() / max(float(np.square(weighted).sum()), 1e-12))}


def project(values, projection):
    return ((values - projection["mean"]) / projection["scale"]) @ projection["components"].T


def projection_error(values, projection):
    centered = (values - projection["mean"]) / projection["scale"]
    basis = projection["components"]
    basis = basis / np.maximum(np.linalg.norm(basis, axis=1, keepdims=True), 1e-12)
    energy = float(np.square(centered).sum())
    retained = float(np.square(centered @ basis.T).sum())
    return max(0., 1 - retained / energy) if energy > 0 else 0.


def profile_matrix(observation, selected=slice(None)):
    profile = observation["profile"][selected]
    count = len(profile)
    return np.column_stack((profile.reshape(count, -1), observation["ffn"][selected].reshape(count, -1),
                            np.log1p(observation["entropy"][selected]).astype(np.float32),
                            np.log1p(observation["surprisal"][selected]).astype(np.float32),
                            observation["candidate_tail_mass"][selected].astype(np.float32)))


def fit_coordinates(observations, source_ids, rank, profile_rank, sample_budget=8192):
    """Equal-source deterministic sampling, never rank dimensions by test AUROC."""
    source_counts = Counter(source_ids)
    rows, weights = [], []
    for observation, source in zip(observations, source_ids):
        count = len(observation["state"])
        budget = max(2, sample_budget // len(source_counts) // source_counts[source])
        selected = np.linspace(0, count - 1, min(count, budget), dtype=int)
        rows.append(selected)
        weights.extend([1 / (len(selected) * source_counts[source])] * len(selected))
    weights = np.asarray(weights)
    result = {}
    for name in ("source_input", "history", "profile"):
        values = [profile_matrix(obs, index) if name == "profile" else obs[name][index]
                  for obs, index in zip(observations, rows)]
        joined = np.concatenate(values)
        result[name] = fit_projection(joined, profile_rank if name == "profile" else rank, weights)
    state = np.concatenate([obs["state"][index] for obs, index in zip(observations, rows)])
    result["state_mean"] = np.average(state, axis=0, weights=weights).astype(np.float32)
    result["state_scale"] = np.maximum(np.sqrt(np.average(
        (state - result["state_mean"]) ** 2, axis=0, weights=weights)), .05).astype(np.float32)
    return result


def make_sequence(observation, coordinates, input_covariance):
    state = (observation["state"] - coordinates["state_mean"]) / coordinates["state_scale"]
    previous = np.concatenate((np.zeros_like(state[:1]), state[:-1]))
    history = np.column_stack((previous, project(observation["history"], coordinates["history"])))
    return {"state": state.astype(np.float32), "history": history.astype(np.float32),
            "input": project(observation["source_input"], coordinates["source_input"]).astype(np.float32),
            "covariance": input_covariance.astype(np.float32),
            "profile": project(profile_matrix(observation), coordinates["profile"]).astype(np.float32)}


def ridge(design, target, weight):
    penalty = np.eye(design.shape[1]) * .1
    return np.linalg.solve(design.T @ (weight[:, None] * design) + penalty,
                           design.T @ (weight[:, None] * target)).T


def initialize(sequences, routing):
    joined = {name: np.concatenate([seq[name] for seq in sequences])
              for name in ("state", "history", "input", "profile")}
    route = np.concatenate(routing)
    low = route <= np.median(route)
    state, history, control, profile = (joined[name] for name in ("state", "history", "input", "profile"))
    transitions, biases, centers, spreads = [], [], [], []
    input_matrix = None
    for mode, selected in enumerate((low, ~low)):
        weight = .1 + .9 * selected
        design = np.column_stack((history, control, np.ones(len(state)))) if mode == 0 else np.column_stack((history, np.ones(len(state))))
        fitted = ridge(design, state, weight)
        transitions.append(fitted[:, :history.shape[1]])
        biases.append(fitted[:, -1])
        if mode == 0:
            input_matrix = fitted[:, history.shape[1]:-1]
        residual = state - design @ fitted.T
        center = np.average(profile, axis=0, weights=weight)
        centers.append(center)
        joint = np.column_stack((residual, profile - center))
        covariance = joint.T @ (joint * weight[:, None]) / weight.sum()
        spreads.append(np.linalg.cholesky(.9 * covariance + .1 * np.eye(joint.shape[1])))
    return {"transition": np.stack(transitions), "injection": input_matrix,
            "bias": np.stack(biases), "profile_mean": np.stack(centers),
            "noise_factor": np.stack(spreads), "mode_logits": np.log([[.94, .06], [.06, .94]]),
            "initial_logits": np.log([.5, .5]), "log_input_variance": np.asarray(-2.3)}


class SwitchingDynamics(torch.nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.parameters_by_name = torch.nn.ParameterDict({
            name: torch.nn.Parameter(torch.as_tensor(value, dtype=torch.float32))
            for name, value in initial.items()
        })

    def emissions(self, sequence):
        params = self.parameters_by_name
        injection = torch.stack((params["injection"], torch.zeros_like(params["injection"])))
        mean = torch.einsum("kdh,th->tkd", params["transition"], sequence["history"])
        mean = mean + torch.einsum("kdu,tu->tkd", injection, sequence["input"]) + params["bias"]
        profile_mean = params["profile_mean"].expand(len(mean), -1, -1)
        mean = torch.cat((mean, profile_mean), dim=-1)
        factor = torch.tril(params["noise_factor"])
        covariance = factor @ factor.transpose(-1, -2) + .03 * torch.eye(
            factor.shape[-1], device=factor.device)
        profile_injection = injection.new_zeros((2, profile_mean.shape[-1], injection.shape[-1]))
        joint_injection = torch.cat((injection, profile_injection), dim=1)
        input_variance = params["log_input_variance"].clamp(-7, 3).exp()
        covariance = covariance[None] + input_variance * torch.einsum(
            "kdu,tuv,kev->tkde", joint_injection, sequence["covariance"], joint_injection)
        observed = torch.cat((sequence["state"], sequence["profile"]), dim=-1)
        return normal_log_density(observed[:, None] - mean, covariance)


def normal_log_density(residual, covariance):
    factor = torch.linalg.cholesky(covariance)
    whitened = torch.linalg.solve_triangular(factor, residual[..., None], upper=False)[..., 0]
    logdet = 2 * torch.diagonal(factor, dim1=-2, dim2=-1).log().sum(-1)
    return -.5 * (residual.shape[-1] * np.log(2 * np.pi) + logdet + whitened.square().sum(-1))


def sequence_log_likelihood(emission, transition, initial):
    state = initial + emission[0]
    for current in emission[1:]:
        state = current + torch.logsumexp(state[:, None] + transition, dim=0)
    return torch.logsumexp(state, dim=0)


def fit_model(sequences, routing, source_ids, epochs, learning_rate, device):
    model = SwitchingDynamics(initialize(sequences, routing)).to(device)
    data = [{name: torch.as_tensor(value, device=device) for name, value in seq.items()} for seq in sequences]
    counts = Counter(source_ids)
    weights = [1 / (len(counts) * counts[source]) for source in source_ids]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trace, best_loss, best = [], np.inf, None
    progress = trange(epochs, desc="unlabelled dynamics fit")
    for epoch in progress:
        optimizer.zero_grad()
        loss = 0.
        params = model.parameters_by_name
        for sequence, weight in zip(data, weights):
            emission = model.emissions(sequence)
            likelihood = sequence_log_likelihood(emission, params["mode_logits"].log_softmax(-1),
                                                  params["initial_logits"].log_softmax(-1))
            current = -weight * likelihood / len(emission)
            current.backward()
            loss += float(current.detach())
        regularizer = 1e-3 * (params["transition"].square().mean() + params["injection"].square().mean())
        regularizer.backward()
        loss += float(regularizer.detach())
        if not np.isfinite(loss):
            raise FloatingPointError("Non-finite dynamics training objective")
        if loss < best_loss:
            best_loss = loss
            best = {name: value.detach().cpu().numpy().copy() for name, value in params.items()}
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        optimizer.step()
        trace.append({"epoch": epoch, "objective": loss})
        progress.set_postfix(nll=f"{loss:.4f}")
    return best, trace


def infer(parameters, sequence, device="cpu"):
    model = SwitchingDynamics(parameters).to(device)
    with torch.no_grad():
        data = {name: torch.as_tensor(value, device=device) for name, value in sequence.items()}
        emission = model.emissions(data).cpu().numpy()
        transition = model.parameters_by_name["mode_logits"].softmax(-1).cpu().numpy()
        initial = model.parameters_by_name["initial_logits"].softmax(-1).cpu().numpy()
    result = mode_posteriors(emission, transition, initial)
    result["emission"] = emission
    return result
