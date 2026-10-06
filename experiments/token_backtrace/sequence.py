"""Source-anchored IOHMM: joint emissions, ordered states and learned boundaries.

States alternate strong/weak source support, then read/history-heavy behavior.
These are observation regimes, not identified factual truth states.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.special import logsumexp


FEATURES = ('source_full', 'source_local', 'present_full', 'present_local',
            'route', 'attention', 'entropy')
INPUTS = ('bias', 'previous_source', 'previous_route', 'source_change', 'route_change')
STATES = ('read_strong', 'read_weak', 'history_strong', 'history_weak')


def lagged_inputs(values):
    """Previous observations only; every answer resets both lags at its start."""
    axes = np.column_stack((values[:, :2].mean(axis=1), values[:, 4]))
    previous = np.vstack((np.zeros((1, 2)), axes[:-1]))
    change = np.vstack((np.zeros((1, 2)), np.diff(previous, axis=0)))
    return np.column_stack((np.ones(len(values)), previous, change))


def batch_sequences(sequences):
    lengths = np.array([len(values) for values in sequences])
    valid = np.arange(lengths.max())[None, :] < lengths[:, None]
    values = np.zeros((*valid.shape, len(FEATURES)))
    inputs = np.zeros((*valid.shape, len(INPUTS)))
    for index, sequence in enumerate(sequences):
        values[index, :len(sequence)] = sequence
        inputs[index, :len(sequence)] = lagged_inputs(sequence)
    return values, inputs, valid


def forward_backward(emission, transition, initial, valid):
    """Exact batched inference with a different transition matrix at each time.

    transition[a,t,i,j] joins t-1 to t; t=0 is unused. Padded answers end
    independently. xi[:,0] and all padding have zero expected transition mass.
    """
    answers, steps, states = emission.shape
    alpha, scales = np.zeros_like(emission), np.zeros((answers, steps))
    first = emission[:, 0] + np.log(initial)
    scales[:, 0] = logsumexp(first, axis=1)
    alpha[:, 0] = first - scales[:, 0, None]
    for time in range(1, steps):
        current = emission[:, time] + logsumexp(
            alpha[:, time - 1, :, None] + transition[:, time], axis=1)
        scale = logsumexp(current, axis=1)
        alpha[:, time] = np.where(valid[:, time, None], current - scale[:, None], alpha[:, time - 1])
        scales[:, time] = np.where(valid[:, time], scale, 0.)
    beta = np.zeros_like(emission)
    for time in range(steps - 2, -1, -1):
        current = logsumexp(transition[:, time + 1] + emission[:, time + 1, None, :]
                            + beta[:, time + 1, None, :], axis=2) - scales[:, time + 1, None]
        beta[:, time] = np.where(valid[:, time + 1, None], current, 0.)
    joint = alpha + beta
    gamma = np.exp(joint - logsumexp(joint, axis=2, keepdims=True)) * valid[..., None]
    xi = np.zeros((answers, steps, states, states))
    joint = (alpha[:, :-1, :, None] + transition[:, 1:] + emission[:, 1:, None, :]
             + beta[:, 1:, None, :])
    xi[:, 1:] = np.exp(joint - logsumexp(joint, axis=(2, 3), keepdims=True)) * valid[:, 1:, None, None]
    return gamma, xi, scales.sum(axis=1)


def viterbi(emission, transition, initial):
    best = emission[0] + np.log(initial)
    parents = np.zeros_like(emission, dtype=int)
    for time in range(1, len(emission)):
        candidates = best[:, None] + transition[time]
        parents[time] = candidates.argmax(axis=0)
        best = candidates.max(axis=0) + emission[time]
    path = np.empty(len(emission), dtype=int)
    path[-1] = best.argmax()
    for time in range(len(emission) - 1, 0, -1):
        path[time - 1] = parents[time, path[time]]
    return path, float(best.max())


def ordering(states, dimensions):
    source = np.zeros(dimensions)
    source[:2] = .5
    behavior = np.eye(dimensions)[4]
    specifications = [(0, 1, source)] if states == 2 else [
        (0, 1, source), (2, 3, source), (0, 2, behavior), (1, 3, behavior)]
    constraints = np.zeros((len(specifications), states, dimensions))
    for row, (low, high, axis) in enumerate(specifications):
        constraints[row, high], constraints[row, low] = axis, -axis
    return constraints.reshape(len(specifications), -1)


def constrained_means(target, mass, precision, margin=.2):
    """The weighted Mahalanobis M-step, solved jointly under linear orders."""
    constraints = ordering(*target.shape)

    def objective(flat):
        difference = flat.reshape(target.shape) - target
        gradient = mass[:, None] * (difference @ precision)
        return .5 * float(np.sum(difference * gradient)), gradient.ravel()

    solution = minimize(objective, target.ravel(), jac=True, method='SLSQP',
                        constraints=LinearConstraint(constraints, margin, np.inf),
                        options=dict(maxiter=200, ftol=1e-10))
    if not solution.success:
        raise RuntimeError('Ordered emission M-step failed: ' + solution.message)
    return solution.x.reshape(target.shape)


def anchored_means(target, mass, precision, margin=.2):
    """Constrain likelihood-ratio directions, rather than just mean orders.

    mu[c,g] = base + g*tau_source*Sigma*a + c*tau_route*Sigma*b.
    Thus Sigma^-1*(mu[c,weak]-mu[c,strong]) = tau_source*a:
    nuisance coordinates cannot reverse the source emission contrast. Positive
    tau values also preserve the original physical mean-order constraints.
    """
    states, dimensions = target.shape
    covariance = np.linalg.inv(precision)
    source = np.zeros(dimensions)
    source[:2] = .5
    axes = [source] if states == 2 else [source, np.eye(dimensions)[4]]
    design = np.zeros((states, dimensions, dimensions + len(axes)))
    design[:, :, :dimensions] = np.eye(dimensions)
    for index, axis in enumerate(axes):
        levels = np.arange(states) % 2 if index == 0 else np.arange(states) // 2
        design[:, :, dimensions + index] = levels[:, None] * (covariance @ axis)
    hessian = np.einsum('kdp,de,keq,k->pq', design, precision, design, mass)
    linear = np.einsum('kdp,de,ke,k->p', design, precision, target, mass)

    def objective(parameters):
        gradient = hessian @ parameters - linear
        return float(.5 * parameters @ hessian @ parameters - linear @ parameters), gradient

    constraint = np.eye(dimensions + len(axes))[dimensions:]
    lower = np.array([margin / (axis @ covariance @ axis) for axis in axes])
    solution = minimize(objective, np.linalg.solve(hessian, linear), jac=True, method='SLSQP',
                        constraints=LinearConstraint(constraint, lower, np.inf),
                        options=dict(maxiter=200, ftol=1e-10))
    if not solution.success:
        raise RuntimeError('Anchored emission M-step failed: ' + solution.message)
    return np.einsum('kdp,p->kd', design, solution.x)


def transition_update(inputs, expected, weights, penalty, maxiter):
    """Weighted soft multinomial regression for each preceding state."""
    states, _, dimensions = weights.shape
    updated, diagnostics = weights.copy(), []
    for state in range(states):
        counts = expected[:, state]

        def objective(flat):
            parameters = flat.reshape(states, dimensions)
            logits = inputs @ parameters.T
            logprob = logits - logsumexp(logits, axis=1, keepdims=True)
            residual = np.exp(logprob) * counts.sum(axis=1, keepdims=True) - counts
            regularized = parameters.copy()
            regularized[:, 0] = 0.
            loss = -np.sum(counts * logprob) + .5 * penalty * np.sum(regularized ** 2)
            gradient = residual.T @ inputs + penalty * regularized
            return float(loss), gradient.ravel()

        result = minimize(objective, weights[state].ravel(), jac=True, method='L-BFGS-B',
                          options=dict(maxiter=maxiter, ftol=1e-10))
        if not np.isfinite(result.fun):
            raise RuntimeError('Nonfinite transition objective')
        updated[state] = result.x.reshape(states, dimensions)
        diagnostics.append(dict(success=bool(result.success), iterations=int(result.nit),
                                message=str(result.message)))
    return updated, diagnostics


@dataclass
class SequenceModel:
    means: np.ndarray
    covariance: np.ndarray
    weights: np.ndarray
    initial: np.ndarray
    margin: float = .2
    penalty: float = .01

    def potentials(self, values, inputs):
        difference = values[..., None, :] - self.means
        precision = np.linalg.inv(self.covariance)
        quadratic = np.einsum('...kd,de,...ke->...k', difference, precision, difference)
        constant = len(FEATURES) * np.log(2 * np.pi) + np.linalg.slogdet(self.covariance)[1]
        emission = -.5 * (quadratic + constant)
        logits = np.einsum('...d,ijd->...ij', inputs[..., :self.weights.shape[-1]], self.weights)
        return emission, logits - logsumexp(logits, axis=-1, keepdims=True)

    def infer(self, batch):
        values, inputs, valid = batch
        emission, transition = self.potentials(values, inputs)
        return forward_backward(emission, transition, self.initial, valid)

    def score(self, values):
        emission, transition = self.potentials(values, lagged_inputs(values))
        gamma, xi, likelihood = forward_backward(emission[None], transition[None], self.initial,
                                                 np.ones((1, len(values)), dtype=bool))
        iid = emission + np.log(self.initial)
        iid = np.exp(iid - logsumexp(iid, axis=1, keepdims=True))
        weak = np.arange(len(self.means)) % 2 == 1
        enter = xi[0][:, ~weak][:, :, weak].sum(axis=(1, 2))
        exit_ = xi[0][:, weak][:, :, ~weak].sum(axis=(1, 2))
        return dict(risk=gamma[0][:, weak].sum(axis=1), posterior=gamma[0],
                    iid=iid[:, weak].sum(axis=1), enter=enter, exit=exit_,
                    path=viterbi(emission, transition, self.initial)[0], loglik=float(likelihood[0]))

    def arrays(self):
        return {name: getattr(self, name) for name in ('means', 'covariance', 'weights', 'initial')}


def initialize(values, token_weights, states, driven, seed, shrinkage, margin, penalty, anchored=False):
    rng = np.random.default_rng(seed)
    source, behavior = values[:, :2].mean(axis=1), values[:, 4]
    assignments = (source > np.median(source)).astype(int)
    if states == 4:
        assignments += 2 * (behavior > np.median(behavior))
    means = np.array([np.average(values[assignments == state], axis=0,
                                weights=token_weights[assignments == state]) for state in range(states)])
    means += rng.normal(0, .05, means.shape)
    covariance = (values * token_weights[:, None]).T @ values
    covariance = (1 - shrinkage) * covariance + shrinkage * np.diag(np.diag(covariance)) + .05 * np.eye(values.shape[1])
    update = anchored_means if anchored else constrained_means
    means = update(means, np.ones(states), np.linalg.inv(covariance), margin)
    probability = np.full((states, states), .1 / (states - 1))
    np.fill_diagonal(probability, .9)
    weights = np.zeros((states, states, len(INPUTS) if driven else 1))
    weights[:, :, 0] = np.log(probability)
    return SequenceModel(means, covariance, weights, np.full(states, 1 / states), margin, penalty)


def fit_sequence(train, development, answer_weights, dev_weights, states=4, driven=True,
                 seed=42, iterations=12, transition_steps=30, shrinkage=.2, margin=.2, penalty=.01,
                 anchored=False):
    batch, dev_batch = batch_sequences(train), batch_sequences(development)
    values, inputs, valid = batch
    token_weights = np.broadcast_to(answer_weights[:, None], valid.shape)[valid]
    model = initialize(values[valid], token_weights, states, driven, seed, shrinkage, margin, penalty, anchored)
    best, best_dev, history = None, -np.inf, []
    precision = np.linalg.inv(model.covariance)
    for iteration in range(iterations + 1):
        gamma, xi, likelihood = model.infer(batch)
        dev_likelihood = model.infer(dev_batch)[2]
        dev_score = float(dev_weights @ dev_likelihood)
        regularization = .5 * penalty * np.sum(model.weights[:, :, 1:] ** 2)
        record = dict(iteration=iteration, train_loglik=float(answer_weights @ likelihood),
                      objective=float(answer_weights @ likelihood - regularization), dev_loglik=dev_score,
                      occupancy=(gamma * answer_weights[:, None, None]).sum(axis=(0, 1)).tolist(),
                      order_gaps=(ordering(states, len(FEATURES)) @ model.means.ravel()).tolist())
        history.append(record)
        if dev_score > best_dev:
            best_dev = dev_score
            best = SequenceModel(**{name: value.copy() for name, value in model.arrays().items()},
                                 margin=margin, penalty=penalty)
        if iteration == iterations:
            break
        responsibility = gamma[valid] * token_weights[:, None]
        mass = responsibility.sum(axis=0)
        target = (responsibility.T @ values[valid]) / mass[:, None]
        update = anchored_means if anchored else constrained_means
        model.means = update(target, mass, precision, margin)
        starts = gamma[:, 0] * answer_weights[:, None]
        model.initial = starts.sum(axis=0) / starts.sum()
        expected = (xi * answer_weights[:, None, None, None])[valid]
        model.weights, record['transition_solver'] = transition_update(
            inputs[valid, :model.weights.shape[-1]], expected, model.weights, penalty, transition_steps)
    return best, history
