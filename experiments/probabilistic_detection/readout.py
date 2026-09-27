"""Controlled discriminative approximation to conditional log odds.

Linear, marginal-square, interaction and source-conditioned designs are nested.
Their comparison tests predictive structure, not causal information transport.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


DESIGNS = ("linear", "squares", "interactions", "conditioned")
REGULARIZATION = (.01, 1.)
ABLATIONS = ("no_logp", "no_route", "context_only", "no_position")


def design_matrix(context, observations, transform, design="linear", ablation=None):
    phi = transform._context_design(context)[:, 1:]
    z = transform.observation_transform_.transform(observations)
    if ablation == "no_position":
        # SplineTransformer emits one contiguous block per context feature.
        width = phi.shape[1] // context.shape[1]
        phi = phi[:, :2 * width]
    if ablation == "context_only":
        return phi
    if ablation == "no_logp":
        z = np.delete(z, [2, 3], axis=1)
    elif ablation == "no_route":
        z = np.delete(z, [4, 5, 7, 9], axis=1)
    blocks = [phi, z]
    if design in ("squares", "interactions", "conditioned"):
        blocks.append(z ** 2)
    if design in ("interactions", "conditioned"):
        left, right = np.triu_indices(z.shape[1], k=1)
        blocks.append(z[:, left] * z[:, right])
    if design == "conditioned":
        anchors = transform.context_scaler_.transform(context)[:, :2]
        blocks.extend(anchors[:, index, None] * z for index in range(2))
    return np.column_stack(blocks)


class QuadraticReadout:
    """L2 logistic readout of frozen, train-fitted native-response features."""

    def __init__(self, transform, design="linear", regularization=1., ablation=None):
        if design not in DESIGNS or ablation not in (None, *ABLATIONS):
            raise ValueError("Unknown controlled readout design or ablation")
        self.transform = transform
        self.design = design
        self.regularization = regularization
        self.ablation = ablation

    def matrix(self, context, observations):
        return design_matrix(context, observations, self.transform, self.design, self.ablation)

    def fit_matrix(self, matrix, labels, weights):
        self.scaler_ = StandardScaler().fit(matrix, sample_weight=weights)
        self.classifier_ = LogisticRegression(C=self.regularization, max_iter=1500, random_state=42)
        self.classifier_.fit(self.scaler_.transform(matrix), labels, sample_weight=weights)
        return self

    def decision_function(self, context, observations):
        matrix = self.matrix(context, observations)
        return self.classifier_.decision_function(self.scaler_.transform(matrix))
