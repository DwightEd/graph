"""Export actual token representations and label-conditioned descriptive data."""

import numpy as np
from state_audit.storage import write_arrays

from .dynamics_fit import profile_matrix, project


def token_vectors(observation, coordinates):
    state = (observation["state"] - coordinates["state_mean"]) / coordinates["state_scale"]
    previous = np.concatenate((np.zeros_like(state[:1]), state[:-1]))
    profile = project(profile_matrix(observation), coordinates["profile"])
    history = np.column_stack((previous, project(observation["history"], coordinates["history"])))
    return {"z": state, "v": profile, "u": project(observation["source_input"], coordinates["source_input"]),
            "h": history, "observation": np.column_stack((state, profile))}


def feature_groups(observation, vectors):
    layers, heads = observation["profile"].shape[1:3]
    rank = observation["state"].shape[1]
    profile_names = [f"layer={layer}/head={head}/{name}" for layer in range(layers)
                     for head in range(heads) for name in observation["profile_channels"]]
    ffn_width = observation["ffn"].shape[1] // layers
    ffn_names = [f"layer={layer}/{('state' if axis < rank else 'choice')}_{axis if axis < rank else axis - rank}"
                 for layer in range(layers) for axis in range(ffn_width)]
    count = len(observation["state"])
    groups = {"head": (observation["profile"].reshape(count, -1), profile_names),
              "ffn": (observation["ffn"], ffn_names)}
    for name, values in vectors.items():
        groups[name] = (values, [f"coordinate={axis}" for axis in range(values.shape[1])])
    names = ("entropy", "surprisal", "candidate_tail_mass", "future_query_count")
    groups["output"] = (np.column_stack([observation[name] for name in names]), names)
    return groups


def accumulate_features(accumulator, observation, vectors, labels, valid, probability):
    """Streaming token-weighted moments; no stacking all answers' head tensors."""
    for family, (values, names) in feature_groups(observation, vectors).items():
        for scope, selected in (("all", valid), ("H_ge_0_9", valid & (probability >= .9))):
            key = (family, scope)
            if key not in accumulator:
                accumulator[key] = {"names": names, "count": np.zeros(2, dtype=int),
                                    "sum": np.zeros((2, values.shape[1])), "square": np.zeros((2, values.shape[1]))}
            item = accumulator[key]
            for label in (0, 1):
                data = values[selected & (labels == label)].astype(np.float64)
                item["count"][label] += len(data)
                item["sum"][label] += data.sum(0)
                item["square"][label] += np.square(data).sum(0)


def feature_rows(accumulator):
    result = []
    for (family, scope), item in accumulator.items():
        count = item["count"]
        mean = item["sum"] / np.maximum(count[:, None], 1)
        variance = np.maximum(0, item["square"] / np.maximum(count[:, None], 1) - mean ** 2)
        scale = np.sqrt(variance.mean(0))
        for axis, name in enumerate(item["names"]):
            both = bool((count > 0).all())
            difference = float(mean[1, axis] - mean[0, axis]) if both else None
            result.append({"family": family, "scope": scope, "feature": str(name),
                "normal_tokens": int(count[0]), "error_tokens": int(count[1]),
                "normal_mean": float(mean[0, axis]) if count[0] else None,
                "error_mean": float(mean[1, axis]) if count[1] else None,
                "normal_std": float(np.sqrt(variance[0, axis])) if count[0] else None,
                "error_std": float(np.sqrt(variance[1, axis])) if count[1] else None,
                "error_minus_normal": difference,
                "standardized_difference": difference / float(scale[axis]) if both and scale[axis] > 0 else None})
    return result


def projection_rows(identity, observation, coordinates, labels, valid, probability):
    result = []
    for name in ("source_input", "history", "profile"):
        values = profile_matrix(observation) if name == "profile" else observation[name]
        projection = coordinates[name]
        centered = ((values - projection["mean"]) / projection["scale"]).astype(np.float64)
        basis = projection["components"].astype(np.float64)
        basis /= np.maximum(np.linalg.norm(basis, axis=1, keepdims=True), 1e-12)
        total = np.square(centered).sum(1)
        retained = np.square(centered @ basis.T).sum(1)
        for scope, selected in (("all", valid), ("H_ge_0_9", valid & (probability >= .9))):
            for label in (0, 1):
                mask = selected & (labels == label)
                energy = float(total[mask].sum())
                result.append({"response_id": identity, "projection": name, "scope": scope, "label": label,
                               "tokens": int(mask.sum()), "total_energy": energy,
                               "retained_energy": float(retained[mask].sum()),
                               "outside_fraction": max(0., 1 - float(retained[mask].sum()) / energy) if energy > 0 else None,
                               "train_retained_variance": float(projection["retained_variance"])})
    return result


def export_vectors(path, observation, vectors, labels, valid, posterior):
    write_arrays(path, **vectors, target=observation["target"], query=observation["query"],
                 token_id=observation["token_id"], labels=labels, valid_tokens=valid,
                 posterior=posterior["smoothed"], log_posterior=posterior["log_smoothed"],
                 log_filtered=posterior["log_filtered"])


def representation_schema(observation, vectors, capture):
    return {
        "unit": "one_prediction_query_per_answer_token; query=P+target-1",
        "graph_neural_network": False, "labels_in_representation": False,
        "raw": {name: {"shape": list(observation[name].shape), "dtype": str(observation[name].dtype)}
                for name in ("state", "profile", "source_input", "history", "ffn")},
        "head_channel_order": observation["profile_channels"].tolist(),
        "raw_token_cache": {"group_effect_axes": ["layer", "head", "source_blocks+4", "state_rank+choices"],
                            "group_order": ["source_blocks", "history", "self", "other", "special"],
                            "mapping": "capture/NNNN/sources.json", "loaded_by_audit": False},
        "ffn_axes": ["layer", f"{capture['rank']} fixed state axes then {capture['choices']} token-local candidate roles"],
        "vectors": {name: {"width": value.shape[1], "dtype": str(value.dtype)} for name, value in vectors.items()},
        "definitions": {
            "z": "standardized fixed orthonormal random projection of final normalized hidden",
            "v": "reference SVD of flattened layer/head profile, FFN responses, log1p entropy/surprisal, tail mass",
            "u": "reference SVD of source response vectors, layer/head identities concatenated",
            "h": "concat(previous z, reference SVD of actual head-weighted historical states)",
            "observation": "concat(z,v)",
            "local_response": "native downstream directional derivative; detached past KV",
            "source_covariance": "per-source joint-head directional mixture; used by emissions; not duplicated by audit",
        },
        "model": {"modes": ["E", "H"], "H_input_matrix": 0,
                  "mean": "concat(A_s h + B_s u + b_s, profile_center_s)",
                  "covariance": "Q_s + rho B_aug,s source_covariance B_aug,s.T",
                  "training": "source-equal reference marginal likelihood; no truth labels",
                  "risk": "posterior(H | complete answer observations)",
                  "binary_truth_classifier": False},
        "statistics": {"feature_weighting": "tokens; descriptive, no independence or causal claims",
                       "H_ge_0_9": "fixed mode-probability diagnostic stratum, not a hallucination threshold",
                       "ffn_sign": "coordinate sign, not semantic support/opposition"},
    }
