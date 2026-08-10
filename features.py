"""Minimal hidden-state and token-stat feature storage."""

from pathlib import Path

import numpy as np
import torch


HIDDEN_FIELDS = ("token_ids", "hidden_layer_ids", "hidden_states")
STAT_FIELDS = ("token_ids", "token_log_prob", "entropy")
NODE_FEATURE_MODES = (
    "none",
    "attention",
    "hidden",
    "stats",
    "attention+hidden",
    "attention+stats",
    "hidden+stats",
    "all",
)


def save_hidden_features(path, token_ids, layer_ids, hidden_states) -> None:
    """Store selected hidden layers as [K,N,D] float16."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        token_ids=torch.as_tensor(token_ids).cpu().to(torch.int32).numpy(),
        hidden_layer_ids=np.asarray(list(layer_ids), dtype=np.int16),
        hidden_states=torch.as_tensor(hidden_states).cpu().to(torch.float16).numpy(),
    )


def load_hidden_features(path, device="cpu"):
    with np.load(Path(path), allow_pickle=False) as arrays:
        if set(arrays.files) != set(HIDDEN_FIELDS):
            raise ValueError(f"hidden sample must contain exactly {HIDDEN_FIELDS}")
        token_ids = torch.from_numpy(arrays["token_ids"].astype(np.int64, copy=False))
        layer_ids = torch.from_numpy(arrays["hidden_layer_ids"].astype(np.int64, copy=False))
        hidden = torch.from_numpy(arrays["hidden_states"])
    return token_ids.to(device), layer_ids.to(device), hidden.to(device)


def save_token_stats(path, token_ids, token_log_prob, entropy) -> None:
    """Store compact statistics derived from logits; full logits are not persisted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        token_ids=torch.as_tensor(token_ids).cpu().to(torch.int32).numpy(),
        token_log_prob=torch.as_tensor(token_log_prob).cpu().to(torch.float32).numpy(),
        entropy=torch.as_tensor(entropy).cpu().to(torch.float32).numpy(),
    )


def load_token_stats(path, device="cpu"):
    with np.load(Path(path), allow_pickle=False) as arrays:
        if set(arrays.files) != set(STAT_FIELDS):
            raise ValueError(f"token-stat sample must contain exactly {STAT_FIELDS}")
        token_ids = torch.from_numpy(arrays["token_ids"].astype(np.int64, copy=False))
        log_prob = torch.from_numpy(arrays["token_log_prob"])
        entropy = torch.from_numpy(arrays["entropy"])
    return token_ids.to(device), log_prob.to(device), entropy.to(device)


def teacher_forced_stats(logits, token_ids, chunk_size=64):
    """Align log p(x_t|x_<t) and LM entropy to token position t.

    Position 0 has no previous-token prediction, so both outputs are zero there.
    Every response token is valid because response_idx > 0.
    """
    values = torch.as_tensor(logits)
    if values.ndim == 3:
        values = values[0]
    tokens = torch.as_tensor(token_ids, device=values.device).long()
    N = len(tokens)
    log_prob = torch.zeros(N, dtype=torch.float32, device=values.device)
    entropy = torch.zeros(N, dtype=torch.float32, device=values.device)

    for start in range(0, max(0, N - 1), chunk_size):
        end = min(N - 1, start + chunk_size)
        z = values[start:end].float()
        log_z = torch.logsumexp(z, dim=-1)
        target = tokens[start + 1 : end + 1]
        log_prob[start + 1 : end + 1] = z.gather(1, target[:, None]).squeeze(1) - log_z
        p = torch.softmax(z, dim=-1)
        entropy[start + 1 : end + 1] = log_z - (p * z).sum(dim=-1)
    return log_prob, entropy


def _check_alignment(reference, other, name):
    if not torch.equal(reference.to(other.device), other):
        raise ValueError(f"{name} token_ids do not match attention token_ids")


def load_node_features(split_root, sample, mode="attention"):
    """Assemble node features without changing graph topology."""
    if mode not in NODE_FEATURE_MODES:
        raise ValueError(f"node feature mode must be one of {NODE_FEATURE_MODES}")
    if mode == "none":
        return torch.empty((sample.num_tokens, 0), dtype=torch.float32, device=sample.token_ids.device)

    parts = []
    requested = {"attention", "hidden", "stats"} if mode == "all" else set(mode.split("+"))
    if "attention" in requested:
        parts.append(
            sample.attention_diagonal.reshape(sample.num_channels, sample.num_tokens)
            .T.contiguous()
        )

    root = Path(split_root)
    if "hidden" in requested:
        token_ids, _, hidden = load_hidden_features(
            root / "hidden" / f"{sample.sample_id}.npz", sample.token_ids.device
        )
        _check_alignment(sample.token_ids, token_ids, "hidden")
        parts.append(hidden.permute(1, 0, 2).reshape(sample.num_tokens, -1))

    if "stats" in requested:
        token_ids, log_prob, entropy = load_token_stats(
            root / "token_stats" / f"{sample.sample_id}.npz", sample.token_ids.device
        )
        _check_alignment(sample.token_ids, token_ids, "token_stats")
        parts.append(torch.stack((log_prob, entropy), dim=1))

    return torch.cat([part.float() for part in parts], dim=1)
