"""Cached attention and strictly prefix-only confidence controls; no value claims."""

import numpy as np
import pandas as pd
from tqdm import tqdm

from .paired_inputs import claim_trace, phase_positions


def saved_entropy_nats(trace):
    """Old sampler caches omit full-vocabulary entropy; top-5 cannot recover it."""
    if "logit_entropy" not in trace:
        return np.full(trace["attention"].shape[2], np.nan)
    return trace["logit_entropy"] * np.log(2.)  # Saved sampler entropy is in bits.


def confidence_controls(trace, vocabulary_size, alpha=.9):
    """RAUQ-style causal adaptation, not the paper's full-answer head selection.

    Stored generation row t is the query P+t-1 predicting response token t.
    Its diagonal key is the previously consumed token, not the unobserved y_t.
    The first answer token has no previous answer token and initializes s only.
    """
    attention = trace["attention"]
    length = attention.shape[2]
    queries = int(trace["prompt_length"]) + np.arange(length) - 1
    previous = attention[:, :, np.arange(length), queries].astype(np.float32)
    entropy = saved_entropy_nats(trace)
    signal = np.maximum(np.log(vocabulary_size) - entropy, np.finfo(float).tiny)
    layers, heads = previous.shape[:2]
    confidence = np.full((layers, heads), signal[0])
    cumulative = np.zeros((layers, heads))
    scores, chosen = [], []
    middle = slice(layers // 3, max(layers // 3 + 1, 2 * layers // 3))
    smooth = signal[0]
    for position in range(length):
        if position:
            cumulative += previous[:, :, position]
            confidence = alpha * signal[position] + (1 - alpha) * previous[:, :, position] * confidence
            smooth = alpha * signal[position] + (1 - alpha) * smooth
        selected = cumulative.argmax(axis=1)
        layer_score = -np.log(confidence[np.arange(layers), selected])
        scores.append((float(-np.log(signal[position])), float(-np.log(smooth)),
                       float(layer_score[middle].max())))
        chosen.append(selected)
    return np.asarray(scores), np.asarray(chosen), previous


def screen_pair(directory, samples, case, vocabulary_size):
    lookup = {(str(row["source_id"]), row["seed"]): row for row in samples}
    rows, traces = [], {}
    for side in ("supported", "unsupported"):
        record = lookup[case["source_id"], case[side]["seed"]]
        trace, span = claim_trace(directory, record, case[side])
        scores, heads, previous = confidence_controls(trace, vocabulary_size)
        prompt = int(trace["prompt_length"])
        positions = phase_positions(span, len(scores), trace["special_mask"][prompt:])
        entropy = saved_entropy_nats(trace)
        entropy_status = "saved_bits" if "logit_entropy" in trace else "not_saved"
        if entropy_status == "not_saved":
            tqdm.write(f"{case['case_id']}/{side}: logit_entropy not saved; entropy/EWMA/RAUQ controls unavailable. Native audit can continue.")
        surprisal = trace["log_normalizer"] - trace["chosen_logit"]
        for phase, position in positions.items():
            rows.append(dict(case_id=case["case_id"], source_id=case["source_id"],
                side=side, phase=phase, position=position, entropy_nats=float(entropy[position]),
                entropy_status=entropy_status,
                surprisal_nats=float(surprisal[position]), local_confidence_score=scores[position, 0],
                ewma_score=scores[position, 1], prefix_rauq_score=scores[position, 2],
                label_scope="reviewed_claim" if phase in ("onset", "back_half") else "unreviewed_neighbor"))
        traces[side + "_scores"] = scores
        traces[side + "_entropy_nats"] = entropy
        traces[side + "_entropy_status"] = np.array(entropy_status)
        traces[side + "_selected_heads"] = heads
        traces[side + "_previous_token_attention"] = previous
    return rows, traces


def screen_pairs(directory, samples, cases, output, vocabulary_size):
    rows = []
    destination = output / "screen"
    destination.mkdir(exist_ok=True)
    for case in tqdm(cases, desc="paired cached controls", unit="pair"):
        measured, arrays = screen_pair(directory, samples, case, vocabulary_size)
        np.savez_compressed(destination / (case["case_id"] + ".npz"), **arrays)
        rows.extend(measured)
    pd.DataFrame(rows).to_csv(output / "confidence_controls.csv", index=False)
