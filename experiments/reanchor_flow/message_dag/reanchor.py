"""Label-free token-level morphology for local-to-remote reanchoring."""

from dataclasses import asdict, dataclass

import numpy as np

REANCHOR_SCHEMA = 1
REANCHOR_TYPES = (
    "broad_diffuse",
    "broad_convergent",
    "broad_diverse",
    "sparse_focal",
    "sparse_diffuse",
)
REQUIRED_SCAN_FIELDS = (
    "remote_positive_gain",
    "remote_gain_focality",
    "remote_gain_effective_sources",
    "remote_gain_distance",
)


@dataclass(frozen=True)
class ReanchorConfig:
    broad_head_fraction: float = 0.50
    head_focality_threshold: float = 0.50
    focal_head_fraction_threshold: float = 0.50
    source_agreement_threshold: float = 0.50

    def check(self):
        values = (
            self.broad_head_fraction,
            self.head_focality_threshold,
            self.focal_head_fraction_threshold,
            self.source_agreement_threshold,
        )
        if not all(np.isfinite(value) and 0 < value <= 1 for value in values):
            raise ValueError("reanchor morphology thresholds must be in (0,1]")
        return self


def _weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


class ReanchorProfiler:
    """Collapse physical read events into one morphology per query token."""

    def __init__(self, config=None):
        self.config = (ReanchorConfig() if config is None else config).check()

    def run(self, scan):
        event = np.asarray(scan["event"], dtype=bool)
        if event.ndim != 3:
            raise ValueError("event must have [layer,head,row] axes")
        row_count = event.shape[2]
        rows = np.asarray(scan["row_position"], dtype=int)
        special = np.asarray(scan["special_mask"], dtype=bool)
        start = int(scan["response_start"])
        if rows.shape != (row_count,) or np.any(np.diff(rows) != 1):
            raise ValueError("reanchor rows must be contiguous model positions")
        if not 0 < start < len(special) or rows.min() < 0 or rows.max() >= len(special):
            raise ValueError("response and row positions are outside the token sequence")
        metrics = {}
        for name in REQUIRED_SCAN_FIELDS:
            value = np.asarray(scan[name], dtype=float)
            if value.shape != event.shape:
                raise ValueError(f"{name} must share the event axes")
            metrics[name] = value
        peak = np.asarray(scan["peak_source"], dtype=int)
        if peak.shape != event.shape:
            raise ValueError("peak_source must share the event axes")
        evidence = (
            np.asarray(scan["evidence_mask"], dtype=bool)
            if "evidence_mask" in scan
            else None
        )
        if evidence is not None and evidence.shape != special.shape:
            raise ValueError("evidence_mask must cover the token sequence")
        token_ids = np.asarray(
            scan.get("token_ids", np.full(len(special), -1)), dtype=np.int64
        )
        token_text = np.asarray(
            scan.get("token_text", np.full(len(special), "")), dtype=str
        )
        if token_ids.shape != special.shape or token_text.shape != special.shape:
            raise ValueError("token IDs and text must cover the token sequence")

        eligible = np.zeros(row_count, dtype=bool)
        if row_count > 1:
            query = rows[1:]
            previous = rows[:-1]
            eligible[1:] = (
                (query >= start)
                & (query + 1 < len(special))
                & ~special[query]
                & ~special[previous]
            )
        reanchor_type = np.full(row_count, "excluded", dtype="<U24")
        reanchor_type[eligible] = "none"
        is_reanchor = eligible & event.any((0, 1))
        dominant_layer = np.full(row_count, -1, dtype=np.int32)
        active_head_fraction = np.full(row_count, np.nan, dtype=np.float32)
        all_layer_head_fraction = np.full(row_count, np.nan, dtype=np.float32)
        active_layer_fraction = np.full(row_count, np.nan, dtype=np.float32)
        head_focality = np.full(row_count, np.nan, dtype=np.float32)
        focal_head_fraction = np.full(row_count, np.nan, dtype=np.float32)
        effective_sources = np.full(row_count, np.nan, dtype=np.float32)
        source_agreement = np.full(row_count, np.nan, dtype=np.float32)
        gain_source_agreement = np.full(row_count, np.nan, dtype=np.float32)
        mean_distance = np.full(row_count, np.nan, dtype=np.float32)
        peak_prompt_fraction = np.full(row_count, np.nan, dtype=np.float32)
        peak_history_fraction = np.full(row_count, np.nan, dtype=np.float32)
        peak_evidence_fraction = np.full(row_count, np.nan, dtype=np.float32)
        dominant_source_position = np.full(row_count, -1, dtype=np.int32)

        positive_gain = metrics["remote_positive_gain"]
        for row in np.flatnonzero(is_reanchor):
            active = event[:, :, row]
            active_gain = positive_gain[:, :, row][active]
            if not np.isfinite(active_gain).all() or np.any(active_gain <= 0):
                raise ValueError("active reanchor heads need positive finite remote gain")
            layer_gain = np.where(active, positive_gain[:, :, row], 0).sum(1)
            layer = int(layer_gain.argmax())
            selected = active[layer]
            weights = positive_gain[layer, selected, row]
            selected_peak = peak[layer, selected, row]
            dominant_layer[row] = layer
            active_head_fraction[row] = selected.mean()
            all_layer_head_fraction[row] = active.mean()
            active_layer_fraction[row] = active.any(1).mean()
            selected_focality = metrics["remote_gain_focality"][layer, selected, row]
            if not np.isfinite(selected_focality).all():
                raise ValueError("active reanchor heads need finite focality")
            focality = _weighted_mean(selected_focality, weights)
            head_focality[row] = focality
            focal_head_fraction[row] = np.mean(
                selected_focality >= self.config.head_focality_threshold
            )
            effective_sources[row] = _weighted_mean(
                metrics["remote_gain_effective_sources"][layer, selected, row],
                weights,
            )
            mean_distance[row] = _weighted_mean(
                metrics["remote_gain_distance"][layer, selected, row], weights
            )
            valid_peak = (selected_peak >= 0) & (selected_peak < len(special))
            if not valid_peak.all() or special[selected_peak].any():
                raise ValueError(
                    "active reanchor heads need valid ordinary peak sources"
                )
            votes = np.bincount(selected_peak, minlength=len(special))
            gain_totals = np.bincount(
                selected_peak, weights=weights, minlength=len(special)
            )
            vote_count = votes.sum()
            gain_total = gain_totals.sum()
            dominant_source_position[row] = int(votes.argmax())
            source_agreement[row] = votes.max() / vote_count
            gain_source_agreement[row] = gain_totals.max() / gain_total
            peak_prompt_fraction[row] = votes[:start].sum() / vote_count
            peak_history_fraction[row] = votes[start:].sum() / vote_count
            if evidence is not None:
                peak_evidence_fraction[row] = votes[evidence].sum() / vote_count
            reanchor_type[row] = self._type(
                active_head_fraction[row],
                focal_head_fraction[row],
                source_agreement[row],
            )

        dominant_source_token_id = np.full(row_count, -1, dtype=np.int64)
        dominant_source_token_text = np.full(row_count, "", dtype=token_text.dtype)
        has_source = dominant_source_position >= 0
        dominant_source_token_id[has_source] = token_ids[
            dominant_source_position[has_source]
        ]
        dominant_source_token_text[has_source] = token_text[
            dominant_source_position[has_source]
        ]

        return {
            "reanchor_schema": REANCHOR_SCHEMA,
            "config": asdict(self.config),
            "row_position": rows,
            "query_token_id": token_ids[rows],
            "query_token_text": token_text[rows],
            "eligible": eligible,
            "is_reanchor": is_reanchor,
            "reanchor_type": reanchor_type,
            "dominant_layer": dominant_layer,
            "active_head_fraction": active_head_fraction,
            "all_layer_head_fraction": all_layer_head_fraction,
            "active_layer_fraction": active_layer_fraction,
            "head_focality": head_focality,
            "focal_head_fraction": focal_head_fraction,
            "effective_sources": effective_sources,
            "source_agreement": source_agreement,
            "gain_source_agreement": gain_source_agreement,
            "mean_distance": mean_distance,
            "peak_prompt_fraction": peak_prompt_fraction,
            "peak_history_fraction": peak_history_fraction,
            "peak_evidence_fraction": peak_evidence_fraction,
            "dominant_source_position": dominant_source_position,
            "dominant_source_token_id": dominant_source_token_id,
            "dominant_source_token_text": dominant_source_token_text,
            "labels_used": False,
            "naming": {
                "query": "Reanchor Query Token (RQT)",
                "source": "Reanchor Source Token (RST)",
            },
        }

    def _type(self, breadth, focal_head_fraction, source_agreement):
        broad = breadth >= self.config.broad_head_fraction
        focal = (
            focal_head_fraction >= self.config.focal_head_fraction_threshold
        )
        if not broad:
            return "sparse_focal" if focal else "sparse_diffuse"
        if not focal:
            return "broad_diffuse"
        if source_agreement >= self.config.source_agreement_threshold:
            return "broad_convergent"
        return "broad_diverse"
