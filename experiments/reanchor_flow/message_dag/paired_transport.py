"""Aligned selective/common decomposition of two closed last-crossing cuts."""

from pathlib import Path
import re

import numpy as np


SCREEN_SCHEMA = 1
_EDGE_CHUNK = re.compile(r"^L(\d+)Q(\d+)$")


def _edge_chunks(archive):
    result = {}
    for name in archive.files:
        match = _EDGE_CHUNK.fullmatch(name)
        if match:
            result[(int(match[1]), int(match[2]))] = name
    return result


def _validate_cut(archive, path):
    if int(archive["cut_schema"]) != 1:
        raise ValueError(f"{path}: unsupported last-crossing cut schema")
    if archive["branch_names"].tolist() != ["V_content", "K_routing"]:
        raise ValueError(f"{path}: unexpected last-crossing branches")
    error = np.asarray(archive["cut_closure_error"], dtype=float)
    if not np.isfinite(error).all():
        raise ValueError(f"{path}: last-crossing cut is not finite and closed")
    if "cut_hop_positive" in archive and "cut_hop_negative" in archive:
        scale = (
            np.asarray(archive["cut_hop_positive"], dtype=float)
            + np.asarray(archive["cut_hop_negative"], dtype=float)
        ).sum(-1)
        limit = 2e-7 + 2e-4 * scale
    else:
        limit = np.full(error.shape, 2e-7)
    if np.any(np.abs(error) > limit):
        raise ValueError(f"{path}: last-crossing cut is not closed")
    signed = np.asarray(archive["cut_signed"], dtype=float)
    rows = np.asarray(archive["row_position"], dtype=int)
    if signed.ndim != 4 or signed.shape[-1] != 2:
        raise ValueError(f"{path}: cut_signed must be [layer,head,target,branch]")
    if len(rows) != signed.shape[2] + 1:
        raise ValueError(f"{path}: row positions do not cover the target axis")
    if not np.isfinite(signed).all():
        raise ValueError(f"{path}: cut_signed contains nonfinite values")
    return signed, rows, int(archive["event_row"])


class PairedRouteScreen:
    """Compare two cut artifacts through one exact coordinate-alignment seam."""

    def compare(self, plus_path, minus_path):
        plus_path, minus_path = Path(plus_path), Path(minus_path)
        with np.load(plus_path, allow_pickle=False) as plus, np.load(
            minus_path, allow_pickle=False
        ) as minus:
            plus_signed, plus_rows, plus_event = _validate_cut(plus, plus_path)
            minus_signed, minus_rows, minus_event = _validate_cut(minus, minus_path)
            if plus_signed.shape != minus_signed.shape:
                raise ValueError("paired cuts do not share layer/head/target axes")
            if not np.array_equal(plus_rows, minus_rows) or plus_event != minus_event:
                raise ValueError("paired cuts do not share frozen row/event positions")
            shape = plus_signed.shape
            matched_plus = np.zeros(shape, dtype=np.float64)
            matched_minus = np.zeros(shape, dtype=np.float64)
            unmatched_plus = np.zeros(shape, dtype=np.float64)
            unmatched_minus = np.zeros(shape, dtype=np.float64)
            matched_abs = unmatched_plus_abs = unmatched_minus_abs = 0.0
            matched_edges = unmatched_plus_edges = unmatched_minus_edges = 0
            top_index = np.empty(0, dtype=np.int32)
            top_effect = top_plus = top_minus = 0.0
            top_strength = -1.0
            plus_chunks, minus_chunks = _edge_chunks(plus), _edge_chunks(minus)
            for coordinate in sorted(set(plus_chunks) | set(minus_chunks)):
                layer, query = coordinate
                plus_name, minus_name = plus_chunks.get(coordinate), minus_chunks.get(coordinate)
                if plus_name is not None and minus_name is not None:
                    plus_values = np.asarray(plus[plus_name], dtype=np.float64)
                    minus_values = np.asarray(minus[minus_name], dtype=np.float64)
                    if plus_values.shape != minus_values.shape:
                        raise ValueError(f"paired edge chunk {plus_name} changes shape")
                    self._check_chunk(plus_values, layer, query, plus_event, shape)
                    stop = query + plus_values.shape[1]
                    matched_plus[layer, :, query:stop] += plus_values.sum(2)
                    matched_minus[layer, :, query:stop] += minus_values.sum(2)
                    matched_abs += 0.5 * (
                        np.abs(plus_values).sum() + np.abs(minus_values).sum()
                    )
                    matched_edges += plus_values.size
                    selective = 0.5 * (plus_values - minus_values)
                    flat = np.abs(selective).reshape(-1)
                    if len(flat) and float(flat.max()) > top_strength:
                        offset = int(flat.argmax())
                        head, target, source, branch = np.unravel_index(
                            offset, selective.shape
                        )
                        top_strength = float(flat[offset])
                        top_effect = float(selective[head, target, source, branch])
                        top_plus = float(plus_values[head, target, source, branch])
                        top_minus = float(minus_values[head, target, source, branch])
                        top_index = np.asarray(
                            [
                                layer,
                                head,
                                query + target,
                                plus_event + source,
                                branch,
                            ],
                            dtype=np.int32,
                        )
                    continue
                archive, name = (
                    (plus, plus_name) if plus_name is not None else (minus, minus_name)
                )
                values = np.asarray(archive[name], dtype=np.float64)
                self._check_chunk(values, layer, query, plus_event, shape)
                stop = query + values.shape[1]
                if plus_name is not None:
                    unmatched_plus[layer, :, query:stop] += values.sum(2)
                    unmatched_plus_abs += float(np.abs(values).sum())
                    unmatched_plus_edges += values.size
                else:
                    unmatched_minus[layer, :, query:stop] += values.sum(2)
                    unmatched_minus_abs += float(np.abs(values).sum())
                    unmatched_minus_edges += values.size

        plus_partition = matched_plus + unmatched_plus
        minus_partition = matched_minus + unmatched_minus
        world_error = max(
            float(np.max(np.abs(plus_partition - plus_signed), initial=0)),
            float(np.max(np.abs(minus_partition - minus_signed), initial=0)),
        )
        world_scale = max(
            float(np.max(np.abs(plus_signed), initial=0)),
            float(np.max(np.abs(minus_signed), initial=0)),
            1.0,
        )
        if world_error > 2e-7 + 2e-4 * world_scale:
            raise ValueError("matched and unmatched edges do not close the world partition")
        selective = 0.5 * (matched_plus - matched_minus)
        common = 0.5 * (matched_plus + matched_minus)
        paired_error = max(
            float(np.max(np.abs(common + selective - matched_plus), initial=0)),
            float(np.max(np.abs(common - selective - matched_minus), initial=0)),
        )
        total_abs = matched_abs + 0.5 * (unmatched_plus_abs + unmatched_minus_abs)
        carrier_response = selective.sum(-1)
        if matched_edges == 0:
            status = "no_edge"
        elif matched_abs == 0:
            status = "no_response"
        elif not np.any(carrier_response):
            status = "no_selective_response"
        else:
            status = "eligible"
        carrier_index = np.empty(0, dtype=np.int32)
        if status == "eligible":
            carrier_index = np.asarray(
                np.unravel_index(np.abs(carrier_response).argmax(), carrier_response.shape),
                dtype=np.int32,
            )
        return {
            "screen_schema": SCREEN_SCHEMA,
            "status": status,
            "row_position": plus_rows,
            "event_row": plus_event,
            "matched_plus_signed": matched_plus.astype(np.float32),
            "matched_minus_signed": matched_minus.astype(np.float32),
            "selective_signed": selective.astype(np.float32),
            "common_signed": common.astype(np.float32),
            "unmatched_plus_signed": unmatched_plus.astype(np.float32),
            "unmatched_minus_signed": unmatched_minus.astype(np.float32),
            "matched_absolute_mass": matched_abs,
            "unmatched_plus_absolute_mass": unmatched_plus_abs,
            "unmatched_minus_absolute_mass": unmatched_minus_abs,
            "matched_fraction": matched_abs / total_abs if total_abs else float("nan"),
            "matched_edges": matched_edges,
            "unmatched_plus_edges": unmatched_plus_edges,
            "unmatched_minus_edges": unmatched_minus_edges,
            "top_edge_index": top_index,
            "top_edge_selective": top_effect,
            "top_edge_plus": top_plus,
            "top_edge_minus": top_minus,
            "carrier_index": carrier_index,
            "world_partition_error": world_error,
            "paired_closure_error": paired_error,
            "labels_used": False,
        }

    @staticmethod
    def _check_chunk(values, layer, query, event, cut_shape):
        layers, heads, targets, branches = cut_shape
        if values.ndim != 4 or values.shape[0] != heads or values.shape[-1] != branches:
            raise ValueError("last-crossing edge chunk has invalid head/branch axes")
        if not (0 <= layer < layers and event < query < targets + 1):
            raise ValueError("last-crossing edge chunk is outside cut coordinates")
        if query + values.shape[1] > targets or event + values.shape[2] > targets:
            raise ValueError("last-crossing edge chunk exceeds the target/source axes")
        if not np.isfinite(values).all():
            raise ValueError("last-crossing edge chunk contains nonfinite values")
