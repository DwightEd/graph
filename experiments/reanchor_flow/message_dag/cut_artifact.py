"""Streaming persistence and closure certification for last-crossing cuts."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

from ..attention_audit import write_array
from .artifacts import AtomicArtifact
from .differential import DifferentialLayer, final_directions


@torch.inference_mode()
def prepare_local_readout(cache, path, *, query_chunk=8, contrasts=None, progress=None):
    """Persist same-position suffix readers shared by all events in a sample."""

    reader = final_directions(cache, contrasts)[0].clone()
    reader[-1] = 0  # the final input row has no captured next-token target
    with ZipFile(path, "w", ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for layer in reversed(range(cache.layers)):
            if progress:
                progress(f"local output readers L{layer+1}/{cache.layers}")
            operator = DifferentialLayer(cache, layer, query_chunk)
            post = reader + operator.mlp_vjp(reader)
            write_array(archive, f"L{layer}", post.cpu().numpy())
            reader = post + operator.attention_same_vjp(post)
            del operator


class CutRecorder:
    """Stream one event's edges and retain exact, small marginal summaries."""

    def __init__(self, path, cache, event_row):
        self.path, self.row = Path(path), int(event_row)
        self.artifact = AtomicArtifact(self.path)
        self.archive = ZipFile(
            self.artifact.temporary,
            "w",
            ZIP_DEFLATED,
            compresslevel=1,
            allowZip64=True,
        )
        layers, heads, targets = cache.layers, cache.heads, cache.rows - 1
        self.signed = np.zeros((layers, heads, targets, 2), np.float32)
        self.positive = np.zeros_like(self.signed)
        self.negative = np.zeros_like(self.signed)
        self.hop_positive = np.zeros((2, targets, 2), np.float32)
        self.hop_negative = np.zeros_like(self.hop_positive)
        self.carriers = np.zeros((targets, cache.rows, 2), np.float32)
        # Display cache only. All edges are written without top-k pruning.
        self.preview_index = np.full((targets, 24, 4), -1, np.int32)
        self.preview_effect = np.zeros((targets, 24), np.float32)
        self.preview_attention = np.zeros_like(self.preview_effect)
        self.rows = cache.trace["row_position"]
        self.saved_layers = set()
        write_array(self.archive, "row_position", self.rows)
        write_array(self.archive, "event_row", np.array(self.row))
        write_array(self.archive, "branch_names", np.array(["V_content", "K_routing"]))

    def write(self, op, begin, end, attention, effects):
        begin_event = max(begin, self.row + 1)
        if begin_event >= end:
            return
        if op.layer not in self.saved_layers:
            if not hasattr(op, "cut_energies"):
                native = op.v[:, op.rows]
                energy = torch.einsum(
                    "hsd,hde,hse->hs", native, op.output_gram, native
                ).clamp_min(0)
                op.cut_energies = (
                    native.square().sum(-1).cpu().numpy(),
                    energy.cpu().numpy(),
                )
            write_array(self.archive, f"value_energy_L{op.layer}", op.cut_energies[0])
            write_array(self.archive, f"write_energy_L{op.layer}", op.cut_energies[1])
            self.saved_layers.add(op.layer)
        # Earlier seed rows cannot be affected; the causal cut excludes
        # future/self sources without an attention threshold.
        selection = slice(begin_event - begin, end - begin)
        values = effects[:, selection, self.row : end - 1].cpu().numpy()
        selected_attention = attention[:, selection, self.row : end - 1].cpu().numpy()
        name = f"L{op.layer}Q{begin_event}"
        write_array(self.archive, name, values)
        write_array(self.archive, "A" + name, selected_attention)
        positive, negative = np.maximum(values, 0), np.maximum(-values, 0)
        self.signed[op.layer, :, begin_event:end] += values.sum(2)
        self.positive[op.layer, :, begin_event:end] += positive.sum(2)
        self.negative[op.layer, :, begin_event:end] += negative.sum(2)
        self.carriers[begin_event:end, self.row : end - 1] += values.sum(0)
        for total, part in (
            (self.hop_positive, positive),
            (self.hop_negative, negative),
        ):
            total[0, begin_event:end] += part[:, :, 0].sum(0)
            total[1, begin_event:end] += part[:, :, 1:].sum((0, 2))
        for index, query in enumerate(range(begin_event, end)):
            flat = values[:, index].reshape(-1)
            count = min(24, len(flat))
            take = np.argpartition(np.abs(flat), len(flat) - count)[-count:]
            take = take[flat[take] != 0]
            head, source, kind = np.unravel_index(take, values[:, index].shape)
            edge_index = np.column_stack(
                (np.full(len(take), op.layer), head, source + self.row, kind)
            )
            candidates = np.r_[self.preview_effect[query], flat[take]]
            order = np.argsort(-np.abs(candidates), kind="stable")[:24]
            self.preview_effect[query] = candidates[order]
            self.preview_attention[query] = np.r_[
                self.preview_attention[query], selected_attention[head, index, source]
            ][order]
            self.preview_index[query] = np.concatenate(
                (self.preview_index[query], edge_index)
            )[order]

    def finish(self, event):
        net = (self.hop_positive - self.hop_negative).sum(-1)
        expected = event["margin_response"][0, 1:]
        error = net - expected
        scale = np.maximum(
            np.abs(expected), (self.hop_positive + self.hop_negative).sum(-1)
        )
        if not (np.isfinite(error).all() and np.isfinite(scale).all()):
            raise ValueError("nonfinite last-crossing response; cannot certify path conservation")
        allowance = 2e-7 + 2e-4 * scale
        if np.any(np.abs(error) > allowance):
            ratio = np.abs(error) / allowance
            hop, target = np.unravel_index(ratio.argmax(), ratio.shape)
            raise ValueError(
                "last-crossing cut does not reconstruct the native 1/2+ hop response: "
                f"event_position={int(self.rows[self.row])}, "
                f"hop={('1', '2+')[hop]}, "
                f"target_position={int(self.rows[target])+1}, "
                f"cut={net[hop, target]:.9g}, expected={expected[hop, target]:.9g}, "
                f"error={error[hop, target]:.9g}, allowance={allowance[hop, target]:.9g}, "
                f"violations={int((ratio > 1).sum())}/{ratio.size}; "
                "no closed artifact was published"
            )
        result = {
            "cut_schema": np.array(1),
            "cut_signed": self.signed,
            "cut_positive": self.positive,
            "cut_negative": self.negative,
            "cut_hop_positive": self.hop_positive,
            "cut_hop_negative": self.hop_negative,
            "cut_carrier_effect": self.carriers,
            "cut_closure_error": error,
            "cut_preview_index": self.preview_index,
            "cut_preview_effect": self.preview_effect,
            "cut_preview_attention": self.preview_attention,
        }
        for name, value in result.items():
            write_array(self.archive, name, value)
        write_array(self.archive, "event_sites", event["event_sites"])
        self.archive.close()
        self.artifact.commit()
        return result

    def close(self):
        self.archive.close()
        self.artifact.discard()
