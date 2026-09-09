"""Bidirectional finite post-WO head-message exchange on frozen carriers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from experiments.common.llama_message_intervention import (
    VALIDATED_ATTRIBUTE,
    MessageGate,
    forward_layers,
    validate_manual_forward,
)


@dataclass(frozen=True)
class Carrier:
    layer: int
    head: int
    position: int


@dataclass(frozen=True)
class WitnessConfig:
    doses: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    query_chunk: int = 8

    def check(self):
        dose = np.asarray(self.doses, dtype=float)
        if len(dose) < 2 or not np.isfinite(dose).all() or np.any((dose < 0) | (dose > 1)):
            raise ValueError("doses must be finite values in [0,1]")
        if np.any(np.diff(dose) <= 0):
            raise ValueError("doses must be strictly increasing")
        if dose[0] != 0 or dose[-1] != 1:
            raise ValueError("doses must include 0 and 1")
        if self.query_chunk < 1:
            raise ValueError("query_chunk must be positive")
        return self


class _HeadCapture:
    def __init__(self, carrier):
        self.carrier = carrier
        self.code = None

    def observe_head_output(self, layer, begin, output):
        carrier = self.carrier
        if layer == carrier.layer and begin <= carrier.position < begin + output.shape[2]:
            self.code = output[0, carrier.head, carrier.position - begin].detach().float()


class MessageExchangeWitness:
    """Test one screen-frozen carrier over both directions and every dose."""

    def __init__(self, model, config=None):
        self.model = model
        self.config = (WitnessConfig() if config is None else config).check()

    @torch.inference_mode()
    def run(self, pair, carrier):
        pair.check()
        self._check_carrier(pair, carrier)
        model = self.model.eval()
        device = model.get_input_embeddings().weight.device
        first = torch.as_tensor(pair.token_ids[0], dtype=torch.long, device=device)
        if not getattr(model, VALIDATED_ATTRIBUTE, False):
            validate_manual_forward(model, first)
        messages = torch.stack(
            [self._native_message(pair.token_ids[world], carrier) for world in range(2)]
        )
        candidates = (pair.candidate_plus_ids, pair.candidate_minus_ids)
        baseline = np.empty((2, 2), dtype=np.float64)
        for world in range(2):
            for candidate, candidate_ids in enumerate(candidates):
                baseline[world, candidate] = self._sequence_log_probability(
                    pair.token_ids[world], candidate_ids, gate=None
                )
        doses = np.asarray(self.config.doses, dtype=np.float64)
        probability = np.empty((2, len(doses), 2), dtype=np.float64)
        mean_probability = np.empty_like(probability)
        for world in range(2):
            donor = 1 - world
            delta = messages[donor] - messages[world]
            for dose_index, dose in enumerate(doses):
                addition = float(dose) * delta
                gate = MessageGate(
                    split_layer=0,
                    attention_write_patch={
                        carrier.layer: {carrier.position: addition}
                    },
                )
                for candidate, candidate_ids in enumerate(candidates):
                    score = self._sequence_log_probability(
                        pair.token_ids[world], candidate_ids, gate=gate
                    )
                    probability[world, dose_index, candidate] = score
                    mean_probability[world, dose_index, candidate] = score / len(candidate_ids)
        baseline_error = float(np.max(np.abs(probability[:, 0] - baseline)))
        dtype = model.get_input_embeddings().weight.dtype
        tolerance = 2e-5 if dtype == torch.float32 else 5e-3
        if not np.isfinite(probability).all():
            raise ValueError("message exchange produced nonfinite sequence probabilities")
        if baseline_error > tolerance:
            raise ValueError("dose 0 does not reproduce the native sequence baseline")
        odds = probability[..., 0] - probability[..., 1]
        evidence_odds = odds * np.asarray([1.0, -1.0])[:, None]
        raw_effect = odds - odds[:, :1]
        return {
            "witness_schema": 1,
            "intervention_kind": "post_wo_head_message_exchange",
            "carrier": np.asarray(
                [carrier.layer, carrier.head, carrier.position], dtype=np.int32
            ),
            "doses": doses,
            "candidate_lengths": np.asarray(
                [len(pair.candidate_plus_ids), len(pair.candidate_minus_ids)],
                dtype=np.int32,
            ),
            "native_head_message_pair": messages.cpu().numpy(),
            "native_head_message_delta": (messages[1] - messages[0]).cpu().numpy(),
            "baseline_candidate_log_probability": baseline,
            "candidate_log_probability": probability,
            "candidate_mean_log_probability": mean_probability,
            "sequence_log_odds": odds,
            "evidence_log_odds": evidence_odds,
            "exchange_effect": raw_effect,
            "intervention_selective": 0.5 * (raw_effect[1] - raw_effect[0]),
            "intervention_common": 0.5 * (raw_effect[1] + raw_effect[0]),
            "baseline_reproduction_error": baseline_error,
            "message_capture_forward_calls": 2,
            "forward_calls": 4 + 4 * len(doses),
            "labels_used": False,
        }

    def _check_carrier(self, pair, carrier):
        layers = len(self.model.model.layers)
        heads = int(self.model.config.num_attention_heads)
        if not 0 <= carrier.layer < layers or not 0 <= carrier.head < heads:
            raise ValueError("carrier is outside the model layer/head axes")
        if not pair.response_start <= carrier.position < pair.token_ids.shape[1]:
            raise ValueError("carrier must be inside the shared response-prefix")
        vocab = int(self.model.config.vocab_size)
        candidates = (*pair.candidate_plus_ids, *pair.candidate_minus_ids)
        if min(candidates) < 0 or max(candidates) >= vocab:
            raise ValueError("candidate token is outside the model vocabulary")

    def _native_message(self, prefix, carrier):
        device = self.model.get_input_embeddings().weight.device
        ids = torch.as_tensor(prefix, dtype=torch.long, device=device)[None]
        observer = _HeadCapture(carrier)
        forward_layers(
            self.model,
            self.model.get_input_embeddings()(ids),
            0,
            observer=observer,
            attention_query_chunk=self.config.query_chunk,
            apply_final_norm=False,
        )
        if observer.code is None:
            raise ValueError("carrier head message was not observed")
        attention = self.model.model.layers[carrier.layer].self_attn
        head_dim = int(getattr(attention, "head_dim", attention.q_proj.out_features // self.model.config.num_attention_heads))
        begin = carrier.head * head_dim
        block = attention.o_proj.weight[:, begin : begin + head_dim].float()
        return block @ observer.code

    def _sequence_log_probability(self, prefix, candidate, *, gate):
        device = self.model.get_input_embeddings().weight.device
        prefix = tuple(map(int, prefix))
        candidate = tuple(map(int, candidate))
        ids = torch.as_tensor((*prefix, *candidate), dtype=torch.long, device=device)[None]
        hidden = forward_layers(
            self.model,
            self.model.get_input_embeddings()(ids),
            0,
            gate=gate,
            attention_query_chunk=self.config.query_chunk,
        )
        begin = len(prefix) - 1
        positions = torch.arange(begin, begin + len(candidate), device=device)
        targets = torch.as_tensor(candidate, dtype=torch.long, device=device)
        logits = self.model.lm_head(hidden[0, positions]).float()
        return float(logits.log_softmax(-1)[torch.arange(len(targets), device=device), targets].sum())
