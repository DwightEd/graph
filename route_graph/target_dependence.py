"""Observed whole-span logP interventions, independent of a correct alternative.

These measurements establish finite target dependence under observer replay.
They do not by themselves establish factual correctness or error adoption.
"""

import hashlib
import math

import torch

from route_graph.frozen_reader import digest
from route_graph.native_audit import NativeGate, native_forward


def _tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def observed_target(row, alignment, span):
    a, b = span
    if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(row["response"]):
        raise ValueError("invalid target span")
    offsets = alignment["response_offsets"]
    relative = [i for i, (l, r) in enumerate(offsets) if l < b and r > a and r > l]
    if not relative or relative != list(range(relative[0], relative[-1] + 1)):
        raise ValueError("target must map to a complete contiguous token event")
    p = row["prompt_length"]
    first, end = p + relative[0], p + relative[-1] + 1
    return {"input_ids": row["token_ids"][:end - 1], "targets": row["token_ids"][first:end],
            "first_prediction": first - 1, "prompt_length": p,
            "response_span": [a, b], "token_span": [first, end],
            "boundary_token_spans": [list(offsets[relative[0]]), list(offsets[relative[-1]])],
            "queries": list(range(p - 1, end - 1)),
            "scope": "full_original_token_span_sum; boundary_overlap_recorded; no_future_response"}


class TargetOracle:
    def __init__(self, model, target, budget=64, artifact_writer=None):
        if type(budget) is not int or budget < 1:
            raise ValueError("target budget must be a positive integer")
        self.model, self.target, self.budget = model, target, budget
        self.calls = self.tokens_processed = 0
        self.records, self.cache = [], {}
        self.artifact_writer = artifact_writer
        self.donor_artifacts = []
        if model.training or not target["targets"] or target["first_prediction"] < 0:
            raise ValueError("invalid observed target or training-mode model")
        if target["first_prediction"] + len(target["targets"]) != len(target["input_ids"]):
            raise ValueError("whole target sequence does not end at final prediction")
        self.baseline_logits = None
        self.base = self.measure([])

    def _forward(self, gates=(), **kwargs):
        def count(module, args):
            if self.calls >= self.budget:
                raise RuntimeError("target_forward_budget_exhausted")
            self.calls += 1
            self.tokens_processed += len(self.target["input_ids"])
        handle = self.model.register_forward_pre_hook(count)
        try:
            return native_forward(self.model, self.target["input_ids"], gates, **kwargs)
        finally:
            handle.remove()

    def measure(self, gates, *, force=False):
        for gate in gates:
            if not set(gate.queries) <= set(self.target["queries"]):
                raise ValueError("gate queries escape observed response prefix")
        specification = []
        for gate in gates:
            item = {k: v for k, v in vars(gate).items() if k != "donor"}
            if gate.donor is not None:
                item["donor_sha256"] = gate.donor.values_sha256
            specification.append(item)
        key = digest(specification)
        if not force and key in self.cache:
            return self.cache[key]
        logits, _, diagnostics = self._forward(gates)
        prefix, sham = True, None
        if self.baseline_logits is None:
            self.baseline_logits = logits.cpu()
            self.baseline_sha256 = _tensor_hash(self.baseline_logits)
        else:
            earliest = min((min(g.queries) for g in gates), default=len(logits))
            prefix = torch.equal(logits[:earliest].cpu(), self.baseline_logits[:earliest])
            is_sham = all(g.expected_input_scale[1] == 1 if g.kind == "donor" else g.strength == 1 for g in gates)
            if is_sham:
                sham = _tensor_hash(logits) == self.baseline_sha256
        if not prefix or sham is False:
            raise RuntimeError("target prefix/sham integrity failure")
        begin = self.target["first_prediction"]
        selected = logits[begin:].float()
        targets = torch.tensor(self.target["targets"], device=logits.device)
        logps = (selected.gather(1, targets[:, None]).squeeze(1) - selected.logsumexp(-1)).cpu().tolist()
        logp = math.fsum(logps)
        if not math.isfinite(logp):
            raise ValueError("nonfinite whole-target log probability")
        result = {"key": key, "logp": logp, "token_logps": logps, "gates": specification,
                  "prefix_exact": prefix, "sham_exact": sham, "diagnostics": diagnostics,
                  "actual_forward_index": self.calls}
        self.records.append(result)
        self.cache[key] = result
        return result

    def gates(self, group, strength=0):
        return [NativeGate(layer=layer, queries=tuple(group["queries"]), kind=group.get("kind", "content"),
                           strength=strength, keys=tuple(group["domain"]), selected=tuple(group["keys"]))
                for layer in group["layers"]]

    def group(self, group, strength=0, force=False):
        result = self.measure(self.gates(group, strength), force=force)
        return {**result, "delta": self.base["logp"] - result["logp"]}

    def origin(self, group, origin, strength, *, force=False):
        """Change only selected raw embeddings, mediate via selected V messages.

        A raw donor pass recomputes its entire prefix; recipient restores every
        unselected value and attention pattern. Each new origin uses two actual
        forwards. Saved native donor bytes are required for run-level auditing.
        """
        if not origin or not set(origin) <= set(group["keys"]):
            raise ValueError("origin must be a nonempty subset of mapped group keys")
        if self.calls + 2 > self.budget:
            raise RuntimeError("target_forward_budget_exhausted_before_origin_pair")
        capture = [(layer, tuple(group["keys"])) for layer in group["layers"]]
        scale = (tuple(origin), strength)
        logits, values, _ = self._forward(input_scale=scale, capture=capture)
        donor_sham = _tensor_hash(logits) == self.baseline_sha256 if strength == 1 else None
        del logits
        if donor_sham is False:
            raise RuntimeError("origin donor sham differs from exact baseline")
        if set(values) != set(group["layers"]):
            raise ValueError("origin donor capture layers differ")
        for layer, value in values.items():
            if (value.keys != tuple(group["keys"]) or value.layer != layer
                    or value.input_ids != tuple(self.target["input_ids"])
                    or value.input_scale != scale or value.model_identity != id(self.model)
                    or value.values.dtype != self.model.dtype
                    or value.values.shape != (len(group["keys"]), self.model.config.num_key_value_heads,
                                               self.model.model.layers[layer].self_attn.head_dim)
                    or not torch.isfinite(value.values).all()
                    or _tensor_hash(value.values) != value.values_sha256):
                raise ValueError("invalid origin donor capture before publication")
        if self.artifact_writer is not None:
            self.donor_artifacts.append(self.artifact_writer(values, "observed", scale))
        gates = [NativeGate(layer, tuple(group["queries"]), "donor", keys=tuple(group["keys"]),
                            selected=tuple(group["keys"]), donor=values[layer], expected_input_scale=scale)
                 for layer in group["layers"]]
        result = self.measure(gates, force=force)
        return {**result, "delta": self.base["logp"] - result["logp"], "origin": list(origin),
                "strength": strength, "donor_sham_exact": donor_sham}


def localize_queries(oracle, group, budget=8):
    """Bounded bisection retains parents and unsearched intervals.

    Search includes the complete earlier response, even across sentence breaks.
    It does not claim a unique node or completeness from a bounded search.
    """
    queries = group["queries"]
    if (not queries or queries != list(range(queries[0], queries[-1] + 1))
            or type(budget) is not int or budget < 0):
        raise ValueError("localization requires a contiguous ordered query interval and nonnegative budget")
    pending = [(float("inf"), group)]
    measured, seen = [], set()
    excluded = []
    start = oracle.calls
    while pending and oracle.calls - start < budget:
        pending.sort(key=lambda item: (-item[0], digest(item[1])))
        _, current = pending.pop(0)
        gid = digest(current)
        if gid in seen:
            continue
        seen.add(gid)
        before = oracle.calls
        result = oracle.group(current)
        measured.append({"id": gid, "group": current, "delta": result["delta"],
                         "measurement": result["key"], "forward_calls": oracle.calls - before})
        queries = current["queries"]
        if len(queries) > 1:
            middle = len(queries) // 2
            for part in (queries[:middle], queries[middle:]):
                if min(current["keys"]) <= max(part):
                    pending.append((abs(result["delta"]), {**current, "queries": part}))
                else:
                    excluded.append({"group": {**current, "queries": part}, "status": "causally_invisible"})
    return {"measured": measured, "unsearched": [g for _, g in pending], "excluded": excluded,
            "complete": not pending, "scope": "query_interval_ranking; no_unique_lookback_ground_truth",
            "actual_forward_calls": oracle.calls - start}
