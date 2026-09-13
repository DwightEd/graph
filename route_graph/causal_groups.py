"""Evidence-candidate-blind group search using actual finite model forwards."""

import heapq

import numpy as np
import torch

from route_graph.causal_contrast import ContinuationContrast
from route_graph.evidence_anchor import text_units
from route_graph.frozen_reader import digest
from route_graph.native_audit import NativeGate, native_forward


def build_contrast(row, question, tokenizer, edit_index=0):
    edit = question["contrast"]["edits"][edit_index]
    if not edit["valid"]:
        raise ValueError("invalid semantic edit")
    end = next(
        unit["end"]
        for unit in text_units(row["response"])
        if unit["end"] >= question["claim_span"][1]
    )
    a, b = question["answer_span"]
    original = row["response"][:end]
    altered = original[:a] + edit["replacement"] + original[b:]
    prompt = row["token_ids"][: row["prompt_length"]]
    sequences = [
        prompt + tokenizer(t, add_special_tokens=False)["input_ids"]
        for t in (original, altered)
    ]
    if sequences[0] != row["token_ids"][: len(sequences[0])]:
        raise ValueError("original event is not an exact prefix of observer replay")
    contrast = ContinuationContrast.from_sequences(sequences, len(prompt))
    if any(len(tokens) > 96 for tokens in contrast.continuations):
        raise ValueError("complete contrast exceeds 96 continuation tokens")
    return contrast, {
        "kind": edit["kind"],
        "original": original,
        "alternative": altered,
        "response_end": end,
        "answer_spans": [[a, b], [a, a + len(edit["replacement"])]],
        "query_scope": "shared_prefix_only",
        "edit": edit,
    }


def _gate_record(gate):
    result = {key: value for key, value in vars(gate).items() if key != "donor"}
    if gate.donor is not None:
        result["donor"] = {
            "values_sha256": gate.donor.values_sha256,
            "input_ids_sha256": digest(gate.donor.input_ids),
            "layer": gate.donor.layer,
            "keys": gate.donor.keys,
            "input_scale": gate.donor.input_scale,
        }
    return result


class CausalOracle:
    """Two full branch forwards per measurement, exact prefix/sham checks.

    The caller owns a single-Claim budget, selects semantic controls in a later
    phase, and serializes records. Native candidate proposal cannot see them.
    """

    def __init__(
        self, model, contrast, budget=256, input_limit=4096, artifact_writer=None
    ):
        self.model, self.contrast = model, contrast
        self.budget, self.calls = budget, 0
        self.tokens_processed = 0
        self.records, self.cache = [], {}
        self.sham_kinds = set()
        self.artifact_writer = artifact_writer
        self.donor_artifacts = []
        self.ids = [
            list(contrast.prefix + continuation)[:-1]
            for continuation in contrast.continuations
        ]
        if len(self.ids) != 2 or max(map(len, self.ids)) > input_limit:
            raise ValueError("native teacher requires two branches within input limit")
        self.baselines = []
        self.profile = None
        self.base = self.measure([[], []], capture_profile=True)
        self.base_f = self.base["F"]

    def _reserve(self, count):
        if self.calls + count > self.budget:
            raise RuntimeError("native_forward_budget_exhausted")

    def _forward(self, ids, *args, **kwargs):
        def count_actual_forward(module, inputs):
            self._reserve(1)
            self.calls += 1
            self.tokens_processed += len(ids)

        handle = self.model.register_forward_pre_hook(count_actual_forward)
        try:
            return native_forward(self.model, ids, *args, **kwargs)
        finally:
            handle.remove()

    def _profile_hooks(self):
        layers = len(self.model.model.layers)
        query_start = self.contrast.prompt_length - 1
        queries, length = len(self.contrast.shared_queries), len(self.contrast.prefix)
        self.profile = np.zeros((3, queries, length), dtype=np.float32)
        counts = np.zeros(3, dtype=np.int32)
        handles = []
        for index, layer in enumerate(self.model.model.layers):
            band = (
                0
                if index < layers * 10 // 32
                else (1 if index < layers * 22 // 32 else 2)
            )
            counts[band] += 1

            def profile_hook(module, args, output, band=band):
                # Head mean is only for control matching, never a truth/readout score.
                for begin in range(0, queries, 64):
                    block = output[1][
                        0,
                        :,
                        query_start + begin : query_start + min(begin + 64, queries),
                        :length,
                    ]
                    self.profile[band, begin : begin + len(block[0])] += (
                        block.float().mean(0).cpu().numpy()
                    )

            handles.append(layer.self_attn.register_forward_hook(profile_hook))
        return handles, counts

    def measure(self, branches, capture_profile=False, force=False):
        for branch in branches:
            for gate in branch:
                self._check_shared_positions(gate.queries, gate.keys)
        key = digest([[_gate_record(g) for g in branch] for branch in branches])
        if not force and key in self.cache:
            return self.cache[key]
        self._reserve(2)
        logs, before_valid, full_same, diagnostics_all = [], True, True, []
        shams = [
            all(
                (g.kind != "donor" and g.strength == 1)
                or (
                    g.kind == "donor"
                    and g.expected_input_scale is not None
                    and g.expected_input_scale[1] == 1
                )
                for g in gs
            )
            for gs in branches
        ]
        for branch, gates in enumerate(branches):
            handles, counts = [], None
            if capture_profile and branch == 0:
                handles, counts = self._profile_hooks()
            try:
                logits, _, diagnostics = self._forward(self.ids[branch], gates)
            finally:
                for handle in handles:
                    handle.remove()
            if counts is not None:
                for band in range(3):
                    self.profile[band] /= max(1, counts[band])
            first = len(self.contrast.prefix) - 1
            targets = torch.tensor(
                self.contrast.continuations[branch], device=logits.device
            )
            chosen = logits[first : first + len(targets)].float()
            logps = (
                (chosen.gather(1, targets[:, None]).squeeze(1) - chosen.logsumexp(-1))
                .cpu()
                .tolist()
            )
            logs.append(logps)
            if capture_profile:
                self.baselines.append(logits.cpu())
            else:
                earliest = min((min(g.queries) for g in gates), default=len(logits))
                before_valid &= torch.equal(
                    logits[:earliest].cpu(), self.baselines[branch][:earliest]
                )
                if shams[branch]:
                    full_same &= torch.equal(logits.cpu(), self.baselines[branch])
            diagnostics_all.append(diagnostics)
            del logits, chosen
        if not before_valid or not full_same:
            raise RuntimeError("native prefix/sham integrity failure")
        result = {
            "key": key,
            "F": self.contrast.score(logs),
            "token_logps": logs,
            "prefix_exact": before_valid,
            "sham_exact": full_same if all(shams) and not capture_profile else None,
            "gates": [[_gate_record(g) for g in branch] for branch in branches],
            "new_forward_calls": 2,
            "diagnostics": diagnostics_all,
        }
        self.records.append(result)
        self.cache[key] = result
        if result["sham_exact"] is True:
            self.sham_kinds.update(g.kind for branch in branches for g in branch)
        return result

    def gates(self, group, strength=0):
        self._check_shared_positions(group["queries"], group.get("domain", ()))
        self._check_shared_positions(group["queries"], group.get("keys", ()))
        result = []
        for layer in group["layers"]:
            result.append(
                NativeGate(
                    layer=layer,
                    queries=tuple(group["queries"]),
                    kind=group["kind"],
                    strength=strength,
                    keys=tuple(group.get("domain", ())),
                    selected=tuple(group.get("keys", ())),
                    destination=tuple(group.get("destination", ())),
                )
            )
        return result

    def _check_shared_positions(self, queries, keys):
        if (
            not queries
            or len(set(queries)) != len(queries)
            or any(
                type(q) is not int
                or not self.contrast.prompt_length - 1 <= q < len(self.contrast.prefix)
                for q in queries
            )
        ):
            raise ValueError(
                "gate queries must remain in the shared prefix decision domain"
            )
        if any(
            type(k) is not int or not 0 <= k < len(self.contrast.prefix) for k in keys
        ):
            raise ValueError("gate keys must remain in the shared prefix")

    def group(self, group, strength=0, force=False, extra=()):
        gates = self.gates(group, strength) + list(extra)
        result = self.measure([gates, gates], force=force)
        return {**result, "delta": self.base_f - result["F"]}

    def mediated(self, group, origin, strength):
        """Branch-matched full donors; only chosen V-message edges are replaced."""
        self.gates(group, 1)  # Validate common-prefix geometry before donors run.
        self._check_shared_positions(group["queries"], origin)
        if self.calls + 4 > self.budget:
            raise RuntimeError("native_forward_budget_exhausted")
        self._reserve(2)  # Two donor forwards; recipient pair counted by measure.
        branches, artifacts = [], []
        scale = (tuple(origin), float(strength))
        for branch in range(2):
            capture = [(layer, tuple(group["keys"])) for layer in group["layers"]]
            logits, captured, _ = self._forward(
                self.ids[branch], input_scale=scale, capture=capture
            )
            if strength == 1 and not torch.equal(logits.cpu(), self.baselines[branch]):
                raise RuntimeError("branch-matched donor sham mismatch")
            del logits
            if self.artifact_writer is not None:
                artifact = self.artifact_writer(captured, branch, scale)
                self.donor_artifacts.append(artifact)
                artifacts.append(artifact)
            branches.append(
                [
                    NativeGate(
                        layer=layer,
                        queries=tuple(group["queries"]),
                        kind="donor",
                        keys=tuple(group["keys"]),
                        selected=tuple(group["keys"]),
                        donor=captured[layer],
                        expected_input_scale=scale,
                    )
                    for layer in group["layers"]
                ]
            )
        result = self.measure(branches, force=True)
        return {
            **result,
            "delta": self.base_f - result["F"],
            "origin": list(origin),
            "eta": strength,
            "donor_artifacts": artifacts,
        }

    def mass(self, group, keys=None):
        self._check_shared_positions(
            group["queries"], group.get("keys", []) if keys is None else keys
        )
        relative = np.asarray(group["queries"]) - self.contrast.prompt_length + 1
        keys = group.get("keys", []) if keys is None else keys
        if not keys:
            return 0.0
        # All-layer or band averages follow the fixed 10/12/10 partition.
        layers = len(self.model.model.layers)
        bands = [
            0 if i < layers * 10 // 32 else (1 if i < layers * 22 // 32 else 2)
            for i in group["layers"]
        ]
        weights = np.bincount(bands, minlength=3)
        profile = np.average(self.profile, axis=0, weights=weights)
        return float(profile[np.ix_(relative, keys)].sum(-1).mean())


def _group_id(group):
    return digest(group)[:20]


def propose_native_groups(oracle, source_keys, screen_calls=64):
    """Bounded deterministic search. No source relevance labels are accepted."""
    prompt = oracle.contrast.prompt_length
    history = list(range(prompt, len(oracle.contrast.prefix)))
    queries = list(oracle.contrast.shared_queries)
    layers = list(range(len(oracle.model.model.layers)))
    domains = {"source": list(source_keys), "history": history}
    pending = []
    enqueued = set()

    def queue(group, priority):
        if _group_id(group) in enqueued:
            return
        enqueued.add(_group_id(group))
        heapq.heappush(pending, (-priority, len(enqueued), _group_id(group), group))

    for role, keys in domains.items():
        if keys and min(keys) <= max(queries):
            queue(
                {
                    "kind": "content",
                    "role": role,
                    "keys": keys,
                    "domain": keys,
                    "queries": queries,
                    "layers": layers,
                },
                float("inf"),
            )
            broad_domain = sorted(set(source_keys) | set(history))
            if len(broad_domain) > len(keys):
                queue(
                    {
                        "kind": "route",
                        "role": role,
                        "keys": keys,
                        "domain": broad_domain,
                        "queries": queries,
                        "layers": layers,
                        "scope": "broad_role_route",
                    },
                    float("inf"),
                )
    queue(
        {"kind": "mlp", "role": "mlp", "queries": queries, "layers": layers},
        float("inf"),
    )
    seen, measured, start_calls = set(), [], oracle.calls
    start_tokens = oracle.tokens_processed
    tree_budget = screen_calls - (8 if screen_calls >= 16 else 0)
    while pending and oracle.calls - start_calls + 2 <= tree_budget:
        _, _, gid, group = heapq.heappop(pending)
        if gid in seen:
            continue
        seen.add(gid)
        before_measure = oracle.calls
        before_tokens = oracle.tokens_processed
        try:
            result = oracle.group(group)
            measured.append(
                {
                    "id": gid,
                    "group": group,
                    "delta": result["delta"],
                    "measurement": result["key"],
                    "new_forward_calls": oracle.calls - before_measure,
                    "new_tokens_processed": oracle.tokens_processed - before_tokens,
                }
            )
        except ValueError as error:
            measured.append(
                {
                    "id": gid,
                    "group": group,
                    "invalid": str(error),
                    "new_forward_calls": oracle.calls - before_measure,
                    "new_tokens_processed": oracle.tokens_processed - before_tokens,
                }
            )
            continue
        children = []
        # Bisect key and query sets; retain every measured parent for synergy.
        for dimension in ("keys", "queries"):
            positions = group.get(dimension, [])
            if len(positions) > 1:
                middle = len(positions) // 2
                for values in (positions[:middle], positions[middle:]):
                    child = {**group, dimension: values}
                    if child.get("keys") and min(child["keys"]) > max(child["queries"]):
                        continue
                    children.append(child)
        if group["kind"] == "content" and len(group["keys"]) < len(group["domain"]):
            children.append({**group, "kind": "route"})
        # Priority is the parent's measured absolute effect. No weak parent is deleted.
        for child in children:
            if _group_id(child) not in seen:
                queue(child, abs(result["delta"]))
    unions = []
    valid = [m for m in measured if "delta" in m]
    for index, left in enumerate(valid):
        for right in valid[index + 1 :]:
            a, b = left["group"], right["group"]
            if {k: v for k, v in a.items() if k != "queries"} != {
                k: v for k, v in b.items() if k != "queries"
            }:
                continue
            if set(a["queries"]) & set(b["queries"]):
                continue
            queries_union = sorted(set(a["queries"]) | set(b["queries"]))
            if len(queries_union) == queries_union[-1] - queries_union[0] + 1:
                continue
            group = {**a, "queries": queries_union}
            unions.append(
                (-(abs(left["delta"]) + abs(right["delta"])), _group_id(group), group)
            )
    for _, gid, group in sorted(unions, key=lambda x: x[:2]):
        if oracle.calls - start_calls + 2 > screen_calls:
            break
        if gid in seen:
            continue
        seen.add(gid)
        before_measure = oracle.calls
        before_tokens = oracle.tokens_processed
        try:
            result = oracle.group(group)
            measured.append(
                {
                    "id": gid,
                    "group": group,
                    "delta": result["delta"],
                    "measurement": result["key"],
                    "family": "nonadjacent_query_union",
                    "new_forward_calls": oracle.calls - before_measure,
                    "new_tokens_processed": oracle.tokens_processed - before_tokens,
                }
            )
        except ValueError as error:
            measured.append(
                {
                    "id": gid,
                    "group": group,
                    "invalid": str(error),
                    "new_forward_calls": oracle.calls - before_measure,
                    "new_tokens_processed": oracle.tokens_processed - before_tokens,
                    "family": "nonadjacent_query_union",
                }
            )
    ranked = sorted(
        [m for m in measured if "delta" in m], key=lambda m: (-abs(m["delta"]), m["id"])
    )
    pending_tree = {item[2] for item in pending} - seen
    pending_union = {item[1] for item in unions} - seen
    return {
        "measured": measured,
        "ranked": ranked,
        "forward_calls": oracle.calls - start_calls,
        "tokens_processed": oracle.tokens_processed - start_tokens,
        "invalid_groups": sum("invalid" in m for m in measured),
        "invalid_forward_calls": sum(
            m["new_forward_calls"] for m in measured if "invalid" in m
        ),
        "invalid_tokens_processed": sum(
            m["new_tokens_processed"] for m in measured if "invalid" in m
        ),
        "unsearched_groups": len(pending_tree | pending_union),
        "unsearched_tree_groups": len(pending_tree),
        "unsearched_union_groups": len(pending_union),
        "root_order": [
            "source_content",
            "source_broad_route",
            "history_content",
            "history_broad_route",
            "MLP",
        ],
        "candidate_scope": "semantic_target_conditioned_evidence_candidate_blind",
    }
