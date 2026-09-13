"""Keep readable envelopes, nested factual slots and graph regions distinct.

No role-anchor or address-window endpoint creates a sentence boundary. These
are deterministic organization rules, not learned factual boundary labels.
"""

import hashlib
import math
import re
from itertools import pairwise

from route_graph.frozen_reader import digest
from route_graph.source_event_graph import validate_inventory, validate_pointer_graph

_TERMINAL = re.compile(r'''[.!?]+["'”’\)\]\}]*?(?=\s|$)''')
_ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e"})
_OPEN_END = frozenset({"a", "an", "the", "in", "of", "to", "for", "from", "between", "and", "or", "with", "by", "than", "that", "which"})


def _content(text):
    return re.findall(r"\w+", text, flags=re.UNICODE)


def sentence_envelopes(text):
    """Conservative original-text envelopes; ambiguous clauses remain together.

    Lists attach their numbering to content. Closing quotation marks/brackets
    stay with their preceding sentence; decimal/internal abbreviation periods
    are not sentence ends. No maximum length or inventory window is consulted.
    """
    cuts = []
    for match in _TERMINAL.finditer(text):
        at, end = match.start(), match.end()
        if text[at] == ".":
            before = text[:at]
            word = re.search(r"([\w.]+)$", before)
            abbreviation = word.group(1).lower() if word else ""
            if abbreviation in _ABBREVIATIONS or re.fullmatch(r"(?:[A-Za-z]\.)*[A-Za-z]", abbreviation):
                continue
            current = text[cuts[-1] if cuts else 0:end]
            if re.fullmatch(r"\s*(?:[-*•]\s*)?\d+[.)]\s*", current):
                continue
        # A quote followed by an attribution clause is kept in one envelope.
        following = re.search(r"\S", text[end:])
        if (match.group()[-1:] in {'"', "'", "”", "’"} and following
                and text[end + following.start()].islower()):
            continue
        cuts.append(end)
    # Blank paragraphs are formatting candidates; single wrapped lines are not.
    cuts.extend(m.end() for m in re.finditer(r"\n\s*\n", text))
    cuts = sorted(set(cuts + [len(text)]))
    spans, left = [], 0
    for end in cuts:
        part = text[left:end]
        if not _content(part):
            continue
        spans.append([left, end])
        left = end
    if left < len(text):
        if spans:
            spans[-1][1] = len(text)
        elif text:
            spans.append([0, len(text)])
    return spans


def base_graph(inventory, pointer_graph):
    """Attach all original role/event pointers under conservative base units."""
    validate_inventory(inventory)
    validate_pointer_graph(inventory, pointer_graph)
    if inventory["side"] != "response" or pointer_graph["schema"] != "pointer-event-graph@1":
        raise ValueError("response pointer event graph required")
    text, bases = inventory["text"], []
    for span in sentence_envelopes(text):
        a, b = span
        address = [u for u in inventory["units"] if u["span"][0] < b and u["span"][1] > a]
        bases.append({"base_id": digest([inventory["sha256"], "sentence_envelope", span]),
                      "kind": "sentence_envelope" if _content(text[a:b]) else "nonassertive",
                      "span": span, "text": text[a:b], "address_unit_ids": [u["id"] for u in address],
                      "address_boundaries_inside": sorted({p for u in address for p in u["span"] if a < p < b}),
                      "content_token_count": len(_content(text[a:b])),
                      "indivisible_for_semantic_scoring": True,
                      "scope": "readable_context_not_verified_single_fact",
                      "clause_status": "safe_clause_split_not_attempted"})
    nested = []
    for event in pointer_graph["events"]:
        a, b = event["anchor_span"]
        overlaps = [u["base_id"] for u in bases if u["span"][0] < b and u["span"][1] > a]
        inside = [u for u in bases if u["span"][0] <= a < b <= u["span"][1]]
        words = _content(text[a:b])
        right_open = bool(words and words[-1].lower() in _OPEN_END)
        status = ("nonassertive_anchor" if not words else
                  "cross_base_anchor" if not inside else
                  "truncated_anchor_candidate" if right_open else "unverified_event_candidate")
        nested.append({"event_id": event["id"], "anchor_span": [a, b], "base_ids": overlaps,
                       "anchor_inside_one_base": bool(inside), "right_open_edge": right_open,
                       "status": status, "roles": dict(event["roles"]),
                       "changes_base_boundaries": False})
    roles = []
    for node in pointer_graph["nodes"]:
        a, b = node["span"]
        owners = [u["base_id"] for u in bases if u["span"][0] <= a < b <= u["span"][1]]
        roles.append({"role_id": node["id"], "role": node["role"], "span": node["span"],
                      "base_id": owners[0] if len(owners) == 1 else None,
                      "status": "nested_role_proposal" if owners else "role_crosses_base_unresolved"})
    result = {"schema": "conservative-base-graph@1", "inventory_sha256": inventory["sha256"],
            "text_sha256": inventory["text_sha256"], "text_length": len(text),
            "pointer_graph_sha256": pointer_graph["sha256"], "base_units": bases,
            "nested_events": nested, "nested_roles": roles,
            "boundary_scope": "sentence_envelopes_then_typed_graph_regions; no_factual_boundary_truth"}
    return {**result, "sha256": digest(result)}


def _validate_graph(graph, text=None):
    if graph["sha256"] != digest({k: v for k, v in graph.items() if k != "sha256"}):
        raise ValueError("base graph content changed")
    if text is not None and (
        graph["text_length"] != len(text) or graph["text_sha256"] != hashlib.sha256(text.encode()).hexdigest()
        or any(b["text"] != text[slice(*b["span"])] for b in graph["base_units"])
    ):
        raise ValueError("text differs from frozen base/role graph")


def localized_words(text, graph, base_bag_risks, local_decisions):
    """A bag-level 'some assertion is wrong' never labels every word as wrong.

    Local decisions must already bind a nested target to an independent finite
    scope check or a validated bridge. No native magnitude supplies truth here.
    Unknown words remain at .5 in every denominator; bag risks stay separate.
    """
    _validate_graph(graph, text)
    bases = {u["base_id"]: u for u in graph["base_units"]}
    roles = {u["role_id"]: u for u in graph["nested_roles"]}
    if set(base_bag_risks) != set(bases):
        raise ValueError("every base needs its separate bag-level score")
    for score in base_bag_risks.values():
        if isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("invalid bag score")
    decisions, seen = [], set()
    for item in local_decisions:
        if item["decision_id"] in seen:
            raise ValueError("duplicate local decision identity")
        seen.add(item["decision_id"])
        role = roles[item["role_id"]]
        if (role["base_id"] is None or item["base_id"] != role["base_id"]
                or item["target_span"] != role["span"]):
            raise ValueError("local decision target differs from nested role/base")
        score = item["risk"]
        if isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("invalid local risk")
        eligible = (item.get("scope_verified") is True
                    and item.get("basis") in {"finite_local_fact_relation", "validated_single_slot_bridge"}
                    and isinstance(item.get("evidence_record_sha256"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", item["evidence_record_sha256"]))
        if eligible and bases[role["base_id"]]["kind"] != "nonassertive":
            decisions.append(item)
    words = []
    for word in re.finditer(r"\S+", text):
        a, b = word.span()
        enclosing = [u for u in bases.values() if u["span"][0] <= a < b <= u["span"][1]]
        if len(enclosing) != 1:
            raise ValueError("a word must belong to exactly one intact base")
        overlapping = [d for d in decisions if d["target_span"][0] < b and d["target_span"][1] > a]
        scores = [d["risk"] for d in overlapping]
        # Opposed local decisions stay unresolved rather than max-pooling truth.
        conflict = bool(scores and min(scores) < .5 < max(scores))
        risk = .5 if not scores or conflict else sum(scores) / len(scores)
        base = enclosing[0]["base_id"]
        nonassertive = enclosing[0]["kind"] == "nonassertive"
        words.append({"span": [a, b], "text": word.group(), "base_id": base,
                      "localized_risk": risk, "base_bag_risk": None if nonassertive else base_bag_risks[base],
                      "local_decision_ids": [d["decision_id"] for d in overlapping],
                      "localization_status": "nonassertive" if nonassertive else
                      "conflicting_local_relations" if conflict else
                      "localized_conditional_prediction" if scores else "unknown_within_base"})
    return words


def graph_regions(graph, edges):
    """Adaptive groups preserve nonadjacent edges without filling skipped text."""
    _validate_graph(graph)
    bases = graph["base_units"]
    order = {u["base_id"]: i for i, u in enumerate(bases)}
    parent = list(range(len(bases)))

    def root(i):
        while i != parent[i]:
            i = parent[i]
        return i

    seen, positive, barriers, retained = set(), set(), set(), []
    for edge in edges:
        if edge["edge_id"] in seen:
            raise ValueError("duplicate boundary edge identity")
        seen.add(edge["edge_id"])
        left, right = order[edge["from_base"]], order[edge["to_base"]]
        probability = edge["probability"]
        if (left >= right or isinstance(probability, bool) or not math.isfinite(probability)
                or not 0 <= probability <= 1):
            raise ValueError("boundary edges require finite probabilities and forward time")
        pair = (left, right)
        status = "unresolved_relation"
        if probability >= .8:
            if edge["kind"] in {"correction", "quote", "new_topic"}:
                barriers.add(pair)
                status = "predicted_boundary_barrier"
            elif edge["kind"] in {"reuse", "same_fact"} and edge.get("specific_fact_link") is True:
                positive.add(pair)
                status = "predicted_fact_connection"
        retained.append({**edge, "grouping_status": status})
    allowed = positive - barriers
    for left, right in sorted(allowed):
        parent[root(right)] = root(left)
    components = {}
    for i in range(len(bases)):
        components.setdefault(root(i), []).append(i)
    contiguous = []
    for i in range(len(bases)):
        if i and (i - 1, i) in allowed:
            contiguous[-1].append(i)
        else:
            contiguous.append([i])

    def describe(indices):
        return {"base_ids": [bases[i]["base_id"] for i in indices],
                "spans": [bases[i]["span"] for i in indices],
                "noncontiguous": any(b != a + 1 or (a, b) not in allowed for a, b in pairwise(indices)),
                "spatially_contiguous": all(b == a + 1 for a, b in pairwise(indices)),
                "jump_edges": [[bases[a]["base_id"], bases[b]["base_id"]] for a, b in sorted(allowed)
                               if a in indices and b in indices and b != a + 1],
                "assigns_word_risk": False}

    return {"connected_regions": [describe(c) for c in components.values()],
            "contiguous_regions": [describe(c) for c in contiguous], "all_edges": retained,
            "conflicting_pairs": [[bases[a]["base_id"], bases[b]["base_id"]] for a, b in sorted(positive & barriers)],
            "scope": "conditional_fact_connectivity; not_error_interval_ground_truth"}
