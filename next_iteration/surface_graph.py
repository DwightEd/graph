"""Exact-text proposal graph without generated SRL coordinates.

Surface types are lexical proposals, never semantic roles. Sentence parents
provide context; they are not assumed to contain only one factual assertion.
"""

import re

from next_iteration.graph_boundaries import sentence_envelopes
from route_graph.frozen_reader import digest
from route_graph.source_event_graph import compile_literal_fields, raw_inventory

_NUM = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_CLOCK = r"\d{1,2}(?::\d{1,2})?\s*(?:[AP]\.?M\.?)?"
_RULES = [
    ("time_range", rf"\b{_CLOCK}\s*(?:to|[-–—])\s*{_CLOCK}(?![\w:])", re.IGNORECASE),
    ("duration", rf"\b{_NUM}(?:[- ](?:years?|months?|weeks?|days?|hours?|minutes?))(?:[- ]old)?\b", re.IGNORECASE),
    ("date", r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?\b", re.IGNORECASE),
    ("time", r"\b\d{1,2}(?::\d{2})?\s*[AP]\.?M\.?\b", re.IGNORECASE),
    ("number", rf"(?<![\w.])(?:[$£€]\s*)?-?{_NUM}(?:\s*%)?(?!\w|\.\d)", 0),
]
_QUOTES = re.compile(r'"[^"\n]+"|“[^”\n]+”|(?<!\w)\x27[^\x27\n]+\x27(?!\w)')
_TOKEN = re.compile(r"[\w]+(?:['’\-][\w]+)*", re.UNICODE)
_CITATION = re.compile(r"\((?:Passage|Source|Document)\s+\d+(?:\s*[,;&]\s*\d+)*\)", re.IGNORECASE)
# Delimiters are an explicit conservative lexical heuristic, not a POS model.
_STOP = frozenset(["a", "an", "the", "this", "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their", "who", "whom", "whose", "which", "what", "where", "when", "why", "how", "and", "or", "but", "nor", "so", "yet", "if", "unless", "although", "because", "while", "as", "at", "by", "for", "from", "in", "into", "of", "on", "onto", "per", "than", "through", "to", "under", "until", "up", "with", "without", "is", "are", "was", "were", "be", "been", "being", "am", "has", "have", "had", "do", "does", "did", "can", "could", "may", "might", "must", "shall", "should", "will", "would", "not", "no", "also", "very", "any", "all", "some", "each", "every", "both", "either", "neither", "about", "across", "after", "before", "between", "during", "over", "such", "then", "there", "here", "instead", "despite", "whether", "says", "said", "say", "tell", "told", "tells", "miss", "misses", "watch", "watches", "watched", "offers", "offer", "offered", "received", "receives", "receive", "discovered", "discover", "finds", "found", "find", "provides", "provide", "provided", "includes", "include", "included", "contains", "contain", "contained", "uses", "use", "used", "became", "become", "becomes", "located", "rated", "serves", "serve", "served", "according", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't"])


def _sealed(value):
    return {**value, "sha256": digest(value)}


def _check(value):
    if value["sha256"] != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("surface graph content changed")


def _proposals(text, span):
    """Nonoverlapping maximal lexical proposals; quotes remain edit-closed."""
    left, right = span
    part, occupied, found = text[left:right], [], []
    occupied.extend(m.span() for m in _CITATION.finditer(part))
    occupied.extend(m.span() for m in re.finditer(r"(?m)^\s*(?:[-*•]\s*)?\d+[.)](?=\s|$)", part))

    def add(a, b, kind, reason):
        if any(a < y and b > x for x, y in occupied):
            return
        if not re.search(r"\w", part[a:b]):
            return
        occupied.append((a, b))
        found.append(([a + left, b + left], kind, reason))

    for match in _QUOTES.finditer(part):
        add(*match.span(), "explicit_quote", "balanced_quote")
    for kind, pattern, flags in _RULES:
        for match in re.finditer(pattern, part, flags):
            a, b = match.span()
            matched_kind = kind
            if kind == "time_range" and not re.search(r":|[ap]\.?m", match.group(), re.IGNORECASE):
                matched_kind = "number_range"
            # A numbered list marker belongs to formatting, not a fact value.
            if kind == "number" and not part[:a].strip(" \t\n-*•") and part[b:b + 1] in {".", ")"}:
                occupied.append((a, b))
                continue
            add(a, b, matched_kind, "regex_" + matched_kind)
    run = []

    def flush():
        if run:
            a, b = run[0].start(), run[-1].end()
            # Do not split a long unknown run into arbitrary editable windows.
            if len(run) <= 6:
                add(a, b, "content_run", "maximal_nonfunction_lexical_run_not_noun_truth")
        run.clear()

    for token in _TOKEN.finditer(part):
        a, b = token.span()
        blocked = token.group().lower() in _STOP or any(a < y and b > x for x, y in occupied)
        gap = part[run[-1].end():a].strip() if run else ""
        named_ampersand = bool(run and gap == "&" and run[-1].group()[0].isupper() and token.group()[0].isupper())
        separated = bool(run and gap and not named_ampersand)
        if blocked or separated:
            flush()
        if not blocked:
            run.append(token)
    flush()
    return sorted(found)


def _surface_envelopes(text):
    """Attach a trailing explicit citation, then the next list marker forward.

    This only repairs formatting boundaries; it does not infer factual scope.
    """
    ends = []
    for _, end in sentence_envelopes(text):
        following = re.match(r"\s*", text[end:]).end() + end
        citation = _CITATION.match(text, following)
        if citation:
            end = citation.end()
        if ends and end <= ends[-1]:
            continue
        left = ends[-1] if ends else 0
        if re.fullmatch(r"\s*(?:[-*•]\s*)?\d+[.)]\s*", text[left:end]):
            continue
        ends.append(end)
    if not ends or ends[-1] != len(text):
        ends.append(len(text))
    return [[a, b] for a, b in zip([0, *ends[:-1]], ends, strict=True) if a < b]


def surface_graph(text, *, sample_id, side):
    if side not in {"source", "response"}:
        raise ValueError("invalid graph side")
    identity = digest([str(sample_id), side, text])
    bases, slots, edges = [], [], []
    for span in _surface_envelopes(text):
        a, b = span
        base_id = digest([identity, "base", span])
        base = {"base_id": base_id, "span": span, "text": text[a:b],
                "kind": "sentence_envelope" if re.search(r"\w", text[a:b]) else "nonassertive",
                "scope": "readable_context_not_single_fact"}
        bases.append(base)
        for target, kind, reason in _proposals(text, span):
            x, y = target
            slot_id = digest([identity, "surface", target, kind])
            mask = "<" + kind.upper() + ">"
            slot = {"slot_id": slot_id, "base_id": base_id, "span": target,
                    "quote": text[x:y], "surface_type": kind, "proposal_reason": reason,
                    "masked_context": text[a:x] + mask + text[y:b],
                    "non_target_spans": [[a, x], [y, b]], "not_semantic_role": True,
                    "edit_policy": "quoted_edit_closed" if kind == "explicit_quote" else "requires_finite_scope_and_preservation"}
            slots.append(slot)
            edges.append({"from": slot_id, "to": base_id, "kind": "contained_in"})
    return _sealed({"schema": "surface-proposal-graph@1", "sample_id": str(sample_id),
                    "side": side, "text": text, "base_units": bases, "slots": slots, "edges": edges,
                    "semantic_role_extraction_required": False, "labels_used": False})


def source_occurrences(text, *, sample_id, task):
    """Retain literal record ancestry or complete natural sentence ancestry."""
    if task == "Data2txt":
        inventory = raw_inventory(text, side="source", sample_id=sample_id)
        try:
            literal = compile_literal_fields(inventory)
        except (TypeError, ValueError, SyntaxError):
            literal = None
        if literal is not None:
            nodes = {n["id"]: n for n in literal["nodes"]}
            ancestors = {e["child"]: e["parent"] for e in literal["edges"]}
            occurrences = []
            for node in literal["nodes"]:
                if node["kind"] != "scalar":
                    continue
                owner_id = ancestors[node["id"]]
                # A list is a container; its nearest record is the useful owner.
                while nodes[owner_id]["kind"] != "dict" and owner_id in ancestors:
                    owner_id = ancestors[owner_id]
                parent = nodes[owner_id]
                a, b = parent["span"]
                x, y = node["span"]
                value = node["value"]
                display = value if type(value) is str else node["raw_literal"]
                kind = "number" if node["value_type"] in {"float", "int"} else "content_run"
                if type(value) is str:
                    candidates = _proposals(value, [0, len(value)])
                    if len(candidates) == 1 and candidates[0][0] == [0, len(value)]:
                        kind = candidates[0][1]
                closed = ("unknown_null" if value is None else "boolean_verbalization_closed" if type(value) is bool else
                          "long_or_multiline_literal_edit_closed" if len(display.split()) > 12 or "\n" in display else
                          "requires_finite_scope_and_preservation")
                occurrences.append({"occurrence_id": node["id"], "span": node["span"],
                    "source_kind": "literal_field", "raw_surface": node["raw_literal"],
                    "display_surface": display, "surface_type": kind, "value_type": node["value_type"],
                    "field_path": node["path"], "parent_id": owner_id, "parent_span": parent["span"],
                    "parent_text": text[a:b], "masked_context": text[a:x] + "<" + kind.upper() + ">" + text[y:b],
                    "epistemic_status": node["epistemic_status"], "edit_policy": closed})
            return _sealed({"schema": "source-occurrences@1", "sample_id": str(sample_id), "text": text,
                "source_kind": "literal_field", "occurrences": occurrences, "literal_graph": literal,
                "source_graph": None, "labels_used": False})
    graph = surface_graph(text, sample_id=sample_id, side="source")
    parents = {b["base_id"]: b for b in graph["base_units"]}
    occurrences = []
    for slot in graph["slots"]:
        parent = parents[slot["base_id"]]
        occurrences.append({"occurrence_id": slot["slot_id"], "span": slot["span"],
            "source_kind": "source_surface", "raw_surface": slot["quote"], "display_surface": slot["quote"],
            "surface_type": slot["surface_type"], "value_type": "text", "field_path": [],
            "parent_id": parent["base_id"], "parent_span": parent["span"], "parent_text": parent["text"],
            "masked_context": slot["masked_context"], "epistemic_status": "observed_text_not_fact_truth",
            "edit_policy": slot["edit_policy"]})
    return _sealed({"schema": "source-occurrences@1", "sample_id": str(sample_id), "text": text,
        "source_kind": "source_surface", "occurrences": occurrences, "source_graph": graph,
        "literal_graph": None, "labels_used": False})


def edit_candidate(response_graph, sources, slot_id, occurrence_id):
    """Construct one exact replacement; finite checks must establish its meaning."""
    _check(response_graph)
    _check(sources)
    slot = next(s for s in response_graph["slots"] if s["slot_id"] == slot_id)
    occurrence = next(o for o in sources["occurrences"] if o["occurrence_id"] == occurrence_id)
    base = next(b for b in response_graph["base_units"] if b["base_id"] == slot["base_id"])
    status = "candidate_not_semantically_verified"
    if slot["edit_policy"] != "requires_finite_scope_and_preservation":
        status = slot["edit_policy"]
    elif occurrence["edit_policy"] != "requires_finite_scope_and_preservation":
        status = occurrence["edit_policy"]
    elif slot["surface_type"] != occurrence["surface_type"]:
        status = "surface_type_mismatch"
    elif slot["quote"] == occurrence["display_surface"]:
        status = "identical_surface"
    replacement = occurrence["display_surface"]
    if not replacement.strip() or not replacement.isprintable():
        status = "nonprintable_or_empty_surface"
    a, b = base["span"]
    x, y = slot["span"]
    text = response_graph["text"]
    edited = text[a:x] + replacement + text[y:b]
    return _sealed({"schema": "surface-edit-candidate@1", "status": status,
        "response_graph_sha256": response_graph["sha256"], "source_graph_sha256": sources["sha256"],
        "slot": slot, "occurrence": occurrence, "base": base, "original_base": text[a:b],
        "edited_base": edited, "earlier_response": text[:a], "replacement": replacement,
        "source": sources["text"], "single_exact_span_edit": True,
        "semantic_edit_kind": "unknown_requires_finite_subject_predicate_condition_gate",
        "labels_used": False, "certificate_count": 0})
