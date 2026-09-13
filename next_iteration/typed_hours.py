"""Conservative Data2txt weekly-hours algebra with exact source provenance.

The schema and response grammar are explicit assumptions. This is a typed
provider, not a general semantic parser or an original-generator explanation.
"""

import ast
import re
from dataclasses import asdict

from next_iteration.surface_graph import (
    _check,
    _sealed,
    source_occurrences,
    surface_graph,
)
from route_graph.audit_alignment import align_row, slot_masks, span_keys
from route_graph.causal_contrast import ContinuationContrast
from route_graph.frozen_reader import digest

DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
CLOCK24 = r"(?:[01]?\d|2[0-3]):[0-5]?\d"
CLOCK12 = r"(?:1[0-2]|[1-9])(?::[0-5]\d)?\s*(?:AM|PM)"
RANGE = rf"(?P<range>(?P<start>{CLOCK12})\s+to\s+(?P<end>{CLOCK12}))"
PROTOCOL = {"schema": "data2txt-hours-provider@1", "days": list(DAYS),
    "source_clock": "explicit 24h H:M-H:M; end must exceed start; overnight unresolved",
    "response_clock": "both endpoints explicit AM/PM; no inferred meridiem",
    "relations": ["root_business_all_week_hours", "root_business_free_wifi_conjunct"],
    "owner": "explicit root business name or unqualified The business in Data2txt single-record task",
    "null": "unknown_never_false", "training": "none", "reader_calls": 0,
    "scope": "typed source interpretation; not general QA/Summary or natural-language truth"}


def _minute24(value):
    if not isinstance(value, str) or not re.fullmatch(CLOCK24, value):
        raise ValueError("invalid 24h clock")
    h, m = map(int, value.split(":"))
    return h * 60 + m


def _minute12(value):
    m = re.fullmatch(r"(1[0-2]|[1-9])(?::([0-5]\d))?\s*(AM|PM)", value, re.IGNORECASE)
    if not m:
        raise ValueError("explicit AM/PM endpoint required")
    return (int(m[1]) % 12 + 12 * (m[3].upper() == "PM")) * 60 + int(m[2] or 0)


def _render_clock(minutes, original):
    h, m = divmod(minutes, 60)
    suffix = "AM" if h < 12 else "PM"
    original_unit = re.search(r"(AM|PM)$", original, re.IGNORECASE)[1]
    if original_unit.islower():
        suffix = suffix.lower()
    colon = ":" in original or m != 0
    number = str(h % 12 or 12) + (f":{m:02d}" if colon else "")
    gap = re.search(r"(\s*)(AM|PM)$", original, re.IGNORECASE)[1]
    return number + gap + suffix


def source_schedule(source, *, source_id):
    graph = source_occurrences(source, sample_id=source_id, task="Data2txt")
    if graph["source_kind"] != "literal_field":
        raise ValueError("typed provider requires validated source literal graph")
    root = ast.literal_eval(source)
    if not isinstance(root, dict) or not isinstance(root.get("name"), str):
        raise TypeError("one named root business required")
    if not isinstance(root.get("hours"), dict):
        raise TypeError("weekly hours need an explicit day map")
    attributes = root.get("attributes")
    fields = {tuple(o["field_path"]): o for o in graph["occurrences"]}
    proofs = []
    for day in DAYS:
        occurrence = fields.get(("hours", day))
        if occurrence is None or occurrence["epistemic_status"] != "observed_literal":
            raise ValueError("weekly hours missing or unknown")
        value = root.get("hours", {}).get(day)
        parsed = re.fullmatch(rf"(?P<start>{CLOCK24})-(?P<end>{CLOCK24})", value or "")
        if not parsed:
            raise ValueError("source hours outside fixed 24h syntax")
        pair = [_minute24(parsed["start"]), _minute24(parsed["end"])]
        if pair[1] <= pair[0]:
            raise ValueError("overnight or ambiguous source interval unresolved")
        raw = occurrence["raw_surface"]
        if raw.count(value) != 1:
            raise ValueError("source clock lacks direct literal character mapping")
        offset = occurrence["span"][0] + raw.index(value)
        endpoint_spans = [[offset + a, offset + b] for a, b in (parsed.span("start"), parsed.span("end"))]
        proofs.append({"day": day, "minutes": pair, "occurrence_id": occurrence["occurrence_id"],
            "field_path": occurrence["field_path"], "literal_span": occurrence["span"],
            "endpoint_spans": endpoint_spans, "raw_value": value})
    return _sealed({"schema": "typed-weekly-schedule@1", "source_graph": graph,
        "business_name": root["name"], "root_owner_id": graph["literal_graph"]["nodes"][0]["id"],
        "days": proofs, "wifi": attributes.get("WiFi") if isinstance(attributes, dict) else None,
        "wifi_occurrence": fields.get(("attributes", "WiFi")), "protocol": PROTOCOL,
        "labels_used": False, "native_certificate_count": 0})


def _match_base(text, name):
    owner = rf"(?P<owner>The business|{re.escape(name)})"
    predicates = [
        rf"{owner} hours are from {RANGE}\s+(?P<days>every day|daily|every day of the week)",
        rf"{owner} operates(?: from)? (?P<days>Monday (?:to|through) Sunday) from {RANGE}",
        rf"{owner} is open (?P<days>seven days a week) from {RANGE}",
    ]
    for pattern in predicates:
        match = re.fullmatch(rf"\s*{pattern}(?P<wifi>, and they offer free WiFi)?[.]\s*", text, re.IGNORECASE)
        if match:
            return match
    return None


def assess_base(row, base, schedule):
    _check(schedule)
    common = {"schema": "typed-hours-fact@1", "response_id": str(row["id"]),
        "source_id": str(row["source_id"]), "row_sha256": digest(row),
        "schedule_sha256": schedule["sha256"], "base": base, "labels_used": False,
        "reader_calls": 0, "native_certificate_count": 0, "protocol": PROTOCOL}
    match = _match_base(base["text"], schedule["business_name"])
    if match is None:
        return _sealed({**common, "status": "response_relation_or_scope_outside_grammar"})
    observed = [_minute12(match["start"]), _minute12(match["end"])]
    if observed[1] <= observed[0]:
        return _sealed({**common, "status": "response_overnight_or_ambiguous_unresolved"})
    mismatched = [d["day"] for d in schedule["days"] if d["minutes"] != observed]
    result = {**common, "observed_minutes": observed, "quantified_days": list(DAYS),
        "mismatched_days": mismatched, "owner_span": [base["span"][0] + x for x in match.span("owner")],
        "owner_basis": "Data2txt_root_business_schema" if match["owner"].lower() == "the business" else "explicit_source_business_name",
        "day_scope_span": [base["span"][0] + x for x in match.span("days")],
        "typed_relation": "conflict" if mismatched else "supported",
        "wifi_conjunct": bool(match["wifi"]),
        "wifi_supported": not match["wifi"] or schedule["wifi"] == "free"}
    if not result["wifi_supported"]:
        return _sealed({**result, "status": "non_target_wifi_not_supported"})
    if not mismatched:
        return _sealed({**result, "status": "typed_supported_no_error_contrast"})
    alternatives = {tuple(d["minutes"]) for d in schedule["days"]}
    if len(alternatives) != 1:
        return _sealed({**result, "status": "schedule_varies_no_single_range_repair"})
    correct = list(next(iter(alternatives)))
    changed = [i for i in (0, 1) if observed[i] != correct[i]]
    if len(changed) != 1:
        return _sealed({**result, "status": "multiple_endpoint_changes_not_single_value_contrast"})
    endpoint = changed[0]
    key = ("start", "end")[endpoint]
    a, b = [base["span"][0] + x for x in match.span(key)]
    replacement = _render_clock(correct[endpoint], match[key])
    left, right = base["span"]
    edited = row["response"][left:a] + replacement + row["response"][b:right]
    restored = _match_base(edited, schedule["business_name"])
    if restored is None or [_minute12(restored["start"]), _minute12(restored["end"])] != correct:
        raise ValueError("typed edit failed round-trip semantics")
    return _sealed({**result, "status": "typed_contrast_available", "correct_minutes": correct,
        "endpoint": key, "answer_span": [a, b], "replacement": replacement,
        "original_base": base["text"], "edited_base": edited,
        "source_value_spans": [d["endpoint_spans"][endpoint] for d in schedule["days"]],
        "constraint_graph": {"owner": schedule["root_owner_id"], "quantifier": "all_seven_days",
            "required_occurrences": [d["occurrence_id"] for d in schedule["days"]],
            "incidence": "every day constrains every source schedule node; no single-day shortcut"},
        "semantic_scope": "single closing/opening endpoint under explicit typed schema assumptions"})


def prepare_row(row):
    if row["task"] != "Data2txt":
        return _sealed({"response_id": str(row["id"]), "status": "task_outside_typed_provider", "facts": [], "labels_used": False})
    source = row["prompt"][slice(*row["source_span"])]
    try:
        schedule = source_schedule(source, source_id=row["source_id"])
    except (ValueError, TypeError, SyntaxError) as error:
        return _sealed({"response_id": str(row["id"]), "status": "source_schedule_unresolved",
            "reason": str(error), "facts": [], "labels_used": False})
    graph = surface_graph(row["response"], sample_id=row["id"], side="response")
    # Every base is retained, including no time phrase and unsupported grammar.
    facts = [assess_base(row, base, schedule) for base in graph["base_units"]]
    return _sealed({"response_id": str(row["id"]), "status": "typed_assessment_complete",
        "schedule": schedule, "response_graph": graph, "facts": facts, "labels_used": False})


def native_contrast(row, prepared, fact_id, tokenizer):
    """Recompute symbolic proof before exact original-token handoff."""
    _check(prepared)
    if prepare_row(row) != prepared:
        raise ValueError("typed proof differs from current source/response/protocol")
    fact = prepared["facts"][fact_id]
    if fact["status"] != "typed_contrast_available":
        raise ValueError("no complete typed contrast")
    a, b = fact["answer_span"]
    end = fact["base"]["span"][1]
    original = row["response"][:end]
    alternative = original[:a] + fact["replacement"] + original[b:]
    prompt_ids = row["token_ids"][:row["prompt_length"]]
    ids = [prompt_ids + tokenizer(text, add_special_tokens=False)["input_ids"] for text in (original, alternative)]
    if ids[0] != row["token_ids"][:len(ids[0])]:
        raise ValueError("typed event is not an exact original token prefix")
    contrast = ContinuationContrast.from_sequences(ids, row["prompt_length"])
    if max(map(len, contrast.continuations)) > 96:
        raise ValueError("typed full contrast exceeds frozen 96-token budget")
    alignment = align_row(row, tokenizer)
    per_day = [span_keys(alignment, "source", s) for s in fact["source_value_spans"]]
    if any(not keys for keys in per_day):
        raise ValueError("source endpoint lacks original token mapping")
    metadata = {"kind": "typed_hours_endpoint", "original": original, "alternative": alternative,
        "response_end": end, "answer_spans": [[a, b], [a, a + len(fact["replacement"])]],
        "query_scope": "all_shared_prefix_queries_only"}
    return _sealed({"schema": "typed-native-contrast@1", "contrast": asdict(contrast),
        "metadata": metadata, "slot_masks": slot_masks(contrast, metadata, tokenizer),
        "source_keys_per_day": per_day, "source_keys": sorted({k for keys in per_day for k in keys}),
        "row_sha256": digest(row), "prepared_sha256": prepared["sha256"], "fact_sha256": fact["sha256"],
        "source_semantics": PROTOCOL, "observer_scope": "Llama3.1 replay; not original generator causality",
        "native_forward_calls": 0, "native_certificate_count": 0, "labels_used": False})
