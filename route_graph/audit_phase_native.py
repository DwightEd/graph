"""Native B/D phases with explicit budgets and no semantic candidate backfill."""

from dataclasses import asdict

from route_graph.audit_alignment import span_keys
from route_graph.audit_artifacts import donor_writer
from route_graph.audit_certificates import (
    validate_fixed_group,
    validate_group,
    validate_layer_bands,
    validate_origin,
)
from route_graph.audit_interactions import mlp_interaction, paired_role
from route_graph.audit_pools import freeze_pools, origin_controls
from route_graph.causal_contrast import ContinuationContrast
from route_graph.causal_groups import CausalOracle, propose_native_groups
from route_graph.frozen_reader import digest


def event_contrast(event):
    spec = event["contrast"]
    contrast = ContinuationContrast.from_sequences(
        [list(spec["prefix"]) + list(c) for c in spec["continuations"]],
        spec["prompt_length"],
    )
    if digest(asdict(contrast)) != digest(spec):
        raise ValueError("event is not in canonical complete-common-prefix form")
    return contrast


def proposal_phase(model, row, question, alignment):
    contrast = event_contrast(question["events"][0])
    oracle = CausalOracle(model, contrast, budget=66)
    search = propose_native_groups(oracle, alignment["source_keys"], screen_calls=64)
    frozen = freeze_pools(oracle, search, row, alignment, question)
    return {
        "status": "proposed",
        "question_id": question["id"],
        "contrast": asdict(contrast),
        "base": oracle.base,
        "search": search,
        "frozen": frozen,
        "records": oracle.records,
        "forward_calls": oracle.calls,
        "tokens_processed": oracle.tokens_processed,
    }


def bounded(oracle, cap, function):
    previous, start = oracle.budget, oracle.calls
    oracle.budget = min(previous, start + cap)
    try:
        result = function()
    except ValueError as error:
        result = {
            "status": "invalid_intervention",
            "passed": False,
            "reason": str(error),
        }
    except RuntimeError as error:
        if str(error) != "native_forward_budget_exhausted":
            raise
        result = {"status": "budget_unresolved", "passed": False}
    finally:
        oracle.budget = previous
    return {**result, "new_forward_calls": oracle.calls - start}


def validation_phase(
    model, row, question, proposal, semantics, alignment, artifact_dir
):
    if (
        digest(proposal["contrast"]) != digest(question["events"][0]["contrast"])
        or proposal["question_id"] != question["id"]
    ):
        raise ValueError("native proposal is not bound to the selected exact A event")
    if semantics["question_id"] != question["id"] or semantics[
        "frozen_pool_sha256"
    ] != digest(proposal["frozen"]):
        raise ValueError("semantic labels are not bound to the frozen native pool")
    is_missing = question["effective_scores"]["N"] >= 0.8
    if is_missing and not semantics["null_source_unit_check"]:
        return {
            "status": "null_semantic_coverage_unresolved",
            "forward_calls": proposal["forward_calls"],
            "new_forward_calls": 0,
            "positions": [],
            "origins": {},
            "records": [],
            "tokens_processed": proposal["tokens_processed"],
        }
    frozen, labels = proposal["frozen"], semantics["labels"]
    oracle = CausalOracle(
        model,
        event_contrast(question["events"][0]),
        budget=320 - proposal["forward_calls"],
        artifact_writer=donor_writer(artifact_dir),
    )
    # Cross-phase replay is checked before any cached finite effect can be reused.
    if oracle.base["token_logps"] != proposal["base"]["token_logps"]:
        raise RuntimeError("B/D baseline replay differs; refusing cached effects")
    for record in proposal["records"]:
        if record["key"] != oracle.base["key"]:
            oracle.cache[record["key"]] = record
    for kind in ("content", "route"):
        item = next(
            (i for i in frozen["finalists"] if i["group"]["kind"] == kind), None
        )
        if item is not None:
            extra = (
                oracle.gates(frozen["mlp"]["group"], 1)
                if kind == "content" and frozen["mlp"]
                else []
            )
            oracle.group(item["group"], 1, extra=extra)
    positions, origins = [], {}
    for item in frozen["finalists"]:
        position = bounded(
            oracle, 12, lambda item=item: validate_group(oracle, item, labels)
        )
        position.update(
            id=item["id"], view_id=item["view_id"], semantic=labels[item["view_id"]]
        )
        positions.append(position)
    # Origins are fixed from the native selected text; history uses the frozen
    # earlier claim when present. No failed pair is replaced by another origin.
    for item, position in zip(frozen["finalists"], positions, strict=True):
        group = item["group"]
        if group["kind"] != "content":
            continue
        origin = group["keys"]
        selection = "native_selected_raw_text"
        if (
            group["role"] == "history"
            and question.get("previous_claim_span") is not None
        ):
            linked = semantics["relation_check"].get("slot_link_valid") is True
            origin_span = (
                semantics["relation_check"]["previous_answer_span"]
                if linked
                else question["previous_claim_span"]
            )
            origin = span_keys(
                alignment,
                "history",
                origin_span,
                len(oracle.contrast.prefix),
            )
            selection = (
                "mapped_previous_answer_slot"
                if linked
                else "frozen_previous_claim_text"
            )
        controls = origin_controls(origin, group, frozen, labels, group["role"])
        result = bounded(
            oracle,
            24,
            lambda group=group, origin=origin, controls=controls, position=position: (
                validate_origin(oracle, group, origin, controls, position)
            ),
        )
        origins[item["id"]] = {**result, "origin_selection": selection}
    source = next(
        (
            p
            for p in positions
            if p.get("passed")
            and p["group"]["kind"] == "content"
            and p["group"]["role"] == "source"
            and p["semantic"].get("relation") == "applicable"
            and p["semantic"].get("conditions_covered") is True
        ),
        {},
    )
    history = next(
        (
            p
            for p in positions
            if p.get("passed")
            and p["group"]["kind"] == "content"
            and p["group"]["role"] == "history"
        ),
        {},
    )
    source_origin, history_origin = (
        origins.get(source.get("id"), {}),
        origins.get(history.get("id"), {}),
    )
    if is_missing:
        paired = {"status": "N_scope_unresolved_not_measured", "passed": False}
        interaction = {"status": "N_scope_unresolved_not_measured", "passed": False}
    else:
        paired = bounded(
            oracle,
            12,
            lambda: paired_role(oracle, source, history, source_origin, history_origin),
        )
        interaction = bounded(
            oracle,
            24,
            lambda: mlp_interaction(
                oracle,
                source,
                frozen["mlp"],
                source_origin,
                question["events"][0]["slot_masks"],
            ),
        )
    primary = next((p for p in positions if p.get("passed")), None)
    template = {"status": "not_applicable", "passed": True}
    alternate_calls = alternate_tokens = 0
    if is_missing:
        primary = next(
            (
                p
                for p in positions
                if p.get("passed")
                and p["group"]["kind"] == "content"
                and (
                    p["semantic"].get("relation")
                    in {"applicable", "nonapplicable", "contradictory"}
                    or (
                        p["group"]["role"] == "history"
                        and origins.get(p["id"], {}).get("passed")
                        and semantics["relation_check"].get("valid")
                    )
                )
            ),
            None,
        )
        template = {"status": "no_primary_certificate", "passed": False}
        if primary is not None:
            # Test a single preselected certificate family under the second
            # withholding template. Other N mechanisms remain template-unresolved.
            remaining = oracle.budget - oracle.calls
            if (
                len(question["events"]) >= 2
                and remaining >= 40
                and event_contrast(question["events"][1]).prefix
                == oracle.contrast.prefix
            ):
                alternate = CausalOracle(
                    model,
                    event_contrast(question["events"][1]),
                    budget=40,
                    artifact_writer=donor_writer(artifact_dir / "alternate"),
                )
                try:
                    alternate.group(primary["group"], 1)
                    alt_position = bounded(
                        alternate,
                        12,
                        lambda: validate_fixed_group(
                            alternate,
                            primary["group"],
                            primary["controls"],
                            primary["id"],
                        ),
                    )
                    alt_origin = {
                        "status": "original_origin_not_certified",
                        "passed": False,
                    }
                    original_origin = origins.get(primary["id"], {})
                    if original_origin.get("passed"):
                        alt_origin = bounded(
                            alternate,
                            24,
                            lambda: validate_origin(
                                alternate,
                                primary["group"],
                                original_origin["origin"],
                                original_origin["controls"],
                                alt_position,
                            ),
                        )
                    passed = (
                        alt_position.get("passed")
                        and alt_position["delta"] * primary["delta"] > 0
                    )
                    origin_passed = bool(
                        alt_origin.get("passed")
                        and original_origin.get("passed")
                        and alt_origin["delta"] * original_origin["delta"] > 0
                    )
                    template = {
                        "status": "tested",
                        "passed": bool(passed),
                        "position": alt_position,
                        "origin": alt_origin,
                        "origin_passed": origin_passed,
                        "primary_id": primary["id"],
                        "primary_selection_reason": "first_allowed_content_position_before_template_effects",
                        "records": alternate.records,
                        "donor_artifacts": alternate.donor_artifacts,
                    }
                finally:
                    alternate_calls = alternate.calls
                    alternate_tokens = alternate.tokens_processed
                    oracle.budget -= alternate_calls
            else:
                template = {
                    "status": "template_budget_or_event_unresolved",
                    "passed": False,
                }
    bands = {"status": "no_primary_certificate", "bands": []}
    if primary is not None:
        bands = bounded(oracle, 42, lambda: validate_layer_bands(oracle, primary))
    return {
        "status": "validated",
        "positions": positions,
        "origins": origins,
        "paired_role": paired,
        "MLP_interaction": interaction,
        "layer_bands": bands,
        "template": template,
        "records": oracle.records,
        "donor_artifacts": oracle.donor_artifacts,
        "forward_calls": proposal["forward_calls"] + oracle.calls + alternate_calls,
        "new_forward_calls": oracle.calls + alternate_calls,
        "budget": 320,
        "tokens_processed": proposal["tokens_processed"]
        + oracle.tokens_processed
        + alternate_tokens,
        "branch_policy": "N_primary_content_template_only"
        if is_missing
        else "C_or_supported_recovery_full_validation",
        "conditional_event_scope": "teacher_forced_shared_prefix_gates",
    }
