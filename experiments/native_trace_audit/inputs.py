"""Original reviewed prefixes and explicitly separate teacher-forced binding probes."""

import numpy as np

GROUP_NAMES = (
    "scope",
    "supported_value",
    "value_source",
    "other_prompt",
    "recent_history",
    "earlier_history",
    "query_self",
    "special",
)
SPECIAL_SPELLINGS = {
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
}
CONTROL_PANELS = {
    "14315_headwear_scope": {
        "name": "parallel_headwear",
        "suffix": " a",
        "candidates": [
            " cap with a folded piece of cloth tied on top",
            " headdress with a special fringe of gold and feathers",
        ],
        "meaning": "Same headwear slot after a forced common determiner; new teacher-forced prefix.",
    },
    "14375_onion_stage": {
        "name": "stage_binding",
        "suffix": (".\nThe quoted 10 to 12 minute interval occurs"),
        "candidates": [
            " before removing the bratwurst from the beer mixture",
            " after removing the bratwurst from the beer mixture",
        ],
        "meaning": "Before/after binding of the quoted interval; an appended diagnostic statement, "
        "not the original duration decision or a replacement answer.",
    },
}


def validate_context(context):
    """Validate the archived identities and semantic-role quotes once, at input."""
    ids, text = context["prefix_ids"], context["token_text"]
    prompt = context["prompt_length"]
    if len(ids) != len(text) or not 0 < prompt <= len(ids):
        raise ValueError(f"{context['case_id']}: prefix/token alignment mismatch")
    used = []
    for role, positions in context["roles"].items():
        if not positions or min(positions) < 0 or max(positions) >= prompt:
            raise ValueError(f"Invalid reviewed prompt role: {role}")
        role_text = "".join(text[index] for index in positions)
        for quote in context["reviewed_case"]["source_roles"][role]:
            if quote not in role_text:
                raise ValueError(
                    f"{role}: saved token positions do not contain reviewed quote"
                )
        used.extend(positions)
    if len(used) != len(set(used)):
        raise ValueError("Reviewed source roles must be disjoint")
    if context["candidates"][0][0] == context["candidates"][1][0]:
        raise ValueError(
            "Natural onset readout requires distinct first candidate tokens"
        )


def inventory(manifest):
    rows = []
    prompts = {}
    for context in manifest["contexts"]:
        validate_context(context)
        case_id = context["case_id"]
        prompt = context["prefix_ids"][: context["prompt_length"]]
        if case_id in prompts and prompts[case_id] != prompt:
            raise ValueError(f"{case_id}: paired sides have different prompts")
        prompts[case_id] = prompt
        rows.append(
            {
                "case_id": case_id,
                "side": context["side"],
                "source_id": context["source_id"],
                "prefix_tokens": len(context["prefix_ids"]),
                "prompt_tokens": context["prompt_length"],
                "history_status": context["history_status"],
                "natural_readout": natural_meaning(case_id),
                "binding_probe": CONTROL_PANELS[case_id]["meaning"],
                "role_quotes_verified": True,
                "model_trajectory_measured": False,
            }
        )
    return rows


def natural_meaning(case_id):
    if case_id == "14375_onion_stage":
        return "over/for: different attribute slots; descriptive wording contrast only"
    return "caps/a: determiner and number confounded; first-token wording contrast only"


def partition_sources(context, prefix_ids, special_ids, recent):
    count, prompt = len(prefix_ids), context["prompt_length"]
    groups = np.full(count, GROUP_NAMES.index("other_prompt"), dtype=np.int64)
    groups[prompt:] = GROUP_NAMES.index("earlier_history")
    groups[max(prompt, count - 1 - recent) : count - 1] = GROUP_NAMES.index(
        "recent_history"
    )
    if count > prompt:
        groups[-1] = GROUP_NAMES.index("query_self")
    for role, positions in context["roles"].items():
        groups[positions] = GROUP_NAMES.index(role)
    groups[np.isin(prefix_ids, special_ids)] = GROUP_NAMES.index("special")
    return groups.tolist()


def compile_panels(context, tokenizer):
    """Preserve archived token IDs; forced text is recorded and never called a natural draw."""
    prefix = context["prefix_ids"]
    decoded = tokenizer.decode(prefix, clean_up_tokenization_spaces=False)
    if decoded != "".join(context["token_text"]):
        raise ValueError(
            f"{context['case_id']}: observer tokenizer does not match saved prefix"
        )
    native = {
        "name": "natural",
        "token_ids": prefix,
        "candidates": context["candidates"],
        "candidate_texts": context["reviewed_case"]["candidates"],
        "suffix": "",
        "meaning": natural_meaning(context["case_id"]),
        "natural": True,
    }
    control = CONTROL_PANELS[context["case_id"]]
    suffix_ids = tokenizer.encode(control["suffix"], add_special_tokens=False)
    candidates = [
        tokenizer.encode(text, add_special_tokens=False)
        for text in control["candidates"]
    ]
    if candidates[0][0] == candidates[1][0]:
        raise ValueError("Binding probe must start at its first divergent token")
    for text, ids in zip(control["candidates"], candidates):
        replay = tokenizer.decode(
            prefix + suffix_ids + ids, clean_up_tokenization_spaces=False
        )
        if replay != decoded + control["suffix"] + text:
            raise ValueError(
                "Forced candidate tokenization does not match declared text"
            )
    controlled = {
        "name": control["name"],
        "token_ids": prefix + suffix_ids,
        "candidates": candidates,
        "candidate_texts": control["candidates"],
        "suffix": control["suffix"],
        "meaning": control["meaning"],
        "natural": False,
    }
    return native, controlled


def query_plans(context, panel, special_ids, window, recent, receiver_budget):
    ids = panel["token_ids"]
    last = len(ids) - 1
    first = (
        max(context["prompt_length"] - 1, last - window) if panel["natural"] else last
    )
    plans = []
    for query in range(first, last + 1):
        if not panel["natural"]:
            observed = None
        elif query < last:
            observed = ids[query + 1]
        else:
            observed = panel["candidates"][int(context["side"] == "unsupported")][0]
        plans.append(
            {
                "query": query,
                "candidate_ids": [values[0] for values in panel["candidates"]],
                "observed_id": observed,
                "group_ids": partition_sources(
                    context, ids[: query + 1], special_ids, recent
                ),
                "group_names": list(GROUP_NAMES),
                "positive_groups": [0, 1],
                "negative_groups": [2],
                "dependencies": query == last,
                "receiver_budget": receiver_budget,
                "decision_query": last,
                "decision_offset": query - last,
                "readout_scope": "decision"
                if query == last
                else "future_candidate_direction_diagnostic",
            }
        )
    return plans
