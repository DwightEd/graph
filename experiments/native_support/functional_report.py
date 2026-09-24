"""Original-answer measurements and explicit coverage; no semantic acceptance gate."""

import numpy as np

from state_audit.storage import read_arrays, read_json, write_csv, write_json

TOKEN_FIELDS = (
    "log_probability", "entropy", "pre_ffn_log_probability", "pre_ffn_entropy",
    "rms_direct_margin", "rms_rescale_margin", "rms_margin_delta",
    "rms_identity_error", "rms_native_rounding_error", "rms_foil_ids",
    "rms_scale_before", "rms_scale_after",
)
TOKEN_COLUMNS = ("response_id", "source_id", "start", "stop", "target", "query", "token_id", "token")
COVERAGE_COLUMNS = ("response_id", "target", "token_id", "token", "planned", "captured")
FFN_COLUMNS = ("response_id", "start", "stop", "target", "layer", "ffn_write_sensitivity")


def load_unit(directory, identity, response, sources):
    """Check cache coordinates, not textual meaning; a mismatch is an error, not a skip."""
    complete = read_json(directory / "complete.json")
    if any(complete[key] != value for key, value in identity.items()):
        raise ValueError(f"{directory}: completed interval differs from plan")
    row = read_arrays(directory / "observed.npz")
    prompt, start, stop = response["prompt_length"], identity["start"], identity["stop"]
    targets = response["token_ids"][prompt + start:prompt + stop]
    expected_groups = sources["group_ids"][:prompt + start]
    expected_groups += [len(sources["blocks"]) + 3] * (stop - start - 1)
    if not np.array_equal(row["target_ids"], targets):
        raise ValueError(f"{directory}: observed target IDs differ")
    if not np.array_equal(row["query"], np.arange(prompt + start - 1, prompt + stop - 1)):
        raise ValueError(f"{directory}: prediction positions differ")
    if not np.array_equal(row["key_group_ids"], expected_groups):
        raise ValueError(f"{directory}: key groups differ")
    if row["head_total"].shape[0] != stop - start:
        raise ValueError(f"{directory}: not every observed query was captured")
    if not all(np.isfinite(array).all() for array in row.values()):
        raise ValueError(f"{directory}: nonfinite measurement")
    return complete, row


def unit_rows(identity, response, captured):
    tokens, ffn = [], []
    common = {key: identity[key] for key in ("response_id", "source_id", "start", "stop")}
    for offset, target in enumerate(range(identity["start"], identity["stop"])):
        position = response["prompt_length"] + target
        tokens.append(dict(**common, target=target, query=position - 1,
            token_id=response["token_ids"][position], token=response["token_text"][position],
            **{name: captured[name][offset].item() for name in TOKEN_FIELDS}))
        for layer, value in enumerate(captured["ffn_write_sensitivity"][offset]):
            ffn.append(dict(response_id=response["id"], start=identity["start"], stop=identity["stop"],
                            target=target, layer=layer, ffn_write_sensitivity=float(value)))
    return tokens, ffn


def coverage_rows(settings, plan, completed):
    rows, summaries = [], []
    for index in plan["response_indices"]:
        response = settings["responses"][index]
        prompt = response["prompt_length"]
        size = len(response["token_ids"]) - prompt
        planned, measured = np.zeros(size, dtype=bool), np.zeros(size, dtype=bool)
        for identity in plan["units"]:
            if identity["response_index"] == index:
                planned[identity["start"]:identity["stop"]] = True
        for identity in completed:
            if identity["response_index"] == index:
                measured[identity["start"]:identity["stop"]] = True
        summaries.append(dict(response_id=response["id"], answer_tokens=size,
            planned_tokens=int(planned.sum()), captured_tokens=int(measured.sum()),
            unselected_tokens=int((~planned).sum()),
            missing_targets=np.flatnonzero(planned & ~measured).tolist()))
        for target in range(size):
            rows.append(dict(response_id=response["id"], target=target,
                token_id=response["token_ids"][prompt + target],
                token=response["token_text"][prompt + target],
                planned=bool(planned[target]), captured=bool(measured[target])))
    return rows, summaries


def report(destination, settings):
    if read_json(destination / "protocol.json")["version"] != 3:
        raise ValueError("Native v3 report requires a v3 output directory; v1/v2 files stay unchanged")
    plan = read_json(destination / "plan.json")
    units, tokens, ffn = [], [], []
    for identity in plan["units"]:
        parent = destination / "responses" / f"{identity['response_index']:04d}"
        directory = parent / f"unit_{identity['start']:06d}"
        if not (directory / "complete.json").is_file():
            continue
        response = settings["responses"][identity["response_index"]]
        complete, captured = load_unit(directory, identity, response, read_json(parent / "sources.json"))
        token_rows, ffn_rows = unit_rows(identity, response, captured)
        units.append({**complete, "log_probability_sum": float(captured["log_probability"].sum(dtype=float)),
            **{f"max_abs_{name}": float(np.abs(captured[name]).max()) for name in
               ("rms_identity_error", "rms_native_rounding_error", "head_reconstruction_error")}})
        tokens.extend(token_rows)
        ffn.extend(ffn_rows)
    coverage, responses = coverage_rows(settings, plan, units)
    write_json(destination / "coverage.json", responses)
    write_csv(destination / "coverage.csv", coverage, list(COVERAGE_COLUMNS))
    write_json(destination / "units.json", units)
    write_csv(destination / "observed_tokens.csv", tokens, list(TOKEN_COLUMNS + TOKEN_FIELDS))
    write_csv(destination / "ffn_layers.csv", ffn, list(FFN_COLUMNS))
    expected = sum(row["planned_tokens"] for row in responses)
    status = "no_selected_tokens" if not expected else "partial_capture"
    if expected and len(tokens) == expected:
        status = "captured"
    return dict(status=status, planned_units=len(plan["units"]), completed_units=len(units),
        expected_tokens=expected, observed_tokens=len(tokens), missing_tokens=expected - len(tokens),
        unselected_tokens=sum(row["unselected_tokens"] for row in responses),
        candidate_generation=False, semantic_filter=False, labels_used_for_measurement=False,
        gradient_objective="observed_unit_log_probability_sum; not_token_truth_score",
        detector_evaluation=False, new_auroc=None)
