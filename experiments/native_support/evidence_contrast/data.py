"""Copy immutable observations at preparation; annotations are opened only later."""

from contextlib import closing

import numpy as np
from state_audit.storage import start_stage, write_arrays

from ..choice_cache import CaptureReader
from .scoring import BASELINES
from .views import prepare_views


def prepare(args):
    with closing(CaptureReader(args.input)) as reader:
        settings = reader.json("settings.json")
        if not settings["responses"]:
            raise ValueError("Input has no observed answers")
        protocol = dict(version="evidence-contrast-v1", input=str(args.input.resolve()),
            model=settings["model"], dtype=args.dtype, max_unit_tokens=args.max_unit_tokens,
            window=args.window, primary_candidate="source_pair", default_risk="raw_route",
            score="-0.5 * (source_effect_full_history + source_effect_local_history)",
            units="text_punctuation_then_token_budget; not_semantic_or_reanchor_boundaries",
            source_removal="delete_saved_prompt_source_token_ids; no_retokenization",
            positions="native_positions_recomputed_after_deletion; not_position_matched_causal_effect",
            labels_used_for_training=False, labels_used_for_scoring=False, parameter_fitting=False,
            automatic_model_selection=False, probability_calibration=False,
            future_tokens_used=True, future_scope="text_boundary_planning_and_offline_span/mean_controls",
            source_role="available_source_blocks_not_verified_applicable_evidence",
            observer_scope="teacher_forcing; may_differ_from_original_generator",
            cohort_role="exploratory_unless_independent_sources_verified_externally",
            conditions=["source_history", "history", "source_local", "local"])
        start_stage(args.output / "protocol.json", protocol, args.resume)
        start_stage(args.output / "settings.json", settings, args.resume)
        for index, response in enumerate(settings["responses"]):
            directory = f"responses/{index:04d}"
            sources = reader.json(directory + "/sources.json")
            views = prepare_views(response, sources, args.max_unit_tokens)
            start_stage(args.output / directory / "views.json", views, args.resume)
            start_stage(args.output / directory / "sources.json", sources, args.resume)
            baseline = reader.arrays(directory + "/scores.npz", (*BASELINES, "target", "token_id"))
            validate_baselines(response, baseline)
            write_arrays(args.output / directory / "baselines.npz", **baseline)
    return settings, protocol


def validate_baselines(response, values):
    tokens = response["token_ids"][response["prompt_length"]:]
    if not np.array_equal(values["token_id"], tokens) or not np.array_equal(values["target"], np.arange(len(tokens))):
        raise ValueError(f"{response['id']}: baseline token alignment differs")
    for name in BASELINES:
        if values[name].shape != (len(tokens),) or not np.isfinite(values[name]).all():
            raise ValueError(f"{response['id']}: incomplete {name} baseline")


def copy_annotations(output, input_path, explicit_path):
    """Called only after all answer scores have been frozen on disk."""
    target = output / "annotations.json"
    if explicit_path:
        target.write_bytes(explicit_path.read_bytes())
    elif input_path.exists():
        with closing(CaptureReader(input_path)) as reader:
            if reader.exists("annotations.json"):
                target.write_bytes(reader.bytes("annotations.json"))
    return target
