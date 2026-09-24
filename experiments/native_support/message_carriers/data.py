"""Reuse frozen contrast units and baselines; never open truth during preparation."""

from contextlib import closing

import numpy as np
from state_audit.storage import start_stage, write_arrays

from ..choice_cache import CaptureReader
from ..evidence_contrast.aggregation import aggregate_scores


def measurement_protocol(args, settings):
    return dict(version="message-carriers-v1", input=str(args.input.resolve()),
        model=settings["model"], dtype=args.dtype, top_k=args.top_k, seed=args.seed,
        candidate="one_history_key_per_head_then_top_k_heads_by_max_abs_two_world_gradient",
        objective="mean_log_probability_of_saved_unit; not_independent_token_gradients",
        key_scope="answer_history_strictly_before_unit_start",
        gate_scope="selected_layer_head_key_at_all_unit_queries; no_renormalization",
        native_effect="sparse_message_subtraction_before_WO; native_downstream_recomputed",
        numerical_limit="SDPA_minus_reconstructed_AV; native_dtype_kernel_roundoff",
        sham="same_heads_and_edge_count; random_other_eligible_key_when_available; not_energy_matched",
        source_removal="reuse_saved_prompts; positions_compact_after_source_deletion",
        representation="fixed_physical_layer_head_axes; unmeasured_head_mask_explicit",
        primary_candidate="source_selected", default_risk="raw_route",
        labels_used_for_scoring=False, labels_used_for_training=False, parameter_fitting=False,
        automatic_model_selection=False, probability_calibration=False, future_tokens_used=True,
        future_scope="whole_unit_gradient_selection_and_unit_mean_readout",
        observer_scope="teacher_forcing; source_effect_is_not_fact_correctness")


def prepare(args):
    with closing(CaptureReader(args.input)) as reader:
        settings = reader.json("settings.json")
        if not settings["responses"]:
            raise ValueError("Input has no observed answers")
        input_protocol = reader.json("protocol.json")
        protocol = token_protocol(args, settings) if args.mode == "token" else measurement_protocol(args, settings)
        if args.mode == "token":
            protocol["unit_carrier_control"] = input_protocol["version"] == "message-carriers-v1"
        start_stage(args.output / "protocol.json", protocol, args.resume)
        start_stage(args.output / "settings.json", settings, args.resume)
        start_stage(args.output / "input_protocol.json", input_protocol, args.resume)
        for index, response in enumerate(settings["responses"]):
            name = f"responses/{index:04d}"
            views = reader.json(name + "/views.json")
            original = response["token_ids"][response["prompt_length"]:]
            if not np.array_equal(original, views["answer_ids"]):
                raise ValueError(f"{response['id']}: original answer token identities differ")
            if views["prompt_with_source"] != response["token_ids"][:response["prompt_length"]]:
                raise ValueError(f"{response['id']}: original prompt identities differ")
            kept = [views["prompt_with_source"][position] for position in views["kept_prompt_positions"]]
            if kept != views["prompt_without_source"]:
                raise ValueError(f"{response['id']}: source-deleted prompt identities differ")
            start_stage(args.output / name / "views.json", views, args.resume)
            original_scores = reader.arrays(name + "/scores.npz")
            saved = aggregate_scores(original_scores, views)
            if args.mode == "token" and protocol["unit_carrier_control"]:
                saved["unit_carrier_v1"] = original_scores["source_selected"]
                saved["unit_carrier_v1_unit_mean"] = original_scores["source_selected_unit_mean"]
            write_arrays(args.output / name / "baselines.npz", **saved)
    return settings, protocol


def token_protocol(args, settings):
    return dict(version="message-carriers-token-v2", input=str(args.input.resolve()),
        model=settings["model"], dtype=args.dtype, top_k=args.top_k, seed=args.seed,
        receiver_budget_per_condition=args.receiver_budget, selection=args.selection,
        objective="independent_target_logit_minus_fixed_non_target_logit",
        foil="highest_non_target_with_source_logit; frozen_across_source_conditions_and_deletions",
        rank="absolute_source_difference_of_edge_margin_derivatives" if args.selection == "conditional"
             else "maximum_absolute_edge_margin_derivative_across_conditions",
        receivers="current_prediction_query_plus_per_head_top_earlier_gradient_norm_times_head_norm; union_two_conditions",
        key_scope="all_causally_visible_answer_history_including_same_text_unit",
        gate_scope="one_exact_layer_head_receiver_key; no_attention_renormalization",
        candidate_limit="one_edge_per_head; bounded_receiver_screen; no_complete_circuit_claim",
        sham="same_head_and_receiver; other_eligible_history_key_when_available; not_energy_or_lag_matched",
        source_removal="reuse_saved_source_deletion; native_positions_compact",
        native_effect="sparse_AV_subtraction_before_WO; downstream_QKV_FFN_recomputed",
        numerical_limit="kernel_and_native_dtype_roundoff; reconstructed_head_error_saved",
        primary_candidate="choice_selected", default_risk="raw_route",
        labels_used_for_scoring=False, labels_used_for_training=False, parameter_fitting=False,
        automatic_model_selection=False, token_future_used=False,
        unit_readouts_future_used=True, semantic_role_filter=False, sink_identification=False)
