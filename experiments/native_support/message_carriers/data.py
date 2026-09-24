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
        protocol = measurement_protocol(args, settings)
        start_stage(args.output / "protocol.json", protocol, args.resume)
        start_stage(args.output / "settings.json", settings, args.resume)
        start_stage(args.output / "input_protocol.json", reader.json("protocol.json"), args.resume)
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
            saved = aggregate_scores(reader.arrays(name + "/scores.npz"), views)
            write_arrays(args.output / name / "baselines.npz", **saved)
    return settings, protocol
