"""Reanalyse completed four-condition captures without changing them or loading a model."""

from contextlib import closing

from state_audit.storage import start_stage, write_arrays, write_json

from ..choice_cache import CaptureReader
from .aggregation import METHODS, aggregate_scores
from .unit_report import evaluate_units


def freeze_aggregation(reader, output, settings):
    for index, response in enumerate(settings["responses"]):
        directory = f"responses/{index:04d}"
        views = reader.json(directory + "/views.json")
        values = aggregate_scores(reader.arrays(directory + "/scores.npz"), views)
        original = response["token_ids"][response["prompt_length"]:]
        if original != views["answer_ids"]:
            raise ValueError(f"{response['id']}: settings and saved unit identities differ")
        write_arrays(output / directory / "scores.npz", **values)
        write_json(output / directory / "views.json", views)


def analyze(args):
    from .run import pack

    protocol = dict(version="evidence-contrast-aggregation-v1", input=str(args.input.resolve()),
        scope="post_hoc_exploratory_cached_analysis", methods=list(METHODS),
        primary_candidate="source_pair", default_risk="raw_route", automatic_model_selection=False,
        labels_used_for_scoring=False, parameter_fitting=False,
        aggregation="arithmetic_mean_on_identical_saved_units_for_all_methods",
        future_tokens_used=True, unit_scores="offline_retrospective_attribution_not_onset_prediction",
        unit_end_delay="lower_bound_excluding_text_boundary_lookahead", new_model_forwards=0)
    start_stage(args.output / "protocol.json", protocol, args.resume)
    with closing(CaptureReader(args.input)) as reader:
        settings = reader.json("settings.json")
        start_stage(args.output / "settings.json", settings, args.resume)
        write_json(args.output / "capture_protocol.json", reader.json("protocol.json"))
        freeze_aggregation(reader, args.output, settings)
        # Truth is accessed only once every new score is saved.
        if args.annotations:
            (args.output / "annotations.json").write_bytes(args.annotations.read_bytes())
        elif reader.exists("annotations.json"):
            (args.output / "annotations.json").write_bytes(reader.bytes("annotations.json"))
    result = evaluate_units(args.output, settings)
    write_json(args.output / "summary.json", dict(protocol=protocol, evaluation=result))
    return dict(output=str(args.output), status=result["status"], review_archive=pack(args.output),
        all_error={name: {key: phases["all_error"][key] for key in ("auroc", "ap")}
                   for name, phases in result.get("methods", {}).items()})
