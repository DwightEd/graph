"""Independent output-token capture/resume followed by label-free vector freezing."""

from pathlib import Path
from time import perf_counter

from tqdm import tqdm
from state_audit.storage import read_arrays, read_json, start_stage, write_arrays, write_json

from ..evidence_contrast.data import copy_annotations
from ..evidence_contrast.run import pack
from ..inputs import validate_tokenizer
from .data import prepare
from .token_representation import assemble_tokens, schema


def capture(args, settings):
    import torch
    from state_audit.model import load_model
    from .token_capture import capture_token

    torch.set_num_threads(args.cpu_threads)
    model, tokenizer = None, None
    for index, response in enumerate(tqdm(settings["responses"], desc="token-choice answers")):
        directory = args.output / "responses" / f"{index:04d}"
        views = read_json(directory / "views.json")
        pending = [target for target in range(len(views["answer_ids"]))
                   if not (directory / f"token_{target:06d}.json").is_file()]
        if not pending:
            continue
        if model is None:
            model, tokenizer = load_model(settings["model"], device=args.device, dtype=args.dtype)
        validate_tokenizer(response, tokenizer)
        for target in tqdm(pending, desc=f"{response['id']} independent targets", leave=False):
            started = perf_counter()
            device = model.native.device
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            values = capture_token(model, views, target, args.top_k, args.receiver_budget,
                                   args.selection, [args.seed, index, target])
            values["foil_text"] = tokenizer.decode([int(values["foil_id"])])
            write_arrays(directory / f"token_{target:06d}.npz", **values)
            write_json(directory / f"token_{target:06d}.json", dict(target=target,
                seconds=perf_counter() - started, device=str(device),
                peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0))


def score_all(output, settings):
    coverage = []
    for index, response in enumerate(tqdm(settings["responses"], desc="freeze independent token vectors")):
        directory = output / "responses" / f"{index:04d}"
        views = read_json(directory / "views.json")
        measured = []
        for target in range(len(views["answer_ids"])):
            read_json(directory / f"token_{target:06d}.json")
            measured.append(read_arrays(directory / f"token_{target:06d}.npz"))
        layout = schema(*measured[0]["head_shape"])
        start_stage(output / "feature_schema.json", layout, resume=True)
        scores, vectors = assemble_tokens(views, read_arrays(directory / "baselines.npz"), measured)
        write_arrays(directory / "scores.npz", **scores)
        write_arrays(directory / "node_features.npz", **vectors)
        coverage.append(dict(response_id=response["id"], planned_tokens=len(measured),
            represented_tokens=len(vectors["node_features"]), intervention_tokens=int((scores["selected_count"] > 0).sum()),
            dimension=layout["dimension"], independent_token_candidates_use_future=False,
            reconstruction_unmeasured_tokens=sum(not value["reconstruction_measured"] for value in measured)))
    write_json(output / "coverage.json", dict(status="complete", responses=coverage,
        represented_tokens=sum(row["represented_tokens"] for row in coverage), dropped_tokens=0))


def finish(args, settings, protocol):
    from .token_report import evaluate

    coverage = read_json(args.output / "coverage.json")
    copy_annotations(args.output, Path(protocol["input"]), args.annotations)
    result = evaluate(args.output, settings, protocol)
    timings = [read_json(path) for path in args.output.glob("responses/*/token_*.json")]
    write_json(args.output / "summary.json", dict(protocol=protocol, coverage=coverage, evaluation=result,
        capture_seconds=sum(row["seconds"] for row in timings), peak_cuda_bytes=max(row["peak_cuda_bytes"] for row in timings)))
    return dict(output=str(args.output), status=result["status"], review_archive=pack(args.output),
        represented_tokens=coverage["represented_tokens"],
        all_error={name: {key: phases["all_error"][key] for key in ("auroc", "ap")}
                   for name, phases in result.get("methods", {}).items()})


def run(args):
    if args.stage in ("run", "capture"):
        settings, protocol = prepare(args)
        capture(args, settings)
    else:
        settings = read_json(args.output / "settings.json")
        protocol = read_json(args.output / "protocol.json")
    if args.stage == "capture":
        return dict(output=str(args.output), status="captured", review_archive=pack(args.output))
    if args.stage in ("run", "score"):
        score_all(args.output, settings)
    return finish(args, settings, protocol)
