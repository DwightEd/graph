"""Audit the existing QA checkpoint, using its saved partitions and prepared graphs."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
import torch
from tqdm import tqdm

from ..model import load_checkpoint
from .controls import MODES, compare_inputs, edge_groups, match_sources, permute_weights
from .explain import evaluate_explanations, fit_explanations, screen_head_pairs
from .features import feature_columns, feature_names, graph_features
from .messages import capture_states, predict, replace_source_states, state_summary, remove_head_interaction
from .report import write_json, write_report


VERSION = 1


def read_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name] for name in saved.files}


def save_sample(path, sample):
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: sample[key] for key in ("features", "logits", "score", "gold", "onset", "offsets", "response")}
    arrays["identity"] = np.asarray(json.dumps({key: sample[key] for key in ("id", "source_id")}))
    arrays["diagnostics"] = np.asarray(json.dumps(sample["diagnostics"]))
    arrays.update({"control_" + key: value for key, value in sample["controls"].items()})
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_sample(path, subsample=False):
    arrays = read_npz(path)
    sample = json.loads(str(arrays.pop("identity")))
    sample["diagnostics"] = json.loads(str(arrays.pop("diagnostics")))
    sample["controls"] = {key[8:]: arrays.pop(key) for key in list(arrays) if key.startswith("control_")}
    sample.update(arrays)
    sample["response"] = str(sample["response"])
    if subsample:
        count = len(sample["logits"])
        positions = np.unique(np.linspace(0, count - 1, min(count, 64), dtype=int))
        for key in ("features", "logits", "score", "gold", "onset", "offsets"):
            sample[key] = sample[key][positions].copy()
    return sample


def load_inputs(args):
    model_dir = Path(args.model_dir)
    training = json.loads((model_dir / "training.json").read_text())
    predictions = json.loads((model_dir / "test/prediction_settings.json").read_text())
    records = json.loads((Path(args.prepared) / "index.json").read_text())
    records = {str(row["id"]): row for row in records}
    if training["variant"] not in ("charm_out", "charm_in"):
        raise ValueError("this audit expects the real-graph charm_out or charm_in checkpoint")
    ids = training["partitions"]
    groups = [{records[str(identity)]["source_id"] for identity in ids[part]} for part in ("fit", "select", "test")]
    if any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i)):
        raise ValueError("fit, selection and test must remain source-disjoint")
    if set(ids["test"]) != set(predictions["records"]):
        raise ValueError("saved predictions are not the test set of this training run")
    checkpoint = model_dir / "checkpoint.pt"
    settings = dict(version=VERSION, checkpoint=[str(checkpoint.resolve()), checkpoint.stat().st_size,
        checkpoint.stat().st_mtime_ns], prepared=str(Path(args.prepared).resolve()),
        partitions=ids, seeds=args.seeds, max_features=args.max_features,
        channels=args.channels, variant=training["variant"], threshold=predictions["threshold"])
    return records, settings


def prepared_graph(args, row):
    path = Path(args.prepared) / "graphs" / row["split"] / (str(row["id"]) + ".npz")
    graph = read_npz(path)
    if graph["edge_attr"].shape[1] != int(graph["layers"]) * int(graph["heads"]):
        raise ValueError("head dimensions differ from prepared edge channels")
    return graph, [path.stat().st_size, path.stat().st_mtime_ns]


def verify_baseline(args, identity, graph, logits):
    """Stop before attribution if this is not the model/run that produced the old scores."""
    path = Path(args.model_dir) / "test/samples" / (str(identity) + ".npz")
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["gold"], graph["gold"])
        np.testing.assert_array_equal(saved["offsets"], graph["offsets"])
        np.testing.assert_allclose(expit(logits), saved["score"], atol=1e-5, rtol=1e-5,
                                   err_msg="checkpoint/graph no longer reproduces saved predictions")
        return saved["score"].copy()


def make_sample(row, graph, stamp, logits, features, score=None):
    return dict(id=str(row["id"]), source_id=str(row["source_id"]), logits=logits, features=features,
        score=expit(logits) if score is None else score,
        gold=graph["gold"], onset=graph["onset"], offsets=graph["offsets"], response=str(graph["response"]),
        diagnostics=dict(graph_stamp=stamp), controls={})


def prepare_features(args, records, settings, model):
    identities = settings["partitions"]["fit"] + settings["partitions"]["select"] + settings["partitions"]["test"]
    for identity in tqdm(identities, desc="读取现有图，计算每头熵", unit="answer"):
        row = records[str(identity)]
        path = Path(args.output) / "features" / (str(identity) + ".npz")
        graph, stamp = prepared_graph(args, row)
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if json.loads(str(saved["diagnostics"]))["graph_stamp"] != stamp:
                    raise ValueError("prepared graph changed; use a separate audit output")
            continue
        logits = predict(model, graph)
        features, _ = graph_features(graph)
        score = None
        if identity in settings["partitions"]["test"]:
            score = verify_baseline(args, identity, graph, logits)
        save_sample(path, make_sample(row, graph, stamp, logits, features, score))


def explain_scores(args, records, settings):
    parts = settings["partitions"]
    fit_samples = [load_sample(Path(args.output) / "features" / (str(identity) + ".npz"), True)
                   for identity in parts["fit"] + parts["select"]]
    graph, _ = prepared_graph(args, records[str(parts["fit"][0])])
    columns = feature_columns(graph["x"].shape[1], int(graph["layers"]))
    names = feature_names(int(graph["layers"]), int(graph["heads"]))
    pairs = screen_head_pairs(fit_samples, parts, int(graph["layers"]), int(graph["heads"]))
    del graph
    models, descriptions = fit_explanations(fit_samples, parts, columns, args.max_features)
    for description in descriptions.values():
        description["selected_features"] = [names[index] for index in description["selected_columns"]]
    del fit_samples
    test_samples = [load_sample(Path(args.output) / "features" / (str(identity) + ".npz")) for identity in parts["test"]]
    result = dict(fitting=descriptions, interaction_pairs=pairs, test=evaluate_explanations(models, test_samples),
        note="Fit/selection targets are CHARM logits, not gold. Selected features are not complete entropy control.")
    save_explanation_predictions(args.output, models, test_samples)
    write_json(Path(args.output) / "score_explanation.json", result)
    return result


def save_explanation_predictions(output, models, samples):
    """Keep predictions and residuals so a claimed explanation can be inspected token by token."""
    arrays = dict(ids=np.concatenate([np.repeat(sample["id"], len(sample["logits"])) for sample in samples]),
                  token=np.concatenate([np.arange(len(sample["logits"])) for sample in samples]),
                  logits=np.concatenate([sample["logits"] for sample in samples]))
    features = np.concatenate([sample["features"] for sample in samples])
    for name, (columns, model) in models.items():
        arrays[name] = model.predict(features[:, columns])
        arrays[name + "_residual"] = arrays["logits"] - arrays[name]
    path = Path(output) / "explanation_predictions.npz"
    with path.with_suffix(".partial").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    path.with_suffix(".partial").replace(path)


def run_pairing_controls(model, graph, sample, seeds):
    for mode in MODES:
        for seed in seeds:
            altered = permute_weights(graph, mode, seed)
            name = mode + "_s" + str(seed)
            sample["controls"][name] = predict(model, altered)
            sample["diagnostics"][name] = compare_inputs(graph, altered)


def run_state_controls(model, graph, sample, states):
    groups = edge_groups(graph)
    donors = match_sources(graph, sample["features"])
    source, target = graph["edge_index"]
    eligible = donors["eligible"]
    sample["diagnostics"]["donors"] = dict(eligible_edges=int(eligible.sum()),
        rr_edges=int((source >= int(graph["prompt_length"])).sum()),
        direct_target_positions=np.unique(target[eligible] - int(graph["prompt_length"])).tolist(),
        eligibility="both same-label and other-label donor satisfy fixed coarse matching")
    sample["diagnostics"]["states"] = state_summary(graph, states)
    for layer_index in range(len(states)):
        sample["controls"][f"group_mean_g{layer_index}"] = replace_source_states(
            model, graph, layer_index, states, groups)
        for name in ("same_label", "other_label"):
            sample["controls"][f"{name}_g{layer_index}"] = replace_source_states(
                model, graph, layer_index, states, groups, donors[name])


def run_head_controls(model, graph, sample, channels):
    """Optional predeclared physical heads; do not choose after inspecting test effects."""
    heads = int(graph["heads"])
    for channel in channels:
        layer, head = map(int, channel.split(":"))
        if not (0 <= layer < int(graph["layers"]) and 0 <= head < heads):
            raise ValueError("channel must be a valid zero-based layer:head")
        index = layer * heads + head
        changed = dict(graph, x=graph["x"].copy(), edge_attr=graph["edge_attr"].copy())
        changed["x"][:, index] = 0
        changed["edge_attr"][:, index] = 0
        sample["controls"][f"remove_L{layer}_H{head}"] = predict(model, changed)


def run_interaction_controls(args, model, graph, sample):
    path = Path(args.output) / "score_explanation.json"
    if not path.exists():
        sample["diagnostics"]["head_interactions"] = "not run: no fit-only head-pair screening yet"
        return
    pairs = json.loads(path.read_text())["interaction_pairs"]
    sample["diagnostics"]["head_interactions"] = pairs
    heads = int(graph["heads"])
    for pair in pairs:
        first = pair["layer"] * heads + pair["first_head"]
        second = pair["layer"] * heads + pair["second_head"]
        for gnn_layer in range(len(model.mp_layers)):
            name = f"interaction_L{pair['layer']}_H{pair['first_head']}_H{pair['second_head']}_g{gnn_layer}"
            logits, details = remove_head_interaction(model, graph, gnn_layer, first, second)
            sample["controls"][name] = logits
            sample["diagnostics"][name] = details


def run_interventions(args, records, settings, model):
    pair_file = Path(args.output) / "score_explanation.json"
    expected_pairs = json.loads(pair_file.read_text())["interaction_pairs"] if pair_file.exists() else "not run: no fit-only head-pair screening yet"
    for identity in tqdm(settings["partitions"]["test"], desc="固定模型，检查多头与邻居", unit="answer"):
        row = records[str(identity)]
        graph, stamp = prepared_graph(args, row)
        path = Path(args.output) / "samples" / (str(identity) + ".npz")
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                details = json.loads(str(saved["diagnostics"]))
                if details["graph_stamp"] != stamp:
                    raise ValueError("prepared graph changed; use a separate audit output")
                if details["head_interactions"] != expected_pairs:
                    raise ValueError("head-pair screening changed; use a new intervention output")
            continue
        logits, states = capture_states(model, graph)
        score = verify_baseline(args, identity, graph, logits)
        features, _ = graph_features(graph)
        sample = make_sample(row, graph, stamp, logits, features, score)
        run_pairing_controls(model, graph, sample, args.seeds)
        run_state_controls(model, graph, sample, states)
        run_head_controls(model, graph, sample, args.channels)
        run_interaction_controls(args, model, graph, sample)
        save_sample(path, sample)


def run(args):
    torch.set_num_threads(args.threads)
    records, settings = load_inputs(args)
    output = Path(args.output)
    if output.resolve() in (Path(args.model_dir).resolve(), (Path(args.model_dir) / "test").resolve(), Path(args.prepared).resolve()):
        raise ValueError("write the audit in a separate directory")
    path = output / "settings.json"
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError("audit settings changed; choose another output directory")
    write_json(path, settings)
    normalization = "out" if settings["variant"] == "charm_out" else "in"
    if args.stage in ("all", "features", "intervene"):
        model, _ = load_checkpoint(Path(args.model_dir) / "checkpoint.pt", args.device, normalization, args.edge_chunk)
        model.requires_grad_(False)
    if args.stage in ("all", "features"):
        prepare_features(args, records, settings, model)
    if args.stage in ("all", "explain"):
        explain_scores(args, records, settings)
    if args.stage in ("all", "intervene"):
        run_interventions(args, records, settings, model)
    if args.stage in ("all", "intervene", "report"):
        if args.completed_only:
            files = sorted((output / "samples").glob("*.npz"))
        else:
            files = [output / "samples" / (str(identity) + ".npz") for identity in settings["partitions"]["test"]]
        samples = [load_sample(file) for file in files]
        if not samples:
            raise ValueError("no completed intervention samples")
        result = write_report(samples, output, settings["threshold"]["value"], args.bootstrap)
        result["completed_preview"] = args.completed_only
        write_json(output / "mechanisms.json", result)
    print("审计结果：", output, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="outputs/charm_structure_audit_qa/QA/seed_0/charm_out")
    parser.add_argument("--prepared", default="outputs/charm_structure_audit_qa/data")
    parser.add_argument("--output", default="outputs/charm_structure_audit_qa/QA/seed_0/charm_out/test/learned_audit")
    parser.add_argument("--stage", choices=["all", "features", "explain", "intervene", "report"], default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--channels", nargs="*", default=[], help="Optional predeclared zero-based layer:head")
    parser.add_argument("--max-features", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--edge-chunk", type=int, default=4096)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--completed-only", action="store_true")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
