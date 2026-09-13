"""Prepare and execute paired full-context reasoned source-graph predictions."""

import argparse
import fcntl
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from next_iteration.reasoned_graph import PROTOCOL, compile_prediction, packet
from route_graph.audit_artifacts import file_sha256
from route_graph.audit_runner import model_manifest, rows
from route_graph.frozen_reader import digest, write_json_once

ROOT = Path(__file__).resolve().parents[1]


def code_files():
    paths = list((ROOT / "route_graph").glob("*.py")) + [ROOT / "next_iteration" / n for n in
        ("__init__.py", "surface_graph.py", "graph_boundaries.py", "reasoned_graph.py", "reasoned_graph_runner.py")]
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in paths}


def checked_json(folder, name, artifacts):
    if file_sha256(folder / name) != artifacts[name]:
        raise ValueError("parent artifact changed: " + name)
    return json.loads((folder / name).read_text())


def prepare(args):
    if args.output.exists():
        raise FileExistsError("fresh reasoned graph output required")
    roster = rows(args)
    if any(r["official_split"] != "train" for r in roster):
        raise ValueError("this development invocation accepts official train only; freeze a separate evaluation invocation")
    parent = json.loads((args.inventory / "manifest.json").read_text())
    if parent["status"] != "complete":
        raise ValueError("complete source inventory required")
    if file_sha256(args.inventory / "settings.json") != parent["settings_file_sha256"]:
        raise ValueError("inventory settings changed")
    parent_settings = json.loads((args.inventory / "settings.json").read_text())
    refs = checked_json(args.inventory, "response_source_refs.json", parent["artifacts"])
    refs = {r["response_id"]: r for r in refs}
    for row in roster:
        expected = {"response_id": str(row["id"]), "source_id": str(row["source_id"]),
            "source_span_in_prompt": row["source_span"], "response_sha256": row["response_sha256"],
            "official_split": row["official_split"]}
        if refs.get(str(row["id"])) != expected:
            raise ValueError("response does not match source inventory roster")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    code = code_files()
    consumed = {f"sources/{r['source_id']}.json": parent["artifacts"][f"sources/{r['source_id']}.json"] for r in roster}
    settings = {"protocol": PROTOCOL, "input_path": str(args.inputs.resolve()), "input_sha256": file_sha256(args.inputs),
        "inventory_path": str(args.inventory.resolve()), "inventory_manifest_sha256": file_sha256(args.inventory / "manifest.json"),
        "inventory_settings_sha256": file_sha256(args.inventory / "settings.json"), "source_artifacts": consumed,
        "model_path": str(args.model.resolve()), "model_files": model_manifest(args.model), "code_sha256": code,
        "torch": str(torch.__version__), "transformers": transformers.__version__, "dtype": "bfloat16", "attention": "sdpa",
        "labels_read": False, "roster": [{k: r[k] for k in ("id", "source_id", "task", "generator", "official_split")} for r in roster],
        "comparison": "same full raw texts/position markers; graph adds observed provenance; paired request seed",
        "split_scope": "development-exposed official train; no final accuracy claim"}
    packets, inventories, lengths, artifacts = [], {}, [], {}
    for index, row in enumerate(roster):
        sid = str(row["source_id"])
        if sid not in inventories:
            source = checked_json(args.inventory, f"sources/{sid}.json", consumed)
            if source["settings_object_digest"] != digest(parent_settings):
                raise ValueError("source inventory settings binding differs")
            inventories[sid] = source["data"]
        modes = PROTOCOL["modes"] if index % 2 == 0 else list(reversed(PROTOCOL["modes"]))
        for mode in modes:
            request = packet(row, inventories[sid], mode)
            messages = [{"role": "system", "content": request["instruction"]},
                {"role": "user", "content": json.dumps(request["payload"], ensure_ascii=False)}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            request.update(rendered_prompt=rendered, input_ids=token_ids,
                source_row_sha256=digest(row), seed=int(digest([PROTOCOL["seed"], str(row["id"])])[:8], 16),
                context_status="available" if len(token_ids) + PROTOCOL["max_new_tokens"] <= PROTOCOL["context_limit"] else "context_unavailable")
            packets.append((f"packets/{row['id']}_{mode}.json", request))
            lengths.append({"response_id": str(row["id"]), "mode": mode, "input_tokens": len(token_ids), "status": request["context_status"]})
    # Publish only after every input is structurally assembled; no model load.
    write_json_once(args.output / "settings.json", settings)
    for name, sha in code.items():
        target = args.output / "executed_code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if file_sha256(target) != sha:
            raise ValueError("code changed while freezing")
    for name, request in packets:
        write_json_once(args.output / name, request)
        artifacts[name] = file_sha256(args.output / name)
    summary = {"status": "prepared", "responses": len(roster), "sources": len(inventories), "requests": len(packets),
        "lengths": lengths, "total_input_tokens": sum(x["input_tokens"] for x in lengths),
        "max_input_tokens": max((x["input_tokens"] for x in lengths), default=0),
        "unavailable": sum(x["status"] != "available" for x in lengths), "labels_read": False, "model_forwards": 0}
    write_json_once(args.output / "prepare_summary.json", summary)
    artifacts["prepare_summary.json"] = file_sha256(args.output / "prepare_summary.json")
    write_json_once(args.output / "prepare_manifest.json", {"status": "prepared", "artifacts": artifacts,
        "settings_sha256": file_sha256(args.output / "settings.json"), "request_order": [name for name, _ in packets]})
    verify(args.output)
    print(json.dumps(summary), flush=True)


def verify(output):
    manifest = json.loads((output / "prepare_manifest.json").read_text())
    if file_sha256(output / "settings.json") != manifest["settings_sha256"]:
        raise ValueError("frozen settings differ")
    settings = json.loads((output / "settings.json").read_text())
    if settings["protocol"] != PROTOCOL or code_files() != settings["code_sha256"]:
        raise ValueError("live protocol/code differs; preserve run and use new revision")
    for name, sha in settings["code_sha256"].items():
        if file_sha256(output / "executed_code" / name) != sha:
            raise ValueError("code snapshot changed")
    for name, sha in manifest["artifacts"].items():
        if file_sha256(output / name) != sha:
            raise ValueError("prepared artifact changed")
    if file_sha256(settings["input_path"]) != settings["input_sha256"]:
        raise ValueError("raw input changed")
    inventory = Path(settings["inventory_path"])
    if (file_sha256(inventory / "manifest.json") != settings["inventory_manifest_sha256"]
            or file_sha256(inventory / "settings.json") != settings["inventory_settings_sha256"]):
        raise ValueError("inventory parent changed")
    for name, sha in settings["source_artifacts"].items():
        if file_sha256(inventory / name) != sha:
            raise ValueError("consumed source graph changed")
    if model_manifest(Path(settings["model_path"])) != settings["model_files"]:
        raise ValueError("model files changed")
    return settings, manifest


def progress(output, **value):
    data = {"pid": os.getpid(), "unix_time": time.time(), **value}
    temporary = output / "progress.partial.json"
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(temporary, output / "progress.json")
    print(json.dumps(data), flush=True)


def run(args):
    settings, prepared = verify(args.output)
    if any((args.output / name).exists() for name in ("results", "summary.json", "manifest.json")):
        raise FileExistsError("fresh execution results required; do not silently resume partial predictions")
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 20 * 1024**3:
        raise RuntimeError("exclusive GPU with 20GiB free required")
    tokenizer = AutoTokenizer.from_pretrained(settings["model_path"], local_files_only=True)
    requests = [(name, json.loads((args.output / name).read_text())) for name in prepared["request_order"]]
    for _, request in requests:
        if tokenizer(request["rendered_prompt"], add_special_tokens=False)["input_ids"] != request["input_ids"]:
            raise ValueError("saved tokenization differs before model load")
    close = tokenizer.encode("</think>", add_special_tokens=False)
    if len(close) != 1:
        raise ValueError("thinking close must be exactly one token")
    progress(args.output, status="loading_model", total=len(requests))
    model = AutoModelForCausalLM.from_pretrained(settings["model_path"], local_files_only=True,
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa").eval()
    model.requires_grad_(False)
    config = GenerationConfig(do_sample=True, temperature=PROTOCOL["temperature"], top_p=PROTOCOL["top_p"],
        top_k=PROTOCOL["top_k"], num_beams=1, num_return_sequences=1, repetition_penalty=1.,
        eos_token_id=model.generation_config.eos_token_id, pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=PROTOCOL["max_new_tokens"], use_cache=True)
    artifacts, counts, model_calls, started = {}, Counter(), 0, time.monotonic()
    forward_counter = [0]
    handle = model.register_forward_pre_hook(lambda module, inputs: forward_counter.__setitem__(0, forward_counter[0] + 1))
    try:
        for index, (name, request) in enumerate(requests):
            begin, before = time.monotonic(), forward_counter[0]
            raw, final, generated = "", "", []
            status = request["context_status"]
            if status == "available":
                torch.manual_seed(request["seed"])
                ids = torch.tensor([request["input_ids"]], device=model.device)
                with torch.inference_mode():
                    output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=config)
                generated = output[0, len(request["input_ids"]):].cpu().tolist()
                raw = tokenizer.decode(generated, skip_special_tokens=False)
                ending = config.eos_token_id if isinstance(config.eos_token_id, list) else [config.eos_token_id]
                if not generated or generated[-1] not in ending:
                    status = "generation_limit"
                elif close[0] not in generated:
                    status = "missing_thinking_close"
                else:
                    # Use first close: a second close in the final answer is invalid
                    # content, not permission to discard an earlier final answer.
                    cut = generated.index(close[0]) + 1
                    final = tokenizer.decode(generated[cut:], skip_special_tokens=True)
                    status = "complete"
                model_calls += 1
                del output, ids
            prediction = compile_prediction(request, final, generation_status=status)
            record = {"packet_file": name, "packet_file_sha256": prepared["artifacts"][name],
                "settings_sha256": prepared["settings_sha256"], "mode": request["mode"],
                "response_id": request["response_id"], "seed": request["seed"], "generation_config": config.to_dict(),
                "generated_ids": generated, "raw_output": raw, "final_output": final, "generation_status": status,
                "prediction": prediction, "seconds": time.monotonic() - begin,
                "actual_forwards": forward_counter[0] - before, "labels_read": False}
            destination = f"results/{request['response_id']}_{request['mode']}.json"
            write_json_once(args.output / destination, record)
            artifacts[destination] = file_sha256(args.output / destination)
            for key, value in prediction["counts"].items():
                counts[request["mode"] + ":" + key] += value
            counts[request["mode"] + ":status:" + status] += 1
            counts[request["mode"] + ":requests"] += 1
            counts[request["mode"] + ":failures"] += len(prediction["failures"])
            progress(args.output, status="running", done=index + 1, total=len(requests), current=destination,
                last_generation_status=status, last_facts=len(prediction["facts"]), actual_forwards=forward_counter[0],
                seconds=time.monotonic() - started)
    finally:
        handle.remove()
    verify(args.output)
    summary = {"status": "complete", "requests": len(requests), "generation_calls": model_calls,
        "actual_forwards": forward_counter[0], "counts": dict(counts), "seconds": time.monotonic() - started,
        "cuda_peak_bytes": torch.cuda.max_memory_allocated(), "labels_read": False, "native_forwards": 0,
        "scope": "paired development semantic predictions; not ground truth or validated method"}
    write_json_once(args.output / "summary.json", summary)
    artifacts["summary.json"] = file_sha256(args.output / "summary.json")
    write_json_once(args.output / "manifest.json", {"status": "complete", "artifacts": artifacts,
        "prepare_manifest_sha256": file_sha256(args.output / "prepare_manifest.json"),
        "settings_sha256": prepared["settings_sha256"]})
    progress(args.output, **summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        if any(getattr(args, key) is None for key in ("inputs", "inventory", "model")):
            parser.error("prepare requires --inputs --inventory --model")
        prepare(args)
    else:
        # Reuse the existing project-wide GPU lock, even with population complete.
        lock_path = ROOT.parent / "reanchor/runs/relation_interleave_20260913.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args)
