"""Execute frozen source-only weak restatement requests on the local Qwen."""

import argparse
import fcntl
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from next_iteration.grounded_graph_data import (
    PROTOCOL,
    ROOT,
    compile_templates,
    inventory_parent,
)
from route_graph.audit_artifacts import file_sha256
from route_graph.audit_runner import model_manifest
from route_graph.frozen_reader import digest, write_json_once


def verify(output):
    prepared = json.loads((output / "prepare_manifest.json").read_text())
    if file_sha256(output / "settings.json") != prepared["settings_sha256"]:
        raise ValueError("weak synthesis settings changed")
    settings = json.loads((output / "settings.json").read_text())
    if settings["protocol"] != PROTOCOL:
        raise ValueError("weak source synthesis protocol changed")
    for name, sha in settings["code_sha256"].items():
        if file_sha256(ROOT / name) != sha or file_sha256(output / "executed_code" / name) != sha:
            raise ValueError("weak synthesis live/snapshot code changed")
    for name, sha in prepared["artifacts"].items():
        if file_sha256(output / name) != sha:
            raise ValueError("prepared source artifact changed")
    if file_sha256(settings["input_path"]) != settings["input_sha256"]:
        raise ValueError("source roster changed")
    inventory = Path(settings["inventory_path"])
    if file_sha256(inventory / "manifest.json") != settings["inventory_manifest_sha256"]:
        raise ValueError("inventory manifest changed")
    parent, inventory_settings = inventory_parent(inventory, settings["input_sha256"])
    if (parent["settings_file_sha256"] != settings["inventory_settings_sha256"]
            or digest(inventory_settings) != settings["inventory_settings_digest"]
            or inventory_settings["protocol"] != settings["inventory_protocol"]):
        raise ValueError("inventory settings/protocol binding changed")
    for name, sha in settings["source_artifacts"].items():
        if file_sha256(inventory / name) != sha:
            raise ValueError("selected inventory changed")
    if model_manifest(Path(settings["model_path"])) != settings["model_files"]:
        raise ValueError("source-only Qwen identity changed")
    return settings, prepared


def progress(output, **values):
    value = {"pid": os.getpid(), "unix_time": time.time(), **values}
    temporary = output / "progress.partial.json"
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, output / "progress.json")
    print(json.dumps(value), flush=True)


def decode_result(tokenizer, generated, eos_ids, close_id):
    padded = list(generated)
    stop = next((i for i, token in enumerate(padded) if token in eos_ids), None)
    effective = padded if stop is None else padded[:stop + 1]
    status, final = "generation_limit" if stop is None else "missing_thinking_close", ""
    if stop is not None and close_id in effective:
        cut = effective.index(close_id) + 1
        status, final = "complete", tokenizer.decode(effective[cut:], skip_special_tokens=True)
    return {"generation_status": status, "generated_padded_ids": padded, "generated_ids": effective,
        "raw_output": tokenizer.decode(effective, skip_special_tokens=False), "final_output": final}


def run(args):
    settings, prepared = verify(args.output)
    if any((args.output / n).exists() for n in ("results", "summary.json", "manifest.json", "execution.json")):
        raise FileExistsError("fresh weak synthesis execution required; preserve partial output")
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 20 * 1024**3:
        raise RuntimeError("exclusive GPU with20GiB free required")
    tokenizer = AutoTokenizer.from_pretrained(settings["model_path"], local_files_only=True, padding_side="left")
    close = tokenizer.encode("</think>", add_special_tokens=False)
    if len(close) != 1:
        raise ValueError("Qwen thinking close is not a single token")
    requests = [(name, json.loads((args.output / name).read_text())) for name in prepared["request_order"]]
    for _, request in requests:
        if tokenizer(request["rendered_prompt"], add_special_tokens=False)["input_ids"] != request["input_ids"]:
            raise ValueError("prepared source tokenization differs")
    progress(args.output, status="loading_model", total_sources=len(requests))
    model = AutoModelForCausalLM.from_pretrained(settings["model_path"], local_files_only=True,
        dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa").eval()
    model.requires_grad_(False)
    config = GenerationConfig(do_sample=True, temperature=PROTOCOL["temperature"], top_p=PROTOCOL["top_p"],
        top_k=PROTOCOL["top_k"], num_beams=1, num_return_sequences=1, repetition_penalty=1.,
        eos_token_id=model.generation_config.eos_token_id, pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=PROTOCOL["max_new_tokens"], use_cache=True)
    eos = config.eos_token_id if isinstance(config.eos_token_id, list) else [config.eos_token_id]
    execution = {"torch": str(torch.__version__), "transformers": transformers.__version__, "dtype": "bfloat16",
        "attention": "sdpa", "generation_config": config.to_dict(), "batch_size": PROTOCOL["batch_size"],
        "seeding": "fixed seed plus ordered request-name digest per batch", "labels_read": False}
    write_json_once(args.output / "execution.json", execution)
    counts, artifacts, calls = Counter(), {"execution.json": file_sha256(args.output / "execution.json")}, [0]
    hook = model.register_forward_pre_hook(lambda module, inputs: calls.__setitem__(0, calls[0] + 1))
    started, batch_count = time.monotonic(), 0
    try:
        for start in range(0, len(requests), PROTOCOL["batch_size"]):
            batch = requests[start:start + PROTOCOL["batch_size"]]
            active = [(n, r) for n, r in batch if r["status"] == "available"]
            decoded, before, begin = {}, calls[0], time.monotonic()
            seed = int(digest([PROTOCOL["seed"], [n for n, _ in batch]])[:8], 16)
            if active:
                torch.manual_seed(seed)
                width = max(len(r["input_ids"]) for _, r in active)
                input_rows = [[tokenizer.eos_token_id] * (width - len(r["input_ids"])) + r["input_ids"] for _, r in active]
                masks = [[0] * (width - len(r["input_ids"])) + [1] * len(r["input_ids"]) for _, r in active]
                ids = torch.tensor(input_rows, device=model.device)
                attention = torch.tensor(masks, device=model.device)
                with torch.inference_mode():
                    output = model.generate(input_ids=ids, attention_mask=attention, generation_config=config)
                generated = output[:, width:].cpu().tolist()
                for (name, _), tokens in zip(active, generated, strict=True):
                    decoded[name] = decode_result(tokenizer, tokens, eos, close[0])
                del output, ids, attention
                batch_count += 1
            for name, request in batch:
                result = decoded.get(name, {"generation_status": "context_or_target_unavailable",
                    "generated_padded_ids": [], "generated_ids": [], "raw_output": "", "final_output": ""})
                compiled = compile_templates(request, result["final_output"], result["generation_status"])
                record = {"request_file": name, "request_file_sha256": prepared["artifacts"][name],
                    "settings_sha256": prepared["settings_sha256"], "execution_sha256": file_sha256(args.output / "execution.json"),
                    "batch_request_names": [n for n, _ in batch], "batch_actual_forwards": calls[0] - before,
                    "batch_seed": seed, "batch_seconds": time.monotonic() - begin, **result, "compiled": compiled,
                    "labels_read": False, "semantic_ground_truth": False}
                destination = f"results/{request['source_id']}.json"
                write_json_once(args.output / destination, record)
                artifacts[destination] = file_sha256(args.output / destination)
                key = request["task"] + ":" + request["split"]
                counts[key + ":sources"] += 1
                counts[key + ":selected_targets"] += compiled["selected_target_count"]
                counts[key + ":usable_weak_examples"] += compiled["mechanically_usable"]
                counts[key + ":failures"] += len(compiled["failures"])
                counts["status:" + result["generation_status"]] += 1
            progress(args.output, status="running", done_sources=start + len(batch), total_sources=len(requests),
                actual_forwards=calls[0], generation_batches=batch_count, counts=dict(counts), seconds=time.monotonic() - started)
    finally:
        hook.remove()
    verify(args.output)
    summary = {"status": "complete", "sources": len(requests), "counts": dict(counts),
        "actual_batched_forwards": calls[0], "generation_batches": batch_count,
        "seconds": time.monotonic() - started, "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
        "labels_read": False, "semantic_ground_truth": False, "scope": "weak source-only reconstruction training data"}
    write_json_once(args.output / "summary.json", summary)
    artifacts["summary.json"] = file_sha256(args.output / "summary.json")
    write_json_once(args.output / "manifest.json", {"status": "complete", "artifacts": artifacts,
        "prepare_manifest_sha256": file_sha256(args.output / "prepare_manifest.json"), "settings_sha256": prepared["settings_sha256"]})
    progress(args.output, **summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (ROOT.parent / "reanchor/runs/relation_interleave_20260913.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
