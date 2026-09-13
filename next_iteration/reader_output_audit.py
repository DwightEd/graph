"""Bounded CPU replay of real reader requests with full-vocabulary readout.

This diagnoses the letter-output interface, not truth or hallucination accuracy.
CUDA is deliberately never selected; old GPU records remain immutable.
"""

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from route_graph.audit_artifacts import file_sha256
from route_graph.frozen_reader import digest, write_json_once


def select_requests(run, per_kind):
    """Freeze the smallest request digests per actual check name, no score filter."""
    by_kind = {}
    for path in sorted((run / "C").glob("*.json")):
        row = json.loads(path.read_text())["data"]
        for attempt in row["attempts"]:
            for name, check in attempt.get("verification", {}).get("checks", {}).items():
                receipt = check["reader_record"]
                by_kind.setdefault(name, {})[Path(receipt["path"]).stem] = receipt
    if set(by_kind) != {"target_original", "edited_base", "selected_occurrence", "preservation", "edit_kind"}:
        raise ValueError("require complete five-check source run")
    return [{"kind": name, "request_digest": key, "receipt": candidates[key]}
            for name, candidates in sorted(by_kind.items()) for key in sorted(candidates)[:per_kind]]


def run_audit(args):
    if args.per_kind < 1 or args.per_kind > 2 or args.max_new_tokens < 1 or args.max_new_tokens > 64:
        raise ValueError("bounded diagnostic: 1-2 per kind, 1-64 generated tokens")
    if args.output.exists():
        raise FileExistsError("use fresh audit output")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    selected = select_requests(args.run, args.per_kind)
    settings = json.loads((args.run / "settings.json").read_text())
    model_path = Path(settings["reader_model"])
    for entry in settings["reader_files"]:
        if file_sha256(model_path / entry["name"]) != entry["sha256"]:
            raise ValueError("frozen reader model files changed")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    frozen = {"schema": "reader-letter-output-audit@1", "upstream_settings_sha256": file_sha256(args.run / "settings.json"),
        "code_sha256": file_sha256(__file__), "selected": selected,
        "selection": "smallest distinct immutable request digest per check; no probability or gold selection",
        "model_path": str(model_path), "model_files": settings["reader_files"],
        "device": "cpu", "dtype": "bfloat16", "attention": "sdpa", "torch": str(torch.__version__),
        "threads": 4, "max_new_tokens": args.max_new_tokens, "labels_read": False,
        "numerical_scope": "CPU replay; not bitwise reproduction of CUDA; logits_to_keep=1",
        "purpose": "unconditional label mass, natural continuation and cached conditional scores"}
    args.output.mkdir(parents=True)
    write_json_once(args.output / "settings.json", frozen)
    print(json.dumps({"status": "loading_cpu_model", "requests": len(selected), "pid": os.getpid()}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa").to("cpu").eval()
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("CPU audit must not use GPU")
    calls = [0]
    def count_forward(module, args, kwargs):
        calls[0] += 1
    hook = model.register_forward_pre_hook(count_forward, with_kwargs=True)
    results = []
    try:
        for index, item in enumerate(selected):
            receipt = item["receipt"]
            path = Path(receipt["path"])
            if file_sha256(path) != receipt["sha256"]:
                raise ValueError("old reader receipt changed")
            saved = json.loads(path.read_text()); request = saved["request"]
            if digest(request) != item["request_digest"]:
                raise ValueError("request identity mismatch")
            rendered = tokenizer.apply_chat_template([
                {"role": "system", "content": request["instruction"]},
                {"role": "user", "content": json.dumps(request["payload"], ensure_ascii=False)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if hashlib.sha256(rendered.encode()).hexdigest() != request["rendered_prompt_sha256"]:
                raise ValueError("rendered prompt differs from exact old request")
            inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
            tokens = [tokenizer.encode(k, add_special_tokens=False) for k in request["labels"]]
            if any(len(t) != 1 for t in tokens):
                raise ValueError("letter tokenization changed")
            start, before = time.monotonic(), calls[0]
            with torch.inference_mode():
                logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                log_probs = logits.log_softmax(-1)
                ids = [t[0] for t in tokens]
                top = logits.topk(12).indices.tolist()
                generated = model.generate(**inputs, generation_config=GenerationConfig.from_dict(request["generation_config"]),
                    do_sample=False, max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.eos_token_id)
            result = {"kind": item["kind"], "receipt": receipt, "request_digest": item["request_digest"],
                "input_tokens": inputs["input_ids"].shape[1], "label_tokens": dict(zip(request["labels"], ids, strict=True)),
                "cached_gpu_prediction": saved["prediction"], "cpu_selected_logits": logits[ids].tolist(),
                "cpu_conditional_probabilities": logits[ids].softmax(-1).tolist(),
                "cpu_unconditional_label_probabilities": log_probs[ids].exp().tolist(),
                "cpu_total_label_mass": log_probs[ids].exp().sum().item(),
                "top_tokens": [{"id": t, "text": tokenizer.decode([t]), "probability": log_probs[t].exp().item()} for t in top],
                "greedy_continuation": tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=False),
                "actual_model_forwards": calls[0] - before, "seconds": time.monotonic() - start,
                "labels_read": False, "native_interventions": 0, "settings_sha256": digest(frozen)}
            write_json_once(args.output / f"{index:02d}.json", result)
            results.append(result)
            print(json.dumps({"done": index + 1, "kind": result["kind"], "label_mass": result["cpu_total_label_mass"],
                "continuation": result["greedy_continuation"], "seconds": result["seconds"]}), flush=True)
        if file_sha256(__file__) != frozen["code_sha256"]:
            raise ValueError("audit code changed during execution")
        write_json_once(args.output / "summary.json", {"status": "complete", "requests": len(results),
            "actual_model_forwards": calls[0], "settings_sha256": digest(frozen), "labels_read": False,
            "native_interventions": 0, "evaluation": "reader-interface diagnostic, no truth metrics"})
    finally:
        hook.remove()
        del model
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-kind", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    run_audit(parser.parse_args())
