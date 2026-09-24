"""Two prompt prefills, native likelihood branches, compact physical-head routing."""

from time import perf_counter

import numpy as np
import torch
from tqdm import tqdm
from state_audit.attribution import _projection_gram
from state_audit.forward_ledger import append_value_energy
from state_audit.model.replay import attention_backend, prefill_cache
from state_audit.native_trace import native_hooks
from state_audit.storage import read_arrays, read_json, write_arrays, write_json

from ..evidence_contrast.capture import collect_condition, score_continuation


def route_block(model, records, cache, grams, energies, prompt, source_mask, start):
    """Reduce messages only after retaining their own physical head's W_O and V."""
    layers = []
    for layer, record in records.items():
        heads, kv_heads, _ = model.head_layout(layer)
        values = cache.layers[layer].values[0].float().repeat_interleave(heads // kv_heads, dim=0)
        energies[layer] = append_value_energy(values, grams[layer], energies.get(layer))
        attention = record["attention"].float().transpose(0, 1)
        magnitude = attention * energies[layer].sqrt()
        count = attention.shape[-1]
        source = torch.zeros(count, device=attention.device)
        source[:prompt] = torch.as_tensor(source_mask, device=attention.device)
        history = (torch.arange(count, device=attention.device) >= prompt).float()
        norm_source, norm_history = magnitude @ source, magnitude @ history
        mass_source, mass_history = attention @ source, attention @ history
        if start == 0 and source_mask[-1]:
            norm_source[0] -= magnitude[0, :, prompt - 1]
            mass_source[0] -= attention[0, :, prompt - 1]
        layers.append(dict(norm_source=norm_source, norm_history=norm_history,
            norm_total=magnitude.sum(-1), mass_source=mass_source, mass_history=mass_history))
    return {name: torch.stack([row[name] for row in layers], 1).cpu().numpy() for name in layers[0]}


@torch.no_grad()
def source_full(model, cache, prefix, answer, source_mask, grams, query_chunk):
    values, routes, entropies = [], [], []
    queries, energies = [prefix[-1], *answer[:-1]], {}
    with attention_backend(model, "eager"):
        for start in range(0, len(answer), query_chunk):
            stop = min(start + query_chunk, len(answer))
            with native_hooks(model, with_grad=False, representations=("attention",)) as records:
                hidden = model.native.model(input_ids=model.input_ids(queries[start:stop]),
                    past_key_values=cache, use_cache=True, return_dict=True).last_hidden_state[0]
            logp = model.native.lm_head(hidden).float().log_softmax(-1)
            ids = model.input_ids(answer[start:stop])[0]
            values.append(logp.gather(-1, ids[:, None])[:, 0].cpu().numpy())
            entropies.append(-(logp.exp() * logp).sum(-1).cpu().numpy())
            routes.append(route_block(model, records, cache, grams, energies, len(prefix), source_mask, start))
    heads = {name: np.concatenate([row[name] for row in routes]) for name in routes[0]}
    numerator = (heads["norm_history"] - heads["norm_source"]).sum(-1)
    denominator = np.maximum(heads["norm_total"].sum(-1), 1e-12)
    scores = dict(full=np.concatenate(values), entropy=np.concatenate(entropies),
        raw_route=(numerator / denominator).mean(-1),
        raw_attention=(heads["mass_history"] - heads["mass_source"]).mean((1, 2)))
    return scores, heads


@torch.no_grad()
def collect_with_source(model, source, response, grams, prefill_chunk, query_chunk):
    prefix, answer, units = source["prompt_with_source"], response["answer_ids"], response["units"]
    if len(prefix) + len(answer) - 1 > model.native.config.max_position_embeddings:
        raise ValueError("Original input exceeds model context; no token truncation is allowed")
    cache = prefill_cache(model, prefix[:-1], prefill_chunk)
    scores, heads = source_full(model, cache, prefix, answer, source["source_mask"], grams, query_chunk)
    local = np.empty_like(scores["full"])
    for unit in units:
        cache.crop(len(prefix) - 1)
        start, stop = unit["start"], unit["stop"]
        local[start:stop] = score_continuation(model, cache, prefix[-1], answer[start:stop], query_chunk)
    return dict(**scores, local=local, token_id=np.asarray(answer)), heads


def capture(args, manifest):
    from state_audit.model import load_model
    torch.set_num_threads(args.cpu_threads)
    model, grams = None, None
    for record in tqdm(manifest["records"], desc="RAGTruth source/routing capture"):
        directory = args.output / record["directory"]
        if (directory / "capture_complete.json").exists():
            continue
        if model is None:
            model, _ = load_model(manifest["model"], device=args.device, dtype=args.dtype)
            grams = {layer: _projection_gram(model, layer) for layer in range(len(model.layers))}
        response = read_json(directory / "response.json")
        source = read_json(args.output / record["source_file"])
        started = perf_counter()
        if not (directory / "with_source_complete.json").exists():
            values, heads = collect_with_source(model, source, response, grams,
                args.prefill_chunk_size, args.query_chunk_size)
            write_arrays(directory / "with_source.npz", **values)
            if args.save_heads:
                write_arrays(directory / "routing_heads.npz", **heads)
            write_json(directory / "with_source_complete.json", dict(complete=True))
        if not (directory / "without_source_complete.json").exists():
            values = collect_condition(model, source["prompt_without_source"], response["answer_ids"],
                response["units"], args.prefill_chunk_size, args.query_chunk_size, record["id"])
            write_arrays(directory / "without_source.npz", **values)
            write_json(directory / "without_source_complete.json", dict(complete=True))
        present, absent = read_arrays(directory / "with_source.npz"), read_arrays(directory / "without_source.npz")
        observed = dict(token_id=present["token_id"],
            source_local=absent["local"].astype(float) - present["local"],
            source_full=absent["full"].astype(float) - present["full"],
            **{name: present[name] for name in ("raw_route", "raw_attention", "entropy")})
        write_arrays(directory / "observations.npz", **observed)
        write_json(directory / "capture_complete.json", dict(tokens=record["tokens"],
            seconds=perf_counter() - started, device=args.device, dtype=args.dtype,
            prefill_chunk_size=args.prefill_chunk_size, query_chunk_size=args.query_chunk_size))
