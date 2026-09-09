"""Native pair capture and constraint-linked lookback trace planning."""

import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .artifacts import atomic_path
from .counterfactual_pairs import CounterfactualPair
from .events import EventConfig, scan


def select_constraint_events(plus, minus, *, changed_positions, row_budget):
    """Freeze union lookbacks whose strongest remote gain names the edit span."""

    if row_budget < 0:
        raise ValueError("event row budget must be nonnegative")
    plus_rows = np.asarray(plus["row_position"], dtype=int)
    minus_rows = np.asarray(minus["row_position"], dtype=int)
    if not np.array_equal(plus_rows, minus_rows):
        raise ValueError("counterfactual scans must share row positions")
    changed = set(map(int, changed_positions))
    if not changed:
        raise ValueError("a constraint-linked event plan needs changed positions")
    event_sets = [
        {tuple(map(int, coordinate)) for coordinate in np.asarray(scan["event_index"]).reshape(-1, 3)}
        for scan in (plus, minus)
    ]
    union = sorted(event_sets[0] | event_sets[1])
    linked = []
    for coordinate in union:
        if any(int(scan["peak_source"][coordinate]) in changed for scan in (plus, minus)):
            linked.append(coordinate)
    candidate_rows = sorted({coordinate[2] for coordinate in linked})
    row_score = {}
    for row in candidate_rows:
        values = [
            float(scan["remote_gain"][coordinate])
            for scan in (plus, minus)
            for coordinate in linked
            if coordinate[2] == row and np.isfinite(scan["remote_gain"][coordinate])
        ]
        row_score[row] = max(values, default=float("-inf"))
    ranked = sorted(candidate_rows, key=lambda row: (-row_score[row], row))
    selected_rows = ranked if row_budget == 0 else ranked[:row_budget]
    selected = set(selected_rows)
    sites = [list(coordinate) for coordinate in linked if coordinate[2] in selected]
    return {
        "candidate_rows": candidate_rows,
        "selected_rows": selected_rows,
        "selected_sites": sites,
        "event_positions": [int(plus_rows[row]) for row in selected_rows],
        "rows_dropped_by_budget": len(candidate_rows) - len(selected_rows),
        "labels_used": False,
    }


def _write_npz(path, **values):
    with atomic_path(Path(path)) as temporary:
        np.savez_compressed(temporary, **values)


def _write_json(path, value):
    with atomic_path(Path(path)) as temporary:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class NativePairTracer:
    """Create standard audit captures, then trace one frozen union event plan."""

    def __init__(self, config):
        self.config = config

    def capture(self, manifest):
        pending = []
        for frozen in manifest["pairs"]:
            pair = CounterfactualPair.from_manifest(frozen)
            folder = self.config.output / "pairs" / pair.pair_id
            for world in ("plus", "minus"):
                if not self._capture_complete(folder / f"{world}.npz", pair, world):
                    pending.append((pair, world))
        if not pending:
            return {"planned_worlds": 2 * len(manifest["pairs"]), "captured_worlds": 0}

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..attention_audit import AuditConfig, capture_audit, special_token_mask

        tokenizer = AutoTokenizer.from_pretrained(
            str(self.config.model), local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(self.config.model),
            local_files_only=True,
            torch_dtype=getattr(torch, self.config.dtype),
            attn_implementation="eager",
        ).to(self.config.device).eval()
        if model.config.model_type != "llama":
            raise ValueError("counterfactual capture is validated only for Llama")
        audit_config = AuditConfig(
            local_window=self.config.window,
            top_k=8,
            query_chunk=self.config.query_chunk,
            save_states=True,
            full_attention=False,
        )
        for pair, world in pending:
            index = 0 if world == "plus" else 1
            ids = pair.token_ids[index]
            evidence = np.zeros(len(ids), dtype=bool)
            evidence[list(pair.changed_positions)] = True
            unit = np.full(len(ids), -1, dtype=np.int32)
            unit[list(pair.changed_positions)] = 0
            special = special_token_mask(tokenizer, ids)
            path = self.config.output / "pairs" / pair.pair_id / f"{world}.npz"
            trace = capture_audit(
                model,
                ids,
                pair.response_start,
                evidence,
                special,
                unit,
                path,
                audit_config,
            )
            trace.update(
                pair_identity=np.array(pair.study_identity),
                pair_id=np.array(pair.pair_id),
                counterfactual_world=np.array(world),
                token_text=np.asarray([tokenizer.decode([int(token)]) for token in ids]),
                labels_used=np.array(False),
            )
            _write_npz(path, **trace)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "planned_worlds": 2 * len(manifest["pairs"]),
            "captured_worlds": len(pending),
        }

    def trace(self, manifest):
        work = []
        event = EventConfig(self.config.window, self.config.gain, self.config.local_floor)
        for frozen in manifest["pairs"]:
            pair = CounterfactualPair.from_manifest(frozen)
            folder = self.config.output / "pairs" / pair.pair_id
            scans = {}
            for world in ("plus", "minus"):
                path = folder / f"{world}.npz"
                if not self._capture_complete(path, pair, world):
                    raise FileNotFoundError(f"{pair.pair_id}/{world}: capture is incomplete")
                scan_path = folder / f"{world}_scan.npz"
                if scan_path.exists():
                    with np.load(scan_path, allow_pickle=False) as stored:
                        scans[world] = dict(stored)
                else:
                    scans[world] = scan(
                        path,
                        event,
                        chunk=self.config.query_chunk,
                        device=self.config.device,
                    )
                    scans[world]["pair_identity"] = np.array(pair.study_identity)
                    _write_npz(scan_path, **scans[world])
                if str(scans[world]["pair_identity"]) != pair.study_identity:
                    raise ValueError(f"{pair.pair_id}/{world}: scan identity changed")
            plan = select_constraint_events(
                scans["plus"],
                scans["minus"],
                changed_positions=pair.changed_positions,
                row_budget=self.config.event_row_budget,
            )
            plan.update(pair_id=pair.pair_id, pair_identity=pair.study_identity)
            plan_path = folder / "event_plan.json"
            if plan_path.exists():
                previous = json.loads(plan_path.read_text(encoding="utf-8"))
                if previous != plan:
                    raise ValueError(f"{pair.pair_id}: frozen event plan changed")
            else:
                _write_json(plan_path, plan)
            sites = np.asarray(plan["selected_sites"], dtype=np.int32).reshape(-1, 3)
            if len(sites):
                work.append((pair, folder, plan, sites))

        if not work:
            return {"pairs": len(manifest["pairs"]), "event_pairs": 0, "traced_cuts": 0}
        from ..message_lineage import CheckpointWeights

        weights = CheckpointWeights(self.config.model, self.config.device)
        traced = 0
        for pair, folder, plan, sites in work:
            position_by_row = dict(zip(plan["selected_rows"], plan["event_positions"]))
            for world in ("plus", "minus"):
                capture_path = folder / f"{world}.npz"
                pending_rows = [
                    row
                    for row in plan["selected_rows"]
                    if not (
                        folder / f"{world}_edges_{int(position_by_row[row])}.npz"
                    ).exists()
                ]
                if not pending_rows:
                    continue
                world_sites = sites[np.isin(sites[:, 2], pending_rows)]
                traced += self._trace_world(
                    pair, world, capture_path, folder, world_sites, weights
                )
        return {"pairs": len(manifest["pairs"]), "event_pairs": len(work), "traced_cuts": traced}

    def _capture_complete(self, path, pair, world):
        required = (path, path.with_suffix(".history.npz"), path.with_suffix(".qk.npz"), path.with_suffix(".states.npz"))
        if not all(item.exists() for item in required):
            if path.exists():
                raise FileNotFoundError(f"{path}: required native companion is missing")
            return False
        with np.load(path, allow_pickle=False) as stored:
            if str(stored["pair_identity"]) != pair.study_identity or str(stored["counterfactual_world"]) != world:
                raise ValueError(f"{path}: counterfactual capture identity changed")
        return True

    def _trace_world(self, pair, world, capture_path, folder, sites, weights):
        from .cache import NativeCache
        from .cut_artifact import CutRecorder, prepare_local_readout
        from .event_trace import trace_events

        rows = np.unique(sites[:, 2])
        with NativeCache(capture_path, weights) as cache, ExitStack() as stack:
            contrasts = [
                {
                    "target": int(target),
                    "positive_id": int(pair.candidate_plus_ids[0]),
                    "negative_id": int(pair.candidate_minus_ids[0]),
                }
                for target in cache.trace["row_position"][:-1] + 1
            ]
            temporary = stack.enter_context(
                TemporaryDirectory(prefix=f"{world}_readout_", dir=folder)
            )
            readout_path = Path(temporary) / "readout.npz"
            prepare_local_readout(
                cache,
                readout_path,
                query_chunk=self.config.query_chunk,
                contrasts=contrasts,
            )
            readout = stack.enter_context(np.load(readout_path, allow_pickle=False))
            recorders = {}
            for row in rows:
                position = int(cache.trace["row_position"][row])
                recorder = CutRecorder(folder / f"{world}_edges_{position}.npz", cache, row)
                recorders[int(row)] = recorder
                stack.callback(recorder.close)
            results = trace_events(
                cache,
                sites,
                window=self.config.window,
                query_chunk=self.config.query_chunk,
                contrasts=contrasts,
                cut_readout=readout,
                cut_recorders=recorders,
                event_batch=len(rows),
            )
            for result in results:
                position = int(result["event_position"])
                _write_npz(
                    folder / f"{world}_event_{position}.npz",
                    pair_identity=np.array(pair.study_identity),
                    **result,
                )
        return len(results)
