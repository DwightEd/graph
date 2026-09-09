"""Linear, resumable orchestration for the counterfactual mediation study."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .artifacts import (
    COUNTERFACTUAL_MEDIATION,
    artifact_header,
    atomic_path,
    output_writer,
)
from .counterfactual_pairs import CounterfactualPair
from .events import EventConfig
from .message_exchange import Carrier, MessageExchangeWitness, WitnessConfig
from .paired_transport import PairedRouteScreen


@dataclass(frozen=True)
class CounterfactualConfig:
    pairs: Path
    model: Path
    output: Path
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    query_chunk: int = 8
    window: int = 10
    gain: float = 0.10
    local_floor: float = 0.50
    event_row_budget: int = 2
    carrier_budget: int = 1
    doses: tuple[float, ...] = (0.0, 0.5, 1.0)

    def check(self):
        EventConfig(self.window, self.gain, self.local_floor)
        WitnessConfig(self.doses, self.query_chunk).check()
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("unsupported model dtype")
        if self.event_row_budget < 0 or self.carrier_budget < 1:
            raise ValueError("event budget must be nonnegative and carrier budget positive")
        return self

    def scientific_settings(self):
        return {
            "model": str(self.model),
            "event": asdict(EventConfig(self.window, self.gain, self.local_floor)),
            "event_row_budget": self.event_row_budget,
            "carrier_budget": self.carrier_budget,
            "doses": list(self.doses),
            "readout": "candidate_plus_minus_candidate_minus_first_token",
            "finite_witness": "bidirectional_post_wo_head_message_exchange",
        }


def _write_json(path, value):
    with atomic_path(Path(path)) as temporary:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _write_npz(path, **values):
    with atomic_path(Path(path)) as temporary:
        np.savez_compressed(temporary, **values)


class CounterfactualStudy:
    """Expose one phase-oriented interface while hiding artifact bookkeeping."""

    PHASES = ("prepare", "capture", "trace", "screen", "witness", "evaluate", "all")

    def __init__(self, config):
        self.config = config.check()

    def run(self, phase="all"):
        if phase not in self.PHASES:
            raise ValueError(f"unknown counterfactual phase: {phase}")
        if phase == "prepare":
            with output_writer(self.config.output, ".counterfactual.lock"):
                return self._prepare()
        if phase == "screen":
            with output_writer(self.config.output, ".counterfactual.lock"):
                return self._screen()
        if phase == "witness":
            with output_writer(self.config.output, ".counterfactual.lock"):
                return self._witness()
        if phase == "evaluate":
            from .counterfactual_report import evaluate

            return evaluate(self.config.output)
        if phase in {"capture", "trace", "all"}:
            return self._run_native(phase)
        raise AssertionError(phase)

    def _prepare(self):
        from transformers import AutoTokenizer

        records = [
            json.loads(line)
            for line in self.config.pairs.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records:
            raise ValueError("counterfactual pair file is empty")
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.config.model), local_files_only=True
        )
        pairs = [CounterfactualPair.compile(record, tokenizer) for record in records]
        identities = [pair.pair_id for pair in pairs]
        if len(set(identities)) != len(identities):
            raise ValueError("counterfactual pair IDs must be unique")
        manifest = {
            **artifact_header(COUNTERFACTUAL_MEDIATION, 1),
            "study_settings": self.config.scientific_settings(),
            "pairs": [pair.to_manifest() for pair in pairs],
            "labels_used": False,
        }
        previous_path = self.config.output / "index.json"
        if previous_path.exists():
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            if previous != manifest:
                raise ValueError("counterfactual inputs changed; use a new output directory")
            return previous
        self.config.output.mkdir(parents=True, exist_ok=True)
        for pair in pairs:
            (self.config.output / "pairs" / pair.pair_id).mkdir(parents=True, exist_ok=True)
        _write_json(previous_path, manifest)
        return manifest

    def _manifest(self):
        path = self.config.output / "index.json"
        if not path.exists():
            raise FileNotFoundError("run the counterfactual prepare phase first")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = artifact_header(COUNTERFACTUAL_MEDIATION, 1)
        if any(manifest.get(name) != value for name, value in expected.items()):
            raise ValueError("output is not a compatible counterfactual mediation study")
        if manifest.get("study_settings") != self.config.scientific_settings():
            raise ValueError("counterfactual scientific settings changed")
        return manifest

    def _screen(self):
        manifest = self._manifest()
        entries = []
        comparator = PairedRouteScreen()
        for frozen in manifest["pairs"]:
            pair = CounterfactualPair.from_manifest(frozen)
            folder = self.config.output / "pairs" / pair.pair_id
            plan_path = folder / "event_plan.json"
            if not plan_path.exists():
                raise FileNotFoundError(f"{pair.pair_id}: run trace before screen")
            event_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            if event_plan.get("pair_identity") != pair.study_identity:
                raise ValueError(f"{pair.pair_id}: event plan identity changed")
            positions = event_plan.get("event_positions", [])
            event_screens, candidates = [], []
            for event_position in positions:
                plus = folder / f"plus_edges_{event_position}.npz"
                minus = folder / f"minus_edges_{event_position}.npz"
                if not plus.exists() or not minus.exists():
                    raise FileNotFoundError(
                        f"{pair.pair_id}: paired cut for event {event_position} is incomplete"
                    )
                result = comparator.compare(plus, minus)
                if int(result["row_position"][result["event_row"]]) != event_position:
                    raise ValueError("event plan and paired cut name different positions")
                destination = folder / f"screen_{event_position}.npz"
                _write_npz(destination, pair_identity=np.array(pair.study_identity), **result)
                event_screens.append(
                    {"event_position": event_position, "status": result["status"]}
                )
                if result["status"] != "eligible":
                    continue
                layer, head, query = map(int, result["carrier_index"])
                local = float(result["selective_signed"].sum(-1)[layer, head, query])
                candidates.append(
                    {
                        "event_position": event_position,
                        "layer": layer,
                        "head": head,
                        "row": query,
                        "position": int(result["row_position"][query]),
                        "local_selective": local,
                        "matched_fraction": float(result["matched_fraction"]),
                    }
                )
            candidates.sort(
                key=lambda item: (
                    -abs(item["local_selective"]),
                    -item["matched_fraction"],
                    item["position"],
                    item["layer"],
                    item["head"],
                )
            )
            carriers = candidates[: self.config.carrier_budget]
            if carriers:
                status = "selected"
            elif not positions:
                status = "no_event"
            elif any(item["status"] == "no_edge" for item in event_screens):
                status = "no_edge"
            else:
                status = "no_response"
            entries.append(
                {
                    "pair_id": pair.pair_id,
                    "pair_identity": pair.study_identity,
                    "status": status,
                    "candidate_event_rows": len(event_plan.get("candidate_rows", [])),
                    "selected_event_rows": len(positions),
                    "rows_dropped_by_budget": event_plan.get("rows_dropped_by_budget", 0),
                    "event_screens": event_screens,
                    "eligible_carriers": len(candidates),
                    "carriers": carriers,
                }
            )
        plan = {
            "screen_schema": 1,
            "method": "counterfactual_last_crossing_mediation_screen",
            "study_settings": manifest["study_settings"],
            "pairs": entries,
            "labels_used": False,
        }
        path = self.config.output / "screen_plan.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
            raise ValueError("frozen counterfactual screen plan changed")
        _write_json(path, plan)
        return plan

    def _witness(self):
        manifest = self._manifest()
        screen_path = self.config.output / "screen_plan.json"
        if not screen_path.exists():
            raise FileNotFoundError("run the counterfactual screen phase first")
        screen = json.loads(screen_path.read_text(encoding="utf-8"))
        frozen = {
            pair.pair_id: pair
            for pair in map(CounterfactualPair.from_manifest, manifest["pairs"])
        }
        planned = [
            (entry, ordinal, carrier)
            for entry in screen["pairs"]
            for ordinal, carrier in enumerate(entry["carriers"])
        ]
        pending = []
        completed = []
        for entry, ordinal, carrier in planned:
            pair = frozen[entry["pair_id"]]
            destination = (
                self.config.output
                / "pairs"
                / pair.pair_id
                / f"witness_{ordinal:02d}.npz"
            )
            if self._witness_complete(destination, pair, carrier):
                completed.append(
                    {
                        "pair_id": pair.pair_id,
                        "carrier": ordinal,
                        "path": str(destination.relative_to(self.config.output)),
                    }
                )
            else:
                pending.append((pair, ordinal, carrier, destination))
        model = self._load_model() if pending else None
        runner = (
            MessageExchangeWitness(
                model, WitnessConfig(self.config.doses, self.config.query_chunk)
            )
            if model is not None
            else None
        )
        for pair, ordinal, carrier, destination in pending:
            result = runner.run(
                pair,
                Carrier(carrier["layer"], carrier["head"], carrier["position"]),
            )
            _write_npz(
                destination,
                pair_identity=np.array(pair.study_identity),
                local_selective=np.array(carrier["local_selective"]),
                event_position=np.array(carrier["event_position"]),
                **result,
            )
            completed.append(
                {
                    "pair_id": pair.pair_id,
                    "carrier": ordinal,
                    "path": str(destination.relative_to(self.config.output)),
                }
            )
        completed.sort(key=lambda item: (item["pair_id"], item["carrier"]))
        result = {
            "witness_schema": 1,
            "planned_witnesses": len(planned),
            "completed_witnesses": len(completed),
            "witnesses": completed,
            "labels_used": False,
        }
        _write_json(self.config.output / "witness_plan.json", result)
        return result

    def _witness_complete(self, path, pair, carrier):
        if not path.exists():
            return False
        with np.load(path, allow_pickle=False) as stored:
            expected_carrier = np.asarray(
                [carrier["layer"], carrier["head"], carrier["position"]]
            )
            valid = (
                int(stored["witness_schema"]) == 1
                and str(stored["pair_identity"]) == pair.study_identity
                and str(stored["intervention_kind"])
                == "post_wo_head_message_exchange"
                and np.array_equal(stored["carrier"], expected_carrier)
                and np.array_equal(stored["doses"], np.asarray(self.config.doses))
                and int(stored["event_position"]) == carrier["event_position"]
                and float(stored["local_selective"])
                == carrier["local_selective"]
                and not bool(stored["labels_used"])
            )
        if not valid:
            raise ValueError(f"{path}: frozen counterfactual witness identity changed")
        return True

    def _load_model(self):
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            str(self.config.model),
            local_files_only=True,
            torch_dtype=getattr(torch, self.config.dtype),
            attn_implementation="eager",
        ).to(self.config.device).eval()
        if model.config.model_type != "llama":
            raise ValueError("counterfactual message exchange is validated only for Llama")
        return model

    def _run_native(self, phase):
        from .counterfactual_capture import NativePairTracer

        with output_writer(self.config.output, ".counterfactual.lock"):
            manifest = self._prepare() if phase == "all" else self._manifest()
            tracer = NativePairTracer(self.config)
            if phase == "capture":
                return tracer.capture(manifest)
            if phase == "trace":
                return tracer.trace(manifest)

            tracer.capture(manifest)
            tracer.trace(manifest)
            self._screen()
            self._witness()
        from .counterfactual_report import evaluate

        return evaluate(self.config.output)
