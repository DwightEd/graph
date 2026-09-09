import json

import numpy as np
import torch

from experiments.reanchor_flow.message_dag.counterfactual_capture import (
    select_constraint_events,
)
from experiments.reanchor_flow.message_dag.counterfactual_pairs import (
    CounterfactualPair,
)
from experiments.reanchor_flow.message_dag.counterfactual_study import (
    CounterfactualConfig,
    CounterfactualStudy,
)


class WordTokenizer:
    def __init__(self):
        self.all_special_ids = [101, 102]
        self.added_tokens_decoder = {}
        self.vocabulary = {}

    def _encode(self, text):
        return [
            self.vocabulary.setdefault(token, len(self.vocabulary) + 3)
            for token in text.strip().split()
        ]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [101] + self._encode(messages[0]["content"]) + [102]

    def encode(self, text, add_special_tokens=False):
        return self._encode(text)

    def decode(self, token_ids):
        return f"<{int(token_ids[0])}>"


def pair_record():
    return {
        "pair_id": "entity-001",
        "constraint_kind": "entity",
        "prompt_plus": "Facts Alice owns red. Question owner Alice",
        "prompt_minus": "Facts Bob owns blue. Question owner Alice",
        "constraint_plus": "Alice owns red.",
        "constraint_minus": "Bob owns blue.",
        "response_prefix": "Based on these facts the requested answer color is now",
        "candidate_plus": "red",
        "candidate_minus": "blue",
        "reviewed": True,
    }


def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(101)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=256,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    ).eval()


def save_cut(path, rows, plus):
    signed = np.zeros((2, 4, len(rows) - 1, 2), np.float32)
    values = np.zeros((4, 1, 1, 2), np.float32)
    values[1, 0, 0, 0] = 3 if plus else 1
    signed[0, :, 2] = values[:, 0].sum(1)
    np.savez_compressed(
        path,
        cut_schema=np.array(1),
        branch_names=np.array(["V_content", "K_routing"]),
        row_position=rows,
        event_row=np.array(1),
        cut_signed=signed,
        cut_closure_error=np.zeros(len(rows) - 1),
        L0Q2=values,
    )


def test_constraint_event_plan_uses_union_then_freezes_budget():
    shape = (2, 2, 4)
    plus = {
        "event_index": np.array([[0, 0, 1], [0, 1, 3]]),
        "peak_source": np.full(shape, -1),
        "remote_gain": np.zeros(shape),
        "row_position": np.array([5, 6, 7, 8]),
    }
    minus = {
        "event_index": np.array([[1, 0, 2], [0, 1, 3]]),
        "peak_source": np.full(shape, -1),
        "remote_gain": np.zeros(shape),
        "row_position": np.array([5, 6, 7, 8]),
    }
    plus["peak_source"][0, 0, 1] = 2
    plus["peak_source"][0, 1, 3] = 2
    minus["peak_source"][1, 0, 2] = 2
    minus["peak_source"][0, 1, 3] = 4
    plus["remote_gain"][0, 0, 1] = 0.2
    plus["remote_gain"][0, 1, 3] = 0.7
    minus["remote_gain"][1, 0, 2] = 0.5

    plan = select_constraint_events(plus, minus, changed_positions=(2,), row_budget=1)

    assert plan["candidate_rows"] == [1, 2, 3]
    assert plan["selected_rows"] == [3]
    assert plan["selected_sites"] == [[0, 1, 3]]
    assert plan["rows_dropped_by_budget"] == 2
    assert plan["labels_used"] is False


def test_study_prepares_screens_witnesses_and_reports_through_public_seam(
    tmp_path, monkeypatch
):
    import transformers

    records = tmp_path / "pairs.jsonl"
    records.write_text(json.dumps(pair_record()) + "\n", encoding="utf-8")
    output = tmp_path / "study"
    model_path = tmp_path / "model"
    tokenizer = WordTokenizer()
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer
    )
    config = CounterfactualConfig(
        pairs=records,
        model=model_path,
        output=output,
        device="cpu",
        dtype="float32",
        event_row_budget=1,
        carrier_budget=1,
        doses=(0.0, 1.0),
    )
    study = CounterfactualStudy(config)

    prepared = study.run("prepare")
    pair = CounterfactualPair.from_manifest(prepared["pairs"][0])
    folder = output / "pairs" / pair.pair_id
    rows = np.arange(pair.response_start - 1, pair.token_ids.shape[1])
    event_position = int(rows[1])
    (folder / "event_plan.json").write_text(
        json.dumps(
            {
                "pair_id": pair.pair_id,
                "pair_identity": pair.study_identity,
                "candidate_rows": [1],
                "selected_rows": [1],
                "selected_sites": [[0, 0, 1]],
                "event_positions": [event_position],
                "rows_dropped_by_budget": 0,
                "labels_used": False,
            }
        ),
        encoding="utf-8",
    )
    save_cut(folder / f"plus_edges_{event_position}.npz", rows, plus=True)
    save_cut(folder / f"minus_edges_{event_position}.npz", rows, plus=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("offline screen cannot load a model")

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", forbidden)
    screen = study.run("screen")
    assert screen["pairs"][0]["status"] == "selected"
    assert screen["pairs"][0]["carriers"][0]["position"] == int(rows[2])

    model = tiny_model()
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: model,
    )
    witness = study.run("witness")
    assert witness["completed_witnesses"] == 1

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", forbidden)
    resumed = study.run("witness")
    assert resumed["completed_witnesses"] == 1

    summary = study.run("evaluate")
    assert summary["pairs"] == 1
    assert summary["selected_pairs"] == 1
    assert summary["completed_witnesses"] == 1
    assert summary["finite_witness_forward_calls"]["total"] == 14
    assert summary["labels_used"] is False
    from experiments.reanchor_flow.message_dag.evaluation import OfflineEvaluator

    assert OfflineEvaluator(output, bootstrap=0).run()["pairs"] == 1


def test_native_capture_and_trace_run_through_public_study_seam(tmp_path, monkeypatch):
    import transformers

    from experiments.reanchor_flow.message_dag import counterfactual_capture

    records = tmp_path / "pairs.jsonl"
    records.write_text(json.dumps(pair_record()) + "\n", encoding="utf-8")
    model_path = tmp_path / "model"
    tiny_model().save_pretrained(model_path, safe_serialization=True)
    tokenizer = WordTokenizer()
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer
    )
    study = CounterfactualStudy(
        CounterfactualConfig(
            pairs=records,
            model=model_path,
            output=tmp_path / "study",
            device="cpu",
            dtype="float32",
            query_chunk=2,
            event_row_budget=1,
            doses=(0.0, 1.0),
        )
    )
    manifest = study.run("prepare")
    pair = CounterfactualPair.from_manifest(manifest["pairs"][0])

    capture = study.run("capture")
    assert capture == {"planned_worlds": 2, "captured_worlds": 2}
    folder = study.config.output / "pairs" / pair.pair_id
    for world in ("plus", "minus"):
        for suffix in (".npz", ".history.npz", ".qk.npz", ".states.npz"):
            assert (folder / f"{world}{suffix}").exists()

    def deterministic_scan(path, config, *, chunk, device):
        with np.load(path, allow_pickle=False) as stored:
            rows = stored["row_position"]
        shape = (2, 4, len(rows))
        peak_source = np.full(shape, -1, dtype=np.int32)
        remote_gain = np.zeros(shape, dtype=np.float32)
        peak_source[0, 0, 1] = pair.changed_positions[0]
        remote_gain[0, 0, 1] = 0.5
        return {
            "event_index": np.array([[0, 0, 1]], dtype=np.int32),
            "peak_source": peak_source,
            "remote_gain": remote_gain,
            "row_position": rows,
        }

    monkeypatch.setattr(counterfactual_capture, "scan", deterministic_scan)
    traced = study.run("trace")

    assert traced == {"pairs": 1, "event_pairs": 1, "traced_cuts": 2}
    plan = json.loads((folder / "event_plan.json").read_text(encoding="utf-8"))
    assert plan["selected_rows"] == [1]
    event_position = plan["event_positions"][0]
    assert (folder / f"plus_edges_{event_position}.npz").exists()
    assert (folder / f"minus_edges_{event_position}.npz").exists()


def test_counterfactual_cli_maps_external_parameters_to_typed_config(
    tmp_path, monkeypatch
):
    from experiments.reanchor_flow.message_dag import counterfactual_run

    observed = {}

    def fake_run(self, phase):
        observed.update(config=self.config, phase=phase)
        return "complete"

    monkeypatch.setattr(CounterfactualStudy, "run", fake_run)
    args = counterfactual_run.parser().parse_args(
        [
            "--pairs",
            str(tmp_path / "pairs.jsonl"),
            "--model",
            str(tmp_path / "model"),
            "--output",
            str(tmp_path / "study"),
            "--phase",
            "screen",
            "--doses",
            "0,0.25,1",
        ]
    )

    assert counterfactual_run.run(args) == "complete"
    assert observed["phase"] == "screen"
    assert observed["config"].doses == (0.0, 0.25, 1.0)
