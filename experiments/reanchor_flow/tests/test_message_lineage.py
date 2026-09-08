"""Native, causal and numerical counterexamples for material lineage."""
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.reanchor_flow.attention_audit import AuditConfig, capture_audit
from experiments.reanchor_flow.message_lineage import CheckpointWeights, norm_component, propagate, swiglu_component


def capture_fixture(tmp_path, dtype=torch.float32, ids=None):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(82)
    cfg = LlamaConfig(vocab_size=29, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2)
    model = LlamaForCausalLM(cfg).to(dtype).eval()
    model.config._attn_implementation = "eager"
    checkpoint = tmp_path / "model"
    model.save_pretrained(checkpoint)
    ids = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]) if ids is None else np.asarray(ids)
    n, start = len(ids), 5
    roots, special = np.isin(np.arange(n), [1, 3]), np.isin(np.arange(n), [0, 4, 8])
    units = np.where(roots, 0, -1)
    path = tmp_path / "sample.npz"
    trace = capture_audit(model, ids, start, roots, special, units, path, AuditConfig(query_chunk=3))
    trace.update(token_ids=ids, token_text=np.array([str(x) for x in ids]), sample_id=np.array("sample"),
                 settings=np.array(json.dumps({"local_window": 10, "dtype": str(dtype).split(".")[-1]})))
    np.savez_compressed(path, **trace)
    return path, trace, CheckpointWeights(checkpoint)


def test_swiglu_allocation_is_additive_signed_and_rule_dependent():
    torch.manual_seed(2)
    x, part = torch.randn(7, 5), torch.randn(7, 5)
    w = {"post_norm": torch.rand(5), "gate": torch.randn(9, 5),
         "up": torch.randn(9, 5), "down": torch.randn(5, 9)}
    z = norm_component(x, x, w["post_norm"], 1e-6)
    expected = F.linear(F.silu(F.linear(z, w["gate"])) * F.linear(z, w["up"]), w["down"])
    a, native = swiglu_component(part, x, w, 1e-6)
    b, _ = swiglu_component(x - part, x, w, 1e-6)
    np.testing.assert_allclose((a + b).numpy(), expected.numpy(), atol=8e-6, rtol=2e-6)
    np.testing.assert_allclose(native.numpy(), expected.numpy(), atol=2e-6)
    negative, _ = swiglu_component(-part, x, w, 1e-6)
    torch.testing.assert_close(negative, -a)
    alternate, _ = swiglu_component(part, x, w, 1e-6, "up")
    assert not torch.allclose(a, alternate)  # closure alone does not identify unique provenance


def test_same_scalar_route_can_have_different_transmitted_content():
    # Material arrives in e1. The reader accepts only e2. The carrier's MLP
    # transforms the material into the reader's subspace; A1*A2 stays one.
    x, material = torch.tensor([[1., 1.]]), torch.tensor([[1., 0.]])
    w = {"post_norm": torch.full((2,), (1 + 1e-6)**.5),
         "gate": torch.tensor([[0., 1.]]), "up": torch.tensor([[1., 0.]]),
         "down": torch.tensor([[0.], [1.]])}
    transformed, _ = swiglu_component(material, x, w, 1e-6)
    assert material[0, 1] == 0
    assert (material + transformed)[0, 1] == pytest.approx(.5 * float(F.silu(torch.tensor(1.))))
    w["down"] = torch.zeros_like(w["down"])
    blocked, _ = swiglu_component(material, x, w, 1e-6)
    assert (material + blocked)[0, 1] == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_replay_chunking_and_real_multihop(tmp_path, dtype):
    path, trace, weights = capture_fixture(tmp_path, dtype)
    a = propagate(path, weights, chunk=3, top_k=2)
    b = propagate(path, weights, chunk=8, top_k=2)
    for key in ("source_margin", "material_residual_margin", "material_history_margin", "score_material_deficit"):
        np.testing.assert_allclose(a[key], b[key], atol=1e-6, rtol=1e-4, equal_nan=True)
    np.testing.assert_allclose(a["source_margin"][..., :-1, :].sum(-1), trace["head_margin"][..., :-1],
                               atol=8e-4 if dtype == torch.bfloat16 else 2e-7, rtol=.02)
    assert np.max(a["replay_relative_error"]) < .05
    assert not a["material_history_margin"][0].any()  # no earlier writer layer
    assert np.abs(a["material_history_margin"][1:]).max() > 1e-8
    np.testing.assert_allclose(a["final_material_margin"] + a["final_remainder_margin"], trace["residual_margin"][-1], atol=1e-7)
    # Special tokens remain in the native computation but are never display hubs/roots.
    for kind in ("direct", "relay"):
        selected = a[f"top_{kind}_position"]
        assert not np.isin(selected[selected >= 0], [0, 4, 8]).any()
    selected = a["top_relay_position"]
    for q, row in enumerate(trace["row_position"]):
        assert ((selected[..., q, :] < row) | (selected[..., q, :] < 0)).all()


def test_root_additivity_legacy_runner_and_no_evidence_is_unknown(tmp_path):
    path, trace, weights = capture_fixture(tmp_path)
    all_roots = propagate(path, weights, chunk=4)
    # Legacy v3 caches use final-state unembedding; identical output target.
    legacy = dict(trace)
    legacy.pop("readout_runner_id")
    np.savez_compressed(path, **legacy)
    old = propagate(path, weights, chunk=4)
    np.testing.assert_array_equal(old["readout_runner_id"], all_roots["readout_runner_id"])
    np.testing.assert_allclose(old["final_material_margin"], all_roots["final_material_margin"], atol=1e-7)
    parts = []
    for root in (1, 3):
        trace["evidence_mask"] = np.arange(len(trace["token_ids"])) == root
        np.savez_compressed(path, **trace)
        parts.append(propagate(path, weights, chunk=4)["final_material_margin"])
    np.testing.assert_allclose(sum(parts), all_roots["final_material_margin"], atol=2e-7, rtol=2e-5)
    trace["evidence_mask"][:] = False
    np.savez_compressed(path, **trace)
    empty = propagate(path, weights, chunk=4)
    assert not empty["final_material_margin"].any()
    assert np.isnan(empty["score_material_deficit"]).all()


def test_suffix_cannot_change_earlier_scores(tmp_path):
    short, long = tmp_path / "short", tmp_path / "long"
    short.mkdir(); long.mkdir()
    prefix = [1, 3, 5, 7, 9, 11, 13, 15, 17]
    p, _, w = capture_fixture(short, ids=prefix)
    q, _, v = capture_fixture(long, ids=prefix + [27, 26, 24])
    a, b = propagate(p, w), propagate(q, v)
    np.testing.assert_allclose(a["score_material_deficit"][:-1], b["score_material_deficit"][:len(prefix)-5], atol=2e-6)


def test_interrupted_pipeline_evaluates_without_weights_or_large_caches(tmp_path, monkeypatch):
    import shutil
    from experiments.reanchor_flow import message_lineage_run as run
    from experiments.reanchor_flow.attention_rhythm_report import save_json
    native = tmp_path / "native"
    native.mkdir()
    path, trace, weights = capture_fixture(tmp_path)
    entries = []
    for split in ("train", "test"):
        for cls in ("normal", "hallucinated"):
            dest = native / split / "QA" / f"{cls}.npz"
            dest.parent.mkdir(parents=True, exist_ok=True)
            for suffix in (".npz", ".states.npz", ".history.npz", ".qk.npz"):
                shutil.copyfile(path.with_suffix(suffix), dest.with_suffix(suffix))
            y = np.zeros(8, dtype=np.int8)
            if cls == "hallucinated":
                y[[1, 2, 5]] = 1
            np.savez_compressed(dest.with_suffix(".labels.npz"), labels=y)
            entries.append(dict(split=split, task_type="QA", sample_id=cls, source_id=split+cls,
                                path=str(dest.relative_to(native)), response_tokens=8))
    entries.append(dict(split="test", task_type="Summary", sample_id="unfinished", source_id="unfinished",
                        path="test/Summary/unfinished.npz", response_tokens=20))
    manifest = dict(audit_schema=3, settings={"save_states": True, "full_attention": False,
                                             "model": str(weights.directory)}, samples=entries)
    save_json(native / "index.json", manifest)
    index_before = (native / "index.json").read_bytes()
    args = run.parser().parse_args(["--audit", str(native), "--device", "cpu", "--completed-only",
                                   "--bootstrap", "2", "--top-k", "2", "--plots-per-class", "1"])
    result = run.run(args)
    assert result["coverage"]["completed_samples"] == 4 and result["coverage"]["partial"]
    assert (native / "index.json").read_bytes() == index_before
    assert set(result["detection"]["test"]["groups"]) == {"QA", "ALL"}
    assert "Summary" not in result["replication"]
    output = native / "message_lineage"
    assert (output / "gallery.html").is_file()
    assert (output / "view_message_lineage.ipynb").is_file()
    for e in entries[:-1]:
        for suffix in (".states.npz", ".history.npz", ".qk.npz"):
            (native/e["path"]).with_suffix(suffix).unlink()
    monkeypatch.setattr(run, "CheckpointWeights", lambda *a, **k: pytest.fail("offline evaluation loaded weights"))
    args = run.parser().parse_args(["--audit", str(native), "--phase", "evaluate", "--completed-only",
                                   "--bootstrap", "0", "--plots-per-class", "0"])
    again = run.run(args)
    assert again["detection"]["test"]["groups"]["ALL"]["scores"] == {
        k: {**v, "auroc_ci95": None, "auprc_ci95": None,
            "bootstrap_valid": {"auroc": 0, "auprc": 0}}
        for k, v in result["detection"]["test"]["groups"]["ALL"]["scores"].items()}
    assert (native / "index.json").read_bytes() == index_before
