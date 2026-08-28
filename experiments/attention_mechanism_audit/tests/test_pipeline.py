from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiment_protocol import FrozenEvaluation, file_sha256
from experiments.attention_mechanism_audit.artifacts import save_artifact
from experiments.attention_mechanism_audit.evaluate import (
    answer_labels,
    validate_canonical_binding,
)
from experiments.attention_mechanism_audit.pipeline import (
    _repeat_identifier,
    aggregate_answer_features,
    checkpoint_fingerprints,
    flatten_token_trajectories,
    preregistered_directions,
    resolve_replay_dtype,
    validate_checkpoint_file_hashes,
    validate_loaded_replay_provenance,
    validate_replay_runtime,
)
from experiments.attention_mechanism_audit.tests.test_artifacts import mechanism_table


class ArrayView:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value.copy()


class FakeAttention:
    def __init__(self, token_ids, response_idx):
        self.token_ids = ArrayView(token_ids)
        self.response_idx = response_idx
        self.num_response_tokens = len(token_ids) - response_idx


class FakeSample:
    def __init__(self, dataset, sample_id, source, task, generator, token_ids, start):
        self.dataset = dataset
        self.sample_id = sample_id
        self.source_id = source
        self.task_type = task
        self.generator_model = generator
        self._attention = FakeAttention(token_ids, start)

    def attention(self):
        return self._attention

    def release_attention(self):
        pass


class FakeDataset:
    def __init__(self, root):
        self.root = root
        self.manifest = {"split": "test"}
        self.samples = {}

    @property
    def sample_ids(self):
        return list(self.samples)

    def __getitem__(self, sample_id):
        return self.samples[str(sample_id)]


def bound_table_and_dataset(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"split": "test"}), encoding="utf-8"
    )
    dataset = FakeDataset(root)
    dataset.samples = {
        "s1": FakeSample(dataset, "s1", "a", "QA", "g1", [1, 2, 3, 11, 12], 3),
        "s2": FakeSample(
            dataset, "s2", "b", "QA", "g2", [4, 5, 6, 7, 21, 22, 23], 4
        ),
    }
    table = mechanism_table()
    metadata = dict(table.metadata)
    metadata["data_root"] = str(root.resolve())
    metadata["dataset_manifest_sha256"] = file_sha256(root / "manifest.json")
    table = replace(table, metadata=metadata).validate()
    path = tmp_path / "mechanisms.npz"
    save_artifact(path, table)
    return table, dataset, path


def test_canonical_binding_rejects_wrong_target_before_any_label_api(tmp_path):
    table, dataset, path = bound_table_and_dataset(tmp_path)
    frozen = FrozenEvaluation.capture(path, expected_split="test")
    validate_canonical_binding(table, dataset, frozen)

    target = table.response_token_id.copy()
    target[0] = 999
    wrong = replace(table, response_token_id=target).validate()
    with pytest.raises(ValueError, match="response token IDs"):
        validate_canonical_binding(wrong, dataset, frozen)


def test_flatten_and_aggregate_preserve_layers_but_use_compact_answer_sources():
    traces = {
        "drift_functional_history_to_grounding_log_ratio": np.asarray(
            [[np.nan, np.nan], [1.0, 3.0], [2.0, 4.0]]
        ),
        "counterfactual_evidence_bypass": np.asarray([0.1, 0.2, 0.3]),
    }
    names, matrix, answer_sources = flatten_token_trajectories(
        traces, response_length=3, layer_count=2
    )

    assert matrix.shape == (3, 4)
    assert "drift_functional_history_to_grounding_log_ratio__layer_000" in names
    np.testing.assert_allclose(
        answer_sources[
            "drift_functional_history_to_grounding_log_ratio__layer_mean"
        ],
        [np.nan, 2.0, 3.0],
    )
    answer_names, answer = aggregate_answer_features(answer_sources)
    assert len(answer_names) == len(answer)
    directions = preregistered_directions(answer_names)
    mean_name = (
        "drift_functional_history_to_grounding_log_ratio__layer_mean__mean"
    )
    assert directions[mean_name] == "high"
    assert directions[mean_name.replace("__mean", "__max")] == "exploratory"


def test_checkpoint_fingerprint_binds_weight_and_tokenizer_bytes(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weight-a")
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")

    first_model, first_tokenizer = checkpoint_fingerprints(model)
    (model / "model.safetensors").write_bytes(b"weight-b")
    second_model, second_tokenizer = checkpoint_fingerprints(model)

    assert first_model != second_model
    assert first_tokenizer == second_tokenizer


def replay_spec(**updates):
    spec = {
        "dtype": "float16",
        "cache_dtype": "torch.float32",
        "attn_implementation": "eager",
        "transformers_version": "4.46.3",
        "torch_version": "2.6.0+cu124",
    }
    spec.update(updates)
    return spec


def test_replay_dtype_auto_uses_compute_dtype_not_storage_dtype():
    spec = replay_spec(dtype="torch.bfloat16", cache_dtype="torch.float16")

    assert resolve_replay_dtype("auto", spec) == "bfloat16"
    assert resolve_replay_dtype("bfloat16", spec) == "bfloat16"
    with pytest.raises(ValueError, match="differs from the cache computation dtype"):
        resolve_replay_dtype("float16", spec)


def test_replay_runtime_requires_exact_versions_and_eager():
    spec = replay_spec()

    assert validate_replay_runtime(
        spec,
        requested_dtype="auto",
        transformers_version="4.46.3",
        torch_version="2.6.0+cu124",
    ) == "float16"
    with pytest.raises(ValueError, match="install transformers==4.46.3"):
        validate_replay_runtime(
            spec,
            requested_dtype="auto",
            transformers_version="4.57.1",
            torch_version="2.6.0+cu124",
        )
    with pytest.raises(ValueError, match="exact extraction environment"):
        validate_replay_runtime(
            spec,
            requested_dtype="auto",
            transformers_version="4.46.3",
            torch_version="2.7.0+cu126",
        )
    with pytest.raises(ValueError, match="attn_implementation=eager"):
        validate_replay_runtime(
            replay_spec(attn_implementation="sdpa"),
            requested_dtype="auto",
            transformers_version="4.46.3",
            torch_version="2.6.0+cu124",
        )


def test_checkpoint_file_hashes_bind_manifest_files(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    shard = model / "model-00001-of-00001.safetensors"
    config.write_text("{}", encoding="utf-8")
    shard.write_bytes(b"weights")
    expected = {
        "config.json": file_sha256(config),
        shard.name: file_sha256(shard),
    }

    assert validate_checkpoint_file_hashes(model, expected) == expected
    shard.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_checkpoint_file_hashes(model, expected)


def test_checkpoint_file_hashes_reject_extra_loadable_file(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    shard = model / "model-00001-of-00001.safetensors"
    config.write_text("{}", encoding="utf-8")
    shard.write_bytes(b"extraction weights")
    expected = {
        config.name: file_sha256(config),
        shard.name: file_sha256(shard),
    }
    (model / "model.safetensors").write_bytes(b"later replacement weights")

    with pytest.raises(ValueError, match="unexpected model.safetensors"):
        validate_checkpoint_file_hashes(model, expected)


def test_checkpoint_file_hashes_allow_snapshot_blob_symlink(tmp_path):
    blob = tmp_path / "blobs" / "weight"
    blob.parent.mkdir()
    blob.write_bytes(b"shared snapshot weights")
    model = tmp_path / "snapshots" / "revision"
    model.mkdir(parents=True)
    shard = model / "model.safetensors"
    shard.symlink_to(blob)
    expected = {shard.name: file_sha256(blob)}

    assert validate_checkpoint_file_hashes(model, expected) == expected


def test_loaded_replay_provenance_checks_class_eager_and_parameter_dtype():
    class FakeParameter:
        def __init__(self, dtype):
            self.dtype = dtype

        def is_floating_point(self):
            return True

    class FakeModel:
        def __init__(self, implementation="eager", dtype="torch.bfloat16"):
            self.config = SimpleNamespace(_attn_implementation=implementation)
            self.parameter = FakeParameter(dtype)

        def named_parameters(self):
            return [("model.layers.0.self_attn.v_proj.weight", self.parameter)]

    expected_class = f"{FakeModel.__module__}.{FakeModel.__qualname__}"
    spec = {"model_class": expected_class}
    replay = SimpleNamespace(model=FakeModel())

    provenance = validate_loaded_replay_provenance(
        replay,
        spec,
        resolved_dtype="bfloat16",
    )
    assert provenance["model_class"] == expected_class
    assert provenance["attention_implementation"] == "eager"
    assert provenance["parameter_dtype"] == "bfloat16"

    with pytest.raises(ValueError, match="model class differs"):
        validate_loaded_replay_provenance(
            replay,
            {"model_class": "elsewhere.OtherModel"},
            resolved_dtype="bfloat16",
        )
    with pytest.raises(ValueError, match="did not instantiate eager"):
        validate_loaded_replay_provenance(
            SimpleNamespace(model=FakeModel(implementation="sdpa")),
            spec,
            resolved_dtype="bfloat16",
        )
    with pytest.raises(ValueError, match="parameter dtype differs"):
        validate_loaded_replay_provenance(
            SimpleNamespace(model=FakeModel(dtype="torch.float16")),
            spec,
            resolved_dtype="bfloat16",
        )


def test_answer_label_includes_positive_unavailable_first_token():
    table = mechanism_table()
    labels = SimpleNamespace(
        response_positive=np.asarray([1, 1, 0, 0, 0], dtype=np.int8),
        source_id=table.token_source_id,
    )

    answer, source = answer_labels(table, labels)

    np.testing.assert_array_equal(answer, [1, 0])
    np.testing.assert_array_equal(source, ["a", "b"])


def test_repeat_identifier_preserves_multicharacter_and_unicode_values():
    np.testing.assert_array_equal(
        _repeat_identifier("回答-1472", 3),
        ["回答-1472", "回答-1472", "回答-1472"],
    )
