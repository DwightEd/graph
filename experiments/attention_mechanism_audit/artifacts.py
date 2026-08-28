"""Deterministic, label-free artifacts for the mechanism audit."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Mapping
import zipfile

import numpy as np

from experiment_protocol import validate_complete_token_rows
from .counterfactuals import COUNTERFACTUAL_NAMES


ARTIFACT_SCHEMA = "attention-hallucination-mechanism-audit"
ARTIFACT_VERSION = 2
ALIGNMENT = "predecessor_query_predicts_response_token"
OBJECTIVE = "hutchinson_diagonal_per_token_chosen_logprob_jacobian"
COUNTERFACTUAL_VARIANTS = COUNTERFACTUAL_NAMES

IMPLEMENTATION_FILES = (
    "aggregation.py",
    "alignment.py",
    "artifacts.py",
    "cache_binding.py",
    "counterfactuals.py",
    "evaluate.py",
    "functional_flow.py",
    "mechanisms.py",
    "pipeline.py",
    "replay.py",
    "roles.py",
    "routing.py",
    "run.py",
)
DEPENDENCY_FILES = (
    "requirements.txt",
    "experiment_protocol.py",
    "research_dataset.py",
    "experiments/grounded_route/config.py",
    "experiments/grounded_route/graph.py",
    "experiments/directed_route_hypergraph/routing_dispersion.py",
)


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def implementation_sha256() -> str:
    """Hash every source file that can change a persisted feature meaning."""

    root = Path(__file__).resolve().parent
    repository = root.parents[1]
    digest = hashlib.sha256()
    for name in IMPLEMENTATION_FILES:
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"mechanism implementation file is missing: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    for name in DEPENDENCY_FILES:
        path = repository / name
        if not path.is_file():
            raise RuntimeError(f"mechanism dependency file is missing: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _digest(value: object, name: str) -> str:
    value = str(value)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lower-case SHA-256 digest")
    return value


@dataclass(frozen=True)
class MechanismArtifact:
    """Complete token trajectories and fixed answer-level summaries."""

    sample_id: np.ndarray
    source_id: np.ndarray
    task_type: np.ndarray
    generator_model: np.ndarray
    prompt_length: np.ndarray
    response_length: np.ndarray
    answer_feature_names: tuple[str, ...]
    answer_feature: np.ndarray
    token_sample_id: np.ndarray
    token_source_id: np.ndarray
    token_index: np.ndarray
    token_response_length: np.ndarray
    response_token_id: np.ndarray
    predictor_position: np.ndarray
    cached_query_index: np.ndarray
    cached_route_available: np.ndarray
    counterfactual_variant_available: np.ndarray
    token_feature_names: tuple[str, ...]
    token_feature: np.ndarray
    metadata: dict[str, object]

    def validate(self) -> "MechanismArtifact":
        answers = len(self.sample_id)
        if answers < 1:
            raise ValueError("mechanism artifact contains no answers")
        answer_columns = (
            self.source_id,
            self.task_type,
            self.generator_model,
            self.prompt_length,
            self.response_length,
            self.answer_feature,
        )
        if any(len(value) != answers for value in answer_columns):
            raise ValueError("answer-level artifact rows are misaligned")
        if len(set(self.sample_id.astype(str).tolist())) != answers:
            raise ValueError("answer-level sample IDs must be unique")
        if bool((np.asarray(self.prompt_length) < 1).any()) or bool(
            (np.asarray(self.response_length) < 1).any()
        ):
            raise ValueError("prompt and response lengths must be positive")
        if self.answer_feature.ndim != 2 or self.answer_feature.shape[1] != len(
            self.answer_feature_names
        ):
            raise ValueError("answer feature matrix and names are misaligned")
        if len(set(self.answer_feature_names)) != len(self.answer_feature_names):
            raise ValueError("answer feature names must be unique")

        tokens = len(self.token_sample_id)
        token_columns = (
            self.token_source_id,
            self.token_index,
            self.token_response_length,
            self.response_token_id,
            self.predictor_position,
            self.cached_query_index,
            self.cached_route_available,
            self.token_feature,
        )
        if any(len(value) != tokens for value in token_columns):
            raise ValueError("token-level artifact rows are misaligned")
        if self.token_feature.ndim != 2 or self.token_feature.shape[1] != len(
            self.token_feature_names
        ):
            raise ValueError("token feature matrix and names are misaligned")
        if (
            self.counterfactual_variant_available.shape
            != (tokens, len(COUNTERFACTUAL_VARIANTS))
            or self.counterfactual_variant_available.dtype != np.bool_
        ):
            raise ValueError(
                "counterfactual availability must be boolean [token, variant]"
            )
        if len(set(self.token_feature_names)) != len(self.token_feature_names):
            raise ValueError("token feature names must be unique")
        validate_complete_token_rows(
            self.token_sample_id,
            self.token_source_id,
            self.token_index,
            self.token_response_length,
        )

        answer_location = {
            sample: index
            for index, sample in enumerate(self.sample_id.astype(str).tolist())
        }
        token_samples = self.token_sample_id.astype(str)
        if set(answer_location) != set(token_samples.tolist()):
            raise ValueError("answer and token sample ID sets must match exactly")
        if not bool(self.counterfactual_variant_available[:, :4].all()):
            raise ValueError("the four non-swap replay branches must cover every token")
        for sample in dict.fromkeys(token_samples.tolist()):
            if sample not in answer_location:
                raise ValueError("token rows contain an unknown answer sample")
            rows = np.flatnonzero(token_samples == sample)
            answer = answer_location[sample]
            if len(rows) != int(self.response_length[answer]):
                raise ValueError("token rows disagree with answer response length")
            if not np.all(
                self.token_source_id[rows].astype(str)
                == str(self.source_id[answer])
            ):
                raise ValueError("token and answer source IDs disagree")
            order = rows[np.argsort(self.token_index[rows])]
            if not np.array_equal(
                self.cached_route_available[order],
                np.arange(len(order), dtype=np.int64) > 0,
            ):
                raise ValueError(
                    "cached routing must be unavailable only for response token zero"
                )
            predictor = self.predictor_position[order].astype(np.int64)
            expected_predictor = (
                int(self.prompt_length[answer])
                - 1
                + np.arange(len(order), dtype=np.int64)
            )
            if not np.array_equal(predictor, expected_predictor):
                raise ValueError("predictor positions are not predecessor aligned")
            if not np.array_equal(
                self.cached_query_index[order].astype(np.int64),
                np.arange(-1, len(order) - 1, dtype=np.int64),
            ):
                raise ValueError("cached queries are not shifted onto predicted tokens")
            swaps = self.counterfactual_variant_available[order, 4:]
            if bool((swaps != swaps[0]).any()):
                raise ValueError("swap availability must be answer-consistent")

        swap_sensitive = [
            index
            for index, name in enumerate(self.token_feature_names)
            if "swapped_evidence" in name or name == "counterfactual_evidence_bypass"
        ]
        unavailable_swap = ~self.counterfactual_variant_available[:, 4:].any(axis=1)
        if swap_sensitive and bool(
            np.isfinite(
                self.token_feature[
                    np.ix_(np.flatnonzero(unavailable_swap), swap_sensitive)
                ]
            ).any()
        ):
            raise ValueError("unavailable evidence swaps must remain NaN, not zero")

        cached_sensitive = [
            index
            for index, name in enumerate(self.token_feature_names)
            if not name.startswith("counterfactual_")
        ]
        if cached_sensitive and bool(
            np.isfinite(
                self.token_feature[
                    np.ix_(np.flatnonzero(~self.cached_route_available), cached_sensitive)
                ]
            ).any()
        ):
            raise ValueError("unavailable cached-query mechanisms must remain NaN")

        metadata = self.metadata
        if bool(metadata.get("labels_used", True)):
            raise ValueError("capture artifacts must not use hallucination labels")
        if metadata.get("audit_scope") not in {"complete_split", "selected_samples"}:
            raise ValueError("artifact audit_scope is missing or invalid")
        if metadata.get("alignment") != ALIGNMENT:
            raise ValueError("artifact uses the wrong predictor alignment")
        if metadata.get("objective") != OBJECTIVE:
            raise ValueError("artifact uses the wrong answer objective")
        for name in (
            "dataset_manifest_sha256",
            "role_index_sha256",
            "source_info_sha256",
            "model_fingerprint",
            "tokenizer_fingerprint",
            "swap_assignment_sha256",
            "attention_binding_sha256",
            "attribution_seed_assignment_sha256",
            "implementation_sha256",
        ):
            _digest(metadata.get(name, ""), name)
        variants = tuple(metadata.get("counterfactual_variants", ()))
        if variants != COUNTERFACTUAL_VARIANTS:
            raise ValueError("artifact counterfactual variants are not preregistered")
        directions = metadata.get("answer_feature_directions")
        if not isinstance(directions, dict) or set(directions) != set(
            self.answer_feature_names
        ):
            raise ValueError("every answer feature must have one frozen direction")
        if any(
            direction not in {"high", "low", "exploratory"}
            for direction in directions.values()
        ):
            raise ValueError("artifact has an invalid answer feature direction")
        onset = tuple(metadata.get("onset_feature_names", ()))
        if len(set(onset)) != len(onset) or not set(onset).issubset(
            self.token_feature_names
        ):
            raise ValueError("onset diagnostics reference unknown token features")
        primary = tuple(metadata.get("primary_answer_feature_names", ()))
        if (
            not primary
            or len(set(primary)) != len(primary)
            or not set(primary).issubset(self.answer_feature_names)
            or any(directions[name] == "exploratory" for name in primary)
        ):
            raise ValueError("artifact has no valid preregistered primary features")
        if not isinstance(metadata.get("mechanism_observability"), dict):
            raise ValueError("artifact mechanism observability is missing")
        if not isinstance(metadata.get("observer_generator_audit"), dict):
            raise ValueError("artifact observer/generator audit is missing")
        binding = metadata.get("cache_replay_attention_binding")
        if not isinstance(binding, dict) or not bool(
            binding.get("verified_every_answer", False)
        ):
            raise ValueError("artifact has no verified cache/replay attention binding")
        tolerance = float(binding.get("absolute_tolerance", np.nan))
        errors = [
            float(binding.get(name, np.nan))
            for name in (
                "retained_max_abs_error",
                "diagonal_max_abs_error",
                "known_mass_max_abs_error",
            )
        ]
        if (
            not 0.0 < tolerance < 0.1
            or not np.isfinite(errors).all()
            or max(errors) > tolerance
            or int(binding.get("diagonal_endpoints_compared", 0)) < tokens
        ):
            raise ValueError("artifact cache/replay attention binding is invalid")
        attribution = metadata.get("functional_attribution")
        if (
            not isinstance(attribution, dict)
            or attribution.get("jacobian_estimator")
            != "iid Rademacher Hutchinson diagonal VJP"
            or int(attribution.get("gradient_probe_count", 0)) < 1
        ):
            raise ValueError("artifact token-diagonal attribution audit is missing")
        runtime = metadata.get("runtime")
        if (
            not isinstance(runtime, dict)
            or runtime.get("attention_implementation") != "eager"
            or not runtime.get("torch")
            or not runtime.get("transformers")
        ):
            raise ValueError("artifact replay runtime provenance is missing")
        return self

    def frozen_rows(self) -> dict[str, np.ndarray]:
        """Return the fields consumed by :class:`FrozenEvaluation`."""

        return {
            "audit_scope": np.asarray(self.metadata["audit_scope"]),
            "dataset_manifest_sha256": np.asarray(
                self.metadata["dataset_manifest_sha256"]
            ),
            "sample_id": self.token_sample_id.astype(str),
            "source_id": self.token_source_id.astype(str),
            "token_index": self.token_index.astype(np.int32),
            "response_length": self.token_response_length.astype(np.int32),
        }


def _payload(table: MechanismArtifact) -> dict[str, np.ndarray]:
    table = table.validate()
    return {
        "schema": np.asarray(ARTIFACT_SCHEMA),
        "version": np.asarray(ARTIFACT_VERSION, dtype=np.int32),
        "sample_id": table.sample_id.astype(str),
        "source_id": table.source_id.astype(str),
        "task_type": table.task_type.astype(str),
        "generator_model": table.generator_model.astype(str),
        "prompt_length": table.prompt_length.astype(np.int32),
        "response_length": table.response_length.astype(np.int32),
        "answer_feature_names": np.asarray(table.answer_feature_names, dtype=str),
        "answer_feature": table.answer_feature.astype(np.float32),
        "token_sample_id": table.token_sample_id.astype(str),
        "token_source_id": table.token_source_id.astype(str),
        "token_index": table.token_index.astype(np.int32),
        "token_response_length": table.token_response_length.astype(np.int32),
        "response_token_id": table.response_token_id.astype(np.int64),
        "predictor_position": table.predictor_position.astype(np.int32),
        "cached_query_index": table.cached_query_index.astype(np.int32),
        "cached_route_available": table.cached_route_available.astype(bool),
        "counterfactual_variant_available": (
            table.counterfactual_variant_available.astype(bool)
        ),
        "token_feature_names": np.asarray(table.token_feature_names, dtype=str),
        "token_feature": table.token_feature.astype(np.float32),
        "metadata_json": np.asarray(
            json.dumps(
                table.metadata,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        ),
    }


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.lib.format.write_array(buffer, np.asarray(value), allow_pickle=False)
    return buffer.getvalue()


def save_artifact(path, table: MechanismArtifact) -> None:
    """Write a byte-deterministic compressed NPZ and atomically install it."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload(table)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name in sorted(payload):
                info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, _npy_bytes(payload[name]), compresslevel=9)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_artifact(path) -> MechanismArtifact:
    with np.load(path, allow_pickle=False) as arrays:
        expected_keys = {
            "schema",
            "version",
            "sample_id",
            "source_id",
            "task_type",
            "generator_model",
            "prompt_length",
            "response_length",
            "answer_feature_names",
            "answer_feature",
            "token_sample_id",
            "token_source_id",
            "token_index",
            "token_response_length",
            "response_token_id",
            "predictor_position",
            "cached_query_index",
            "cached_route_available",
            "counterfactual_variant_available",
            "token_feature_names",
            "token_feature",
            "metadata_json",
        }
        if set(arrays.files) != expected_keys:
            missing = sorted(expected_keys.difference(arrays.files))
            extra = sorted(set(arrays.files).difference(expected_keys))
            raise ValueError(
                f"mechanism artifact keys differ: missing={missing}, extra={extra}"
            )
        if (
            str(arrays["schema"].item()) != ARTIFACT_SCHEMA
            or int(arrays["version"].item()) != ARTIFACT_VERSION
        ):
            raise ValueError("unsupported mechanism artifact")
        table = MechanismArtifact(
            sample_id=arrays["sample_id"].astype(str),
            source_id=arrays["source_id"].astype(str),
            task_type=arrays["task_type"].astype(str),
            generator_model=arrays["generator_model"].astype(str),
            prompt_length=arrays["prompt_length"].astype(np.int32),
            response_length=arrays["response_length"].astype(np.int32),
            answer_feature_names=tuple(
                arrays["answer_feature_names"].astype(str).tolist()
            ),
            answer_feature=arrays["answer_feature"].astype(np.float32),
            token_sample_id=arrays["token_sample_id"].astype(str),
            token_source_id=arrays["token_source_id"].astype(str),
            token_index=arrays["token_index"].astype(np.int32),
            token_response_length=arrays["token_response_length"].astype(np.int32),
            response_token_id=arrays["response_token_id"].astype(np.int64),
            predictor_position=arrays["predictor_position"].astype(np.int32),
            cached_query_index=arrays["cached_query_index"].astype(np.int32),
            cached_route_available=arrays["cached_route_available"].astype(bool),
            counterfactual_variant_available=arrays[
                "counterfactual_variant_available"
            ].astype(bool),
            token_feature_names=tuple(
                arrays["token_feature_names"].astype(str).tolist()
            ),
            token_feature=arrays["token_feature"].astype(np.float32),
            metadata=json.loads(str(arrays["metadata_json"].item())),
        )
    return table.validate()


def require_metadata(table: MechanismArtifact, expected: Mapping[str, object]) -> None:
    """Reject evaluation when any frozen provenance field differs."""

    for name, value in expected.items():
        if table.metadata.get(name) != value:
            raise ValueError(f"artifact metadata differs for {name}")
