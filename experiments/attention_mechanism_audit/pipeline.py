"""Label-sealed orchestration for the three-axis mechanism audit."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Mapping

import numpy as np

from experiment_protocol import canonical_source_group, dataset_manifest_sha256

from .aggregation import ANSWER_STATISTICS, aggregate_trajectory
from .alignment import predecessor_alignment
from .artifacts import (
    ALIGNMENT,
    COUNTERFACTUAL_VARIANTS,
    OBJECTIVE,
    MechanismArtifact,
    file_sha256,
    implementation_sha256,
    save_artifact,
)
from .counterfactuals import build_counterfactual_variants, choose_donors
from .roles import (
    CachedPrompt,
    build_role_index,
    load_role_jsonl,
    position_stratified_role_permutation,
    read_source_info,
    sample_role_permutation_seed,
    source_record_sha256,
    write_role_jsonl,
)


MODEL_CONFIGURATION_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
)
WEIGHT_PATTERNS = ("*.safetensors", "pytorch_model*.bin")
REPLAY_DTYPES = ("float32", "float16", "bfloat16")


def _hash_named_files(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(files), key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 << 20), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def checkpoint_fingerprints(model_path) -> tuple[str, str]:
    """Hash checkpoint bytes and tokenizer bytes, not path names or mtimes."""

    root = Path(model_path).resolve()
    if not root.is_dir():
        raise ValueError("mechanism capture requires a local checkpoint directory")
    configuration = [root / name for name in MODEL_CONFIGURATION_FILES]
    configuration = [path for path in configuration if path.is_file()]
    weights: list[Path] = []
    for pattern in WEIGHT_PATTERNS:
        weights.extend(path for path in root.glob(pattern) if path.is_file())
    if not weights or not (root / "config.json").is_file():
        raise ValueError("checkpoint has no config.json or local model weight shards")
    tokenizer = [root / name for name in TOKENIZER_FILES]
    tokenizer = [path for path in tokenizer if path.is_file()]
    if not tokenizer:
        raise ValueError("checkpoint has no local tokenizer files")
    return (
        _hash_named_files(root, configuration + weights),
        _hash_named_files(root, tokenizer),
    )


def _normalized_replay_dtype(value: object) -> str:
    """Normalize a manifest/CLI dtype without confusing it with storage dtype."""

    name = str(value).strip().casefold().removeprefix("torch.")
    aliases = {"half": "float16", "bf16": "bfloat16", "float": "float32"}
    name = aliases.get(name, name)
    if name not in REPLAY_DTYPES:
        raise ValueError(f"unsupported attention-cache computation dtype: {value!r}")
    return name


def resolve_replay_dtype(requested_dtype: str, cache_spec: Mapping[str, object]) -> str:
    """Resolve ``auto`` from computation dtype, never serialized cache dtype."""

    requested = str(requested_dtype).strip().casefold()
    if requested not in {"auto", *REPLAY_DTYPES}:
        raise ValueError(
            "--torch-dtype must be auto, float32, float16, or bfloat16"
        )
    if "dtype" not in cache_spec:
        raise ValueError(
            "attention_cache_spec does not record the model computation dtype; "
            "use a provenance-complete cache or regenerate it"
        )
    expected = _normalized_replay_dtype(cache_spec["dtype"])
    if requested != "auto" and requested != expected:
        raise ValueError(
            f"requested --torch-dtype {requested} differs from the cache "
            f"computation dtype {expected}; use --torch-dtype auto (or {expected})"
        )
    return expected


def validate_replay_runtime(
    cache_spec: Mapping[str, object],
    *,
    requested_dtype: str,
    transformers_version: str,
    torch_version: str,
) -> str:
    """Validate runtime fields that can alter eager attention numerics."""

    resolved_dtype = resolve_replay_dtype(requested_dtype, cache_spec)
    implementation = str(cache_spec.get("attn_implementation", "")).casefold()
    if implementation != "eager":
        raise ValueError(
            "attention cache was not extracted with attn_implementation=eager; "
            "this audit cannot mix its attention with an eager replay"
        )

    expected_transformers = str(cache_spec.get("transformers_version", ""))
    if not expected_transformers:
        raise ValueError(
            "attention_cache_spec does not record transformers_version; "
            "use a provenance-complete cache or regenerate it"
        )
    if str(transformers_version) != expected_transformers:
        raise ValueError(
            "Transformers version differs from the attention-cache extraction "
            f"runtime: expected {expected_transformers}, found "
            f"{transformers_version}. Activate the extraction environment or "
            f"install transformers=={expected_transformers}; do not relax the "
            "attention-binding tolerance"
        )

    expected_torch = str(cache_spec.get("torch_version", ""))
    if not expected_torch:
        raise ValueError(
            "attention_cache_spec does not record torch_version; use a "
            "provenance-complete cache or regenerate it"
        )
    if str(torch_version) != expected_torch:
        raise ValueError(
            "PyTorch version/build differs from the attention-cache extraction "
            f"runtime: expected {expected_torch}, found {torch_version}. "
            "Activate the exact extraction environment (including its CUDA "
            "build); do not relax the attention-binding tolerance"
        )
    return resolved_dtype


def validate_checkpoint_file_hashes(
    model_path, expected_hashes: object
) -> dict[str, str]:
    """Bind the exact extraction-time root-file inventory and its bytes."""

    root = Path(model_path).resolve()
    if not root.is_dir():
        raise ValueError("mechanism capture requires a local checkpoint directory")
    if not isinstance(expected_hashes, Mapping) or not expected_hashes:
        raise ValueError(
            "attention_cache_spec has no model_files_sha256 map; use a "
            "provenance-complete cache or regenerate it"
        )

    expected_by_name: dict[str, str] = {}
    for raw_name, raw_digest in sorted(
        expected_hashes.items(), key=lambda item: str(item[0])
    ):
        name = str(raw_name)
        expected = str(raw_digest).casefold()
        relative = Path(name)
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.name in {"", ".", ".."}
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError(
                "attention_cache_spec contains an invalid model_files_sha256 entry"
            )
        if name in expected_by_name:
            raise ValueError(
                "attention_cache_spec contains duplicate model file names"
            )
        expected_by_name[name] = expected

    # The upstream extractor hashes every regular file in MODEL_PATH's root.
    # Match that inventory exactly: an added weight/config/remote-code file may
    # change from_pretrained's resolution even when every old file still exists.
    # ``is_file`` intentionally follows Hugging Face snapshot symlinks into the
    # shared blobs directory; lexical single-component names already prevent
    # manifest path traversal.
    actual_names = {path.name for path in root.iterdir() if path.is_file()}
    expected_names = set(expected_by_name)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        problems = [*(f"missing {name}" for name in missing)]
        problems.extend(f"unexpected {name}" for name in unexpected)
        detail = "; ".join(problems[:5])
        if len(problems) > 5:
            detail += f"; and {len(problems) - 5} more"
        raise ValueError(
            "MODEL_PATH root-file inventory differs from the exact checkpoint "
            f"recorded by the attention cache ({detail}). Point MODEL_PATH at "
            "the original checkpoint or regenerate the cache with the current "
            "checkpoint"
        )

    actual: dict[str, str] = {}
    problems: list[str] = []
    for name, expected in sorted(expected_by_name.items()):
        path = root / name
        digest = file_sha256(path)
        actual[name] = digest
        if digest.casefold() != expected:
            problems.append(f"SHA256 mismatch for {name}")
    if problems:
        detail = "; ".join(problems[:5])
        if len(problems) > 5:
            detail += f"; and {len(problems) - 5} more"
        raise ValueError(
            "MODEL_PATH is not the exact checkpoint recorded by the attention "
            f"cache ({detail}). Point MODEL_PATH at the original checkpoint or "
            "regenerate the cache with the current checkpoint"
        )
    return actual


def validate_loaded_replay_provenance(
    replay,
    cache_spec: Mapping[str, object],
    *,
    resolved_dtype: str,
) -> dict[str, object]:
    """Verify what ``from_pretrained`` instantiated, not only its source files."""

    model = replay.model
    expected_class = str(cache_spec.get("model_class", ""))
    if not expected_class:
        raise ValueError(
            "attention_cache_spec does not record model_class; use a "
            "provenance-complete cache or regenerate it"
        )
    actual_class = f"{type(model).__module__}.{type(model).__qualname__}"
    if actual_class != expected_class:
        raise ValueError(
            "loaded replay model class differs from attention-cache extraction: "
            f"expected {expected_class}, found {actual_class}. Use the exact "
            "extraction code path and trust_remote_code setting"
        )

    implementation = str(
        getattr(getattr(model, "config", None), "_attn_implementation", "")
    ).casefold()
    if implementation != "eager":
        raise ValueError(
            "loaded replay model did not instantiate eager attention: "
            f"found {implementation or '<missing>'}"
        )

    mismatched: list[str] = []
    floating_count = 0
    for name, parameter in model.named_parameters():
        is_floating = getattr(parameter, "is_floating_point", None)
        if not callable(is_floating) or not bool(is_floating()):
            continue
        floating_count += 1
        actual_dtype = _normalized_replay_dtype(parameter.dtype)
        if actual_dtype != resolved_dtype:
            mismatched.append(f"{name}={actual_dtype}")
    if floating_count == 0:
        raise ValueError("loaded replay model exposes no floating-point parameters")
    if mismatched:
        detail = ", ".join(mismatched[:5])
        if len(mismatched) > 5:
            detail += f", and {len(mismatched) - 5} more"
        raise ValueError(
            "loaded replay parameter dtype differs from the cache computation "
            f"dtype {resolved_dtype} ({detail})"
        )
    return {
        "model_class": actual_class,
        "attention_implementation": implementation,
        "parameter_dtype": resolved_dtype,
        "floating_parameter_tensors": floating_count,
    }


def select_samples(dataset, task: str, limit: int | None) -> tuple[str, ...]:
    """Select metadata rows without touching attention labels."""

    selected = []
    for sample_id in dataset.sample_ids:
        sample = dataset[sample_id]
        if task.casefold() == "all" or str(sample.task_type).casefold() == task.casefold():
            selected.append(str(sample_id))
    if limit is not None:
        if int(limit) < 1:
            raise ValueError("limit must be positive")
        selected = selected[: int(limit)]
    if not selected:
        raise ValueError("no samples match the requested mechanism-audit task")
    return tuple(selected)


def _require_fast_tokenizer(tokenizer_path, *, trust_remote_code: bool):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "prompt-role reconstruction requires Hugging Face transformers"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
        use_fast=True,
        trust_remote_code=bool(trust_remote_code),
    )
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise ValueError("prompt-role reconstruction requires tokenizer offsets")
    return tokenizer


def build_roles(
    data_root,
    source_info_path,
    tokenizer_path,
    output_path,
    *,
    task: str = "QA",
    limit: int | None = None,
    trust_remote_code: bool = False,
) -> dict[str, object]:
    """Reconstruct exact prompt roles using source_info, never response labels."""

    from research_dataset import open_research_dataset

    dataset = open_research_dataset(
        data_root,
        device="cpu",
        verify_hashes=True,
        retain_embedded_labels=False,
    )
    sample_ids = select_samples(dataset, task, limit)
    tokenizer = _require_fast_tokenizer(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
    )
    cached = []
    for sample_id in sample_ids:
        sample = dataset[sample_id]
        try:
            attention = sample.attention()
            cached.append(
                CachedPrompt(
                    source_id=canonical_source_group(sample),
                    token_ids=attention.token_ids.detach().cpu().numpy(),
                    response_idx=int(attention.response_idx),
                )
            )
        finally:
            sample.release_attention()
    roles = build_role_index(source_info_path, tokenizer, cached)
    path = write_role_jsonl(roles, output_path)
    return {
        "roles": str(path.resolve()),
        "sha256": file_sha256(path),
        "samples": len(sample_ids),
        "sources": len(roles),
        "labels_used": False,
    }


def _finite_mean(value: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(value)
    count = finite.sum(axis=axis)
    total = np.where(finite, value, 0.0).sum(axis=axis)
    return np.divide(
        total,
        count,
        out=np.full(np.shape(total), np.nan, dtype=np.float64),
        where=count > 0,
    )


def _repeat_identifier(value: object, count: int) -> np.ndarray:
    """Repeat a complete identifier without NumPy's ``dtype=str`` truncation."""

    text = str(value)
    count = int(count)
    if not text or count < 1:
        raise ValueError("identifier and repeat count must be non-empty")
    return np.repeat(np.asarray(text), count)


def flatten_token_trajectories(
    trajectories: Mapping[str, np.ndarray],
    *,
    response_length: int,
    layer_count: int,
) -> tuple[tuple[str, ...], np.ndarray, dict[str, np.ndarray]]:
    """Persist every token×layer trace and expose compact layer means."""

    names: list[str] = []
    columns: list[np.ndarray] = []
    answer_sources: dict[str, np.ndarray] = {}
    for name in sorted(trajectories):
        value = np.asarray(trajectories[name])
        if not np.issubdtype(value.dtype, np.number):
            continue
        value = value.astype(np.float64, copy=False)
        if value.shape == (response_length,):
            names.append(name)
            columns.append(value)
            answer_sources[name] = value
        elif value.shape == (response_length, layer_count):
            for layer in range(layer_count):
                names.append(f"{name}__layer_{layer:03d}")
                columns.append(value[:, layer])
            mean_name = f"{name}__layer_mean"
            mean = _finite_mean(value, axis=1)
            names.append(mean_name)
            columns.append(mean)
            answer_sources[mean_name] = mean
        else:
            raise ValueError(
                f"mechanism trajectory {name!r} has unsupported shape {value.shape}"
            )
    if not columns:
        raise ValueError("mechanism capture produced no numeric trajectories")
    return tuple(names), np.column_stack(columns), answer_sources


def aggregate_answer_features(
    trajectories: Mapping[str, np.ndarray],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Apply fixed response statistics without availability/length features."""

    names: list[str] = []
    values: list[float] = []
    for name in sorted(trajectories):
        summaries = aggregate_trajectory(np.asarray(trajectories[name]))
        for statistic in ANSWER_STATISTICS:
            result = np.asarray(summaries[statistic])
            if result.ndim != 0:
                raise ValueError("answer source trajectory must be scalar per token")
            names.append(f"{name}__{statistic}")
            values.append(float(result))
    return tuple(names), np.asarray(values, dtype=np.float64)


def _base_direction(name: str) -> str:
    if name.startswith("role_position_adjusted__"):
        name = name.removeprefix("role_position_adjusted__")
    if name == "drift_functional_history_to_grounding_log_ratio":
        return "high"
    if name in {
        "drift_functional_prompt_fraction",
        "drift_functional_evidence_fraction",
        "routing_direct_evidence_ancestry",
        "routing_relayed_evidence_ancestry",
        "routing_total_evidence_ancestry",
    }:
        return "low"
    if name.startswith("dispersion_functional_hhi") or name.startswith(
        "routing_concentration"
    ):
        return "low"
    if name.startswith("dispersion_") or name.startswith("routing_entropy"):
        return "high"
    if name in {
        "routing_head_role_js",
        "counterfactual_evidence_bypass",
        "counterfactual_no_evidence_delta",
        "counterfactual_swapped_evidence_delta",
    }:
        return "high"
    if name == "counterfactual_no_history_delta":
        return "low"
    return "exploratory"


def preregistered_directions(names: tuple[str, ...]) -> dict[str, str]:
    """Freeze directions from names alone, never from observed labels."""

    directions: dict[str, str] = {}
    for name in names:
        source, statistic = name.rsplit("__", 1)
        base = source.removesuffix("__layer_mean")
        direction = _base_direction(base)
        if statistic in {"max", "max_adjacent_drop"}:
            direction = "exploratory"
        directions[name] = direction
    return directions


def _primary_features(
    names: tuple[str, ...], directions: Mapping[str, str]
) -> tuple[str, ...]:
    # One endpoint per falsifiable hypothesis.  History reliance alone is not
    # directional: correct reasoning can also be highly self-consistent, so it
    # remains an exploratory attractor control rather than a primary test.
    requested = (
        "drift_functional_history_to_grounding_log_ratio"
        "__layer_mean__late_minus_early",
        "dispersion_functional_entropy_observed__layer_mean__late_minus_early",
        "dispersion_functional_cancellation__layer_mean__late_minus_early",
        "routing_entropy_upper__layer_mean__late_minus_early",
        "routing_total_evidence_ancestry__layer_mean__late_minus_early",
        "counterfactual_evidence_bypass__mean",
    )
    missing = [name for name in requested if name not in names]
    if missing:
        raise ValueError(f"preregistered primary mechanism features are missing: {missing}")
    if any(directions[name] == "exploratory" for name in requested):
        raise ValueError("a preregistered primary feature has no frozen direction")
    return requested


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalized_model_name(value: object) -> str:
    text = Path(str(value)).name.casefold()
    return "".join(character for character in text if character.isalnum())


def _swap_assignments(dataset, sample_ids, roles):
    assignments = []
    donors = {}
    for sample_id in sample_ids:
        sample = dataset[sample_id]
        source_id = canonical_source_group(sample)
        if source_id not in roles:
            raise ValueError(f"prompt-role index misses source {source_id}")
        selected = choose_donors(roles[source_id], roles, count=3)
        donors[sample_id] = selected
        assignments.append(
            {
                "sample_id": sample_id,
                "source_id": source_id,
                "donor_source_ids": [donor.source_id for donor in selected],
            }
        )
    return donors, assignments


def capture_mechanisms(
    data_root,
    role_index_path,
    source_info_path,
    model_path,
    output_path,
    *,
    device: str = "cuda",
    torch_dtype: str = "auto",
    task: str = "QA",
    limit: int | None = None,
    vocab_chunk_size: int = 4096,
    gradient_probes: int = 8,
    attribution_seed: int = 20260828,
    role_null_bin_width: int = 32,
    trust_remote_code: bool = False,
) -> dict[str, object]:
    """Capture all mechanisms while hallucination labels remain inaccessible."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise RuntimeError("mechanism capture requires PyTorch") from error
    from research_dataset import open_research_dataset
    from experiments.grounded_route.graph import build_graph

    from .functional_flow import functional_flow
    from .mechanisms import combine_mechanisms
    from .replay import FrozenCausalReplay
    from .routing import routing_flow

    if int(vocab_chunk_size) < 1:
        raise ValueError("vocab_chunk_size must be positive")
    if int(gradient_probes) < 1:
        raise ValueError("gradient_probes must be positive")
    if int(attribution_seed) < 0:
        raise ValueError("attribution_seed must be non-negative")
    if int(role_null_bin_width) < 2:
        raise ValueError("role_null_bin_width must be at least two")
    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dataset = open_research_dataset(
        data_root,
        device="cpu",
        verify_hashes=True,
        retain_embedded_labels=False,
    )
    cache_spec = getattr(dataset, "spec", {})
    try:
        import transformers
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise RuntimeError("mechanism capture requires Transformers") from error
    resolved_torch_dtype = validate_replay_runtime(
        cache_spec,
        requested_dtype=torch_dtype,
        transformers_version=str(transformers.__version__),
        torch_version=str(torch.__version__),
    )
    validate_checkpoint_file_hashes(
        model_path,
        cache_spec.get("model_files_sha256"),
    )
    sample_ids = select_samples(dataset, task, limit)
    roles = load_role_jsonl(role_index_path)
    source_rows = read_source_info(source_info_path)
    donors, assignments = _swap_assignments(dataset, sample_ids, roles)
    model_fingerprint, tokenizer_fingerprint = checkpoint_fingerprints(model_path)
    cache_observer = str(dataset.manifest.get("observer_model") or "")
    replay_name = Path(model_path).name
    normalized_replay = _normalized_model_name(replay_name)
    normalized_observer = _normalized_model_name(cache_observer)
    observer_name_matches_replay = bool(normalized_observer) and (
        normalized_observer == normalized_replay
    )
    replay = FrozenCausalReplay.from_pretrained(
        model_path,
        device=device,
        torch_dtype=dtype_by_name[resolved_torch_dtype],
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    loaded_replay_provenance = validate_loaded_replay_provenance(
        replay,
        cache_spec,
        resolved_dtype=resolved_torch_dtype,
    )
    if replay.head_count != int(dataset.manifest["num_heads"]):
        raise ValueError("replay model head count differs from attention cache")
    layer_count = int(getattr(replay.model.config, "num_hidden_layers"))
    if layer_count != int(dataset.manifest["num_layers"]):
        raise ValueError("replay model layer count differs from attention cache")

    answer_sample: list[str] = []
    answer_source: list[str] = []
    answer_task: list[str] = []
    answer_generator: list[str] = []
    answer_prompt_length: list[int] = []
    answer_response_length: list[int] = []
    answer_rows: list[np.ndarray] = []
    token_sample: list[np.ndarray] = []
    token_source: list[np.ndarray] = []
    token_index: list[np.ndarray] = []
    token_response_length: list[np.ndarray] = []
    response_token_id: list[np.ndarray] = []
    predictor_position: list[np.ndarray] = []
    cached_query_index: list[np.ndarray] = []
    cached_route_available: list[np.ndarray] = []
    variant_available: list[np.ndarray] = []
    token_rows: list[np.ndarray] = []
    token_feature_names: tuple[str, ...] | None = None
    answer_feature_names: tuple[str, ...] | None = None
    generator_models = []
    probe_seed_assignments: list[dict[str, object]] = []
    attention_bindings: list[dict[str, object]] = []

    for sample_number, sample_id in enumerate(sample_ids, start=1):
        sample = dataset[sample_id]
        try:
            attention = sample.attention()
            graph = build_graph(sample)
            tokens = attention.token_ids.detach().cpu().numpy().astype(np.int64)
            response_start = int(attention.response_idx)
            alignment = predecessor_alignment(
                tokens,
                response_start,
                cached_query_count=int(attention.num_response_tokens),
            )
            source_id = canonical_source_group(sample)
            # FormalResearchSample lazily reloads its PT payload after
            # release_attention().  Cache every metadata field while the
            # canonical attention view is live so one sample is loaded only
            # once and the released payload cannot be observed later.
            sample_task = str(sample.task_type or "")
            sample_generator = str(sample.generator_model or "")
            role_map = roles.get(source_id)
            if role_map is None:
                raise ValueError(f"prompt-role index misses source {source_id}")
            if role_map.task_type.casefold() != sample_task.casefold():
                raise ValueError("prompt-role task differs from cached sample")
            role_map.validate(tokens)
            source_row = source_rows.get(source_id)
            if source_row is None or source_record_sha256(source_row) != (
                role_map.source_info_sha256
            ):
                raise ValueError("prompt roles differ from bound source_info row")
            variants = build_counterfactual_variants(
                tokens,
                role_map,
                donors[sample_id],
            )
            sample_attribution_seed = sample_role_permutation_seed(
                attribution_seed,
                sample_id,
                tokens,
            )
            role_null_seed = sample_role_permutation_seed(
                attribution_seed + 1,
                sample_id,
                tokens,
            )
            null_role_ids = position_stratified_role_permutation(
                role_map.role_ids,
                bin_width=role_null_bin_width,
                seed=role_null_seed,
            )
        finally:
            sample.release_attention()

        print(
            f"capture sample {sample_number}/{len(sample_ids)}: {sample_id}",
            flush=True,
        )
        baseline = replay.capture_baseline(
            tokens,
            response_start,
            allowed_attention=variants["full"].allowed_attention,
            vocab_chunk_size=vocab_chunk_size,
            gradient_probes=gradient_probes,
            attribution_seed=sample_attribution_seed,
            expected_graph=graph,
        )
        if not baseline.attention_cache_binding or not bool(
            baseline.attention_cache_binding.get("verified", False)
        ):
            raise RuntimeError(
                "functional capture was not numerically bound to cached attention"
            )
        attention_bindings.append(dict(baseline.attention_cache_binding))
        probe_seed_assignments.append(
            {
                "sample_id": sample_id,
                "attribution_seed": sample_attribution_seed,
                "role_null_seed": role_null_seed,
            }
        )
        replay_result = replay.replay(
            tokens,
            response_start,
            np.flatnonzero(role_map.role_mask("evidence")),
            variants=variants,
            baseline_capture=baseline,
            vocab_chunk_size=vocab_chunk_size,
        )
        baseline_logp = baseline.chosen_logprob.detach().cpu().numpy()
        if not np.allclose(
            baseline_logp,
            replay_result.variants["full"].chosen_logprob,
            atol=2e-4,
            rtol=2e-4,
        ):
            raise RuntimeError("baseline gradient replay and full replay scores differ")

        functional = functional_flow(graph, role_map, baseline)
        routing = routing_flow(graph, role_map)
        mechanisms = combine_mechanisms(functional, routing, replay_result)
        null_functional = functional_flow(graph, null_role_ids, baseline)
        null_routing = routing_flow(graph, null_role_ids)
        null_mechanisms = combine_mechanisms(
            null_functional,
            null_routing,
            counterfactual=None,
        )
        role_sensitive_prefixes = (
            "drift_",
            "dispersion_functional_head_role_js",
            "functional_signed_",
            "functional_absolute_",
            "routing_head_role_js",
            "routing_direct_",
            "routing_relayed_",
            "routing_total_",
            "routing_mean_mass_",
        )
        for name, null_value in null_mechanisms.items():
            if not name.startswith(role_sensitive_prefixes):
                continue
            if name.endswith("_estimator_se"):
                continue
            actual = np.asarray(mechanisms[name], dtype=np.float64)
            null_value = np.asarray(null_value, dtype=np.float64)
            if actual.shape != null_value.shape:
                raise RuntimeError("role-position null trajectory shape changed")
            mechanisms[f"role_position_null__{name}"] = null_value
            mechanisms[f"role_position_adjusted__{name}"] = actual - null_value
        current_token_names, current_token_rows, answer_sources = (
            flatten_token_trajectories(
                mechanisms,
                response_length=alignment.response_length,
                layer_count=layer_count,
            )
        )
        current_answer_names, current_answer_row = aggregate_answer_features(
            answer_sources
        )
        if token_feature_names is None:
            token_feature_names = current_token_names
            answer_feature_names = current_answer_names
        elif (
            token_feature_names != current_token_names
            or answer_feature_names != current_answer_names
        ):
            raise RuntimeError("mechanism feature schema changed between samples")

        response_length = alignment.response_length
        availability = np.asarray(
            [replay_result.variants[name].available for name in COUNTERFACTUAL_VARIANTS],
            dtype=np.bool_,
        )
        answer_sample.append(sample_id)
        answer_source.append(source_id)
        answer_task.append(sample_task)
        answer_generator.append(sample_generator)
        generator_models.append(sample_generator)
        answer_prompt_length.append(response_start)
        answer_response_length.append(response_length)
        answer_rows.append(current_answer_row)
        # ``np.full(..., dtype=str)`` creates ``<U1`` and silently truncates
        # identifiers such as "1472" to "1".  Repeating a scalar preserves
        # the complete inferred Unicode width.
        token_sample.append(_repeat_identifier(sample_id, response_length))
        token_source.append(_repeat_identifier(source_id, response_length))
        token_index.append(np.arange(response_length, dtype=np.int32))
        token_response_length.append(
            np.full(response_length, response_length, dtype=np.int32)
        )
        response_token_id.append(alignment.target_token_id)
        predictor_position.append(alignment.predictor_position)
        cached_query_index.append(alignment.cached_query_index)
        cached_route_available.append(alignment.cached_route_available)
        variant_available.append(
            np.broadcast_to(availability, (response_length, len(availability))).copy()
        )
        token_rows.append(current_token_rows)
        del (
            baseline,
            replay_result,
            functional,
            routing,
            null_functional,
            null_routing,
            null_mechanisms,
            mechanisms,
            graph,
        )

    assert token_feature_names is not None and answer_feature_names is not None
    candidate_directions = preregistered_directions(answer_feature_names)
    primary = _primary_features(answer_feature_names, candidate_directions)
    directions = {
        name: candidate_directions[name] if name in primary else "exploratory"
        for name in answer_feature_names
    }
    onset_candidates = (
        "drift_functional_history_to_grounding_log_ratio__layer_mean",
        "dispersion_functional_entropy_observed__layer_mean",
        "dispersion_functional_cancellation__layer_mean",
        "routing_entropy_upper__layer_mean",
        "routing_total_evidence_ancestry__layer_mean",
        "counterfactual_evidence_bypass",
        "counterfactual_history_necessity",
    )
    onset = tuple(name for name in onset_candidates if name in token_feature_names)
    verified_observer_name = normalized_observer or normalized_replay
    generator_matches = [
        _normalized_model_name(name) == verified_observer_name
        for name in generator_models
    ]
    swap_available_count = sum(bool(selected) for selected in donors.values())
    complete_swap_ensemble_count = sum(
        len(selected) == 3 for selected in donors.values()
    )
    swap_donor_slots = sum(len(selected) for selected in donors.values())
    binding_payload = [
        {
            "sample_id": sample_id,
            **binding,
        }
        for sample_id, binding in zip(sample_ids, attention_bindings, strict=True)
    ]
    attention_binding_summary = {
        "verified_every_answer": len(attention_bindings) == len(sample_ids),
        "absolute_tolerance": float(attention_bindings[0]["absolute_tolerance"]),
        "retained_endpoints_compared": int(
            sum(int(value["retained_endpoints_compared"]) for value in attention_bindings)
        ),
        "diagonal_endpoints_compared": int(
            sum(int(value["diagonal_endpoints_compared"]) for value in attention_bindings)
        ),
        "retained_max_abs_error": float(
            max(float(value["retained_max_abs_error"]) for value in attention_bindings)
        ),
        "diagonal_max_abs_error": float(
            max(float(value["diagonal_max_abs_error"]) for value in attention_bindings)
        ),
        "known_mass_max_abs_error": float(
            max(float(value["known_mass_max_abs_error"]) for value in attention_bindings)
        ),
    }
    metadata: dict[str, object] = {
        "labels_used": False,
        "label_boundary": (
            "capture uses retain_embedded_labels=False, discards the legacy "
            "unified-cache label tensor, and never calls a label API"
        ),
        "data_root": str(Path(data_root).resolve()),
        "split": str(dataset.manifest["split"]),
        "task": task,
        "audit_scope": (
            "complete_split"
            if set(sample_ids) == set(map(str, dataset.sample_ids))
            else "selected_samples"
        ),
        "dataset_manifest_sha256": dataset_manifest_sha256(dataset),
        "role_index_sha256": file_sha256(role_index_path),
        "source_info_sha256": file_sha256(source_info_path),
        "model_fingerprint": model_fingerprint,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "swap_assignment_sha256": _canonical_json_sha256(assignments),
        "attention_binding_sha256": _canonical_json_sha256(binding_payload),
        "attribution_seed_assignment_sha256": _canonical_json_sha256(
            probe_seed_assignments
        ),
        "implementation_sha256": implementation_sha256(),
        "alignment": ALIGNMENT,
        "objective": OBJECTIVE,
        "counterfactual_variants": list(COUNTERFACTUAL_VARIANTS),
        "intervention_specification": {
            "no_evidence": "block evidence attention keys for strict later queries",
            "no_history": (
                "block prior response attention keys; query embedding/residual remains"
            ),
            "swapped_evidence_ensemble": (
                "up to three target-hashed, same-task, source-disjoint, "
                "exact-length donor evidence crops"
            ),
            "targets": "all branches score the unchanged factual response tokens",
        },
        "functional_attribution": {
            "formula": (
                "attention * dot(d_logp_token/d_same_predictor_o_proj_input, "
                "value_state)"
            ),
            "objective_scope": (
                "per-token chosen-logprob Jacobian diagonal; no future-token "
                "gradient is intentionally included"
            ),
            "jacobian_estimator": "iid Rademacher Hutchinson diagonal VJP",
            "gradient_probe_count": int(gradient_probes),
            "global_attribution_seed": int(attribution_seed),
            "sample_seed_scheme": (
                "SHA256(global_seed, complete sample_id, ordered token_ids)"
            ),
            "endpoint_scope": "retained sparse endpoints plus exact diagonal only",
            "unresolved_policy": "coverage only; never imputed as zero contribution",
            "operator_geometry_artifact_used": False,
        },
        "role_position_null": {
            "enabled": True,
            "distance_bin_width": int(role_null_bin_width),
            "permutation": "within distance-to-response bins",
            "preserves": "each prompt-role count within every position bin",
            "diagnostic_only": True,
            "primary_drift_uses_actual_minus_null": False,
            "limitation": (
                "long contiguous evidence spans make many bins role-constant; "
                "the null cannot identify evidence effects perfectly separately "
                "from prompt position"
            ),
        },
        "prompt_partition": "exact_ragtruth_evidence_question_constraint_other",
        "answer_feature_directions": directions,
        "primary_answer_feature_names": list(primary),
        "onset_feature_names": list(onset),
        "mechanism_observability": {
            "routing_drift": True,
            "routing_dispersion": True,
            "functional_contribution": True,
            "counterfactual_evidence_bypass": swap_available_count > 0,
            "evidence_swap_available_answers": swap_available_count,
            "evidence_swap_available_fraction": swap_available_count / len(sample_ids),
            "evidence_swap_complete_three_donor_answers": (
                complete_swap_ensemble_count
            ),
            "evidence_swap_donor_slots_available": swap_donor_slots,
            "prompt_query_attention_rows_observed": False,
            "prompt_to_prompt_role_relay_decomposed": False,
            "transformer_residual_mlp_paths_decomposed": False,
            "parametric_bias_directly_observed": False,
        },
        "cache_replay_attention_binding": attention_binding_summary,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "attention_implementation": "eager",
            "requested_device": device,
            "actual_embedding_device": str(replay._embedding_device()),
            "requested_torch_dtype": torch_dtype,
            "torch_dtype": resolved_torch_dtype,
            "cache_computation_dtype": _normalized_replay_dtype(cache_spec["dtype"]),
            "cache_storage_dtype": str(cache_spec.get("cache_dtype", "")),
            "cache_model_files_sha256_verified": True,
            "cache_files_verified_against_manifest_sha256": True,
            "loaded_model_class": loaded_replay_provenance["model_class"],
            "loaded_parameter_dtype": loaded_replay_provenance["parameter_dtype"],
            "loaded_floating_parameter_tensors": loaded_replay_provenance[
                "floating_parameter_tensors"
            ],
        },
        "observer_generator_audit": {
            "cache_observer_model": cache_observer,
            "replay_checkpoint": str(Path(model_path).resolve()),
            "cache_observer_name_matches_replay": observer_name_matches_replay,
            "checkpoint_identity_verified_without_path_name": True,
            "cache_attention_values_match_replay": True,
            "generator_models": sorted(set(generator_models)),
            "generator_matches_replay_answers": int(sum(generator_matches)),
            "generator_matches_replay_fraction": float(np.mean(generator_matches)),
            "teacher_forced_observer_audit": True,
            "generator_checkpoint_identity_verified": False,
            "original_generator_formation_claim": False,
        },
        "claim_boundary": (
            "A×gradient is local first-order token attribution estimated with "
            "finite stochastic probes. The cache contains response-query rows, "
            "so prompt-to-prompt attention ancestry and residual/MLP path mass "
            "are not separately decomposed. Evidence/history masks measure "
            "sensitivity; evidence-independent persistence is not proof that "
            "knowledge came from model parameters."
        ),
    }
    table = MechanismArtifact(
        sample_id=np.asarray(answer_sample, dtype=str),
        source_id=np.asarray(answer_source, dtype=str),
        task_type=np.asarray(answer_task, dtype=str),
        generator_model=np.asarray(answer_generator, dtype=str),
        prompt_length=np.asarray(answer_prompt_length, dtype=np.int32),
        response_length=np.asarray(answer_response_length, dtype=np.int32),
        answer_feature_names=answer_feature_names,
        answer_feature=np.vstack(answer_rows).astype(np.float32),
        token_sample_id=np.concatenate(token_sample),
        token_source_id=np.concatenate(token_source),
        token_index=np.concatenate(token_index),
        token_response_length=np.concatenate(token_response_length),
        response_token_id=np.concatenate(response_token_id),
        predictor_position=np.concatenate(predictor_position),
        cached_query_index=np.concatenate(cached_query_index),
        cached_route_available=np.concatenate(cached_route_available),
        counterfactual_variant_available=np.concatenate(variant_available),
        token_feature_names=token_feature_names,
        token_feature=np.vstack(token_rows).astype(np.float32),
        metadata=metadata,
    ).validate()
    save_artifact(output_path, table)
    return {
        "artifact": str(Path(output_path).resolve()),
        "sha256": file_sha256(output_path),
        "samples": len(answer_sample),
        "tokens": len(table.token_sample_id),
        "labels_used": False,
    }


__all__ = [
    "aggregate_answer_features",
    "build_roles",
    "capture_mechanisms",
    "checkpoint_fingerprints",
    "flatten_token_trajectories",
    "preregistered_directions",
    "select_samples",
]
