"""Dataset pipeline for attention-only multiplex representations."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
from tqdm.auto import tqdm

from .representation import (
    REPRESENTATION_SCHEMA,
    MultiplexConfig,
    represent_attention_multiplex,
)


MANIFEST_SCHEMA = "attention-dynamic-multiplex-dataset-v1"
RUN_STATE_SCHEMA = "attention-dynamic-multiplex-run-state-v1"


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _output_lock(output_dir: Path):
    """Prevent two new runners from writing the same split concurrently."""

    path = output_dir / ".attention_multiplex.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another attention-multiplex process owns {output_dir}"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _json_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _selected_ids(dataset, limit=None):
    sample_ids = list(dataset.sample_ids)
    if limit is not None:
        limit = int(limit)
        if limit < 1:
            raise ValueError("limit must be positive")
        sample_ids = sample_ids[:limit]
    if not sample_ids:
        raise ValueError("no attention samples selected")
    return sample_ids


def _sample_id_digest(sample_ids) -> str:
    payload = json.dumps(list(map(str, sample_ids)), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_contract(dataset, sample_ids, config: MultiplexConfig):
    return {
        "schema": RUN_STATE_SCHEMA,
        "representation_schema": REPRESENTATION_SCHEMA,
        "attention_root": str(Path(dataset.root).resolve()),
        "split": str(getattr(dataset, "split_name", dataset.manifest.get("split"))),
        "sample_count": len(sample_ids),
        "sample_ids_sha256": _sample_id_digest(sample_ids),
        "rank": int(config.rank),
        "random_seed": int(config.random_seed),
        "include_diagonal": bool(config.include_diagonal),
    }


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _validate_run_state(path: Path, contract, *, resume: bool) -> None:
    if not path.exists():
        return
    state = _load_json(path)
    if not isinstance(state, dict):
        raise ValueError(f"invalid run state: {path}")
    mismatches = [key for key, value in contract.items() if state.get(key) != value]
    if mismatches:
        raise ValueError(
            "output directory belongs to an incompatible run; mismatched "
            + ", ".join(mismatches)
        )
    if not resume:
        raise FileExistsError(
            f"output directory already contains a run: {path.parent}; use --resume"
        )


def _scalar(archive, name):
    value = np.asarray(archive[name])
    if value.size != 1:
        raise ValueError(f"{name} is not scalar")
    return value.reshape(()).item()


def _existing_index_row(
    output_path: Path,
    output_dir: Path,
    sample_id: str,
    config: MultiplexConfig,
    geometry,
):
    """Validate an atomic artifact and recover the row needed for resume."""

    required = {
        "schema",
        "labels_included",
        "sample_id",
        "source_id",
        "response_idx",
        "token_ids",
        "mass_query_by_layer",
        "mass_source_by_head",
        "mass_singular_values",
        "mass_captured_energy",
        "shape_query_by_layer",
        "shape_source_by_head",
        "shape_singular_values",
        "shape_captured_energy",
        "self_attention",
        "retained_row_mass",
        "unresolved_row_mass",
        "attention_floor",
    }
    try:
        with np.load(output_path, allow_pickle=False) as archive:
            if required.difference(archive.files):
                return None
            if str(_scalar(archive, "schema")) != REPRESENTATION_SCHEMA:
                return None
            if bool(_scalar(archive, "labels_included")):
                return None
            if str(_scalar(archive, "sample_id")) != str(sample_id):
                return None
            mass_rank = int(np.asarray(archive["mass_singular_values"]).size)
            shape_rank = int(np.asarray(archive["shape_singular_values"]).size)
            response_idx = int(_scalar(archive, "response_idx"))
            tokens = int(np.asarray(archive["token_ids"]).size)
            if not 0 <= response_idx < tokens:
                return None
            expected_rank = min(
                int(config.rank),
                int(geometry[0]) * (tokens - response_idx),
                int(geometry[1]) * tokens,
            )
            if mass_rank != expected_rank or shape_rank != expected_rank:
                return None
            if "config_rank" in archive.files and int(
                _scalar(archive, "config_rank")
            ) != int(config.rank):
                return None
            if "config_seed" in archive.files and int(
                _scalar(archive, "config_seed")
            ) != int(config.random_seed):
                return None
            if "config_include_diagonal" in archive.files and bool(
                _scalar(archive, "config_include_diagonal")
            ) != bool(config.include_diagonal):
                return None
            task_type = (
                str(_scalar(archive, "task_type"))
                if "task_type" in archive.files
                else ""
            )
            data_source = (
                str(_scalar(archive, "data_source"))
                if "data_source" in archive.files
                else ""
            )
            retained_edges = (
                int(_scalar(archive, "retained_off_diagonal_edges"))
                if "retained_off_diagonal_edges" in archive.files
                else None
            )
            return {
                "sample_id": str(sample_id),
                "source_id": str(_scalar(archive, "source_id")),
                "path": output_path.relative_to(output_dir).as_posix(),
                "response_tokens": tokens - response_idx,
                "tokens": tokens,
                "retained_off_diagonal_edges": retained_edges,
                "mass_captured_energy": float(
                    _scalar(archive, "mass_captured_energy")
                ),
                "shape_captured_energy": float(
                    _scalar(archive, "shape_captured_energy")
                ),
                "task_type": task_type or None,
                "data_source": data_source or None,
            }
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def _write_index(output_dir: Path, sample_ids, rows) -> Path:
    path = output_dir / "index.jsonl"
    text = "".join(
        json.dumps(rows[sample_id], sort_keys=True) + "\n"
        for sample_id in sample_ids
        if sample_id in rows
    )
    _atomic_text(path, text)
    return path


def _write_state(path: Path, contract, *, state: str, completed: int, adopted: int):
    payload = {
        **contract,
        "state": state,
        "completed_samples": int(completed),
        "adopted_pre_checkpoint_samples": int(adopted),
        "updated_unix_time": time.time(),
    }
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _encode_sample(dataset, sample_id, output_dir: Path, config: MultiplexConfig):
    sample = dataset[sample_id]
    started = time.perf_counter()
    try:
        attention = sample.attention()
        representation = represent_attention_multiplex(sample, config=config)
        output_path = output_dir / "samples" / f"multiplex_{sample_id}.npz"
        token_ids = attention.token_ids.detach().cpu().numpy().astype(
            np.int64, copy=False
        )
        task_type = _json_value(sample.task_type)
        data_source = _json_value(sample.data_source)
        _atomic_npz(
            output_path,
            schema=np.asarray(REPRESENTATION_SCHEMA),
            labels_included=np.asarray(False),
            sample_id=np.asarray(str(sample_id)),
            source_id=np.asarray(str(sample.source_id)),
            response_idx=np.asarray(representation.response_idx, dtype=np.int32),
            token_ids=token_ids,
            config_rank=np.asarray(config.rank, dtype=np.int32),
            config_seed=np.asarray(config.random_seed, dtype=np.int64),
            config_include_diagonal=np.asarray(config.include_diagonal),
            mass_query_by_layer=representation.mass.query_by_layer,
            mass_source_by_head=representation.mass.source_by_head,
            mass_singular_values=representation.mass.singular_values,
            mass_captured_energy=np.asarray(
                representation.mass.captured_energy, dtype=np.float32
            ),
            shape_query_by_layer=representation.shape.query_by_layer,
            shape_source_by_head=representation.shape.source_by_head,
            shape_singular_values=representation.shape.singular_values,
            shape_captured_energy=np.asarray(
                representation.shape.captured_energy, dtype=np.float32
            ),
            self_attention=representation.self_attention,
            retained_row_mass=representation.retained_row_mass,
            unresolved_row_mass=representation.unresolved_row_mass,
            retained_off_diagonal_edges=np.asarray(
                representation.retained_off_diagonal_edges, dtype=np.int64
            ),
            attention_floor=np.asarray(attention.attention_floor, dtype=np.float32),
            task_type=np.asarray("" if task_type is None else str(task_type)),
            data_source=np.asarray("" if data_source is None else str(data_source)),
        )
        row = {
            "sample_id": str(sample_id),
            "source_id": str(sample.source_id),
            "path": output_path.relative_to(output_dir).as_posix(),
            "response_tokens": int(attention.num_response_tokens),
            "tokens": int(attention.num_tokens),
            "retained_off_diagonal_edges": int(
                representation.retained_off_diagonal_edges
            ),
            "mass_captured_energy": float(representation.mass.captured_energy),
            "shape_captured_energy": float(representation.shape.captured_energy),
            "task_type": task_type,
            "data_source": data_source,
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        _atomic_text(
            output_path.with_suffix(".json"),
            json.dumps(row, sort_keys=True) + "\n",
        )
        return row
    finally:
        sample.release_attention()


def build_dataset_representations(
    dataset,
    output_dir,
    *,
    config: MultiplexConfig | None = None,
    limit=None,
    resume: bool = False,
    workers: int = 1,
    checkpoint_every: int = 10,
):
    """Encode a split without opening or serializing hallucination labels."""

    config = MultiplexConfig() if config is None else config
    config.validate()
    workers = int(workers)
    checkpoint_every = int(checkpoint_every)
    if workers < 1:
        raise ValueError("workers must be positive")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    output_dir = Path(output_dir)
    node_dir = output_dir / "samples"
    node_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = list(map(str, _selected_ids(dataset, limit)))
    contract = _run_contract(dataset, sample_ids, config)
    state_path = output_dir / "run_state.json"
    geometry = (
        int(dataset.manifest["num_layers"]),
        int(dataset.manifest["num_heads"]),
    )

    with _output_lock(output_dir):
        _validate_run_state(state_path, contract, resume=bool(resume))
        existing_paths = list(node_dir.glob("multiplex_*.npz"))
        if existing_paths and not resume:
            raise FileExistsError(
                f"{output_dir} already contains sample artifacts; use --resume"
            )

        rows = {}
        adopted = 0
        if resume:
            for sample_id in tqdm(
                sample_ids,
                desc="validate checkpoints",
                unit="sample",
                leave=False,
            ):
                output_path = node_dir / f"multiplex_{sample_id}.npz"
                if not output_path.is_file():
                    continue
                row = _existing_index_row(
                    output_path, output_dir, sample_id, config, geometry
                )
                if row is not None:
                    sidecar = output_path.with_suffix(".json")
                    sidecar_row = _load_json(sidecar) if sidecar.is_file() else None
                    if (
                        isinstance(sidecar_row, dict)
                        and sidecar_row.get("sample_id") == sample_id
                        and sidecar_row.get("path") == row["path"]
                    ):
                        rows[sample_id] = sidecar_row
                    else:
                        rows[sample_id] = row
                    if not sidecar.is_file():
                        adopted += 1

        index_path = _write_index(output_dir, sample_ids, rows)
        _write_state(
            state_path,
            contract,
            state="running",
            completed=len(rows),
            adopted=adopted,
        )
        pending = [sample_id for sample_id in sample_ids if sample_id not in rows]
        resumed_count = len(rows)
        completed_since_checkpoint = 0

        def record(sample_id, row):
            nonlocal completed_since_checkpoint
            rows[sample_id] = row
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= checkpoint_every:
                _write_index(output_dir, sample_ids, rows)
                _write_state(
                    state_path,
                    contract,
                    state="running",
                    completed=len(rows),
                    adopted=adopted,
                )
                completed_since_checkpoint = 0

        try:
            with tqdm(
                total=len(sample_ids),
                initial=len(rows),
                desc="attention multiplex",
                unit="sample",
            ) as progress:
                if workers == 1:
                    for sample_id in pending:
                        record(
                            sample_id,
                            _encode_sample(dataset, sample_id, output_dir, config),
                        )
                        progress.update()
                else:
                    executor = ThreadPoolExecutor(max_workers=workers)
                    futures = {}
                    try:
                        futures = {
                            executor.submit(
                                _encode_sample,
                                dataset,
                                sample_id,
                                output_dir,
                                config,
                            ): sample_id
                            for sample_id in pending
                        }
                        for future in as_completed(futures):
                            sample_id = futures[future]
                            record(sample_id, future.result())
                            progress.update()
                    except BaseException:
                        for future in futures:
                            future.cancel()
                        executor.shutdown(wait=True, cancel_futures=True)
                        raise
                    else:
                        executor.shutdown(wait=True)
        except BaseException:
            _write_index(output_dir, sample_ids, rows)
            _write_state(
                state_path,
                contract,
                state="interrupted",
                completed=len(rows),
                adopted=adopted,
            )
            raise

        index_path = _write_index(output_dir, sample_ids, rows)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "representation_schema": REPRESENTATION_SCHEMA,
            "state": "complete",
            "labels_included": False,
            "samples": len(rows),
            "layers": geometry[0],
            "heads": geometry[1],
            "rank": int(config.rank),
            "include_diagonal": bool(config.include_diagonal),
            "attention_input": "canonical ResearchDataset/ResearchSample only",
            "query_role": "layer x response_token",
            "source_role": "head x prompt_plus_response_token",
            "pp_policy": "not_available_not_fabricated",
            "censoring_policy": (
                "legal missing edges equal attention_floor in the reconstructed view; "
                "spectral input stores only excess over that floor"
            ),
            "cross_sample_alignment": False,
            "resumable": True,
            "workers": workers,
            "index": index_path.name,
        }
        _atomic_text(
            output_dir / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
        _write_state(
            state_path,
            contract,
            state="complete",
            completed=len(rows),
            adopted=adopted,
        )
        return {
            "output_dir": str(output_dir),
            "manifest": str(output_dir / "manifest.json"),
            "run_state": str(state_path),
            "index": str(index_path),
            "sample_directory": str(node_dir),
            "samples": len(rows),
            "resumed_samples": resumed_count,
            "computed_samples": len(pending),
            "layers": geometry[0],
            "heads": geometry[1],
            "rank": int(config.rank),
            "workers": workers,
        }
