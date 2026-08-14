"""Canonical dataset access for attention-graph experiments.

This module is the single data boundary for research experiments. It owns file
format detection, validated loading, sparse CSR decoding, dense/mean attention
views, metadata, and evaluation-only labels. Experimental code must not parse
canonical NPZ or formal PT attention files directly.

Graph construction, spectral analysis, feature learning, anomaly scoring, and
visualization belong outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch

from cache import AttentionDataset, load_attention_sample, sha256


@dataclass(frozen=True)
class SparseAttentionBlock:
    """A bounded decoded block from canonical response-query CSR rows.

    ``row`` is the global CSR row index. ``query`` is response-relative while
    ``target`` and ``source`` are absolute prompt+response token positions.
    Missing CSR entries remain censored; they are never materialized here.
    """

    row: torch.Tensor
    layer: torch.Tensor
    head: torch.Tensor
    query: torch.Tensor
    target: torch.Tensor
    source: torch.Tensor
    weight: torch.Tensor


@dataclass(frozen=True)
class CensoredCausalAttentionChannel:
    """One reconstructed response-query attention channel.

    The formal cache stores exact diagonal values and off-diagonal response
    rows whose values meet ``attention_floor``. It does *not* store prompt
    query rows or the exact values of censored edges. Consequently this view
    is ``[response query, prompt + response source]`` rather than a fabricated
    square ``[all tokens, all tokens]`` matrix.

    ``values`` contains exact retained values and the requested imputation for
    eligible-but-censored entries. ``observed`` is therefore part of the data
    contract: callers must not mistake an imputed floor value for an observed
    original attention value. ``eligible`` marks the causal support, including
    the diagonal only when it was requested.
    """

    layer: int
    head: int
    response_idx: int
    attention_floor: float
    censored_fill: float
    values: torch.Tensor
    observed: torch.Tensor
    eligible: torch.Tensor

    @property
    def num_response_tokens(self) -> int:
        return int(self.values.shape[0])

    @property
    def num_tokens(self) -> int:
        return int(self.values.shape[1])

    @property
    def prompt_to_response(self) -> torch.Tensor:
        """Causal source-in-prompt mask for response query rows."""

        mask = torch.zeros_like(self.eligible)
        mask[:, : self.response_idx] = True
        return mask & self.eligible

    @property
    def response_to_response(self) -> torch.Tensor:
        """Causal source-in-response mask, including an eligible diagonal."""

        mask = torch.zeros_like(self.eligible)
        mask[:, self.response_idx :] = True
        return mask & self.eligible

    @property
    def censored(self) -> torch.Tensor:
        return self.eligible & ~self.observed

    def excess_over_floor(self) -> torch.Tensor:
        """Return the sparse-equivalent signal above the censoring floor.

        Censored causal entries become zero. Retained off-diagonal edges are
        represented by their margin above ``attention_floor``. Exact diagonal
        values are left unchanged because they were never censored by the
        off-diagonal floor.
        """

        result = torch.zeros_like(self.values)
        retained_off_diagonal = self.observed.clone()
        query = torch.arange(self.num_response_tokens, device=result.device)
        target = self.response_idx + query
        diagonal = retained_off_diagonal[query, target]
        retained_off_diagonal[query, target] = False
        result[retained_off_diagonal] = (
            self.values[retained_off_diagonal] - self.attention_floor
        ).clamp_min(0)
        if bool(diagonal.any()):
            selected = query[diagonal]
            result[selected, target[diagonal]] = self.values[
                selected, target[diagonal]
            ]
        return result

    def square_with_unavailable_pp(self):
        """Place this view in ``[N,N]`` coordinates for inspection only.

        Prompt-query rows are zero in ``values`` but also false in both masks,
        which means *unavailable*, not observed zero. Algorithms should prefer
        the bounded ``[R,N]`` representation and avoid this extra allocation.
        """

        values = torch.zeros(
            (self.num_tokens, self.num_tokens),
            dtype=self.values.dtype,
            device=self.values.device,
        )
        observed = torch.zeros(
            (self.num_tokens, self.num_tokens),
            dtype=torch.bool,
            device=self.values.device,
        )
        eligible = torch.zeros_like(observed)
        values[self.response_idx :] = self.values
        observed[self.response_idx :] = self.observed
        eligible[self.response_idx :] = self.eligible
        return values, observed, eligible


@dataclass(frozen=True)
class CensoredCausalAttentionRow:
    """One streamed row from :class:`CensoredCausalAttentionChannel`."""

    layer: int
    head: int
    query: int
    target: int
    response_idx: int
    attention_floor: float
    censored_fill: float
    values: torch.Tensor
    observed: torch.Tensor
    eligible: torch.Tensor


def open_research_dataset(
    split_root,
    *,
    device="cpu",
    verify_hashes=False,
    retain_embedded_labels=False,
):
    """Open canonical NPZ or the formal sparse PT cache without repacking it."""

    root = Path(split_root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"split has no manifest: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("attention_cache_spec") is not None:
        return FormalResearchDataset(
            root,
            device=device,
            retain_labels=retain_embedded_labels,
        )
    return ResearchDataset(root, device=device, verify_hashes=verify_hashes)


class _ResearchSampleViews:
    """Format-independent attention views shared by all research samples."""

    def iter_sparse_attention_blocks(self, block_rows=4096):
        """Stream decoded sparse response-query entries in bounded row blocks.

        This is the preferred interface for experiments that can operate on
        sparse attention. It avoids allocating ``[L,H,R,N]`` and preserves the
        distinction between retained values and cache-censored entries.
        """

        block_rows = int(block_rows)
        if block_rows < 1:
            raise ValueError("block_rows must be positive")
        attention = self.attention()
        response_count = attention.num_response_tokens
        rows_per_layer = attention.num_heads * response_count
        total_rows = attention.num_channels * response_count
        row_ptr = attention.response_row_ptr.long()
        for row_start in range(0, total_rows, block_rows):
            row_stop = min(row_start + block_rows, total_rows)
            pointer = row_ptr[row_start : row_stop + 1]
            lengths = pointer[1:] - pointer[:-1]
            rows = torch.repeat_interleave(
                torch.arange(
                    row_start,
                    row_stop,
                    dtype=torch.long,
                    device=row_ptr.device,
                ),
                lengths,
            )
            value_start = int(pointer[0].item())
            value_stop = int(pointer[-1].item())
            query = rows.remainder(response_count)
            yield SparseAttentionBlock(
                row=rows,
                layer=torch.div(rows, rows_per_layer, rounding_mode="floor"),
                head=torch.div(
                    rows.remainder(rows_per_layer),
                    response_count,
                    rounding_mode="floor",
                ),
                query=query,
                target=attention.response_idx + query,
                source=attention.response_column_indices[value_start:value_stop].long(),
                weight=attention.response_values[value_start:value_stop],
            )

    def dense_response_channel(
        self,
        layer,
        head,
        *,
        include_diagonal=False,
        dtype=torch.float32,
    ):
        """Return one cache-censored ``[R,N]`` layer/head attention matrix.

        Unretained entries are filled with zero only for this requested dense
        view. They still mean ``<= attention_floor`` rather than known original
        attention of exactly zero. The full multi-channel dense tensor is never
        allocated.
        """

        return self.censored_causal_response_channel(
            layer,
            head,
            include_diagonal=include_diagonal,
            censored_fill=0.0,
            dtype=dtype,
        ).values

    def censored_causal_response_channel(
        self,
        layer,
        head,
        *,
        include_diagonal=True,
        censored_fill="floor",
        dtype=torch.float32,
    ):
        """Restore one bounded causal channel with explicit censoring masks.

        The matrix is ``[R,N]``: every query is a response token and sources
        contain prompt tokens plus the causal response prefix. Prompt-query
        rows (the PP block) cannot be reconstructed because they were never
        stored and are deliberately not fabricated.

        ``censored_fill='floor'`` imputes every legal but unretained
        off-diagonal edge with the cache floor (normally ``0.01``). A numeric
        fill or ``'zero'`` may be requested for an algorithm that treats the
        observed sparse graph separately. In every case ``observed`` and
        ``eligible`` distinguish measurements, censoring, and structural zeros.
        """

        attention = self.attention()
        layer, head = int(layer), int(head)
        if not 0 <= layer < attention.num_layers:
            raise IndexError("layer is outside attention geometry")
        if not 0 <= head < attention.num_heads:
            raise IndexError("head is outside attention geometry")
        if censored_fill == "floor":
            fill = float(attention.attention_floor)
        elif censored_fill == "zero":
            fill = 0.0
        else:
            fill = float(censored_fill)
        if not 0.0 <= fill <= float(attention.attention_floor):
            raise ValueError(
                "censored_fill must be zero, 'floor', or a value in "
                "[0, attention_floor]"
            )

        response_count = attention.num_response_tokens
        device = attention.response_values.device
        query = torch.arange(response_count, device=device)
        target = attention.response_idx + query
        source = torch.arange(attention.num_tokens, device=device)
        eligible = source.unsqueeze(0) < target.unsqueeze(1)
        if include_diagonal:
            eligible[query, target] = True
        values = torch.zeros(
            (response_count, attention.num_tokens), dtype=dtype, device=device
        )
        values[eligible] = fill
        observed = torch.zeros_like(eligible)

        channel = layer * attention.num_heads + head
        row_offset = channel * response_count
        row_ptr = attention.response_row_ptr.long()
        for response_query in range(response_count):
            row = row_offset + response_query
            start = int(row_ptr[row].item())
            stop = int(row_ptr[row + 1].item())
            if stop <= start:
                continue
            columns = attention.response_column_indices[start:stop].long()
            values[response_query, columns] = attention.response_values[
                start:stop
            ].to(dtype=dtype)
            observed[response_query, columns] = True

        if include_diagonal:
            values[query, target] = attention.attention_diagonal[
                layer, head, target
            ].to(dtype=dtype)
            observed[query, target] = True

        return CensoredCausalAttentionChannel(
            layer=layer,
            head=head,
            response_idx=int(attention.response_idx),
            attention_floor=float(attention.attention_floor),
            censored_fill=fill,
            values=values,
            observed=observed,
            eligible=eligible,
        )

    def iter_censored_causal_response_channels(
        self,
        *,
        include_diagonal=True,
        censored_fill="floor",
        dtype=torch.float32,
    ):
        """Yield ``[R,N]`` channels without allocating ``[L,H,R,N]``."""

        attention = self.attention()
        for layer in range(attention.num_layers):
            for head in range(attention.num_heads):
                yield self.censored_causal_response_channel(
                    layer,
                    head,
                    include_diagonal=include_diagonal,
                    censored_fill=censored_fill,
                    dtype=dtype,
                )

    def iter_censored_causal_response_rows(
        self,
        *,
        include_diagonal=True,
        censored_fill="floor",
        dtype=torch.float32,
    ):
        """Yield one reconstructed row at a time in layer/head/query order."""

        for channel in self.iter_censored_causal_response_channels(
            include_diagonal=include_diagonal,
            censored_fill=censored_fill,
            dtype=dtype,
        ):
            for query in range(channel.num_response_tokens):
                yield CensoredCausalAttentionRow(
                    layer=channel.layer,
                    head=channel.head,
                    query=query,
                    target=channel.response_idx + query,
                    response_idx=channel.response_idx,
                    attention_floor=channel.attention_floor,
                    censored_fill=channel.censored_fill,
                    values=channel.values[query],
                    observed=channel.observed[query],
                    eligible=channel.eligible[query],
                )

    def mean_response_attention(
        self,
        *,
        include_diagonal=False,
        dtype=torch.float32,
        block_rows=4096,
    ):
        """Return the channel-mean cache-censored response attention ``[R,N]``.

        This is a reusable data view for spectral/SVD feasibility experiments.
        It averages retained layer/head entries over the *total* channel count,
        so absent channels contribute only through the explicit zero-fill view;
        they must still be interpreted as censored below ``attention_floor``.
        """

        attention = self.attention()
        output = torch.zeros(
            (attention.num_response_tokens, attention.num_tokens),
            dtype=dtype,
            device=attention.response_values.device,
        )
        for block in self.iter_sparse_attention_blocks(block_rows=block_rows):
            if block.weight.numel():
                output.index_put_(
                    (block.query, block.source),
                    block.weight.to(dtype=dtype),
                    accumulate=True,
                )
        output /= float(attention.num_channels)
        if include_diagonal:
            query = torch.arange(attention.num_response_tokens, device=output.device)
            target = attention.response_idx + query
            diagonal = attention.attention_diagonal[
                :, :, attention.response_idx :
            ].to(dtype=dtype).mean(dim=(0, 1))
            output[query, target] = diagonal
        return output


class ResearchDataset:
    """Lazy access to one canonical attention split."""

    def __init__(self, split_root, *, device="cpu", verify_hashes=False):
        self.root = Path(split_root)
        self.device = device
        self.verify_hashes = bool(verify_hashes)
        self.attention_dataset = AttentionDataset(
            self.root, device=device, verify_hashes=verify_hashes
        )
        self.manifest = self.attention_dataset.manifest
        self.rows = {
            str(row["sample_id"]): row for row in self.attention_dataset.rows
        }
        self.source_to_ids: dict[str, list[str]] = {}
        for sample_id, row in self.rows.items():
            self.source_to_ids.setdefault(str(row["source_id"]), []).append(sample_id)

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        for sample_id in self.rows:
            yield self[sample_id]

    def __contains__(self, sample_id):
        return str(sample_id) in self.rows

    def __getitem__(self, sample_id):
        sample_id = str(sample_id)
        if sample_id not in self.rows:
            raise KeyError(sample_id)
        return ResearchSample(self, sample_id)

    @property
    def sample_ids(self):
        return list(self.rows)

    @property
    def source_ids(self):
        return list(self.source_to_ids)

    def samples_from_source(self, source_id):
        return [self[sample_id] for sample_id in self.source_to_ids.get(str(source_id), [])]

    def filter(self, **metadata):
        return [
            self[sample_id]
            for sample_id, row in self.rows.items()
            if all(row.get(key) == value for key, value in metadata.items())
        ]

    def labels(self):
        return LabelStore(self)


class ResearchSample(_ResearchSampleViews):
    """One canonical attention sample plus lightweight metadata."""

    def __init__(self, dataset: ResearchDataset, sample_id: str):
        self.dataset = dataset
        self.sample_id = sample_id
        self.row = dataset.rows[sample_id]
        self._attention = None

    @property
    def source_id(self):
        return str(self.row["source_id"])

    @property
    def split(self):
        return self.row.get("split", self.dataset.manifest.get("split"))

    @property
    def task_type(self):
        return self.row.get("task_type")

    @property
    def data_source(self):
        return self.row.get("data_source")

    @property
    def generator_model(self):
        return self.row.get(
            "generator_model", self.dataset.manifest.get("generator_model")
        )

    @property
    def observer_model(self):
        return self.dataset.manifest.get("observer_model")

    @property
    def temperature(self):
        return self.row.get("temperature")

    @property
    def quality(self):
        return self.row.get("quality")

    @property
    def metadata(self):
        return {
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "split": self.split,
            "task_type": self.task_type,
            "data_source": self.data_source,
            "generator_model": self.generator_model,
            "observer_model": self.observer_model,
            "temperature": self.temperature,
            "quality": self.quality,
        }

    def attention(self):
        if self._attention is not None:
            return self._attention
        path = self.dataset.root / self.row["path"]
        if not path.is_file() or path.stat().st_size != int(self.row["bytes"]):
            raise ValueError("attention sample byte count does not match index")
        if self.dataset.verify_hashes and sha256(path) != self.row["sha256"]:
            raise ValueError("attention sample SHA256 does not match index")
        sample = load_attention_sample(
            path,
            sample_id=self.sample_id,
            source_id=self.source_id,
            attention_floor=self.dataset.attention_dataset.attention_floor,
            device=self.dataset.device,
        )
        if (
            sample.num_layers != int(self.dataset.manifest["num_layers"])
            or sample.num_heads != int(self.dataset.manifest["num_heads"])
        ):
            raise ValueError("attention geometry does not match split manifest")
        self._attention = sample
        return sample

    def release_attention(self):
        self._attention = None


class LabelStore:
    """Evaluation-only token labels from ``labels.jsonl``."""

    def __init__(self, dataset: ResearchDataset):
        self.dataset = dataset
        path = dataset.root / "labels.jsonl"
        if not path.is_file():
            raise ValueError("labels.jsonl is missing")
        expected_hash = dataset.manifest.get("labels_sha256")
        if expected_hash is not None and sha256(path) != expected_hash:
            raise ValueError("labels_sha256 does not match labels.jsonl")
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.rows = {str(row["sample_id"]): row for row in rows}
        if len(self.rows) != len(rows) or set(self.rows) != set(dataset.rows):
            raise ValueError("label/canonical sample ID sets do not match")

    def positive_runs(self, sample_id, *, response_count=None):
        runs = self.rows[str(sample_id)].get("positive_runs", [])
        previous_end = 0
        normalized = []
        if response_count is None:
            response_count = self.dataset[str(sample_id)].attention().num_response_tokens
        for run in runs:
            if len(run) != 2:
                raise ValueError("each positive run must contain [start, end)")
            start, end = map(int, run)
            if not 0 <= start < end <= response_count or start < previous_end:
                raise ValueError("positive runs must be sorted valid response-relative spans")
            normalized.append([start, end])
            previous_end = end
        return normalized

    def response_labels(self, sample: ResearchSample):
        self._check_dataset(sample)
        attention = sample.attention()
        labels = torch.zeros(
            attention.num_response_tokens,
            dtype=torch.long,
            device=attention.token_ids.device,
        )
        for start, end in self.positive_runs(
            sample.sample_id, response_count=attention.num_response_tokens
        ):
            labels[start:end] = 1
        return labels

    def _check_dataset(self, sample: ResearchSample):
        if sample.dataset is not self.dataset:
            raise ValueError("labels belong to a different dataset")


class FormalResearchDataset:
    """Lazy adapter over the existing sparse ``attention_*.pt`` archive.

    No attention tensor is copied or re-serialized. The adapter normalizes the
    in-memory object to ``AttentionSample`` and keeps embedded labels sealed
    until ``labels()`` is explicitly requested after pattern discovery.
    """

    def __init__(self, split_root, *, device="cpu", retain_labels=False):
        from formal_cache import read_formal_manifest

        self.root = Path(split_root)
        formal_manifest, spec, files, split = read_formal_manifest(self.root)
        self.device = device
        self.split_name = split
        self.spec = spec
        self.retain_labels = bool(retain_labels)
        self._label_cache = {}
        self.manifest = {
            "schema": formal_manifest["attention_cache_spec"][
                "attention_cache_schema"
            ],
            "split": split,
            "num_layers": int(spec["num_hidden_layers"]),
            "num_heads": int(spec["num_attention_heads"]),
            "attention_floor": float(spec["attention_floor"]),
            "count": len(files),
            "generator_model": spec.get("generator_model"),
            "observer_model": Path(str(spec.get("model_path", ""))).name or None,
        }
        self.rows = {}
        for path, digest in files:
            stem = path.stem
            sample_id = stem[len("attention_"):] if stem.startswith("attention_") else stem
            if not sample_id or sample_id in self.rows:
                raise ValueError("formal cache file names do not identify unique samples")
            self.rows[sample_id] = {
                "sample_id": sample_id,
                "path": path,
                "sha256": digest,
            }

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        for sample_id in self.rows:
            yield self[sample_id]

    def __contains__(self, sample_id):
        return str(sample_id) in self.rows

    def __getitem__(self, sample_id):
        sample_id = str(sample_id)
        if sample_id not in self.rows:
            raise KeyError(sample_id)
        return FormalResearchSample(self, sample_id)

    @property
    def sample_ids(self):
        return list(self.rows)

    def labels(self):
        if not self.retain_labels:
            raise RuntimeError("embedded labels were not retained for this dataset")
        if len(self._label_cache) != len(self.rows):
            raise RuntimeError(
                "formal labels become available only after every attention sample "
                "has been processed"
            )
        return FormalLabelStore(self)


class FormalResearchSample(_ResearchSampleViews):
    def __init__(self, dataset: FormalResearchDataset, sample_id: str):
        self.dataset = dataset
        self.sample_id = sample_id
        self.row = dataset.rows[sample_id]
        self._attention = None
        self._metadata = None

    def _load(self):
        if self._attention is not None:
            return
        from formal_cache import load_formal_sample

        sample, labels, payload = load_formal_sample(
            self.row["path"],
            self.row["sha256"],
            split=self.dataset.split_name,
            spec=self.dataset.spec,
        )
        if sample.sample_id != self.sample_id:
            raise ValueError(
                "formal cache response_id does not match its attention file name"
            )
        for name in (
            "token_ids",
            "attention_diagonal",
            "response_row_ptr",
            "response_column_indices",
            "response_values",
        ):
            setattr(sample, name, getattr(sample, name).to(self.dataset.device))
        self._attention = sample
        self._metadata = {
            "source_id": sample.source_id,
            "task_type": payload.get("task_type", self.dataset.spec.get("task_type")),
            "data_source": payload.get(
                "data_source", payload.get("source", self.dataset.spec.get("data_source"))
            ),
            "generator_model": payload.get(
                "generator_model", self.dataset.spec.get("generator_model")
            ),
            "temperature": payload.get("temperature"),
            "quality": payload.get("quality"),
        }
        if self.dataset.retain_labels:
            self.dataset._label_cache[self.sample_id] = (
                labels[sample.response_idx:].to(dtype=torch.long, device="cpu")
            )

    @property
    def source_id(self):
        self._load()
        return str(self._metadata["source_id"])

    @property
    def split(self):
        return self.dataset.split_name

    @property
    def task_type(self):
        self._load()
        return self._metadata["task_type"]

    @property
    def data_source(self):
        self._load()
        return self._metadata["data_source"]

    @property
    def generator_model(self):
        self._load()
        return self._metadata["generator_model"]

    @property
    def observer_model(self):
        return self.dataset.manifest.get("observer_model")

    @property
    def temperature(self):
        self._load()
        return self._metadata["temperature"]

    @property
    def quality(self):
        self._load()
        return self._metadata["quality"]

    @property
    def metadata(self):
        return {
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "split": self.split,
            "task_type": self.task_type,
            "data_source": self.data_source,
            "generator_model": self.generator_model,
            "observer_model": self.observer_model,
            "temperature": self.temperature,
            "quality": self.quality,
        }

    def attention(self):
        self._load()
        return self._attention

    def release_attention(self):
        self._attention = None


class FormalLabelStore:
    def __init__(self, dataset: FormalResearchDataset):
        self.dataset = dataset

    def response_labels(self, sample: FormalResearchSample):
        if sample.dataset is not self.dataset:
            raise ValueError("labels belong to a different dataset")
        return self.dataset._label_cache[sample.sample_id]
