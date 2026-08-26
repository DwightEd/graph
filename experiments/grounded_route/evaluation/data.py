"""Load saved node embeddings and align construction variants."""

from dataclasses import dataclass

import numpy as np

from research_dataset import open_research_dataset


@dataclass(frozen=True)
class EmbeddingTable:
    sample_id: np.ndarray
    source_id: np.ndarray
    token_index: np.ndarray
    response_length: np.ndarray
    response_token_id: np.ndarray
    embedding: np.ndarray

    @classmethod
    def load(cls, path) -> "EmbeddingTable":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                sample_id=data["sample_id"].astype(str),
                source_id=data["source_id"].astype(str),
                token_index=data["token_index"].astype(np.int32),
                response_length=data["response_length"].astype(np.int32),
                response_token_id=data["response_token_id"].astype(np.int64),
                embedding=data["embedding"].astype(np.float32),
            )

    def select(self, order: np.ndarray) -> "EmbeddingTable":
        return EmbeddingTable(
            sample_id=self.sample_id[order],
            source_id=self.source_id[order],
            token_index=self.token_index[order],
            response_length=self.response_length[order],
            response_token_id=self.response_token_id[order],
            embedding=self.embedding[order],
        )

    def keys(self) -> list[tuple[str, int]]:
        return list(zip(self.sample_id.tolist(), self.token_index.tolist()))


def align_table(reference: EmbeddingTable, table: EmbeddingTable) -> EmbeddingTable:
    location = {key: row for row, key in enumerate(table.keys())}
    order = np.asarray([location[key] for key in reference.keys()], dtype=np.int64)
    return table.select(order)


def load_variants(paths: dict[str, str]) -> dict[str, EmbeddingTable]:
    tables = {name: EmbeddingTable.load(path) for name, path in paths.items()}
    reference = tables["real"]
    return {
        name: reference if name == "real" else align_table(reference, table)
        for name, table in tables.items()
    }


def load_labels(table: EmbeddingTable, test_root: str) -> np.ndarray:
    dataset = open_research_dataset(
        test_root,
        device="cpu",
        retain_embedded_labels=True,
    )
    label_store = dataset.prepare_evaluation_labels()
    labels = np.empty(len(table.sample_id), dtype=np.int8)

    for sample_id in dict.fromkeys(table.sample_id.tolist()):
        rows = np.flatnonzero(table.sample_id == sample_id)
        sample = dataset[sample_id]
        try:
            current = label_store.response_labels(sample).cpu().numpy().astype(np.int8)
            labels[rows] = current[table.token_index[rows]]
        finally:
            sample.release_attention()
    return labels
