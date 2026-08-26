import numpy as np

from experiments.grounded_route.evaluation import data as data_module
from experiments.grounded_route.evaluation.data import EmbeddingTable, align_table


def table(order):
    return EmbeddingTable(
        sample_id=np.asarray(["a", "a", "b"])[order],
        source_id=np.asarray(["x", "x", "y"])[order],
        token_index=np.asarray([0, 1, 0], dtype=np.int32)[order],
        response_length=np.asarray([2, 2, 1], dtype=np.int32)[order],
        response_token_id=np.asarray([10, 11, 12], dtype=np.int64)[order],
        embedding=np.eye(3, dtype=np.float32)[order],
    )


def test_align_table_restores_reference_token_order():
    reference = table(np.asarray([0, 1, 2]))
    candidate = table(np.asarray([2, 0, 1]))
    aligned = align_table(reference, candidate)
    assert np.array_equal(aligned.sample_id, reference.sample_id)
    assert np.array_equal(aligned.token_index, reference.token_index)


def test_load_labels_reads_only_index_samples_and_preserves_row_order(monkeypatch):
    class LabelVector:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.int8)

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Sample:
        def __init__(self, sample_id):
            self.sample_id = sample_id

        def release_attention(self):
            pass

    class Labels:
        def response_labels(self, sample):
            return LabelVector({"a": [0, 1], "b": [1]}[sample.sample_id])

    class Dataset:
        def __init__(self):
            self.selected = None

        def prepare_evaluation_labels(self, sample_ids):
            self.selected = sample_ids
            return Labels()

        def __getitem__(self, sample_id):
            return Sample(sample_id)

    dataset = Dataset()
    monkeypatch.setattr(
        data_module,
        "open_research_dataset",
        lambda *args, **kwargs: dataset,
    )
    embeddings = table(np.asarray([2, 1, 0]))

    labels = data_module.load_labels(embeddings, "split")

    assert dataset.selected == ["b", "a"]
    np.testing.assert_array_equal(labels, [1, 1, 0])
