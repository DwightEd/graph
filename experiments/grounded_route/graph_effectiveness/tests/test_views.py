import numpy as np

from experiments.grounded_route.graph_effectiveness.tests.helpers import write_bundle
from experiments.grounded_route.graph_effectiveness.views import load_embedding_views


def test_independently_encoded_variants_align_only_final_node_embeddings(tmp_path):
    real = write_bundle(tmp_path / "real", variant="real")
    rewired = write_bundle(
        tmp_path / "rewired",
        variant="endpoint_rewire",
        embedding_shift=0.5,
    )
    views = load_embedding_views(
        {
            "real": real,
            "endpoint_rewire": rewired,
        }
    )

    assert views.variants == ("real", "endpoint_rewire")
    assert views.embedding("real").shape == (25, 8)
    assert np.allclose(
        views.embedding("endpoint_rewire") - views.embedding("real"),
        0.5,
    )
    assert views.views["endpoint_rewire"].changed_fraction == 0.25


def test_no_message_view_is_a_separately_encoded_real_graph(tmp_path):
    real = write_bundle(tmp_path / "real")
    no_message = write_bundle(
        tmp_path / "no_message",
        message_mode="row_local",
        embedding_shift=0.25,
    )
    views = load_embedding_views({"real": real, "no_message": no_message})

    assert views.views["no_message"].graph_variant == "real"
    assert views.views["no_message"].message_mode == "row_local"
    assert np.allclose(views.embedding("no_message") - views.embedding("real"), 0.25)
