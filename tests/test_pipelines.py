"""Spec §3.1 / §3.2 — graph_embedding returns a LayoutResult, and instruments the run."""

import networkx as nx
import numpy as np
import pytest

import twin.graphmatrix as gm
from twin import embedding as emb
from twin.pipelines import batch_graph_embedding, graph_embedding
from twin.results import LayoutResult

REPRESENTATIONS = [
    ("vertex", gm.VertexMatrix),
    ("edge", gm.EdgeMatrix),
    ("twin", gm.TwinEmbeddingMatrix),
]


@pytest.mark.parametrize(("name", "cls"), REPRESENTATIONS)
def test_returns_layout_result_in_canonical_orientation(karate, name, cls):
    res = graph_embedding(karate, representation=cls, embedding=emb.SpringLayout)

    assert isinstance(res, LayoutResult)
    assert res.V.shape == (karate.number_of_nodes(), 2)
    assert res.E.shape == (karate.number_of_edges(), 2)
    assert res.slopes.shape == (karate.number_of_edges(),)


@pytest.mark.parametrize(("name", "cls"), REPRESENTATIONS)
def test_n_points_reflects_what_was_actually_laid_out(karate, name, cls):
    """Spec §3.2: n for A_V, m for A_E, n+m for A_tne, visible without recomputation."""
    n, m = karate.number_of_nodes(), karate.number_of_edges()
    expected = {"vertex": n, "edge": m, "twin": n + m}[name]

    res = graph_embedding(karate, representation=cls, embedding=emb.SpringLayout)
    assert res.n_points == expected


@pytest.mark.parametrize(("name", "cls"), REPRESENTATIONS)
def test_provenance_is_carried_into_the_result(karate, name, cls):
    res = graph_embedding(karate, representation=cls, embedding=emb.SpringLayout)
    assert res.point_provenance == cls.point_provenance


def test_timings_are_split_between_representation_and_embed(karate):
    res = graph_embedding(
        karate, representation=gm.TwinEmbeddingMatrix, embedding=emb.SpringLayout
    )
    assert res.representation_s > 0
    assert res.runtime_s > 0
    # Distinct measurements, not the same clock reading twice.
    assert res.representation_s != res.runtime_s


def test_memory_is_not_measured_unless_asked(karate):
    assert graph_embedding(karate, embedding=emb.SpringLayout).peak_mem_mb is None

    res = graph_embedding(karate, embedding=emb.SpringLayout, measure_memory=True)
    assert res.peak_mem_mb is not None and res.peak_mem_mb >= 0


def test_records_identity_and_parameters(karate):
    res = graph_embedding(
        karate,
        representation=gm.TwinEmbeddingMatrix,
        embedding=emb.SGtSNELayout,
        representation_kw={"alpha": 0.5, "k": 3},
        embedding_kw={"lambda_par": 2.0},
        representation_name="twin",
        embedding_name="sgtsne",
        seed=7,
    )
    assert (res.representation, res.embedding, res.seed) == ("twin", "sgtsne", 7)
    assert res.params["representation_kw"] == {"alpha": 0.5, "k": 3}
    assert res.params["embedding_kw"] == {"lambda_par": 2.0}


def test_undeclared_provenance_is_refused(karate):
    class UndeclaredMatrix(gm.VertexMatrix):
        point_provenance = None

    with pytest.raises(TypeError, match="does not declare point_provenance"):
        graph_embedding(karate, representation=UndeclaredMatrix, embedding=emb.SpringLayout)


def test_batch_returns_layout_results_with_registry_names():
    spec = [
        {
            "graph": "karate_club",
            "representation": ["vertex", "edge", "twin"],
            "embedding": "spring",
        }
    ]
    results = batch_graph_embedding(spec)

    assert len(results) == 3
    assert all(isinstance(r, LayoutResult) for r in results.values())
    assert {r.representation for r in results.values()} == {"vertex", "edge", "twin"}
    assert {r.embedding for r in results.values()} == {"spring"}
    assert {r.n_points for r in results.values()} == {34, 78, 34 + 78}
