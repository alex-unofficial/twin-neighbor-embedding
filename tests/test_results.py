"""Spec §3.1 / §3.3 — the (n, d) boundary, and the twin block ordering."""

import networkx as nx
import numpy as np
import pytest

import twin.graphmatrix as gm
from twin.datasets import Dataset, Unweighted
from twin.results import LayoutResult

VERTEX_PROVENANCE = {"V": "primary", "E": "derived"}


def _dataset(G) -> Dataset:
    """Wrap a bare graph so a shape test does not have to reach for a registered dataset."""
    return Dataset(name="fixture", graph=G, affinity=Unweighted(), distance=Unweighted())


def _raw(G, d=2, seed=0):
    """(d, n) and (d, m) arrays as the embedding layer returns them."""
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(d, G.number_of_nodes())),
        rng.normal(size=(d, G.number_of_edges())),
        rng.normal(size=G.number_of_edges()),
    )


def _make(G, **kw):
    V, E, S = _raw(G)
    return LayoutResult.from_raw(
        V=V,
        E=E,
        slopes=S,
        dataset=_dataset(G),
        representation="vertex",
        embedding="sgtsne",
        point_provenance=VERTEX_PROVENANCE,
        **kw,
    )


# --------------------------------------------------------------------- shapes


def test_from_raw_transposes_once(karate):
    V, E, S = _raw(karate)
    r = _make(karate)
    assert r.V.shape == (karate.number_of_nodes(), 2)
    assert r.E.shape == (karate.number_of_edges(), 2)
    assert np.array_equal(r.V, V.T)
    assert np.array_equal(r.E, E.T)
    assert (r.n, r.m, r.d) == (34, 78, 2)


def test_from_raw_rejects_already_transposed_input(karate):
    """The failure the sniffing convention used to paper over."""
    V, E, S = _raw(karate)
    with pytest.raises(ValueError, match=r"expected \(d, 34\)"):
        LayoutResult.from_raw(
            V=V.T,
            E=E,
            slopes=S,
            dataset=_dataset(karate),
            representation="vertex",
            embedding="sgtsne",
            point_provenance=VERTEX_PROVENANCE,
        )


def test_square_input_is_not_sniffed():
    """n == d is exactly where shape inspection guesses wrong. It must not guess at all."""
    G = nx.path_graph(2)  # n = 2, so a (2, 2) array is ambiguous
    V = np.array([[0.0, 1.0], [2.0, 3.0]])  # (d=2, n=2)
    r = LayoutResult.from_raw(
        V=V,
        E=None,
        slopes=None,
        dataset=_dataset(G),
        representation="vertex",
        embedding="spring",
        point_provenance=VERTEX_PROVENANCE,
    )
    assert np.array_equal(r.V, V.T), "from_raw must transpose unconditionally, not conditionally"


def test_rejects_wrong_edge_count(karate):
    V, E, S = _raw(karate)
    with pytest.raises(ValueError, match=r"expected \(d, 78\)"):
        LayoutResult.from_raw(
            V=V,
            E=E[:, :10],
            slopes=S,
            dataset=_dataset(karate),
            representation="vertex",
            embedding="sgtsne",
            point_provenance=VERTEX_PROVENANCE,
        )


def test_edge_points_may_be_absent(karate):
    V, _, _ = _raw(karate)
    r = LayoutResult.from_raw(
        V=V,
        E=None,
        slopes=None,
        dataset=_dataset(karate),
        representation="vertex",
        embedding="spring",
        point_provenance={"V": "primary"},
    )
    assert r.E is None


def test_the_result_carries_the_dataset_not_just_the_graph(karate):
    """A metric or a figure is a function of the embedded points plus a `Dataset` (§3.4)."""
    r = _make(karate)
    assert r.dataset.name == "fixture"
    assert r.G is r.dataset.graph


# ---------------------------------------------------------------- provenance


def test_provenance_per_representation():
    """The pre-restructure classes, which `paper_figures` uses until step 10."""
    assert gm.VertexMatrix.point_provenance == {"V": "primary", "E": "derived"}
    assert gm.EdgeMatrix.point_provenance == {"V": "derived", "E": "primary"}
    assert gm.TwinEmbeddingMatrix.point_provenance == {"V": "primary", "E": "primary"}
    assert gm.GraphMatrixMethod.point_provenance is None


def test_provenance_lookup_and_validation(karate):
    r = _make(karate)
    assert r.provenance_of("V") == "primary"
    assert r.provenance_of("E") == "derived"

    with pytest.raises(ValueError, match="primary/derived"):
        LayoutResult(
            V=r.V,
            E=None,
            slopes=None,
            dataset=_dataset(karate),
            representation="x",
            embedding="y",
            point_provenance={"V": "laid-out"},
        )


def test_provenance_missing_key_raises_clearly(karate):
    r = LayoutResult(
        V=_raw(karate)[0].T,
        E=None,
        slopes=None,
        dataset=_dataset(karate),
        representation="x",
        embedding="y",
        point_provenance={"V": "primary"},
    )
    with pytest.raises(KeyError, match="not declared"):
        r.provenance_of("E")


# ------------------------------------------------------- twin block ordering


def test_twin_block_ordering_is_edges_then_vertices(karate):
    """Spec §3.3: twin_adjacency builds [[0, Bt], [B, 0]] -> rows are [edges(m); vertices(n)].

    Easy to invert, and inverting it produces a plausible-looking but wrong layout. Tag each
    column of a synthetic (d, n+m) layout with its index and check which half comes back as
    which point set.
    """
    Gm = gm.TwinEmbeddingMatrix(karate)
    nv, ne = Gm.incidence.shape

    tagged = np.vstack([np.arange(ne + nv, dtype=float), np.zeros(ne + nv)])
    X_v, X_e, _ = Gm.vertex_and_edge_embeddings(tagged)

    assert np.array_equal(X_e[0], np.arange(ne)), "edge points must be the first ne columns"
    assert np.array_equal(X_v[0], np.arange(ne, ne + nv)), "vertices must be the last nv columns"


def test_common_init_slicing_is_columnwise(karate):
    """Appendix A is authoritative over the §3.3 code block: slice columns, not rows."""
    from twin import embedding as emb

    Gm = gm.TwinEmbeddingMatrix(karate)
    nv, ne = Gm.incidence.shape
    d = 2

    y0 = emb.SGtSNELayout(seed=0, d=d).init_embedding(nv + ne, d)
    assert y0.shape == (d, nv + ne)

    y0_edge, y0_vertex = y0[:, :ne], y0[:, ne:]
    assert y0_edge.shape == (d, ne)
    assert y0_vertex.shape == (d, nv)

    # The §3.3 code block sliced rows. On a (d, n+m) array that slices the *dimension* axis:
    # y0[:ne] keeps all d rows and every column, and y0[ne:] is empty.
    assert y0[:ne].shape == (d, ne + nv)
    assert y0[ne:].shape == (0, ne + nv)


def test_incidence_column_order_matches_graph_edges(karate):
    """Spec §0.5 (4): assert it, do not rely on it.

    Edge points are indexed by incidence-matrix column; every consumer that pairs an edge point
    with an edge (curve construction, slopes, crossings) assumes those agree with G.edges().
    """
    Gm = gm.VertexMatrix(karate)
    B = Gm.incidence.tocsc()
    for col, (u, v) in enumerate(karate.edges()):
        assert set(B[:, [col]].nonzero()[0]) == {u, v}, f"column {col} is not edge {(u, v)}"
