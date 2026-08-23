"""Spec §3.4 / §3.5 — `LayoutGraph`, the three mappings, and where derivation happens.

The half of the layering these hold: a `LayoutGraph` is laid out and is **never labelled**, its
affinity and distance forms are both always available, and a missing side is derived exactly
once, in `vertex_graph`.
"""

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

import twin.graphmatrix as gm
from twin.config import REAL_WORLD_DIR
from twin.datasets import Dataset, Stored, StoredMatrix, Unweighted, load_dataset
from twin.layoutgraph import (
    LayoutGraph,
    affinity_matrix_to_distance,
    edge_graph,
    twin_graph,
    vertex_graph,
)

TABULA = REAL_WORLD_DIR / "tabula-liver.mat"
FOOTBALL = REAL_WORLD_DIR / "football_2023.mat"


@pytest.fixture
def strength_dataset() -> Dataset:
    """A 5-path with strengths 1..4: one stored side, so the distance side is derived."""
    G = nx.path_graph(5)
    for k, (u, v) in enumerate(G.edges()):
        G[u][v]["weight"] = float(k + 1)
    return Dataset(name="strengths", graph=G, affinity=Stored(meaning="strength"))


@pytest.fixture
def combinatorial_dataset() -> Dataset:
    return Dataset(
        name="star", graph=nx.star_graph(4), affinity=Unweighted(), distance=Unweighted()
    )


def edge_weights(G: nx.Graph) -> np.ndarray:
    return np.array([d["weight"] for _, _, d in G.edges(data=True)])


# --------------------------------------------------------------------------- #
#                     Both forms, on every representation                      #
# --------------------------------------------------------------------------- #


def test_every_layout_graph_exposes_all_three_forms(strength_dataset):
    """A layout method asks for the form it needs; nothing converts at the boundary (§3.5)."""
    vg = vertex_graph(strength_dataset)
    for g in (vg, edge_graph(vg), twin_graph(vg)):
        assert sp.issparse(g.affinity)
        assert sp.issparse(g.combinatorial)
        assert isinstance(g.distance, nx.Graph)
        assert g.distance.number_of_nodes() == g.n_points


@pytest.mark.parametrize("normalize", [False, True])
def test_the_three_forms_share_one_support(strength_dataset, normalize):
    """One support, three valuations: combinatorial (1), affinity (w), distance (d)."""
    vg = vertex_graph(strength_dataset, normalize=normalize)
    for g in (vg, edge_graph(vg), twin_graph(vg)):
        combinatorial, affinity = sp.csr_array(g.combinatorial), sp.csr_array(g.affinity)
        assert (combinatorial.astype(bool) != affinity.astype(bool)).nnz == 0, g.name
        assert set(np.unique(combinatorial.data)) == {1.0}, g.name


def test_the_combinatorial_form_of_the_vertex_graph_is_the_graphs_own_adjacency(
    strength_dataset,
):
    got = sp.csr_array(vertex_graph(strength_dataset).combinatorial).toarray()
    expected = nx.to_scipy_sparse_array(strength_dataset.graph, weight=None).toarray()
    np.testing.assert_array_equal(got, expected)


def test_the_combinatorial_form_is_structural_not_a_threshold(strength_dataset):
    """Derived by rerunning the algebra on the unweighted incidence.

    `normalize=True` scales entries by 1/sqrt(deg), so hanging a support on a float comparison
    against zero would make it a function of the weights. It must not be.
    """
    plain = sp.csr_array(vertex_graph(strength_dataset).combinatorial)
    scaled = sp.csr_array(vertex_graph(strength_dataset, normalize=True).combinatorial)
    assert (plain != scaled).nnz == 0


def test_n_points_is_n_then_m_then_n_plus_m(strength_dataset):
    vg = vertex_graph(strength_dataset)
    n, m = strength_dataset.n, strength_dataset.m
    assert (vg.n_points, edge_graph(vg).n_points, twin_graph(vg).n_points) == (n, m, n + m)


def test_a_layout_graph_carries_no_labels(strength_dataset):
    """The whole point of the split: a length-n label vector must not reach a length-m node set."""
    vg = vertex_graph(strength_dataset)
    for g in (vg, edge_graph(vg), twin_graph(vg)):
        assert not hasattr(g, "labels")
        assert not hasattr(g, "label_sets")


def test_derived_graphs_must_come_off_the_vertex_graph(strength_dataset):
    eg = edge_graph(vertex_graph(strength_dataset))
    with pytest.raises(ValueError, match="affinity algebra"):
        edge_graph(eg)
    with pytest.raises(ValueError, match="affinity algebra"):
        twin_graph(eg)


def test_the_incidence_matrix_is_built_once_and_shared(strength_dataset):
    """Three representations used to construct their own; the derived ones now reuse A_V's."""
    vg = vertex_graph(strength_dataset)
    eg, tg = edge_graph(vg), twin_graph(vg)

    assert eg.incidence is vg.incidence and tg.incidence is vg.incidence
    assert eg.incidence_weighted is vg.incidence_weighted
    assert tg.incidence_weighted is vg.incidence_weighted


def test_incidence_columns_are_graph_edges(strength_dataset):
    B = sp.csc_array(vertex_graph(strength_dataset).incidence)
    for col, (u, v) in enumerate(strength_dataset.graph.edges()):
        assert set(B[:, [col]].nonzero()[0]) == {u, v}


# --------------------------------------------------------------------------- #
#                    Derivation: one side stored, one derived                  #
# --------------------------------------------------------------------------- #


def test_strength_derives_one_over_w_and_round_trips(strength_dataset):
    """$1/w$ is the resistance model, and it composes back to the stored side exactly."""
    vg = vertex_graph(strength_dataset)
    stored = np.array([1.0, 2.0, 3.0, 4.0])

    np.testing.assert_allclose(edge_weights(vg.distance), 1.0 / stored)
    assert "1/w" in vg.distance_derivation

    # A_V is faithful, so its edge entries are the stored affinities; the two forms invert.
    A = sp.csr_array(vg.affinity)
    affinity_w = np.array([A[u, v] for u, v in strength_dataset.graph.edges()])
    np.testing.assert_allclose(affinity_w * edge_weights(vg.distance), 1.0)


def test_distance_derives_the_exponential_affinity_and_round_trips():
    """The exponential is the exact inverse of $-\\log(w/w_{max})$, so the two forms agree.

    This is what keeps the affinities SG-t-SNE is given and the ideal distances SNS scores
    against describing the *same* geometry. A Gaussian forward with a log reverse would not.
    """
    ds = load_dataset("digits", n_neighbors=5)
    vg = vertex_graph(ds)

    stored = np.array([d["weight"] for _, _, d in ds.graph.edges(data=True)])
    d_scale = float(np.median(stored))

    A = sp.csr_array(vg.affinity)
    affinity_w = np.array([A[u, v] for u, v in ds.graph.edges()])

    np.testing.assert_allclose(affinity_w, np.exp(-stored / d_scale), rtol=1e-12)
    np.testing.assert_allclose(-np.log(affinity_w) * d_scale, stored, rtol=1e-10)
    assert "exp(-d/d_scale)" in vg.derivation


def test_the_squared_rule_is_selectable_but_not_the_default():
    """Kept so the choice can be compared empirically once, not left as an assumption."""
    ds = load_dataset("digits", n_neighbors=5)
    plain = vertex_graph(ds)
    squared = vertex_graph(ds, affinity_rule="exp_decay_sq")

    assert "exp(-(d/d_scale)^2)" in squared.derivation
    assert (sp.csr_array(plain.affinity) != sp.csr_array(squared.affinity)).nnz > 0


def test_unweighted_distance_is_hop_counts_with_non_edges_absent(combinatorial_dataset):
    """Not an all-ones dense matrix: absent pairs must stay absent so paths count hops."""
    vg = vertex_graph(combinatorial_dataset)
    D = vg.distance

    assert D.number_of_edges() == combinatorial_dataset.m
    np.testing.assert_array_equal(edge_weights(D), np.ones(combinatorial_dataset.m))

    # star_graph(4): leaves are two hops apart, and that pair is not an edge.
    assert not D.has_edge(1, 2)
    assert nx.shortest_path_length(D, 1, 2, weight="weight") == 2


def test_an_adaptive_kernel_without_a_stored_distance_raises():
    """Tabula must fail loudly if `D` is ever dropped, not fall back to -log(w/w_max).

    UMAP's per-point rho_i and sigma_i are not stored, so no global transform of `A` recovers
    `d` — applying the kernel rule anyway would assign ideal distance 0 to one edge per vertex.
    """
    G = nx.path_graph(4)
    for u, v in G.edges():
        G[u][v]["weight"] = 0.5
    ds = Dataset(name="adaptive", graph=G, affinity=Stored(meaning="adaptive_kernel"))

    with pytest.raises(ValueError, match="globally non-invertible"):
        vertex_graph(ds)


@pytest.mark.skipif(not TABULA.exists(), reason="tabula-liver.mat not present")
def test_dropping_tabulas_D_makes_the_mapping_raise_rather_than_transform():
    """The concrete case the refusal exists for — not a synthetic one.

    Applying `-log(w/w_max)` to tabula's UMAP memberships would assign ideal distance exactly 0
    to one edge per vertex, requiring 93.1% of its vertices to coincide
    (`docs/DATASET_PROVENANCE.md`). It must fail loudly instead.
    """
    ds = load_dataset("tabula")
    ds.distance = None

    with pytest.raises(ValueError, match="must supply a stored distance side"):
        vertex_graph(ds)


def test_a_stored_side_that_does_not_span_the_edge_set_raises():
    G = nx.path_graph(4)
    partial = sp.csr_array(np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], float))
    ds = Dataset(name="gappy", graph=G, affinity=StoredMatrix("A", meaning="kernel"),
                 matrices={"A": partial})

    with pytest.raises(ValueError, match="zero on 2 of the graph's 3 edges"):
        vertex_graph(ds)


# --------------------------------------------------------------------------- #
#                         Which distance form, and why                         #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not TABULA.exists(), reason="tabula-liver.mat not present")
def test_the_vertex_graph_of_tabula_carries_its_measured_distances():
    """The point of storing both sides: a classical layout gets the *measured* geometry."""
    ds = load_dataset("tabula")
    vg = vertex_graph(ds)

    D = ds.stored_matrix("distance")
    expected = np.array([D[u, v] for u, v in ds.graph.edges()])
    np.testing.assert_allclose(edge_weights(vg.distance), expected)

    assert vg.distance_derivation.startswith("stored")
    assert vg.distance_chain() == [
        "dataset:tabula",
        'A_V: stored (stored matrix "A" (adaptive_kernel))',
        'distance: stored (stored matrix "D" (distance))',
    ]


def test_derived_graphs_fall_back_and_the_chain_records_it(strength_dataset):
    """Three derivations deep in the worst case — the nature of the method, not a defect."""
    tg = twin_graph(vertex_graph(strength_dataset))

    assert tg.distance_derivation == "generic fallback: normalise, symmetrise, -log"
    assert tg.distance_chain() == [
        "dataset:strengths",
        'A_V: stored (graph edge attribute "weight" (strength))',
        "A_tne: sum_k alpha^k A~^k, alpha=0.85, k=2",
        "distance: generic fallback: normalise, symmetrise, -log",
    ]
    assert tg.parent.parent is None and tg.parent.name == "A_V"


def test_the_generic_fallback_symmetrises():
    """Row normalisation alone leaves the matrix asymmetric, and nx then keeps one of each pair.

    That is the live bug in `KamadaKawaiLayout(mode="similarity")`; the lifted version fixes it
    (spec §0.5 decision 7).
    """
    A = sp.csr_array(np.array([[0.0, 1.0, 3.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
    D = affinity_matrix_to_distance(A).toarray()

    np.testing.assert_allclose(D, D.T)
    assert (D >= 0).all(), "a normalised affinity cannot exceed 1, so -log cannot go negative"

    # Row-normalising alone would give 0.25 one way and 1.0 the other for the pair (0, 2).
    assert D[0, 2] == pytest.approx(-np.log((3 / 4 + 1.0) / 2))


def test_faithfulness_is_declared_not_inferred(strength_dataset):
    vg = vertex_graph(strength_dataset)
    assert vg.is_faithful
    assert not vertex_graph(strength_dataset, normalize=True).is_faithful
    assert not edge_graph(vg).is_faithful and not twin_graph(vg).is_faithful


# --------------------------------------------------------------------------- #
#                                   Points                                     #
# --------------------------------------------------------------------------- #


def test_twin_block_ordering_is_edges_then_vertices(strength_dataset):
    """`twin_adjacency` builds [[0, B^T], [B, 0]], so rows are [edges (m); vertices (n)]."""
    tg = twin_graph(vertex_graph(strength_dataset))
    n, m = strength_dataset.n, strength_dataset.m

    tagged = np.vstack([np.arange(n + m, dtype=float), np.zeros(n + m)])
    X_v, X_e, _ = tg.vertex_and_edge_points(tagged)

    np.testing.assert_array_equal(X_e[0], np.arange(m))
    np.testing.assert_array_equal(X_v[0], np.arange(m, m + n))


@pytest.mark.parametrize("kind", ["vertex", "edge", "twin"])
def test_points_come_back_in_the_original_indexing(strength_dataset, kind):
    """Which is why the dataset's labels apply to a layout of any representation."""
    vg = vertex_graph(strength_dataset)
    g = {"vertex": vg, "edge": edge_graph(vg), "twin": twin_graph(vg)}[kind]

    X = np.random.default_rng(0).normal(size=(2, g.n_points))
    X_v, X_e, slopes = g.vertex_and_edge_points(X)

    assert X_v.shape == (2, strength_dataset.n)
    assert X_e.shape == (2, strength_dataset.m)
    assert slopes.shape == (strength_dataset.m,)


def test_a_layout_of_the_wrong_width_is_refused(strength_dataset):
    vg = vertex_graph(strength_dataset)
    with pytest.raises(ValueError, match=r"expected a \(d, 5\) layout"):
        vg.vertex_and_edge_points(np.zeros((2, 5)).T)


# --------------------------------------------------------------------------- #
#              Equivalence with the pre-restructure representations            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("name", ["karate_weighted", "small_world"])
def test_the_mappings_reproduce_the_legacy_representation_matrices(name, normalize):
    """A refactor, and this is what makes that checkable.

    The legacy classes read the graph's own ``weight`` attribute; the mappings read the declared
    affinity side. Where those coincide — which is every dataset except digits, whose stored
    side is the distance one — the three matrices must be identical.
    """
    ds = load_dataset(name)
    vg = vertex_graph(ds, normalize=normalize)

    legacy_v = gm.VertexMatrix(ds.graph, normalize=normalize)
    legacy_e = gm.EdgeMatrix(ds.graph, normalize=normalize)
    legacy_t = gm.TwinEmbeddingMatrix(ds.graph, normalize=normalize, alpha=0.85, k=2)

    for new, old in (
        (vg.affinity, legacy_v.adj),
        (edge_graph(vg).affinity, legacy_e.adj),
        (twin_graph(vg, alpha=0.85, k=2).affinity, legacy_t.adj),
    ):
        difference = abs(sp.csr_array(new) - sp.csr_array(old))
        assert difference.nnz == 0 or difference.max() < 1e-15


@pytest.mark.parametrize("kind", ["vertex", "edge", "twin"])
def test_the_mappings_reproduce_the_legacy_point_derivations(kind):
    ds = load_dataset("karate_weighted")
    vg = vertex_graph(ds)
    new = {"vertex": vg, "edge": edge_graph(vg), "twin": twin_graph(vg, alpha=0.85, k=2)}[kind]
    legacy = {
        "vertex": gm.VertexMatrix(ds.graph),
        "edge": gm.EdgeMatrix(ds.graph),
        "twin": gm.TwinEmbeddingMatrix(ds.graph, alpha=0.85, k=2),
    }[kind]

    X = np.random.default_rng(0).normal(size=(2, new.n_points))
    for a, b in zip(new.vertex_and_edge_points(X), legacy.vertex_and_edge_embeddings(X)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-15)


def test_the_legacy_classes_still_satisfy_the_embed_interface():
    """`graph_embedding_all` and `paper_figures` keep working until step 10 (spec §3.6).

    The shim must stay bit-identical to the old `mode="distance"` default, which is what makes
    deleting `mode=` a no-op on the legacy path. It is not a principled distance form — that is
    the whole reason `LayoutGraph` exists — so nothing new may route through it.
    """
    ds = load_dataset("karate_weighted")
    legacy = gm.VertexMatrix(ds.graph)

    assert legacy.affinity is legacy.adj
    assert legacy.n_points == ds.n

    shim = legacy.distance
    expected = nx.from_scipy_sparse_array(legacy.adj)
    assert shim.number_of_edges() == expected.number_of_edges()
    np.testing.assert_array_equal(edge_weights(shim), edge_weights(expected))


def test_point_provenance_is_declared_per_representation(strength_dataset):
    """Scoring derived point sets is correct; the asymmetry must stay visible (§0.5 (5))."""
    vg = vertex_graph(strength_dataset)
    assert vg.point_provenance == {"V": "primary", "E": "derived"}
    assert edge_graph(vg).point_provenance == {"V": "derived", "E": "primary"}
    assert twin_graph(vg).point_provenance == {"V": "primary", "E": "primary"}
