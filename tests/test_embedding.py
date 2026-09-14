"""Spec §3.5 — every layout method takes a `LayoutGraph` and asks for the side it needs."""

import networkx as nx
import numpy as np
import pytest

from twin.datasets import load_dataset
from twin.embedding import EMBEDDING_REGISTRY
from twin.layoutgraph import edge_graph, twin_graph, vertex_graph

NAMES = sorted(EMBEDDING_REGISTRY)


# --------------------------------------------------------------------------- #
#          graphviz `sfdp` is broken in 16.0.0 — probe it, do not assume        #
# --------------------------------------------------------------------------- #

#: The smallest graph found that triggers it: 5 nodes, 5 edges.
#:
#: ⚠️ **This is an upstream crash, not a fault in `YifanHuLayout`.** graphviz 16.0.0's `sfdp`
#: corrupts the heap and aborts (SIGABRT, `sysmalloc` assertion) inside its **multilevel
#: coarsening**: `-Glevels=0` and `-Glevels=1` survive, `-Glevels>=2` does not, and no other
#: attribute (`quadtree`, `mode`, `overlap`, `smoothing`, `K`, `maxiter`) changes anything.
#: `neato`, `dot`, `fdp`, `twopi` and `circo` handle every input that kills it, and graphviz
#: 14.1.4 handles them too — it is a regression.
#:
#: It depends on the order graphviz *creates* the nodes in, not on the graph: this same graph
#: survives when its nodes are created in the order 0,1,3,4,2 and aborts when they are created
#: 0,1,2,3,4. `nx_pydot` always writes node statements in sorted order, so every call from
#: `YifanHuLayout` lands on the failing order — which is why it looked size-dependent at first
#: (`complete_graph(10)` is fine, `karate_club_graph` is not).
#:
#: Upstream: forum.graphviz.org/t/sfdp-crash-regression-in-15-1-1/3392 and GitLab issue #2852 —
#: introduced in 15.1.1, **fixed in 16.1.0**. Arch currently ships 16.0.0-1.
SFDP_TRIGGER_EDGES = [(0, 1), (1, 3), (1, 4), (2, 3), (2, 4)]


def _probe_sfdp() -> str | None:
    """Run the trigger through the exact path `YifanHuLayout` uses. None means sfdp is healthy.

    Deliberately **narrow**. It exercises the backend and the `nx_pydot` plumbing and nothing of
    ours, so if sfdp handles this graph but a real layout still fails, the tests below run and
    fail loudly — which is the right outcome, because that would be a different bug.
    """
    trigger = nx.Graph()
    trigger.add_nodes_from(range(5))
    trigger.add_edges_from(SFDP_TRIGGER_EDGES)
    try:
        nx.nx_pydot.graphviz_layout(trigger, prog="sfdp")
    except Exception as error:  # pydot raises AssertionError carrying the return code
        message = str(error).strip().splitlines()[0][:160]
        # Only a *crash* justifies skipping: pydot reports a signal death as a negative return
        # code. A missing binary, a missing pydot or a permissions problem must fail loudly
        # instead, the way they already do for the `neato` tests, rather than quietly disabling
        # a method forever because the probe happened to raise.
        return message if "returned code: -" in message else None
    return None


SFDP_FAILURE = _probe_sfdp()

#: The same parametrisation, with `yifanhu` skipped when the backend under it is broken. Only
#: these two tests actually invoke `sfdp`; `test_yifan_hu_is_handed_the_structure_and_nothing
#: _else` inspects the dot source instead and keeps running either way, so the method does not
#: lose all of its coverage — only the part that needs a working binary.
BACKEND_NAMES = [
    pytest.param(
        name,
        marks=pytest.mark.skipif(
            name == "yifanhu" and SFDP_FAILURE is not None,
            reason=(
                f"graphviz sfdp is broken in this environment and aborts on a 5-node graph "
                f"({SFDP_FAILURE}). Upstream regression, fixed in graphviz 16.1.0; this "
                f"un-skips by itself once the installed graphviz is new enough. "
                f"Do not 'fix' it by changing YifanHuLayout."
            ),
        ),
    )
    for name in NAMES
]


class RecordingGraph:
    """A `LayoutGraph` that remembers which of its three forms was read.

    `expects` is declared metadata that nothing branches on, so nothing keeps it honest either.
    This is what does (`DECISIONS.md`).
    """

    FORMS = ("combinatorial", "affinity", "distance")

    def __init__(self, inner):
        self._inner = inner
        self.touched: set[str] = set()

    @property
    def combinatorial(self):
        self.touched.add("combinatorial")
        return self._inner.combinatorial

    @property
    def affinity(self):
        self.touched.add("affinity")
        return self._inner.affinity

    @property
    def distance(self):
        self.touched.add("distance")
        return self._inner.distance

    def __getattr__(self, name):
        return getattr(self._inner, name)


@pytest.fixture(scope="module")
def small_vertex_graph():
    return vertex_graph(load_dataset("karate_weighted"))


@pytest.mark.parametrize("name", BACKEND_NAMES)
def test_every_method_touches_exactly_the_form_it_declares(small_vertex_graph, name):
    spec = EMBEDDING_REGISTRY[name]
    recorder = RecordingGraph(small_vertex_graph)

    Y = spec.cls(seed=0, d=2).embed(recorder)

    assert recorder.touched == {spec.expects}, (
        f"{name} declares expects={spec.expects!r} but read {sorted(recorder.touched)}"
    )
    assert Y.shape == (2, small_vertex_graph.n_points)


@pytest.mark.parametrize("name", BACKEND_NAMES)
def test_every_method_lays_out_every_representation(small_vertex_graph, name):
    """A_E and A_tne have different node counts; the output must follow n_points, not n."""
    spec = EMBEDDING_REGISTRY[name]
    for graph in (edge_graph(small_vertex_graph), twin_graph(small_vertex_graph)):
        Y = spec.cls(seed=0, d=2).embed(graph)
        assert Y.shape == (2, graph.n_points)


@pytest.mark.parametrize("name", NAMES)
def test_init_embedding_is_d_by_n(name):
    y0 = EMBEDDING_REGISTRY[name].cls(seed=0, d=2).init_embedding(11, 2)
    assert y0.shape == (2, 11)


def test_mode_is_gone_from_the_distance_consuming_methods():
    """It never belonged to the layout class — it is a property of the graph (spec §3.5)."""
    for name in ("kamada-kawai", "neato"):
        cls = EMBEDDING_REGISTRY[name].cls
        assert "mode" not in cls.__dataclass_fields__
        with pytest.raises(TypeError):
            cls(mode="similarity")


def test_the_registry_declares_which_methods_sweep_lambda_and_which_are_stochastic():
    assert EMBEDDING_REGISTRY["sgtsne"].has_lambda
    assert not any(s.has_lambda for n, s in EMBEDDING_REGISTRY.items() if n != "sgtsne")
    assert {n for n, s in EMBEDDING_REGISTRY.items() if s.stochastic} == {
        "spring",
        "forceatlas2",
        "sgtsne",
    }


def test_a_distance_consuming_method_uses_the_datasets_own_distances(small_vertex_graph):
    """The reason the side is a property of the graph: 1/w is not the same layout as w.

    karate_weighted stores strengths, so its distance side is the derived 1/w. Laying it out
    against the affinity read as lengths — which is what `mode="distance"` did — puts the
    *strongest* ties furthest apart, and the two layouts must therefore differ.
    """
    from twin.embedding import KamadaKawaiLayout

    correct = KamadaKawaiLayout(seed=0, d=2).embed(small_vertex_graph)

    as_if_lengths = nx.from_scipy_sparse_array(small_vertex_graph.affinity)
    inverted = np.zeros_like(correct)
    for node, position in nx.kamada_kawai_layout(as_if_lengths, dim=2).items():
        inverted[:, node] = position

    assert not np.allclose(correct, inverted)


# --------------------------------------------------------------------------- #
#       What actually reaches the backend — which `expects` cannot check       #
# --------------------------------------------------------------------------- #


def dot_source(graph) -> str:
    return nx.nx_pydot.to_pydot(graph).to_string()


def test_yifan_hu_is_handed_the_structure_and_nothing_else(small_vertex_graph):
    """Hu's method is combinatorial — attractive d²/K, repulsive −CK²/d, K a *global* constant.

    `nx.nx_pydot` writes every edge attribute into the dot source and graphviz reads `weight` as
    a spring-strength multiplier, so an input graph carrying any valuation would silently hand
    one to a method with no use for it.
    """
    from twin.embedding import _structure_only

    graph = _structure_only(small_vertex_graph)
    assert graph.number_of_nodes() == small_vertex_graph.n_points
    assert all(not data for _, _, data in graph.edges(data=True)), "no edge attributes"

    source = dot_source(graph)
    assert "weight" not in source and "len" not in source


def test_neato_receives_desired_lengths_not_spring_strengths(small_vertex_graph):
    """The inversion this closes: for graphviz, `len` is the target and `weight` is how hard it
    is held to that target. A distance passed as `weight` says "far apart means pull harder"."""
    from twin.embedding import _desired_lengths

    graph = _desired_lengths(small_vertex_graph)
    source = dot_source(graph)

    assert "len=" in source and "weight=" not in source

    expected = {
        frozenset((u, v)): d
        for u, v, d in small_vertex_graph.distance.edges(data="weight")
    }
    got = {frozenset((u, v)): d for u, v, d in graph.edges(data="len")}
    assert got == pytest.approx(expected)


@pytest.mark.parametrize("name", ["karate_weighted", "football_2023"])
def test_desired_lengths_are_strictly_positive_on_every_representation(name):
    """graphviz has no use for a zero-length edge; the generic fallback could in principle
    produce one where a normalised symmetric entry lands at exactly 1."""
    from twin.datasets import load_dataset
    from twin.embedding import _desired_lengths

    vg = vertex_graph(load_dataset(name))
    for graph in (vg, edge_graph(vg), twin_graph(vg)):
        lengths = np.array([d for _, _, d in _desired_lengths(graph).edges(data="len")])
        assert (lengths > 0).all(), graph.name


def test_neato_responds_to_the_distances_it_is_given(small_vertex_graph):
    """Witness for the behaviour change: the layout is a function of `len`, so reversing the
    ranking of the distances must move it."""
    from twin.embedding import NeatoLayout, _desired_lengths

    correct = NeatoLayout(seed=0, d=2).embed(small_vertex_graph)

    reversed_graph = _desired_lengths(small_vertex_graph)
    longest = max(d for _, _, d in reversed_graph.edges(data="len"))
    for u, v, data in reversed_graph.edges(data=True):
        data["len"] = longest + 0.01 - data["len"]

    inverted = np.zeros_like(correct)
    for node, position in nx.nx_pydot.graphviz_layout(reversed_graph, prog="neato").items():
        inverted[:, node] = position

    assert not np.allclose(correct, inverted)


# --------------------------------------------------------------------------- #
#             SpectralLayout — Hall's formulation (thesis eqs. 12-14)          #
# --------------------------------------------------------------------------- #


def test_spectral_solves_halls_problem_not_the_normalised_one(small_vertex_graph):
    r"""The default is $L^w = D - W$, so the layout satisfies $X^\top\mathbf{1} = 0$.

    That constraint is eq. (14) and it is the cheap witness for *which* Laplacian was used:
    $L_{\text{sym}}$'s trivial eigenvector is $D^{1/2}\mathbf{1}$, so its remaining
    eigenvectors are orthogonal to that instead and the layout is not centred.
    """
    from twin.embedding import SpectralLayout

    X = SpectralLayout().embed(small_vertex_graph)
    assert np.abs(X.mean(axis=1)).max() < 1e-10

    normalised = SpectralLayout(variant="degree_normalized").embed(small_vertex_graph)
    assert np.abs(normalised.mean(axis=1)).max() > 1e-6


def test_spectral_matches_a_dense_eigendecomposition_of_D_minus_W(small_vertex_graph):
    """Against `numpy.linalg.eigh` on the dense Laplacian — the definition, not another solver.

    Compared as a *subspace*: eigenvector signs are arbitrary and a repeated eigenvalue makes
    the individual vectors arbitrary within their eigenspace, so comparing them elementwise
    would pin an ARPACK convention rather than the mathematics.
    """
    import scipy.sparse as sp

    from twin.embedding import SpectralLayout

    A = small_vertex_graph.affinity
    L = sp.csgraph.laplacian(A, normed=False).toarray()
    _, dense = np.linalg.eigh(L)

    X = SpectralLayout().embed(small_vertex_graph).T
    q_dense = np.linalg.qr(dense[:, 1:3])[0]
    q_arpack = np.linalg.qr(X)[0]
    overlap = np.linalg.svd(q_dense.T @ q_arpack, compute_uv=False).min()
    assert overlap == pytest.approx(1.0, abs=1e-8)


def test_spectral_is_deterministic_as_the_registry_declares(small_vertex_graph):
    """`stochastic=False` has to be true bit-wise, not just up to a reflection.

    ARPACK draws its starting vector at random unless given one, and successive runs then
    differ by eigenvector sign. No metric here can see that — they are all
    reflection-invariant — but a persisted layout or a committed figure would not reproduce.
    """
    from twin.embedding import SpectralLayout

    assert not EMBEDDING_REGISTRY["spectral"].stochastic
    first = SpectralLayout().embed(small_vertex_graph)
    for _ in range(3):
        assert np.array_equal(SpectralLayout().embed(small_vertex_graph), first)


def test_the_degree_normalized_variant_is_koren_not_the_L_sym_embedding(small_vertex_graph):
    r"""$D^{-1/2}u$, not $u$. The difference is the whole of what was wrong before.

    Returning $u$ directly is the $L_{\text{sym}}$ embedding used in spectral clustering; its
    extra $D^{1/2}$ factor scales each vertex by the square root of its own degree, which
    pushes hubs outward instead of drawing them in.
    """
    import scipy.sparse as sp

    from twin.embedding import SpectralLayout

    A = small_vertex_graph.affinity
    degrees = np.asarray(A.sum(axis=1)).ravel()
    X = SpectralLayout(variant="degree_normalized").embed(small_vertex_graph).T

    L_sym = sp.csgraph.laplacian(A, normed=True)
    _, u = sp.linalg.eigsh(L_sym, k=3, sigma=0.0, which="LM", v0=np.ones(A.shape[0]))

    q_expected = np.linalg.qr(u[:, 1:] / np.sqrt(degrees)[:, None])[0]
    q_actual = np.linalg.qr(X)[0]
    assert np.linalg.svd(q_expected.T @ q_actual, compute_uv=False).min() == pytest.approx(
        1.0, abs=1e-8
    )


def test_spectral_does_not_place_hubs_further_out_than_leaves():
    """The behaviour thesis §2.2 describes, on the roster's most degree-heterogeneous graph.

    `barabasi_albert` is the scale-free case — a 20x spread of weighted degree — and Hall's
    formulation draws its hubs inward: Spearman(degree, radius) is -0.52. The previous
    implementation returned +0.16 on the same graph, and was positive on all ten registered
    datasets, i.e. it did the opposite of what the text describing it says.
    """
    from scipy.stats import spearmanr

    from twin.embedding import SpectralLayout

    graph = vertex_graph(load_dataset("barabasi_albert"))
    degrees = np.asarray(graph.affinity.sum(axis=1)).ravel()
    X = SpectralLayout().embed(graph).T

    radius = np.linalg.norm(X - X.mean(axis=0), axis=1)
    assert spearmanr(degrees, radius).statistic < -0.3


def test_spectral_refuses_an_unknown_variant(small_vertex_graph):
    from twin.embedding import SpectralLayout

    with pytest.raises(ValueError, match="variant must be"):
        SpectralLayout(variant="normalized").embed(small_vertex_graph)
