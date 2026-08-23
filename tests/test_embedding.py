"""Spec §3.5 — every layout method takes a `LayoutGraph` and asks for the side it needs."""

import networkx as nx
import numpy as np
import pytest

from twin.datasets import load_dataset
from twin.embedding import EMBEDDING_REGISTRY
from twin.layoutgraph import edge_graph, twin_graph, vertex_graph

NAMES = sorted(EMBEDDING_REGISTRY)


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


@pytest.mark.parametrize("name", NAMES)
def test_every_method_touches_exactly_the_form_it_declares(small_vertex_graph, name):
    spec = EMBEDDING_REGISTRY[name]
    recorder = RecordingGraph(small_vertex_graph)

    Y = spec.cls(seed=0, d=2).embed(recorder)

    assert recorder.touched == {spec.expects}, (
        f"{name} declares expects={spec.expects!r} but read {sorted(recorder.touched)}"
    )
    assert Y.shape == (2, small_vertex_graph.n_points)


@pytest.mark.parametrize("name", NAMES)
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
