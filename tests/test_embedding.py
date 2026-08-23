"""Spec §3.5 — every layout method takes a `LayoutGraph` and asks for the side it needs."""

import networkx as nx
import numpy as np
import pytest

from twin.datasets import load_dataset
from twin.embedding import EMBEDDING_REGISTRY
from twin.layoutgraph import edge_graph, twin_graph, vertex_graph

NAMES = sorted(EMBEDDING_REGISTRY)


class RecordingGraph:
    """A `LayoutGraph` that remembers which of its two forms was read.

    `expects` is declared metadata that nothing branches on, so nothing keeps it honest either.
    This is what does (`DECISIONS.md`).
    """

    def __init__(self, inner):
        self._inner = inner
        self.touched: set[str] = set()

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
def test_every_method_touches_exactly_the_side_it_declares(small_vertex_graph, name):
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
