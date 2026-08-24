"""Spec §2.1 — ADR, the $k$-NN neighbourhood family, SNS.

The analytic cases build their graphs by hand and check against values computed in the test
file, per spec §6. That is deliberate: a metric checked only against its own past output pins
whatever it does, and a metric checked against a hand computation pins what it is *for*.
"""

import networkx as nx
import numpy as np
import pytest

from twin.datasets import load_dataset
from twin.forms import stamp_form
from twin.layoutgraph import vertex_graph
from twin.metrics.common import mean_pairwise_distance
from twin.metrics.structural import adjacency_distance_ratio


# --------------------------------------------------------------------------- #
#                      ADR — hand-computed analytic cases                      #
# --------------------------------------------------------------------------- #


def test_adr_of_a_complete_graph_is_exactly_one(k4):
    """Both averages run over the same pairs, so the ratio is 1 for *any* layout.

    The strongest analytic case available: it holds for a square, for a random cloud, and for
    a degenerate line, so it pins the normalisation rather than one layout.
    """
    square = np.array([[0.0, 0], [1, 0], [1, 1], [0, 1]])
    random = np.random.default_rng(7).normal(size=(4, 3))
    for X in (square, random):
        assert adjacency_distance_ratio(X, k4) == pytest.approx(1.0)


def test_adr_of_a_collinear_path(path5):
    """Edges are the four unit steps; all ten pairs average 2. ADR = 1/2."""
    X = np.column_stack([np.arange(5.0), np.zeros(5)])
    assert adjacency_distance_ratio(X, path5) == pytest.approx(0.5)


def test_adr_of_a_star(star5):
    """Four unit spokes over the mean of ten pairs: 4 at 1, 4 at sqrt(2), 2 at 2."""
    X = np.array([[0.0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]])
    expected = 1.0 / ((4 * 1 + 4 * np.sqrt(2) + 2 * 2) / 10)
    assert adjacency_distance_ratio(X, star5) == pytest.approx(expected)
    assert adjacency_distance_ratio(X, star5) == pytest.approx(10 / (8 + 4 * np.sqrt(2)))


def test_adr_of_a_triangle_is_one_because_it_is_complete(triangle):
    X = np.array([[0.0, 0], [1, 0], [0.5, 3.0]])
    assert adjacency_distance_ratio(X, triangle) == pytest.approx(1.0)


def test_adr_falls_when_adjacent_vertices_are_drawn_closer(path5):
    """Low is good: the direction of the metric, pinned."""
    spread = np.column_stack([np.arange(5.0), np.zeros(5)])
    clustered = np.column_stack([np.array([0.0, 0.1, 0.2, 4.0, 4.1]), np.zeros(5)])
    assert adjacency_distance_ratio(clustered, path5) < adjacency_distance_ratio(spread, path5)


# --------------------------------------------------------------------------- #
#                          ADR — weights, and eq. (29)                         #
# --------------------------------------------------------------------------- #


def eq29(X: np.ndarray, G: nx.Graph) -> float:
    """Eq. (29) written out independently: mean edge length over mean pairwise distance."""
    lengths = [float(np.linalg.norm(X[u] - X[v])) for u, v in G.edges()]
    return float(np.mean(lengths) / mean_pairwise_distance(X))


def test_eq27_reduces_to_eq29_on_an_unweighted_graph(karate_structured_layout):
    """No unweighted branch exists in the source; this is the test that replaces one."""
    dataset = load_dataset("karate")
    X = karate_structured_layout
    assert adjacency_distance_ratio(X, dataset.graph) == pytest.approx(eq29(X, dataset.graph))


def test_the_weighted_case_is_not_the_unweighted_one(karate_structured_layout):
    """Otherwise the eq.-(29) agreement above would be vacuous."""
    dataset = load_dataset("karate_weighted")
    X = karate_structured_layout
    weighted = adjacency_distance_ratio(X, dataset.graph)
    assert weighted != pytest.approx(eq29(X, dataset.graph))


def test_the_source_has_no_unweighted_branch():
    """Eq. (29) is eq. (27) with w in {0, 1}; `nx.adjacency_matrix` already fills the 1s."""
    import inspect

    source = inspect.getsource(adjacency_distance_ratio)
    body = source[source.index('"""', source.index('"""') + 3) :]
    assert "number_of_edges" not in body or "weight is None" not in body
    assert body.count("nx.adjacency_matrix") == 1


def test_adr_reads_the_weights_it_is_given():
    """Doubling one edge's affinity moves the number in the direction its weight implies."""
    G = nx.path_graph(3)
    X = np.array([[0.0, 0], [1, 0], [4, 0]])
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    even = adjacency_distance_ratio(X, G)

    G[0][1]["weight"] = 9.0  # nearly all the mass on the short edge
    assert adjacency_distance_ratio(X, G) < even


# --------------------------------------------------------------------------- #
#                              ADR — invariance                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("transform", ["scale", "rotate", "reflect", "translate"])
def test_adr_is_invariant_under_similarity_transforms(karate, karate_structured_layout, transform):
    """Numerator and denominator are both lengths, so a uniform scaling cancels exactly."""
    X = karate_structured_layout
    theta = 0.7
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    moved = {
        "scale": 7.0 * X,
        "rotate": X @ rotation.T,
        "reflect": X * np.array([1.0, -1.0]),
        "translate": X + np.array([3.0, -2.0]),
    }[transform]

    assert adjacency_distance_ratio(moved, karate) == pytest.approx(
        adjacency_distance_ratio(X, karate), rel=1e-12
    )


# --------------------------------------------------------------------------- #
#                    ADR — what it refuses, and what it accepts                #
# --------------------------------------------------------------------------- #


def test_adr_refuses_the_digits_graph():
    """digits' `weight` is a stored Euclidean distance, 5.3-37.8 — bug §1.6 one layer up.

    Silently accepting it would make ADR reward putting graph-*distant* pairs close together,
    and unlike §1.6 there is no Spearman of -1 to make it visible afterwards.
    """
    dataset = load_dataset("digits")
    X = np.random.default_rng(8).normal(size=(dataset.n, 2))
    with pytest.raises(ValueError, match="affinity-valued") as excinfo:
        adjacency_distance_ratio(X, dataset.graph)
    assert "'distance'" in str(excinfo.value)


def test_adr_accepts_the_affinity_form_of_the_same_dataset():
    """The correct call: the affinity graph, where a large weight means *coupled*."""
    dataset = load_dataset("digits")
    X = np.random.default_rng(8).normal(size=(dataset.n, 2))
    affinity = vertex_graph(dataset).affinity_graph()
    assert np.isfinite(adjacency_distance_ratio(X, affinity))


def test_adr_refuses_a_layout_graph_distance_form(karate_dataset, karate_structured_layout):
    with pytest.raises(ValueError, match="affinity-valued"):
        adjacency_distance_ratio(
            karate_structured_layout, vertex_graph(karate_dataset).distance
        )


def test_adr_accepts_a_bare_unstamped_graph(k4):
    """§6's analytic cases build a K_4 by hand and must not need a Dataset."""
    assert adjacency_distance_ratio(np.eye(4)[:, :2], k4) == pytest.approx(1.0)


def test_adr_accepts_a_combinatorial_graph(karate_structured_layout):
    dataset = load_dataset("karate")
    assert np.isfinite(adjacency_distance_ratio(karate_structured_layout, dataset.graph))


def test_adr_checks_orientation(karate, karate_structured_layout):
    with pytest.raises(ValueError, match="n_points"):
        adjacency_distance_ratio(karate_structured_layout.T, karate)


def test_adr_of_an_edgeless_graph_is_nan():
    G = nx.empty_graph(4)
    assert np.isnan(adjacency_distance_ratio(np.random.default_rng(9).normal(size=(4, 2)), G))


def test_adr_of_a_collapsed_layout_is_nan(k4):
    """Every pairwise distance is zero, so the denominator is 0 — say so rather than divide."""
    assert np.isnan(adjacency_distance_ratio(np.zeros((4, 2)), k4))


def test_adr_uses_the_original_vertex_indexing():
    """`nx.adjacency_matrix` follows `G.nodes()` order; X rows are 0..n-1.

    Every current graph happens to be inserted in order, so `nodelist=range(n)` is invisible
    today and stays invisible until a dataset relabels. This is the test that sees it.
    """
    G = nx.Graph()
    G.add_nodes_from([2, 0, 1])  # insertion order is not vertex order
    G.add_edge(0, 1)
    X = np.array([[0.0, 0], [1, 0], [5, 0]])

    # Edge 0-1 has length 1; the three pairs average 10/3. Under `G.nodes()` order the edge
    # would land on rows 1 and 2 instead, giving length 4 and an ADR of 1.2.
    assert adjacency_distance_ratio(X, G) == pytest.approx(1.0 / (10 / 3))


def test_the_docstring_states_the_comparability_rule():
    """§5.3 enforces it at step 11; the docstring is what makes that read as a rule."""
    doc = adjacency_distance_ratio.__doc__
    assert "fixed graph under a fixed weighting" in doc
    assert "across nothing" in doc
