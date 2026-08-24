"""Shared fixtures.

Layout arrays handed to anything under ``twin.metrics`` are ``(n_points, d)``; see CLAUDE.md.
Every layout fixture here is deterministic without invoking a layout algorithm, so the pinned
values in ``test_characterisation.py`` do not move when a backend changes.
"""

import networkx as nx
import numpy as np
import pytest


# --------------------------------------------------------------------------- #
#                         Analytic graphs (spec §6)                            #
# --------------------------------------------------------------------------- #


@pytest.fixture
def k4() -> nx.Graph:
    return nx.complete_graph(4)


@pytest.fixture
def path5() -> nx.Graph:
    return nx.path_graph(5)


@pytest.fixture
def star5() -> nx.Graph:
    """Centre 0 with four leaves."""
    return nx.star_graph(4)


@pytest.fixture
def triangle() -> nx.Graph:
    return nx.cycle_graph(3)


# --------------------------------------------------------------------------- #
#                                   Karate                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture
def karate() -> nx.Graph:
    """``nx.karate_club_graph`` — carries Zachary's 1-7 interaction weights.

    Registered as ``karate_weighted``, whose affinity side is ``Stored(meaning="strength")``.
    The unweighted ``karate.mat`` variant is a different dataset (§3.4).
    """
    return nx.karate_club_graph()


@pytest.fixture
def karate_dataset():
    """The registered ``karate_weighted`` dataset, whose graph is the ``karate`` fixture."""
    from twin.datasets import load_dataset

    return load_dataset("karate_weighted")


@pytest.fixture
def karate_labels(karate) -> np.ndarray:
    """Club membership, integer-coded: 0 = "Mr. Hi", 1 = "Officer"."""
    return np.array([0 if karate.nodes[i]["club"] == "Mr. Hi" else 1 for i in karate.nodes()])


@pytest.fixture
def karate_random_layout(karate) -> np.ndarray:
    """An unstructured (n, 2) layout. PCG64 is stream-stable, so this is reproducible."""
    return np.random.default_rng(0).normal(size=(karate.number_of_nodes(), 2))


@pytest.fixture
def isolated_vertex_graph() -> nx.Graph:
    """A weighted path 0-1-2-3-4 plus a zero-degree vertex 5.

    No registered dataset has an isolated vertex — every one of them is connected — so the
    NaN-exclusion path in the neighbourhood metrics is unreachable from the roster and invisible
    to the karate pins. The path is weighted 1..4 so that ``weighted recall`` is not vacuously
    equal to ``recall`` here.
    """
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 2.0), (2, 3, 3.0), (3, 4, 4.0)])
    G.add_node(5)
    return G


@pytest.fixture
def isolated_vertex_layout(isolated_vertex_graph) -> np.ndarray:
    """An unstructured (6, 2) layout for :func:`isolated_vertex_graph`."""
    return np.random.default_rng(2).normal(size=(isolated_vertex_graph.number_of_nodes(), 2))


@pytest.fixture
def karate_structured_layout(karate, karate_labels) -> np.ndarray:
    """A structured (n, 2) layout: club along x, degree along y, with fixed jitter.

    Not a real layout, but it partially separates the labels and correlates with degree, so
    metrics land away from chance and a regression moves the number visibly. The separation is
    deliberately incomplete (0.6 against jitter scale 0.25) so that label accuracy is not a
    degenerate 1.0 that a broken implementation could also produce.
    """
    degrees = np.array([d for _, d in karate.degree()], dtype=float)
    jitter = np.random.default_rng(1).normal(scale=0.25, size=(karate.number_of_nodes(), 2))
    return np.column_stack([karate_labels * 0.6, degrees * 0.2]) + jitter


@pytest.fixture
def karate_coincident_layout(karate_structured_layout) -> np.ndarray:
    """``karate_structured_layout`` with vertices 0-8 collapsed onto one point.

    Nine coincident points, deliberately more than ``k + 1`` at both pinned ``k``, because that
    is the regime where the two leave-one-out conventions in the package diverge: sklearn's
    ``kneighbors(X=None)`` masks column 0 for a row whose own index did not survive into the
    ``k + 1`` query, while ``twin.metrics._loo_neighbors`` masks the last column by design. No
    registered dataset has a coincident vertex pair (`STATUS.md`), so nothing else can see it.
    """
    X = karate_structured_layout.copy()
    X[1:9] = X[0]
    return X


@pytest.fixture
def disjoint_paths() -> nx.Graph:
    """A 4-path and a triangle, disconnected — 7 nodes, two components.

    Every registered dataset is connected, so without this fixture the ``"lcc"`` / ``"mask"``
    policies and ``finite_pair_mask`` are exercised only on graphs where every pair is finite,
    and an inverted comparison inside either is invisible. Both counts are hand-computable:
    6 finite pairs under ``"lcc"`` (the 4-path alone), 9 under ``"mask"`` (6 + 3).
    """
    return nx.disjoint_union(nx.path_graph(4), nx.cycle_graph(3))
