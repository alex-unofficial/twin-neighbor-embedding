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
