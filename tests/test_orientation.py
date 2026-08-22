"""Spec §1.3 — one orientation convention, enforced at entry rather than sniffed."""

import networkx as nx
import numpy as np
import pytest

from twin.metrics import knn_accuracy, knn_neighbor_preservation_accuracy, require_n_by_d


def test_require_n_by_d_accepts_canonical_orientation():
    X = np.zeros((10, 2))
    assert require_n_by_d(X, 10) is not None


def test_require_n_by_d_rejects_transposed_input_with_a_useful_message():
    X = np.zeros((2, 10))
    with pytest.raises(ValueError, match=r"X\.T would fit"):
        require_n_by_d(X, 10)


def test_require_n_by_d_rejects_non_2d():
    with pytest.raises(ValueError, match="must be 2-D"):
        require_n_by_d(np.zeros(10), 10)


def test_require_n_by_d_does_not_sniff_square_input():
    """The n == d case: a (2, 2) array is ambiguous, so it must be taken at face value."""
    X = np.arange(4.0).reshape(2, 2)
    out = require_n_by_d(X, 2)
    assert np.array_equal(out, X), "must not transpose a square array"


@pytest.mark.parametrize(
    "metric",
    [
        lambda X, G, labels: knn_neighbor_preservation_accuracy(X, G, 5),
        lambda X, G, labels: knn_accuracy(X, labels, 5),
    ],
    ids=["neighbor_preservation", "knn_accuracy"],
)
def test_metrics_reject_d_by_n_layouts(karate, karate_labels, karate_random_layout, metric):
    """Previously these silently transposed. They must now refuse."""
    assert metric(karate_random_layout, karate, karate_labels) is not None
    with pytest.raises(ValueError, match=r"\(n_points, d\)"):
        metric(karate_random_layout.T, karate, karate_labels)


def test_metric_rejects_wrong_point_count(karate, karate_random_layout):
    with pytest.raises(ValueError, match="n_points=34"):
        knn_neighbor_preservation_accuracy(karate_random_layout[:20], karate, 5)


def test_square_layout_is_not_reoriented_by_a_metric():
    """A 2-vertex graph in 2-D: n == d, so sniffing would have been a coin flip."""
    G = nx.path_graph(2)
    X = np.array([[0.0, 0.0], [1.0, 1.0]])  # (n=2, d=2)
    assert knn_neighbor_preservation_accuracy(X, G, 1) == pytest.approx(1.0)
