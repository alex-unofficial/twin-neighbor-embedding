"""Layout quality metrics.

Every layout array here is ``(n_points, d)``. Conversion from the ``(d, n)`` orientation the
embedding layer returns happens exactly once, in ``twin.results.LayoutResult.from_raw``; see
CLAUDE.md. Orientation is asserted at entry, never inferred from ``shape`` — inference silently
picks the wrong axis whenever ``n == d`` or the input is square (spec §1.3).
"""

import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier


def require_n_by_d(X: np.ndarray, n_expected: int, name: str = "X") -> np.ndarray:
    """Assert that ``X`` is ``(n_expected, d)``. Never reorients.

    Args:
        X: the layout array.
        n_expected: the number of points the caller's graph or label vector implies.
        name: used in the error message.

    Returns:
        ``X`` as a 2-D float array.

    Raises:
        ValueError: if ``X`` is not 2-D or its first axis is not ``n_expected``. If the
            transpose would have fitted, the message says so — that is the common call-site
            mistake and it should be fixed at the boundary, not here.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_points, d), got shape {X.shape}")
    if X.shape[0] != n_expected:
        hint = ""
        if X.shape[1] == n_expected:
            hint = (
                f" -- {name}.T would fit, so this is probably a (d, n) array. Convert it once "
                "at the boundary via LayoutResult.from_raw rather than here."
            )
        raise ValueError(
            f"{name} must be (n_points, d) with n_points={n_expected}, got {X.shape}{hint}"
        )
    return X


def knn_classification_accuracy(X: np.ndarray, target: np.ndarray, n_neighbors):
    # Handle both scalar and iterable inputs
    if isinstance(n_neighbors, (list, np.ndarray)):
        accuracies = np.array([knn_classification_accuracy(X, target, n) for n in n_neighbors])

        return accuracies

    X = require_n_by_d(X, len(target), "X")

    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    knn_classifier.fit(X, target)

    # Generate target prediction from kNN classifier
    knn_prediction = knn_classifier.predict(None)

    # Compare predicted targets vs true targets to compute accuracy
    prediction_accuracy = np.mean(knn_prediction == target)

    return prediction_accuracy


def knn_neighbor_preservation_accuracy(X: np.ndarray, G: nx.Graph, n_neighbors, mode="recall"):
    # Handle both scalar and iterable inputs
    if isinstance(n_neighbors, (list, np.ndarray)):
        accuracies = np.array(
            [knn_neighbor_preservation_accuracy(X, G, n, mode) for n in n_neighbors]
        )

        return accuracies

    # Ensure mode is set correctly and set parameters accordingly
    if mode in {"recall", "jaccard", "precision"}:
        weight = None
    elif mode == "weighted recall":
        weight = "weight"
    else:
        raise ValueError(
            "mode must be one of 'recall', 'weighted recall', 'jaccard' or 'precision'"
        )

    X = require_n_by_d(X, G.number_of_nodes(), "X")

    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(X)

    # connectivity matrices (sparse)
    knn_adjacency_matrix = knn.kneighbors_graph(mode="connectivity")
    true_adjacency_matrix = nx.adjacency_matrix(G, weight=weight)

    # ensure CSR format for elementwise operations
    knn_adj = knn_adjacency_matrix.tocsr()
    true_adj = true_adjacency_matrix.tocsr()

    # element-wise multiply to count matching neighbors
    matching_neighbors = knn_adj.multiply(true_adj)

    # row-wise sums -> 1D numpy arrays
    matching_counts = np.asarray(matching_neighbors.sum(axis=1)).ravel()

    if mode in {"recall", "weighted recall"}:
        normalization = np.asarray(true_adj.sum(axis=1)).ravel()
    elif mode == "jaccard":
        normalization = np.asarray(true_adj.maximum(knn_adj).sum(axis=1)).ravel()
    elif mode == "precision":
        normalization = n_neighbors

    # avoid division by zero: mark degree-0 nodes as NaN and use nanmean
    with np.errstate(divide="ignore", invalid="ignore"):
        neighbor_accuracy = matching_counts / normalization

    neighbor_accuracy = np.where(normalization == 0, np.nan, neighbor_accuracy)

    mean_accuracy = np.nanmean(neighbor_accuracy)

    return mean_accuracy


# `point_distance_metric` lived here. Removed per spec §1.2: its `adjacency=True` branch
# raised (a sparse matrix reached `np.nanmean`) and was unreachable, and its `adjacency=False`
# branch normalised by the maximum pairwise distance rather than the mean, so it was a rough
# ancestor of ADR rather than ADR itself. Superseded by
# `twin.metrics.structural.adjacency_distance_ratio` (spec §2.1, step 3); until that lands the
# point-distance tables are simply absent rather than filled with a stand-in.
