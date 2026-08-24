"""Structural metrics (spec §2.1): ADR, the $k$-NN neighbourhood family, SNS.

A metric here is a function of the embedded points and the **graph** — no labels. All layout
arrays are ``(n_points, d)``; see :mod:`twin.metrics.common`.
"""

import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors

from twin.metrics.common import require_n_by_d


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
