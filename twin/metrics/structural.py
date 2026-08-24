"""Structural metrics (spec §2.1): ADR, the $k$-NN neighbourhood family, SNS.

A metric here is a function of the embedded points and the **graph** — no labels. All layout
arrays are ``(n_points, d)``; see :mod:`twin.metrics.common`.
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from twin.forms import SOURCE_KEY, graph_side
from twin.metrics.common import mean_pairwise_distance, require_n_by_d


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


def adjacency_distance_ratio(X: np.ndarray, G: nx.Graph, weight: str = "weight") -> float:
    r"""ADR — the weighted mean edge length over the mean pairwise distance (eqs. 27, 29).

    .. math::

        \mathrm{ADR}(X;G)=\frac{\frac{1}{W}\sum_{i<j} w_{ij}\lVert x_i-x_j\rVert}
        {\frac{2}{n(n-1)}\sum_{i<j}\lVert x_i-x_j\rVert},\qquad W=\sum_{i<j}w_{ij}

    Low is good: adjacent vertices are close relative to the layout's own scale. A complete
    graph gives exactly 1 for any layout, since the two averages are then over the same pairs.

    **Comparable across layouts of a fixed graph under a fixed weighting, and across nothing
    else.** Not across datasets — the denominator is a property of the layout and the numerator
    ranges over a different edge set. Not across a re-parameterised weighting either: $w$ and
    $w^2$ put different mass on the same edges, so the two numbers are not on a common scale.
    The reporting layer enforces this at spec §5.3; this paragraph is what makes that
    enforcement read as a rule rather than a restriction.

    There is deliberately **no unweighted branch**: eq. (29) is eq. (27) with
    $w_{ij}\in\{0,1\}$, and ``nx.adjacency_matrix`` already yields unit weights when the
    attribute is absent. The eq.-(29) agreement is a *test*, not a code path.

    Args:
        X: an ``(n, d)`` layout, in the original vertex indexing.
        G: the graph, **affinity-valued**. A graph stamped ``twin_side="distance"`` by the
            `LayoutGraph` layer is refused: `dataset.graph`'s ``weight`` attribute is a stored
            Euclidean *distance* on digits, and passing it here would make ADR reward putting
            graph-distant pairs close together — bug §1.6 recurring one layer up, this time with
            no Spearman of −1 to make it visible. Take the affinity graph from the evaluation
            context. A bare unstamped graph is accepted, so §6's analytic cases need no
            `Dataset`.
        weight: the edge attribute holding the affinity. Absent means 1.

    Returns:
        The ratio. ``nan`` if the graph has no edges or the layout is a single point.
    """
    n = G.number_of_nodes()
    X = require_n_by_d(X, n, "X")

    side = graph_side(G)
    if side == "distance":
        raise ValueError(
            f"adjacency_distance_ratio needs an affinity-valued graph, got one stamped "
            f"'distance' ({G.graph.get(SOURCE_KEY)!r}). ADR weights each edge by w_ij, so "
            "feeding it lengths inverts the metric: it would reward placing graph-distant pairs "
            "close together (bug §1.6 one layer up). Pass the evaluation context's affinity "
            "graph."
        )

    A = sp.coo_array(nx.adjacency_matrix(G, nodelist=range(n), weight=weight))
    upper = A.coords[0] < A.coords[1]
    rows, cols, w = A.coords[0][upper], A.coords[1][upper], A.data[upper]

    W = w.sum()
    if W == 0:
        return float("nan")

    edge_lengths = np.linalg.norm(X[rows] - X[cols], axis=1)
    mean_pairwise = mean_pairwise_distance(X)
    if mean_pairwise == 0:
        return float("nan")

    return float((w @ edge_lengths) / W / mean_pairwise)
