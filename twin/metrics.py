"""Layout quality metrics.

Every layout array here is ``(n_points, d)``. Conversion from the ``(d, n)`` orientation the
embedding layer returns happens exactly once, in ``twin.results.LayoutResult.from_raw``; see
CLAUDE.md. Orientation is asserted at entry, never inferred from ``shape`` — inference silently
picks the wrong axis whenever ``n == d`` or the input is square (spec §1.3).
"""

from dataclasses import dataclass

import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors


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


@dataclass
class KnnAccuracyReport:
    """Score plus the diagnostics needed to read it (spec §2.3).

    Attributes:
        k: the neighbourhood sizes evaluated.
        accuracy: one score per ``k``.
        majority_class_rate: the chance floor — the rate obtained by always predicting the
            most common label. Accuracy on imbalanced labels is misleading without it.
        n_ties: per ``k``, how many vertices had more than one label tied for the argmax.
        n_zero_distance: per ``k``, how many neighbour distances were clamped to ``eps``.
            Non-zero means coincident points, which is what crowding produces.
    """

    k: np.ndarray
    accuracy: np.ndarray
    majority_class_rate: float
    n_ties: np.ndarray
    n_zero_distance: np.ndarray


def majority_class_rate(labels: np.ndarray) -> float:
    """The chance floor: the accuracy of always predicting the most frequent label."""
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / labels.size)


def _loo_neighbors(X: np.ndarray, k_max: int):
    """Indices and distances of the ``k_max`` nearest *other* points, per point.

    Queries ``k_max + 1`` and drops the self column. With coincident points the self entry is
    not guaranteed to sort first, so it is removed by index identity rather than by position.
    """
    n = X.shape[0]
    knn = NearestNeighbors(n_neighbors=k_max + 1).fit(X)
    dist, idx = knn.kneighbors(X, n_neighbors=k_max + 1)

    drop = idx == np.arange(n)[:, None]
    # A row can miss its own index only under exact duplicates; drop its furthest entry so
    # every row contributes exactly k_max neighbours.
    missing = ~drop.any(axis=1)
    drop[missing, -1] = True

    # Stable sort on the drop flag: False (keep) sorts first, in original distance order.
    order = np.argsort(drop, axis=1, kind="stable")[:, :k_max]
    return (
        np.take_along_axis(dist, order, axis=1),
        np.take_along_axis(idx, order, axis=1),
    )


def knn_accuracy_report(
    X: np.ndarray,
    labels: np.ndarray,
    k=10,
    weighting: str = "uniform",
    tie_rule: str = "fractional",
    eps: float = 1e-12,
) -> KnnAccuracyReport:
    """Leave-one-out $k$-NN label accuracy of a layout, with diagnostics.

    Implements thesis eqs. (57)-(60) directly rather than via ``KNeighborsClassifier``, so that
    the tie rule is under our control. Reimplementation for control, not a bug fix: the previous
    ``predict(None)`` call was valid sklearn and already left each point out of its own
    neighbour set (spec §0.5).

    Args:
        X: ``(n, d)`` layout.
        labels: ``(n,)`` integer-coded labels.
        k: neighbourhood size, scalar or array-like. Queried once at ``max(k)`` and sliced.
        weighting: ``"uniform"`` (eq. 57) or ``"distance"``, $f(d) = 1/d$ (eq. 58).
        tie_rule: ``"fractional"`` — a vertex with $p$ labels tied for the argmax contributes
            $1/p$ if its true label is among them, else 0. This is the expected accuracy under
            uniform random tie-breaking, in closed form: no variance, no seed, and therefore
            **an expectation over tie-breakings rather than a literal fraction of vertices**.
            ``"first"`` breaks toward the lowest label, matching ``KNeighborsClassifier``.
        eps: distances below this are clamped before inversion, so coincident points do not
            produce infinite weights. Occurrences are counted, not silently absorbed.

    Returns:
        A :class:`KnnAccuracyReport`.

    Note:
        The clamp differs from sklearn, which gives an exactly-coincident neighbour weight 1
        and every other neighbour weight 0. The two agree whenever no distance is zero.
    """
    if weighting not in {"uniform", "distance"}:
        raise ValueError("weighting must be 'uniform' or 'distance'")
    if tie_rule not in {"fractional", "first"}:
        raise ValueError("tie_rule must be 'fractional' or 'first'")

    labels = np.asarray(labels)
    X = require_n_by_d(X, labels.size, "X")
    n = labels.size

    ks = np.atleast_1d(np.asarray(k, dtype=int))
    if ks.min() < 1:
        raise ValueError(f"k must be >= 1, got {ks.min()}")
    if ks.max() > n - 1:
        raise ValueError(f"k={ks.max()} exceeds n-1={n - 1}; there are not that many neighbours")

    classes, codes = np.unique(labels, return_inverse=True)
    dist, idx = _loo_neighbors(X, int(ks.max()))
    neighbor_codes = codes[idx]

    accuracy = np.empty(ks.size)
    n_ties = np.zeros(ks.size, dtype=int)
    n_zero = np.zeros(ks.size, dtype=int)

    rows = np.arange(n)[:, None]
    for i, kk in enumerate(ks):
        d_k = dist[:, :kk]
        if weighting == "uniform":
            w = np.ones_like(d_k)
        else:
            n_zero[i] = int((d_k <= eps).sum())
            w = 1.0 / np.maximum(d_k, eps)

        # Per-vertex vote mass for each label.
        scores = np.zeros((n, classes.size))
        np.add.at(scores, (np.broadcast_to(rows, d_k.shape), neighbor_codes[:, :kk]), w)

        # Exact equality: under uniform weighting the scores are integer counts, and under
        # distance weighting a near-tie is not a tie.
        tied = scores == scores.max(axis=1, keepdims=True)
        p = tied.sum(axis=1)
        n_ties[i] = int((p > 1).sum())

        if tie_rule == "fractional":
            correct = np.where(tied[rows.ravel(), codes], 1.0 / p, 0.0)
        else:
            correct = (np.argmax(scores, axis=1) == codes).astype(float)

        accuracy[i] = float(correct.mean())

    return KnnAccuracyReport(
        k=ks,
        accuracy=accuracy,
        majority_class_rate=majority_class_rate(labels),
        n_ties=n_ties,
        n_zero_distance=n_zero,
    )


def knn_accuracy(
    X: np.ndarray,
    labels: np.ndarray,
    k=10,
    weighting: str = "uniform",
    tie_rule: str = "fractional",
    eps: float = 1e-12,
):
    """Score only — see :func:`knn_accuracy_report` for the diagnostics and full docstring.

    Returns a float for scalar ``k`` and an array for array-like ``k``, matching the metric
    signature shape declared in spec §2.
    """
    report = knn_accuracy_report(X, labels, k, weighting, tie_rule, eps)
    return report.accuracy if np.ndim(k) else float(report.accuracy[0])


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
