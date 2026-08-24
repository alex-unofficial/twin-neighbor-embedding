"""Label-based metrics (spec §2.3): $k$-NN label accuracy, and the silhouette family later.

A metric here is a function of the embedded points and a **label vector**, which is per
*original vertex* — `LayoutGraph.vertex_and_edge_points` returns points in the original
indexing precisely so that a layout of $A_E$ or $A_{tne}$ can be scored against the dataset's
own labels (CLAUDE.md).

Only the $k$-NN accuracy family lives here so far; the silhouette metrics of §2.3 land at
spec §7 step 6.
"""

from dataclasses import dataclass

import numpy as np

from twin.metrics.common import _loo_neighbors, require_n_by_d


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
