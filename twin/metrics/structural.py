"""Structural metrics (spec §2.1): ADR, the $k$-NN neighbourhood family, SNS.

A metric here is a function of the embedded points and the **graph** — no labels. All layout
arrays are ``(n_points, d)``; see :mod:`twin.metrics.common`.
"""

from dataclasses import dataclass

import numpy as np
import networkx as nx
import scipy.sparse as sp

from twin.forms import SOURCE_KEY, graph_side
from twin.metrics.common import _loo_neighbors, mean_pairwise_distance, require_n_by_d


@dataclass
class NeighborPreservationReport:
    """Score plus the diagnostics needed to read it (spec §2.1, `DECISIONS.md` (18)).

    Attributes:
        k: the neighbourhood sizes evaluated.
        value: one score per ``k``.
        n_excluded: per ``k``, how many vertices had a zero normalisation and were dropped
            before the mean. §3.1.1 says a zero-degree vertex has no neighbourhood to recover,
            so excluding it is correct — but absorbing it silently is how a table comes to
            average over a denominator nobody stated. Which vertices those are depends on the
            criterion: recall and weighted recall normalise by graph degree and exclude the
            isolated ones; precision normalises by ``k`` and jaccard by the union with the
            embedded neighbourhood, and neither is ever zero.
        n_scored: per ``k``, how many vertices the mean actually ran over.
    """

    k: np.ndarray
    value: np.ndarray
    n_excluded: np.ndarray
    n_scored: np.ndarray

    def __post_init__(self):
        # `np.nanmean` over an all-NaN row warns and returns NaN, which reaches a table as a
        # blank rather than as an error. The two counts cannot disagree if they are checked.
        total = self.n_excluded + self.n_scored
        if len(set(total.tolist())) != 1:
            raise ValueError(
                f"n_excluded + n_scored must be the vertex count for every k, got {total}"
            )


def _preservation_report(
    X: np.ndarray, G: nx.Graph, k, criterion: str
) -> NeighborPreservationReport:
    """The shared core of the four neighbourhood metrics — eqs. (30)-(39).

    One ``NearestNeighbors`` fit and **one query at max(k)**, sliced per ``k``. The recursive
    re-fit per ``k`` it replaces was the dominant cost of a swept call.

    Note:
        Slicing is not identical to re-querying when the layout has exact duplicates: which of
        several coincident points count as "the k nearest" depends on how many were asked for,
        and there is nothing to break the tie with. On a layout with no duplicates — every
        registered dataset, every real layout so far — the two agree exactly.
    """
    n = G.number_of_nodes()
    X = require_n_by_d(X, n, "X")

    ks = np.atleast_1d(np.asarray(k, dtype=int))
    if ks.min() < 1:
        raise ValueError(f"k must be >= 1, got {ks.min()}")
    if ks.max() > n - 1:
        raise ValueError(f"k={ks.max()} exceeds n-1={n - 1}; there are not that many neighbours")

    weight = "weight" if criterion == "weighted_recall" else None
    if weight is not None:
        side = graph_side(G)
        if side == "distance":
            raise ValueError(
                f"weighted_recall weights a recovered neighbour by its affinity, but this graph "
                f"is stamped 'distance' ({G.graph.get(SOURCE_KEY)!r}) — on digits that attribute "
                "is a stored Euclidean distance, so the metric would reward recovering the "
                "*least* coupled neighbours. Pass the evaluation context's affinity graph "
                "(`LayoutGraph.affinity_graph()`)."
            )
        if side == "combinatorial":
            # The declaration wins over whatever is on the edges. football_2023 declares
            # `Unweighted()` on both sides and still carries a stray `weight` attribute — 732
            # edges at 1 and three at 2 — from the .mat it was built from. Reading those would
            # make eq. (39) fail on a graph the dataset says is unweighted.
            weight = None

    true_adj = sp.csr_array(nx.adjacency_matrix(G, nodelist=range(n), weight=weight))
    _, idx = _loo_neighbors(X, int(ks.max()))

    value = np.empty(ks.size)
    n_excluded = np.empty(ks.size, dtype=int)

    for i, kk in enumerate(ks):
        rows = np.repeat(np.arange(n), kk)
        knn_adj = sp.csr_array((np.ones(n * kk), (rows, idx[:, :kk].ravel())), shape=(n, n))
        matched = np.asarray(knn_adj.multiply(true_adj).sum(axis=1)).ravel()

        if criterion in ("recall", "weighted_recall"):
            normalisation = np.asarray(true_adj.sum(axis=1)).ravel()
        elif criterion == "jaccard":
            normalisation = np.asarray(true_adj.maximum(knn_adj).sum(axis=1)).ravel()
        elif criterion == "precision":
            normalisation = np.full(n, float(kk))
        else:
            raise ValueError(f"unknown criterion {criterion!r}")

        with np.errstate(divide="ignore", invalid="ignore"):
            per_vertex = matched / normalisation
        per_vertex = np.where(normalisation == 0, np.nan, per_vertex)

        n_excluded[i] = int(np.isnan(per_vertex).sum())
        value[i] = float(np.nanmean(per_vertex)) if n_excluded[i] < n else np.nan

    return NeighborPreservationReport(
        k=ks, value=value, n_excluded=n_excluded, n_scored=n - n_excluded
    )


def _score(report: NeighborPreservationReport, k):
    """A float for scalar ``k``, an array for array-like ``k`` — as `knn_accuracy` does.

    ``np.ndim(k)``, not ``isinstance(k, (list, np.ndarray))``: a tuple used to fall through the
    latter into the scalar path and then into an obscure sklearn error. Nothing here infers
    *sweep versus distribution* from the shape — `MetricSpec` declares that at step 7
    (`DECISIONS.md` (20)).
    """
    return report.value if np.ndim(k) else float(report.value[0])


def knn_recall_report(X: np.ndarray, G: nx.Graph, k=10) -> NeighborPreservationReport:
    """Eq. (30)-(32): the fraction of each vertex's graph neighbours found among its k nearest."""
    return _preservation_report(X, G, k, "recall")


def knn_recall(X: np.ndarray, G: nx.Graph, k=10):
    """$R^k$ — recovered graph neighbours over graph degree, averaged over vertices (eq. 32).

    Zero-degree vertices are excluded rather than scored at 0; see
    :class:`NeighborPreservationReport` for the count.
    """
    return _score(knn_recall_report(X, G, k), k)


def knn_precision_report(X: np.ndarray, G: nx.Graph, k=10) -> NeighborPreservationReport:
    """Eq. (33): recovered graph neighbours over ``k``."""
    return _preservation_report(X, G, k, "precision")


def knn_precision(X: np.ndarray, G: nx.Graph, k=10):
    """$P^k$ — recovered graph neighbours over ``k``, averaged over vertices (eq. 33).

    Nothing is excluded: the normalisation is ``k``, so an isolated vertex scores 0 rather than
    dropping out. That asymmetry with recall is deliberate and is what the two measure.
    """
    return _score(knn_precision_report(X, G, k), k)


def knn_jaccard_report(X: np.ndarray, G: nx.Graph, k=10) -> NeighborPreservationReport:
    """Eq. (34)-(35): intersection over union of the graph and embedded neighbourhoods."""
    return _preservation_report(X, G, k, "jaccard")


def knn_jaccard(X: np.ndarray, G: nx.Graph, k=10):
    """$J^k$ — |graph ∩ embedded| / |graph ∪ embedded|, averaged over vertices (eqs. 34-35).

    Nothing is excluded: the union always contains the ``k`` embedded neighbours.
    """
    return _score(knn_jaccard_report(X, G, k), k)


def weighted_recall_report(X: np.ndarray, G: nx.Graph, k=10) -> NeighborPreservationReport:
    """Eqs. (36)-(38): recall with each neighbour weighted by its affinity."""
    return _preservation_report(X, G, k, "weighted_recall")


def weighted_recall(X: np.ndarray, G: nx.Graph, k=10):
    r"""$R_w^k$ — recall with each recovered neighbour weighted by $w_{ij}$ (eqs. 36-38).

    Equals :func:`knn_recall` exactly on an unweighted graph (eq. 39), which is the cheap test
    that says the weighting is wired up rather than merely present.

    ``G`` must be **affinity-valued**: a graph stamped ``twin_side="distance"`` is refused.
    §2.1's literal signature implies `dataset.graph`'s ``weight`` attribute, but on digits that
    attribute is a stored Euclidean distance, and a metric that reads whichever number happens
    to be on the edge is a metric that inverts silently on the one dataset where the two differ
    (`DECISIONS.md`, step-3 decisions). Take the affinity graph from the evaluation context.
    """
    return _score(weighted_recall_report(X, G, k), k)


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
