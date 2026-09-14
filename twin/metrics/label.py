r"""Label-based metrics (spec §2.3): the silhouette family and $k$-NN label accuracy.

A metric here is a function of the embedded points and a **label vector**, which is per
*original vertex* — `LayoutGraph.vertex_and_edge_points` returns points in the original
indexing precisely so that a layout of $A_E$ or $A_{tne}$ can be scored against the dataset's
own labels (CLAUDE.md).

**Singletons score 0, not ``nan``** — Rousseeuw's own convention, thesis §3.1.2. ``sklearn``
implements it, verified directly rather than assumed; what it does *not* implement is either
degenerate end, and those are handled here (see :func:`silhouette_report`).

The graph-side counterparts of §2.3 — Sil\* and the 1-hop accuracy — land in the two commits
after this one.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import silhouette_samples

from twin.metrics.common import _loo_neighbors, require_n_by_d


def _as_labels(labels, name: str = "labels") -> np.ndarray:
    """A 1-D label vector.

    The dimensionality check is not decoration: :func:`silhouette_graph` takes ``labels``
    *first* (`DECISIONS.md`), so a call written from muscle memory as
    ``silhouette_graph(X, labels)`` would otherwise reach ``np.unique`` with a layout array and
    return a number computed from nothing.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(
            f"{name} must be a 1-D vector, one entry per original vertex, got shape "
            f"{labels.shape}. A 2-D array here is usually a layout passed in the wrong "
            "position."
        )
    return labels


# --------------------------------------------------------------------------- #
#                        Silhouette (spec §2.3, eqs. 49-55)                    #
# --------------------------------------------------------------------------- #


@dataclass
class SilhouetteReport:
    r"""A silhouette score plus the class-size diagnostics needed to read it.

    Spec §0.5 decision 9 asks for these columns because a flat composite score can be
    class-size collapse rather than absent substructure: singleton classes contribute exactly 0
    by convention, so enough of them drag the mean toward 0 whatever the layout does.

    ⚠️ **The guard is these four numbers and nothing else** — no minimum-size filtering. §0.5
    decision 9's premise was measured against the roster and does not hold: `football_2023` has
    **11** conferences rather than 14, its composite has 15 classes with a minimum size of 4 and
    **no singletons**, and `tabula`'s composite is redundant (`compartment` is a function of
    `cell_ontology_class`, so it has the same 13 classes). A filter would be untested code for a
    case that does not occur — §0.5 decision 3's precedent. `DECISIONS.md` has the table.

    Attributes:
        value: the score, in $[-1, 1]$; ``nan`` where it is undefined (see
            :func:`silhouette_report`).
        n_classes: how many distinct labels the scored vertices carry.
        min_class_size: the smallest class among them.
        median_class_size: the median class size.
        n_singletons: how many classes have exactly one member, i.e. how many vertices
            contribute a conventional 0 regardless of the layout.
        n_excluded: vertices this metric itself dropped — always 0 on the layout side, and the
            out-of-component count for :func:`silhouette_graph` (`DECISIONS.md` (18)). The four
            class-size figures above describe the vertices that were **scored**, so a class
            lying entirely outside the largest component is absent from them and visible only
            here.
    """

    value: float
    n_classes: int
    min_class_size: int
    median_class_size: float
    n_singletons: int
    n_excluded: int = 0


def _silhouette_report(data, labels: np.ndarray, *, metric: str, n_excluded: int = 0):
    r"""The mean of Rousseeuw's per-vertex scores, plus diagnostics — eqs. (49)-(52).

    One core for both sides of the metric: ``metric="euclidean"`` over an ``(n, d)`` layout is
    the layout score, ``metric="precomputed"`` over an ``(n, n)`` distance matrix is Sil\*. The
    equations differ only in which $d$ they read, so implementing them twice would be two
    chances to get $b_i$'s $1/|C|$ wrong (it is not $1/(|C|-1)$ — vertex $i$ is not in $C$).

    **Both degenerate ends are handled here, because ``sklearn`` refuses both.** Its accepted
    range is $2 \le |\mathcal{L}| \le n-1$:

    * $|\mathcal{L}| = 1$ returns ``nan``. A label vector with one class is a property of the
      *data*, not a caller error, and this package's convention is ``nan`` for that and a raise
      for caller error (``require_n_by_d``). ``n_classes=1`` in the report makes it visible.
    * $|\mathcal{L}| = n$ — every class a singleton — returns exactly **0.0**, not ``nan``.
      Rousseeuw's convention already covers it: every $s_i$ is 0, so the mean is 0. A vector
      with one pair and $n-2$ singletons scores through ``sklearn`` normally, so returning
      ``nan`` one step further along would be a discontinuity of ours rather than of the metric.
    """
    labels = _as_labels(labels)
    classes, counts = np.unique(labels, return_counts=True)
    diagnostics = {
        "n_classes": int(classes.size),
        "min_class_size": int(counts.min()) if counts.size else 0,
        "median_class_size": float(np.median(counts)) if counts.size else float("nan"),
        "n_singletons": int((counts == 1).sum()),
        "n_excluded": int(n_excluded),
    }

    if classes.size < 2:
        return SilhouetteReport(value=float("nan"), **diagnostics)
    if classes.size == labels.size:
        return SilhouetteReport(value=0.0, **diagnostics)

    samples = silhouette_samples(data, labels, metric=metric)
    return SilhouetteReport(value=float(samples.mean()), **diagnostics)


def silhouette_report(X: np.ndarray, labels: np.ndarray) -> SilhouetteReport:
    r"""Sil — the layout-side silhouette score, with its class-size diagnostics (eqs. 49-52).

    .. math::

        a_i = \frac{1}{|C_i| - 1}\sum_{j \in C_i, j \neq i}\lVert x_j - x_i\rVert,
        \qquad
        b_i = \min_{C \neq C_i}\frac{1}{|C|}\sum_{j \in C}\lVert x_j - x_i\rVert,
        \qquad
        \mathrm{Sil}_i = \frac{b_i - a_i}{\max\{a_i, b_i\}}

    and $\mathrm{Sil}(X; \mathcal{l}) = \frac{1}{n}\sum_i \mathrm{Sil}_i$ — the mean over
    **every** vertex, singletons included at their conventional 0.

    Backed by ``sklearn.metrics.silhouette_samples``. Unlike :func:`knn_accuracy`, which was
    hand-rolled at step 3 to control the tie rule, there is no knob here worth reimplementing
    for — so the validation is an independent $a_i$/$b_i$ oracle in the test module, which
    shares no code with this (`tests/test_silhouette.py`). That is the session-4 lesson: a test
    comparing two callers of one function proves nothing, which is how `twin_adjacency` went
    three sessions untested.

    Args:
        X: ``(n, d)`` layout. Orientation is asserted against ``labels``, never inferred.
        labels: ``(n,)`` integer-coded labels, per original vertex.

    Returns:
        A :class:`SilhouetteReport`.

    Note:
        ⚠️ **Invariant in exact arithmetic, invariant to about eight digits here.**
        ``silhouette_samples`` computes Euclidean distances by the
        $\lVert x\rVert^2 + \lVert y\rVert^2 - 2x\cdot y$ expansion, which loses digits for a
        layout sitting far from the origin relative to its own extent: measured
        $1.8\times10^{-8}$ relative on `tabula`'s stored UMAP at an offset/extent ratio of 4000,
        against $1.2\times10^{-14}$ when the same code is handed exact distances.
        ``metric="precomputed"`` would remove it and is deliberately not used —
        ``silhouette_samples`` chunks its Euclidean reduction and never materialises $(n, n)$,
        where a precomputed matrix is 200 MB at tabula's size, once per label set. Real layouts
        sit within a few multiples of their extent of the origin, where the deviation is
        $10^{-14}$ or below (`tests/test_silhouette.py`).

    Note:
        ⚠️ **Coincident points inflate this metric**, as spec §0.5 decision 1 predicts: two
        points at one location give $a_i = 0$ and hence $s_i = 1$, which is the strongest
        possible score for the least informative layout. No registered dataset has a coincident
        *vertex pair*, but crowding at large λ can produce one in a layout, and then Sil rewards
        it. The behaviour is recorded in the tests rather than special-cased — there is no
        defensible repair inside the metric, and an epsilon would only move the threshold.
    """
    labels = _as_labels(labels)
    X = require_n_by_d(X, labels.size, "X")
    return _silhouette_report(X, labels, metric="euclidean")


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Score only — see :func:`silhouette_report` for the diagnostics and full docstring."""
    return silhouette_report(X, labels).value


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
