r"""Label-based metrics (spec §2.3): the silhouette family and $k$-NN label accuracy.

A metric here is a function of the embedded points and a **label vector**, which is per
*original vertex* — `LayoutGraph.vertex_and_edge_points` returns points in the original
indexing precisely so that a layout of $A_E$ or $A_{tne}$ can be scored against the dataset's
own labels (CLAUDE.md).

**Singletons score 0, not ``nan``** — Rousseeuw's own convention, thesis §3.1.2. ``sklearn``
implements it, verified directly rather than assumed; what it does *not* implement is either
degenerate end, and those are handled here (see :func:`silhouette_report`).

⚠️ **:func:`silhouette_graph` scores the graph, not the layout, and it is not a baseline.**
It answers "are these labels reflected in the topology at all?", on a scale incomparable with
the layout score (spec §2.3, §5.3): it says when a low :func:`silhouette` is uninformative
because the labels were never recoverable. It is not a target a layout is trying to reach and
not a number to subtract from one. :func:`graph_neighborhood_accuracy` is the same thing for
the accuracy family and carries the same warning.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import silhouette_samples

from twin.forms import SOURCE_KEY, graph_side
from twin.metrics.common import _loo_neighbors, ideal_distances, require_n_by_d
from twin.metrics.context import EvaluationContext


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


#: The message :func:`silhouette_graph` refuses ``disconnected="mask"`` with, in one place
#: because three call paths can carry the policy in (an explicit kwarg, a context built under
#: it, and a raw array bearing its footprint).
_MASK_REFUSAL = (
    "silhouette_graph refuses disconnected='mask' (spec §4.4, §0.5 decision 4). Under 'mask' a "
    "vertex whose nearest *other* cluster lies in another component has b(i) = inf, so "
    "s(i) -> 1 and a **disconnected** layout scores better: the policy inflates exactly the "
    "quantity this metric measures. Acceptable for SNS, dangerous for Sil*. The refusal is "
    "unconditional rather than conditional on the graph actually being disconnected — every "
    "registered dataset is connected today, so a conditional guard would be inert until the "
    "first dataset that needs it, which is the one case where it must not be. Use "
    "disconnected='lcc' (the default); n_excluded reports what that discards."
)


def _ideal_for_silhouette(labels: np.ndarray, G, ideal, ideal_distance_kw: dict):
    r"""The $(n, n)$ ideal distances Sil\* scores against, resolved and policy-checked.

    :func:`~twin.metrics.structural._stress_pairs`' contract, with two differences forced by
    §4.4 and by ``sklearn``:

    * ``disconnected="mask"`` is refused on all three paths it can arrive by;
    * the **square** form is needed, not the condensed one, because ``metric="precomputed"``
      takes a matrix. :attr:`~twin.metrics.context.EvaluationContext.ideal` is therefore the
      right attribute to read, where the stresses read ``ideal_condensed``.
    """
    if ideal_distance_kw.get("disconnected") == "mask":
        raise ValueError(_MASK_REFUSAL)

    if ideal is None:
        if G is None:
            raise ValueError(
                "pass either a graph or precomputed ideal distances (as an array or an "
                "EvaluationContext)"
            )
        return ideal_distances(G, **ideal_distance_kw)

    if ideal_distance_kw:
        raise ValueError(
            f"ideal distances were supplied, so {sorted(ideal_distance_kw)} would be ignored. "
            "They configure how ideal distances are computed, which has already happened."
        )

    if isinstance(ideal, EvaluationContext):
        if ideal.disconnected == "mask":
            raise ValueError(_MASK_REFUSAL)
        if ideal.n != labels.size:
            raise ValueError(
                f"the context is for {ideal.n} vertices and labels has {labels.size} entries"
            )
        return ideal.ideal

    D = np.asarray(ideal, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(
            f"Sil* needs the square (n, n) ideal distances for metric='precomputed', got shape "
            f"{D.shape}. The condensed form the stress metrics take will not do here."
        )
    if D.shape[0] != labels.size:
        raise ValueError(
            f"ideal distances are ({D.shape[0]}, {D.shape[0]}) and labels has {labels.size} "
            "entries"
        )
    return D


def silhouette_graph_report(
    labels: np.ndarray, G: nx.Graph | None = None, *, ideal=None, **ideal_distance_kw
) -> SilhouetteReport:
    r"""Sil\* — the same silhouette over graph geodesics (eqs. 53-55). **Not a baseline.**

    .. math::

        a^*_i = \frac{1}{|C_i| - 1}\sum_{j \in C_i, j \neq i} d_{ij},
        \qquad
        b^*_i = \min_{C \neq C_i}\frac{1}{|C|}\sum_{j \in C} d_{ij},
        \qquad
        \mathrm{Sil}^*_i = \frac{b^*_i - a^*_i}{\max\{a^*_i, b^*_i\}}

    ⚠️ **This is a label-validity check on a scale incomparable with the layout score**
    (spec §2.3, §5.3). It answers "are these labels reflected in the topology at all?" — it says
    when a low :func:`silhouette` is uninformative because the labels were never recoverable
    from the graph. It is **not** a target the layout is trying to reach and not a number to
    subtract from the layout score: $d_{ij}$ is a shortest-path length in the graph's own units
    and $\lVert x_i - x_j\rVert$ is a Euclidean distance in the layout's, and the two are related
    by no fixed factor. §5.3 guard 2 keeps the pair out of adjacent columns of one table for
    that reason — column adjacency is itself an invitation to difference two incomparable
    quantities.

    Args:
        labels: ``(n,)`` integer-coded labels, per original vertex. **First**, unlike spec
            §2.3's ``(G, labels)``, because it is the only argument that is always required —
            the graph is optional once ``ideal`` is supplied. `DECISIONS.md` records the
            reorder, with (5)'s precedent; ``labels`` is checked for 1-D so the two orders
            cannot blur silently.
        G: the **distance**-valued graph, if ``ideal`` is not supplied. Refused if it is stamped
            ``"affinity"`` — by :func:`~twin.metrics.common.ideal_distances`, which owns that
            guard.
        ideal: precomputed ideal distances: a square ``(n, n)`` array or the
            :class:`~twin.metrics.context.EvaluationContext` itself. **The context is the right
            call on the results path** — it computes the closure once per dataset, where
            recomputing it per label set is 200 MB and seconds at tabula's size, and tabula has
            four label sets. Note this reads :attr:`~twin.metrics.context.EvaluationContext.
            ideal` and not ``ideal_condensed``: ``sklearn``'s ``metric="precomputed"`` takes a
            matrix.
        **ideal_distance_kw: forwarded to :func:`~twin.metrics.common.ideal_distances`; refused
            when ``ideal`` is supplied, since it would silently do nothing.

    Returns:
        A :class:`SilhouetteReport`. ``n_excluded`` counts the vertices outside the largest
        connected component, which the ``"lcc"`` default scores nothing for.

    Raises:
        ValueError: on ``disconnected="mask"`` (see :data:`_MASK_REFUSAL`), on a condensed or
            mis-shaped ``ideal``, or on an ``ideal`` whose infinities are not the ``"lcc"``
            footprint.

    Note:
        **Under ``"lcc"``, ``ideal_distances`` returns the full ``(n, n)`` with ``inf`` outside
        the component — it does not subset, so this does** (`DECISIONS.md`). That is not
        optional tidiness: ``silhouette_samples`` rejects a precomputed matrix containing
        ``inf``, so the alternative is not an inflated number but an sklearn error message about
        input validation. The explicit subset is what makes ``n_excluded`` reportable.
    """
    labels = _as_labels(labels)
    D = _ideal_for_silhouette(labels, G, ideal, ideal_distance_kw)

    finite = np.isfinite(D)
    if finite.all():
        keep = np.ones(labels.size, dtype=bool)
    else:
        # The largest set of mutually reachable vertices, read off the row with the most finite
        # entries. Policy-independent, and then checked against the "lcc" footprint rather than
        # trusted: under "lcc" an excluded vertex is finite only on its own diagonal, so more
        # than one finite entry in an excluded row means the array came from a policy that kept
        # other components' internal distances — i.e. "mask", arriving as a bare array where
        # neither of the two guards above can see it.
        keep = finite[finite.sum(axis=1).argmax()]
        if (finite[~keep].sum(axis=1) > 1).any():
            raise ValueError(
                "these ideal distances are finite within a component other than the largest, "
                "which is the footprint of disconnected='mask'. " + _MASK_REFUSAL
            )
        if not finite[np.ix_(keep, keep)].all():
            raise ValueError(
                "the ideal distances have no single mutually-reachable block, so Sil* has no "
                "well-defined support. ideal_distances under 'lcc' produces one by "
                "construction; this array did not come from it."
            )

    n_excluded = int((~keep).sum())
    D_scored = np.ascontiguousarray(D[np.ix_(keep, keep)])
    # `metric="precomputed"` requires an exactly zero diagonal, and subsetting preserves it —
    # asserted rather than assumed, because the failure is an sklearn message about the
    # diagonal in the middle of a batch run.
    np.fill_diagonal(D_scored, 0.0)

    return _silhouette_report(
        D_scored, labels[keep], metric="precomputed", n_excluded=n_excluded
    )


def silhouette_graph(
    labels: np.ndarray, G: nx.Graph | None = None, *, ideal=None, **ideal_distance_kw
) -> float:
    r"""Score only — see :func:`silhouette_graph_report` for the full docstring.

    ⚠️ A **label-validity check, not a baseline**: incomparable in scale with
    :func:`silhouette`, and kept out of the same table by §5.3 guard 2.
    """
    return silhouette_graph_report(labels, G, ideal=ideal, **ideal_distance_kw).value


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


def _vote_outcome(scores: np.ndarray, codes: np.ndarray, tie_rule: str):
    r"""Resolve a ``(n, |L|)`` vote mass into per-vertex correctness — eqs. (59)-(60), (63)-(64).

    Shared by the layout-side and graph-side classifiers, because the *voting* is the same in
    both and only the neighbourhood differs: $\mathcal{N}^k_X(i)$ for one, $\mathcal{N}_G(i)$
    for the other. Restating it would be a second place for the fractional rule to drift.

    Exact equality decides a tie: under uniform weighting the scores are integer counts, and
    under either weighted scheme a near-tie is not a tie.

    Args:
        scores: vote mass per vertex per class. Rows the caller means to exclude must be
            removed **before** this call — an all-zero row reads as every class tied.
        codes: the true class code per row, in the same ordering as ``scores``' columns.
        tie_rule: ``"fractional"`` or ``"first"``.

    Returns:
        ``(correct, n_ties)`` — per-row values in $[0, 1]$, and how many rows had more than one
        class tied for the argmax.
    """
    tied = scores == scores.max(axis=1, keepdims=True)
    p = tied.sum(axis=1)

    if tie_rule == "fractional":
        correct = np.where(tied[np.arange(codes.size), codes], 1.0 / p, 0.0)
    else:
        correct = (np.argmax(scores, axis=1) == codes).astype(float)

    return correct, int((p > 1).sum())


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

        correct, n_ties[i] = _vote_outcome(scores, codes, tie_rule)
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


# --------------------------------------------------------------------------- #
#             The 1-hop graph-side accuracy (spec §2.3, eqs. 61-64)            #
# --------------------------------------------------------------------------- #


@dataclass
class GraphAccuracyReport:
    r"""Acc\* plus the diagnostics needed to read it (spec §2.3, `DECISIONS.md` (18)).

    Attributes:
        accuracy: the score, over the vertices actually scored; ``nan`` if none were.
        majority_class_rate: the chance floor, **over those same vertices** rather than over
            all of them, so that the two numbers share a denominator. Accuracy on imbalanced
            labels is misleading without it — `football_2023`'s 11 conferences range from 4
            teams to 18.
        n_ties: how many scored vertices had more than one class tied for the argmax. A
            variable-size neighbourhood ties far more readily than a fixed $k$ does: a
            degree-2 vertex with one neighbour of each of two classes is a tie, and under
            uniform weighting every even split is one.
        n_excluded: vertices with no vote mass, which the score does not average over
            (§2.3). Under ``weighting="uniform"`` that is exactly the isolated ones.
        n_scored: how many vertices the mean actually ran over.
        n_isolated: the structural count — vertices of degree 0. Equal to ``n_excluded``
            except under affinity weighting on a graph carrying a zero affinity, which no
            registered dataset does.
    """

    accuracy: float
    majority_class_rate: float
    n_ties: int
    n_excluded: int
    n_scored: int
    n_isolated: int

    def __post_init__(self):
        # Every isolated vertex is excluded, and possibly more under affinity weighting. The
        # counts come from two different expressions -- a nonzero-count and a row-sum -- so the
        # one relation that must hold between them is worth checking, in the spirit of
        # `NeighborPreservationReport.__post_init__`.
        if self.n_isolated > self.n_excluded:
            raise ValueError(
                f"n_isolated={self.n_isolated} exceeds n_excluded={self.n_excluded}: an "
                "isolated vertex has no vote mass and must always be excluded"
            )


def graph_neighborhood_accuracy_report(
    labels: np.ndarray,
    G: nx.Graph,
    *,
    weighting: str = "uniform",
    tie_rule: str = "fractional",
) -> GraphAccuracyReport:
    r"""Acc\* — label recovery from the 1-hop graph neighbourhood (eqs. 61-64).
    **Not a baseline.**

    .. math::

        s^*_i(\ell) = |\mathcal{N}_G(i) \cap C(\ell)|
        \quad\text{(eq. 61)},
        \qquad
        s^*_i(\ell) = \sum_{j \in \mathcal{N}_G(i) \cap C(\ell)} w_{ij}
        \quad\text{(eq. 62)}

    and $\mathrm{Acc}^*(G; \ell) = \frac{1}{n}\sum_i [\hat\ell^*_i = \ell_i]$ over the
    non-isolated vertices.

    The graph twin of :func:`knn_accuracy`, and the difference is the neighbourhood rather than
    the voting: $\mathcal{N}_G(i)$ of variable size $\deg(i)$ in place of a fixed $k$. The
    fractional-tie rule, the argmax and the chance floor are literally the same code
    (:func:`_vote_outcome`).

    ⚠️ **A label-validity check on an incomparable scale** (spec §2.3, §5.3), like
    :func:`silhouette_graph`. It answers "are these labels recoverable from the topology at
    all?" — it says when a low :func:`knn_accuracy` is uninformative. Its neighbourhood is not
    the layout's: variable-size against fixed $k$, so even the units of "how much evidence per
    vertex" differ. §5.3 guard 2 keeps it out of any table adjacent to the layout score.

    Args:
        labels: ``(n,)`` integer-coded labels, per original vertex. **First**, for the same
            reason as :func:`silhouette_graph` and so that the two graph-side label metrics read
            alike; spec §2.3 writes ``(G, labels)``. `DECISIONS.md` records the reorder.
        G: the graph. ⚠️ **Pass the evaluation context's affinity graph —
            ``ctx.affinity`` — never ``dataset.graph``.** On `digits` that attribute is a
            stored Euclidean *distance*, so eq. (62) would vote with an inverted sign: the
            least-coupled neighbours would count most. That is bug §1.6 one layer up, a bug
            this project has already shipped once. `tabula`'s graph is unstamped, so the guard
            below is a backstop and not a substitute for asking the context.
        weighting: ``"uniform"`` is eq. (61) — one vote per neighbour, reading no valuation at
            all. ``"affinity"`` is eq. (62), voting with $w_{ij}$. **Not** ``"distance"``,
            which is :func:`knn_accuracy`'s second mode and the opposite quantity: there
            $f(d) = 1/d$ turns a length into a weight, whereas here the edge value *is* the
            coupling. Naming the graph mode ``"distance"`` would invite exactly the substitution
            the guard refuses.
        tie_rule: as :func:`knn_accuracy_report` — ``"fractional"`` (the expectation over
            uniform random tie-breaking, in closed form) or ``"first"``.

    Returns:
        A :class:`GraphAccuracyReport`.

    Raises:
        ValueError: on an unknown ``weighting`` or ``tie_rule``, a label/graph size mismatch,
            or — under ``weighting="affinity"`` — a graph stamped ``"distance"``.

    Note:
        A ``"combinatorial"``-stamped graph is voted on uniformly even when asked for
        ``"affinity"``: the *declaration* wins over whatever is on the edges. `football_2023`
        declares ``Unweighted()`` on both sides and still carries a stray ``weight`` attribute
        from the ``.mat`` it was built from — 732 edges at 1 and three at 2 — and reading those
        would make eq. (62) disagree with eq. (61) on a graph the dataset says is unweighted.
        :func:`~twin.metrics.structural.weighted_recall` takes the same position.
    """
    if weighting not in {"uniform", "affinity"}:
        raise ValueError(
            f"weighting must be 'uniform' (eq. 61) or 'affinity' (eq. 62), got {weighting!r}. "
            "'distance' is knn_accuracy's mode and is the opposite quantity here."
        )
    if tie_rule not in {"fractional", "first"}:
        raise ValueError("tie_rule must be 'fractional' or 'first'")

    labels = _as_labels(labels)
    n = G.number_of_nodes()
    if labels.size != n:
        raise ValueError(
            f"labels has {labels.size} entries and the graph has {n} vertices; labels are "
            "per original vertex"
        )

    weight = None
    if weighting == "affinity":
        side = graph_side(G)
        if side == "distance":
            raise ValueError(
                f"graph_neighborhood_accuracy with weighting='affinity' votes with w_ij, but "
                f"this graph is stamped 'distance' ({G.graph.get(SOURCE_KEY)!r}) — on digits "
                "that attribute is a stored Euclidean distance, so the vote would count the "
                "*least* coupled neighbours most (bug §1.6 one layer up). Pass the evaluation "
                "context's affinity graph (`ctx.affinity`)."
            )
        weight = None if side == "combinatorial" else "weight"

    classes, codes = np.unique(labels, return_inverse=True)
    A = sp.csr_array(nx.adjacency_matrix(G, nodelist=range(n), weight=weight), dtype=float)

    # Vote mass per vertex per class: one sparse product, no per-vertex loop over neighbours.
    onehot = sp.csr_array(
        (np.ones(n), (np.arange(n), codes)), shape=(n, classes.size), dtype=float
    )
    scores = (A @ onehot).toarray()

    degrees = np.asarray((A != 0).sum(axis=1)).ravel()
    scored = np.asarray(A.sum(axis=1)).ravel() > 0
    n_scored = int(scored.sum())

    if n_scored == 0:
        return GraphAccuracyReport(
            accuracy=float("nan"),
            majority_class_rate=float("nan"),
            n_ties=0,
            n_excluded=n,
            n_scored=0,
            n_isolated=int((degrees == 0).sum()),
        )

    correct, n_ties = _vote_outcome(scores[scored], codes[scored], tie_rule)

    return GraphAccuracyReport(
        accuracy=float(correct.mean()),
        # Over the scored vertices, so the score and its floor share a denominator.
        majority_class_rate=majority_class_rate(labels[scored]),
        n_ties=n_ties,
        n_excluded=n - n_scored,
        n_scored=n_scored,
        n_isolated=int((degrees == 0).sum()),
    )


def graph_neighborhood_accuracy(
    labels: np.ndarray,
    G: nx.Graph,
    *,
    weighting: str = "uniform",
    tie_rule: str = "fractional",
) -> float:
    r"""Score only — see :func:`graph_neighborhood_accuracy_report` for the full docstring.

    ⚠️ A **label-validity check, not a baseline**: its neighbourhood is the graph's 1-hop one
    and its scale is incomparable with :func:`knn_accuracy`'s fixed $k$. §5.3 guard 2 keeps the
    two out of one table.
    """
    return graph_neighborhood_accuracy_report(
        labels, G, weighting=weighting, tie_rule=tie_rule
    ).accuracy
