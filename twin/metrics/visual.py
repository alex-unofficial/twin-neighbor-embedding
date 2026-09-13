r"""Visual-quality metrics (spec §2.2): point crowding, distance distributions, edge lengths.

These measure *readability of the drawing*, independently of any structural correspondence to
the graph — thesis §3.1.3. A layout can preserve every neighbourhood and still be unreadable
because half its vertices sit on top of each other, and nothing in :mod:`twin.metrics.structural`
would say so.

**These metrics take ``X`` alone; only :func:`edge_lengths` takes a graph, and it reads only the
edge set and its order.** No `EvaluationContext`: no ideal distances, no affinities, no labels.
So there is deliberately **no stamp guard** here — every other graph-consuming metric in the
package has one, and its absence in this module is a consequence of reading no valuation rather
than an omission.

**Edge length here is the chord, never the arc** (CLAUDE.md, thesis eq. 79). Step 5's LWCr
weights each crossing by the lengths of the two edges involved, and it takes them from
:func:`edge_lengths`; measuring the arc instead would score the curved variant of a layout
against a different set of weights than the straight one and confound "fewer crossings" with
"longer curves". The arc-over-chord ratio is its own metric — ``detour_ratio``, step 5.

⚠️ **A metric of ``X`` alone cannot check its own orientation.** Everywhere else in the package
``require_n_by_d`` catches a ``(d, n)`` array against the graph's node count; here there is
nothing to check against, so a transposed layout yields a number rather than an error. That is
the strongest reason for these to be reached through
:class:`~twin.results.LayoutResult`, which converts exactly once (CLAUDE.md).

All layout arrays are ``(n_points, d)``.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

from twin.metrics.common import (
    _loo_neighbors,
    edge_index_array,
    mean_pairwise_distance,
    pairwise_distances,
    require_n_by_d,
)

#: Row budget for the chunked row-wise maximum, in bytes. The ``normalize="max"`` branch of
#: :func:`node_resolution_vertexwise` is the **only** $O(n^2)$-memory computation in this module,
#: and squaring up tabula's condensed vector would be 200 MB on top of the 100 MB vector it is
#: already holding. Chunking bounds it at this instead, for one extra pass over the data.
MAX_NORMALISER_CHUNK_BYTES = 32 << 20


def _as_layout(X: np.ndarray, name: str = "X") -> np.ndarray:
    """A 2-D float view of a layout. Not an orientation check — see the module docstring."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_points, d), got shape {X.shape}")
    return X


def _neighborhood_ks(k, n: int) -> np.ndarray:
    """``k`` as a 1-D int array, validated against how many neighbours exist."""
    ks = np.atleast_1d(np.asarray(k, dtype=int))
    if ks.min() < 1:
        raise ValueError(f"k must be >= 1, got {ks.min()}")
    if ks.max() > n - 1:
        raise ValueError(f"k={ks.max()} exceeds n-1={n - 1}; there are not that many neighbours")
    return ks


def _scalar_or_array(values: np.ndarray, k):
    """A float for scalar ``k``, an array for array-like ``k`` — as the kNN family does.

    ``np.ndim(k)``, not ``isinstance``: a tuple falls through the latter into the scalar path.
    Nothing infers *sweep versus distribution* from the return shape; `MetricSpec` declares that
    at step 7 (`DECISIONS.md` (20)).
    """
    return values if np.ndim(k) else float(values[0])


# --------------------------------------------------------------------------- #
#                        Point crowding: NND (eqs. 65-67)                      #
# --------------------------------------------------------------------------- #


def normalized_nn_distance(X: np.ndarray, k=10):
    r"""NND$_k$ — local spacing against the layout's global scale (eqs. 65-67).

    .. math::

        \mathrm{NND}_k(X) = \frac{d_k(X)}{\bar d}
        = \frac{\frac{1}{n}\sum_{i\in V}\frac{1}{k}
                \sum_{j\in\mathcal{N}^k_X(i)}\lVert x_i-x_j\rVert}
               {\frac{2}{n(n-1)}\sum_{i<j}\lVert x_i-x_j\rVert}

    **Small is crowded**: the nearest embedded neighbours of a typical vertex are close relative
    to the size of the drawing. Larger means vertices are better separated locally and are
    easier to tell apart. It is scale-invariant, since numerator and denominator carry the same
    units — which is the whole reason for the normalisation, layouts having no common unit.

    Note that unlike every metric in :mod:`twin.metrics.structural`, this says nothing about
    whether the neighbours are the *right* ones. A layout that crowds the wrong vertices
    together scores exactly like one that crowds the right ones.

    Args:
        X: an ``(n, d)`` layout.
        k: neighbourhood size, scalar or array-like. Swept by querying $\max(k)$ once and
            slicing, as the kNN family does. Unlike there, slicing is exactly equivalent here
            even on a layout with duplicates: this reads the neighbour *distances*, whose sorted
            multiset is unique, and never the neighbour identities, which are not
            (`DECISIONS.md`, step-3 departures). A test pins that on the coincident fixture.

    Returns:
        A float for scalar ``k``, an array for array-like ``k``. ``nan`` for a layout collapsed
        to a single point, where $\bar d = 0$: the limit exists but reporting a limit as a score
        is how an undefined quantity reaches a table (as in SNS).
    """
    X = _as_layout(X)
    ks = _neighborhood_ks(k, X.shape[0])

    mean_pairwise = mean_pairwise_distance(X)
    if mean_pairwise == 0:
        return _scalar_or_array(np.full(ks.size, np.nan), k)

    dist, _ = _loo_neighbors(X, int(ks.max()))
    values = np.array([dist[:, :kk].mean() / mean_pairwise for kk in ks])
    return _scalar_or_array(values, k)


# --------------------------------------------------------------------------- #
#                    Node resolution: NR and VNR (eqs. 68, 69)                 #
# --------------------------------------------------------------------------- #


def node_resolution(X: np.ndarray) -> float:
    r"""NR — the closest pair of vertices against the widest (eq. 68).

    .. math::

        \mathrm{NR}(X) = \frac{\min_{i<j}\lVert x_i-x_j\rVert}
                              {\max_{i<j}\lVert x_i-x_j\rVert}

    ⚠️ **Not one of the reported metrics — VNR is.** The thesis introduces eq. (68) in order to
    argue against it: it is determined by a single pair, so one nearly coincident pair drives it
    to near zero however readable the rest of the drawing is, and "for this reason, we use a
    vertex-wise averaged version". Its summary of the suite names NND$_k$ and VNR as the
    point-crowding scores and does not include NR. This function exists because
    :func:`node_resolution_vertexwise` is *defined* by contrast with it, and because a quantity
    the argument rests on should be computable rather than asserted.

    Measured on `digits`, the argument holds quantitatively: NR is $1\times10^{-5}$ over 1797
    vertices with a 54 % relative IQR across seeds, where NND's is 0.4 % — it reports which seed
    happened to place two points nearest each other (`DECISIONS.md`). Scale-invariant, and
    useful as a diagnostic of the lower tail beside the distributions; not a score to rank
    layouts by.

    Returns:
        The ratio, 0 when any two vertices coincide, ``nan`` for a layout collapsed to a single
        point — where the denominator is 0 and the quantity is undefined rather than 0 or 1.
    """
    X = _as_layout(X)
    if X.shape[0] < 2:
        return float("nan")

    distances = pairwise_distances(X)
    widest = float(distances.max())
    if widest == 0:
        return float("nan")
    return float(distances.min()) / widest


def _rowwise_denominator(X: np.ndarray, normalize: str) -> np.ndarray:
    r"""The per-vertex normaliser $N_i$ of eq. (69), in chunks.

    Chunked because this is the only $O(n^2)$-memory computation in the module: squaring up
    tabula's condensed vector would be 200 MB on top of the 100 MB vector already held.

    The diagonal is handled differently by the two per-vertex forms, and getting that wrong is
    silent:

    * ``"max"`` — $\lVert x_i-x_i\rVert = 0$ can only be a row's maximum when every distance in
      the row is 0, which is the collapsed layout the caller has already refused. No masking
      needed.
    * ``"median"`` — a spurious zero **does** shift the median, by half a rank position. The
      self entry is pushed to $+\infty$ so it sorts last, and the order statistics are then
      taken over the $n-1$ genuine distances.
    """
    n = X.shape[0]
    rows = max(1, MAX_NORMALISER_CHUNK_BYTES // (8 * max(n, 1)))
    out = np.empty(n)

    # Order statistics of the n-1 *other* distances, 0-indexed into the sorted row.
    m = n - 1
    lower, upper = (m // 2, m // 2) if m % 2 else (m // 2 - 1, m // 2)

    for start in range(0, n, rows):
        stop = min(start + rows, n)
        block = cdist(X[start:stop], X)
        if normalize == "max":
            out[start:stop] = block.max(axis=1)
            continue
        block[np.arange(stop - start), np.arange(start, stop)] = np.inf
        if lower == upper:
            out[start:stop] = np.partition(block, lower, axis=1)[:, lower]
        else:
            picked = np.partition(block, [lower, upper], axis=1)[:, [lower, upper]]
            out[start:stop] = picked.mean(axis=1)
    return out


@dataclass
class NodeResolutionReport:
    """VNR plus the diagnostic needed to read it (spec §2.2, `DECISIONS.md` (18)).

    Attributes:
        value: the score.
        normalize: which normaliser produced it. Carried because the three are not on a common
            scale and a bare number does not say which was used.
        n_zero_denominator: how many vertices had $N_i = 0$ and were clamped to ``eps``.
            Reachable only under ``"median"``, and only where **more than half** the other
            points coincide with that vertex — pathological, but it is the extreme collapse
            this metric exists to detect, so it is counted rather than absorbed. Belongs with
            `n_ties` and `n_zero_distance` in the diagnostics table, not with `n_excluded`.
    """

    value: float
    normalize: str
    n_zero_denominator: int


def node_resolution_vertexwise_report(
    X: np.ndarray, normalize: str = "median", *, eps: float = 1e-12
) -> NodeResolutionReport:
    """VNR with its clamp count — see :func:`node_resolution_vertexwise`."""
    if normalize not in ("median", "max", "mean"):
        raise ValueError(
            f"normalize must be 'median', 'max' or 'mean', got {normalize!r}"
        )

    X = _as_layout(X)
    n = X.shape[0]
    if n < 2:
        return NodeResolutionReport(float("nan"), normalize, 0)

    # One collapse test for all three forms, before any per-vertex work. A layout with every
    # point at one location has no scale to normalise by under any of them, and the limit is
    # not the score (`DECISIONS.md`, as in SNS).
    if mean_pairwise_distance(X) == 0:
        return NodeResolutionReport(float("nan"), normalize, 0)

    nearest = _loo_neighbors(X, 1)[0][:, 0]
    if normalize == "mean":
        denominator = np.full(n, mean_pairwise_distance(X))
    else:
        denominator = _rowwise_denominator(X, normalize)

    n_zero = int((denominator <= eps).sum())
    ratios = nearest / np.maximum(denominator, eps)
    return NodeResolutionReport(float(ratios.mean()), normalize, n_zero)


def node_resolution_vertexwise(
    X: np.ndarray, normalize: str = "median", *, eps: float = 1e-12
) -> float:
    r"""VNR — per-vertex nearest-neighbour distance against a per-vertex normaliser (eq. 69).

    .. math::

        \mathrm{VNR}(X) = \frac{1}{n}\sum_{i\in V}
            \frac{\min_{j\neq i}\lVert x_i-x_j\rVert}{N_i}

    Unlike :func:`node_resolution` it is not decided by a single pair, so it stays sensitive to
    crowding while reflecting the typical readability of the layout. Larger is better separated.

    ⚠️ **Not a "higher is better" score on its own.** VNR and NND reward *uniform* spread, and
    density variation is the signature of a good layout of clustered data — so on `digits` a
    random Gaussian control tops both while coming last on ADR and recall. Read them jointly
    with the structural metrics, never alone (`DECISIONS.md`).

    Args:
        X: an ``(n, d)`` layout.
        normalize: what $N_i$ is.

            ``"median"`` **(default)** — $\operatorname{median}_{j\neq i}\lVert x_i-x_j\rVert$.
            **[THESIS EDIT]**, the only one on this metric: eq. (69) specifies ``"max"``.
            Per-vertex, so it varies between core and periphery; and robust, because one
            distant vertex moves a median essentially not at all. Measured: pushing an outlier
            from 10 to 1000 units moves this by 2.1 % where ``"max"`` falls by a factor of 2.5.

            ``"max"`` — the thesis's $\max_{j\neq i}\lVert x_i-x_j\rVert$. Kept selectable, and
            it is what eq. (69) says, but it fails in **both** regimes: with a distant vertex
            every denominator is approximately the diameter, and *without* one the maxima are
            near-homogeneous and it reduces to $\mathrm{NND}_1\cdot\bar d/M$ — a rescaled
            NND$_1$, within 5.8 % of that closed form on all five real layouts measured. Either
            outlier-driven or redundant, never a clean second measurement.

            ``"mean"`` — the global $\bar d$. Spec §0.5's original edit, **superseded**: a
            constant factors out of the sum, so this is *exactly*
            ``normalized_nn_distance(X, 1)`` and is not vertex-wise at all. Kept selectable
            only so the identity stays demonstrable.

        eps: denominators at or below this are clamped before division, and counted. See
            :class:`NodeResolutionReport`.

    Returns:
        The mean ratio, or ``nan`` for a layout collapsed to a single point.

    Note:
        Under ``"median"`` and ``"max"`` the score is bounded in $[0, 1]$, since
        $\min_{j\neq i}$ is no greater than any order statistic of the same set; 1 is attained
        under ``"median"`` only when more than half the other points sit at exactly the minimum
        distance. Under ``"mean"`` it is not bounded above. A naturally bounded score is easier
        to read across layouts, which is a small further argument for the default.
    """
    return node_resolution_vertexwise_report(X, normalize, eps=eps).value


# --------------------------------------------------------------------------- #
#                   Distance distributions (eqs. 70, 71)                       #
# --------------------------------------------------------------------------- #


def pairwise_distance_distribution(X: np.ndarray) -> np.ndarray:
    r"""$\{\lVert x_i-x_j\rVert/\bar d : i<j\}$ — every pair, normalised (eq. 70).

    **A diagnostic, not a score.** §3.1.3 is explicit: these produce no scalar by which layouts
    can be ordered, and they are returned whole rather than summarised because summarising them
    here would pre-empt the choice the reporting layer has to make from the data (step 11).
    They tell you whether one layout has a longer lower tail of very small distances, whether
    its distances are more evenly spread, and whether NND or VNR is being driven by a handful of
    near-overlaps.

    ⚠️ **Large.** One value per pair: 561 for karate, 12.5 million — 100 MB — for tabula.
    `DECISIONS.md` (20) is how step 7 persists it, in ``distributions/<key>.npz`` rather than in
    the metric frame, with ``status="distribution"``.

    Returns:
        The condensed distances over $\bar d$, in
        :data:`~twin.metrics.common.CONDENSED_ORDERING`, so it lines up elementwise with an
        ideal-distance vector. All-``nan`` for a collapsed layout.
    """
    X = _as_layout(X)
    distances = pairwise_distances(X)
    mean_pairwise = distances.mean() if distances.size else np.nan
    if not mean_pairwise:
        return np.full(distances.shape, np.nan)
    # A copy, not a view: `pairwise_distances` hands back the cached array itself.
    return distances / mean_pairwise


def nn_distance_distribution(X: np.ndarray) -> np.ndarray:
    r"""$\{\min_{j\neq i}\lVert x_i-x_j\rVert/\bar d : i\in V\}$ — per vertex (eq. 71).

    The per-vertex terms whose mean is VNR under the ``"mean"`` normaliser, and the local
    spacing values NND$_1$ averages. One value per **vertex**, so it is ``n`` long rather than
    $\binom{n}{2}$ — 5007 values on tabula against 12.5 million — and it is the cheaper of the
    two diagnostics by four orders of magnitude.

    A diagnostic, not a score; see :func:`pairwise_distance_distribution`.

    Returns:
        One normalised nearest-neighbour distance per vertex, in vertex order. All-``nan`` for a
        collapsed layout.
    """
    X = _as_layout(X)
    if X.shape[0] < 2:
        return np.full(X.shape[0], np.nan)

    nearest, _ = _loo_neighbors(X, 1)
    mean_pairwise = mean_pairwise_distance(X)
    if mean_pairwise == 0:
        return np.full(X.shape[0], np.nan)
    return nearest[:, 0] / mean_pairwise


# --------------------------------------------------------------------------- #
#                      Edge lengths (eqs. 79, 80)                              #
# --------------------------------------------------------------------------- #


def edge_lengths(X: np.ndarray, G: nx.Graph) -> np.ndarray:
    r"""$L(e_{ij}) = \lVert x_i - x_j\rVert$ per edge, **in ``G.edges()`` order** (eq. 79).

    **The chord, never the arc** (CLAUDE.md). The thesis is explicit that the length of an edge
    is the distance between its endpoints "regardless of the shape of the curve used to draw
    it", because it is the distance spanned across the layout, not the path taken, that decides
    how much of the drawing a crossing on that edge can disrupt. This is what keeps the
    straight-versus-curved comparison interpretable: under the arc, the curved variant of a
    layout would be scored against a *different* set of weights than the straight one, and LWCr
    would confound "fewer crossings" with "longer curves". The arc-over-chord ratio is a metric
    in its own right — ``detour_ratio``, step 5 — and is deliberately not folded in here.

    ⚠️ **The order is ``G.edges()``, and that is load-bearing.** ADR computes the same chord
    lengths off the CSR upper triangle (:func:`~twin.metrics.structural.adjacency_distance_ratio`),
    which is a *different* order: on `digits`, 12377 of 14732 positions disagree. Edge points,
    incidence-matrix columns and step 5's per-edge crossing weights are all indexed by
    ``G.edges()``, so this must be too. The two are not unified: ADR keeps the CSR order because
    its lengths have to align with the weights it reads out of the same matrix.

    Args:
        X: an ``(n, d)`` layout, in the original vertex indexing.
        G: the graph. Only its **edge set and their order** are read — no weights, no valuation,
            hence no side stamp to check (see the module docstring). Nodes must be
            ``0..n-1``, the indexing convention the whole package uses.

    Returns:
        One length per edge, ``(m,)``, in ``G.edges()`` order. Empty for an edgeless graph.
    """
    n = G.number_of_nodes()
    X = require_n_by_d(X, n, "X")

    if G.number_of_edges() == 0:
        return np.empty(0)

    edges = edge_index_array(G, n)
    return np.linalg.norm(X[edges[:, 0]] - X[edges[:, 1]], axis=1)


def edge_length_cv(X: np.ndarray, G: nx.Graph) -> float:
    r"""The coefficient of variation of the edge lengths: $\sigma_L/\bar L$.

    Uniform edge length is a classical drawing aesthetic, and $\bar L$ is eq. (80) — but the
    **CV itself is not a thesis equation**; it is spec §2.2's addition, a scale-invariant
    summary of the spread that eq. (80)'s mean alone does not carry. 0 means every edge is drawn
    at the same length; large means the drawing mixes very short and very long edges.

    The standard deviation is the population one (``ddof=0``), so the CV of a fixed drawing is a
    property of that drawing rather than an estimate of a population it was sampled from.

    Returns:
        $\sigma_L/\bar L$, or ``nan`` for a graph with no edges or a layout whose every edge has
        collapsed to zero length.
    """
    lengths = edge_lengths(X, G)
    if lengths.size == 0:
        return float("nan")
    mean = float(lengths.mean())
    if mean == 0:
        return float("nan")
    return float(lengths.std()) / mean
