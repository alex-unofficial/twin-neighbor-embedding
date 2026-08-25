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

⚠️ **A metric of ``X`` alone cannot check its own orientation.** Everywhere else in the package
``require_n_by_d`` catches a ``(d, n)`` array against the graph's node count; here there is
nothing to check against, so a transposed layout yields a number rather than an error. That is
the strongest reason for these to be reached through
:class:`~twin.results.LayoutResult`, which converts exactly once (CLAUDE.md).

All layout arrays are ``(n_points, d)``.
"""

import numpy as np
from scipy.spatial.distance import cdist

from twin.metrics.common import _loo_neighbors, mean_pairwise_distance, pairwise_distances

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

    Scale-invariant, and a **worst-case overlap detector**: it is determined by a single pair,
    so one nearly coincident pair drives it to near zero however readable the rest of the
    drawing is. That is what :func:`node_resolution_vertexwise` exists to soften, and it is why
    NR is reported beside it rather than instead of it.

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


def _rowwise_max_distance(X: np.ndarray) -> np.ndarray:
    r"""$\max_{j}\lVert x_i-x_j\rVert$ per row, in chunks.

    The diagonal needs no masking: $\lVert x_i-x_i\rVert = 0$ can only be the row maximum when
    every distance in the row is 0, and that is the collapsed case the caller refuses anyway.
    """
    n = X.shape[0]
    rows = max(1, MAX_NORMALISER_CHUNK_BYTES // (8 * max(n, 1)))
    out = np.empty(n)
    for start in range(0, n, rows):
        stop = min(start + rows, n)
        out[start:stop] = cdist(X[start:stop], X).max(axis=1)
    return out


def node_resolution_vertexwise(X: np.ndarray, normalize: str = "mean") -> float:
    r"""VNR — per-vertex nearest-neighbour distance against a normaliser (eq. 69).

    .. math::

        \mathrm{VNR}(X) = \frac{1}{n}\sum_{i\in V}
            \frac{\min_{j\neq i}\lVert x_i-x_j\rVert}{N_i}

    Unlike :func:`node_resolution` it is not decided by a single pair, so it stays sensitive to
    crowding while reflecting the typical readability of the layout. Larger is better separated.

    Args:
        X: an ``(n, d)`` layout.
        normalize: what $N_i$ is.

            ``"mean"`` **(default)** — the global $\bar d$, so every vertex is judged against
            one scale. **[THESIS EDIT]**: eq. (69) specifies ``"max"``, and this deliberately
            differs (spec §0.5). With a single distant vertex, *every* per-vertex maximum is
            approximately the diameter, so the ``"max"`` form stops being per-vertex at all and
            collapses back toward the global NR it was introduced to fix — measured below.

            ``"max"`` — the thesis's $\max_{j\neq i}\lVert x_i-x_j\rVert$, each vertex against
            its own visual range. Kept selectable so the two can be compared on real layouts, as
            §0.5 requires. Chunked: it is the only $O(n^2)$-memory computation in this module.

    Returns:
        The mean ratio, or ``nan`` for a layout collapsed to a single point.

    Note:
        ⚠️ **With ``normalize="mean"`` this is exactly** ``normalized_nn_distance(X, 1)``.
        Both are $\frac{1}{n}\sum_i \min_{j\neq i}\lVert x_i-x_j\rVert$ over $\bar d$; the sum
        and the division commute, so the edit does not merely put NND and VNR in one family, it
        makes VNR the $k=1$ member of it. That is not a defect — NND$_k$ is reported at $k$ =
        mean degree, which is a different number — but it does mean the ``"mean"`` form carries
        no information NND$_k$ does not, whereas the thesis's ``"max"`` form does. A test pins
        the identity, and `DECISIONS.md` records it as something the author should know when
        deciding what to report.
    """
    if normalize not in ("mean", "max"):
        raise ValueError(f"normalize must be 'mean' or 'max', got {normalize!r}")

    X = _as_layout(X)
    if X.shape[0] < 2:
        return float("nan")

    nearest, _ = _loo_neighbors(X, 1)
    nearest = nearest[:, 0]

    if normalize == "mean":
        denominator = mean_pairwise_distance(X)
        if denominator == 0:
            return float("nan")
        return float(nearest.mean() / denominator)

    widest = _rowwise_max_distance(X)
    if widest.min() == 0:
        # Only possible when every distance is 0: a row's maximum is 0 iff that vertex is
        # coincident with every other, which for n > 1 means the whole layout has collapsed.
        return float("nan")
    return float((nearest / widest).mean())


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
