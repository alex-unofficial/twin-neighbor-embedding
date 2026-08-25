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

from twin.metrics.common import _loo_neighbors, mean_pairwise_distance


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
