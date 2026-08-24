"""Shared metric machinery: orientation checks, the ideal-distance model, distance helpers.

Every layout array in this package is ``(n_points, d)``. Conversion from the ``(d, n)``
orientation the embedding layer returns happens exactly once, in
``twin.results.LayoutResult.from_raw``; see CLAUDE.md. Orientation is asserted at entry, never
inferred from ``shape`` — inference silently picks the wrong axis whenever ``n == d`` or the
input is square (spec §1.3).
"""

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph

from twin.layoutgraph import SOURCE_KEY, graph_side

#: One pair ordering for the whole package: scipy's condensed ordering, the row-major upper
#: triangle. ``pdist`` produces it, :func:`condensed` converts into it, SNS sums over it. Every
#: "which triangle, in what order" bug in a stress implementation comes from having two.
CONDENSED_ORDERING = "scipy condensed (row-major upper triangle, i < j)"

#: Edge-length models for :func:`ideal_distances` (spec §2.0, thesis §2.1.1).
WEIGHT_TRANSFORMS = {
    "identity": lambda w: w,
    "inverse": lambda w: 1.0 / w,
    "one_minus": lambda w: 1.0 - w,
    "neg_log": lambda w: -np.log(w),
}

class _Unset:
    """Sentinel: ``weight_transform`` was not passed at all.

    Distinguishing "not passed" from an explicit ``"identity"`` is what lets the escape-hatch
    guard in :func:`ideal_distances` fire on the second — a caller that names a transform has
    made a claim about the edge lengths, and that claim can contradict a declared source.
    """

    def __repr__(self) -> str:
        return "identity"


_UNSET = _Unset()


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


# --------------------------------------------------------------------------- #
#                          Ideal distances (spec §2.0)                         #
# --------------------------------------------------------------------------- #


def ideal_distances(
    G: nx.Graph,
    weight_transform: str | _Unset = _UNSET,
    weight: str = "weight",
    *,
    disconnected: str = "lcc",
    max_n: int = 20000,
) -> np.ndarray:
    r"""All-pairs graph distances $d_{ij}$ — the target SNS and Sil\* score a layout against.

    **Who should call this.** The runner, once per dataset, on
    ``vertex_graph(dataset).distance`` — and nothing else. That form is by construction the
    least-derived distance available: the stored distance side if the dataset has one, otherwise
    the single derivation from its stored affinity (spec §3.4). Never the generic fallback
    `edge_graph` and `twin_graph` expose. The reason is the central ablation and not tidiness:
    if the target came from whichever `LayoutGraph` produced the layout, $A_V$, $A_E$ and
    $A_{tne}$ would each be scored against a *different* target and comparing them would mean
    nothing. One target per dataset is what makes the comparison legitimate.

    Non-adjacent pairs get the shortest-path distance under the edge lengths; for an unweighted
    graph every edge length is 1 and this is the hop count.

    Args:
        G: the distance-valued graph. Non-edges must be **absent**, not present at 1.
        weight_transform: a key of :data:`WEIGHT_TRANSFORMS`, default ``"identity"``. A
            low-level escape hatch for the analytic tests of §6 — the pipeline never passes it,
            because on the path that produces results the transform has already been applied by
            `vertex_graph` (`DECISIONS.md` (10)). Passing it *together with* a graph stamped by
            the `LayoutGraph` layer raises rather than silently overriding a declared source.
        weight: the edge attribute holding the length. Absent means 1.
        disconnected: ``"lcc"`` (default) scores only the largest component and marks every
            pair touching another component infinite; ``"mask"`` marks only genuinely
            unreachable pairs infinite; ``"components"`` raises (spec §0.5 decisions 3, 4).
        max_n: refuse graphs above this size. The guard lives here, in the primitive, so there
            is exactly one place it can be forgotten.

    Returns:
        A dense ``(n, n)`` float array, symmetric, zero on the diagonal, with ``np.inf`` in
        every entry the policy excludes or the graph cannot reach. Both policies return the
        **full** ``(n, n)`` array and differ only in where the infinities are, so a consumer
        never has to subset its ``(n, d)`` layout in step (`DECISIONS.md`).

        ``inf`` rather than ``nan`` deliberately: it propagates through arithmetic, so a
        consumer that forgets to mask produces ``inf`` or ``nan`` rather than a plausible
        number. ``np.nanmean`` would absorb ``nan`` quietly, which is how `point_distance_metric`
        (spec §1.2) came to return a number that meant nothing.

    Raises:
        ValueError: on an oversized graph, a non-positive edge length, an unknown transform, or
            an explicit ``weight_transform`` applied to a stamped `LayoutGraph` form.
        NotImplementedError: for ``disconnected="components"``.
    """
    n = G.number_of_nodes()
    if n > max_n:
        raise ValueError(
            f"ideal_distances refuses n={n} > max_n={max_n}: the all-pairs array alone would "
            f"be {n * n * 8 / 1e9:.2f} GB, and condensing it transiently needs both. Raise "
            "max_n explicitly if that is what you meant — note that the default already admits "
            "3.2 GB, so it is an escape valve and not a memory policy."
        )

    explicit_transform = weight_transform is not _UNSET
    transform_name = "identity" if not explicit_transform else weight_transform
    if transform_name not in WEIGHT_TRANSFORMS:
        raise ValueError(
            f"unknown weight_transform {transform_name!r}; expected one of "
            f"{sorted(WEIGHT_TRANSFORMS)}"
        )

    side = graph_side(G)
    if explicit_transform and side is not None:
        raise ValueError(
            f"this graph is stamped {side!r} by the LayoutGraph layer "
            f"({G.graph.get(SOURCE_KEY)!r}), so its edge lengths are already the declared "
            f"distance model — applying weight_transform={weight_transform!r} on top of it "
            "would silently override the dataset's own source. weight_transform is a low-level "
            "escape hatch for graphs built by hand (DECISIONS.md (10)); drop one or the other."
        )
    if side == "affinity":
        raise ValueError(
            f"ideal_distances needs a distance-valued graph, got one stamped 'affinity' "
            f"({G.graph.get(SOURCE_KEY)!r}). Reading affinities as lengths inverts the target: "
            "close pairs become far ones. Pass LayoutGraph.distance."
        )

    if disconnected == "components":
        raise NotImplementedError(
            "disconnected='components' is deliberately not built: it would be untested code for "
            "a case that does not occur, since every registered dataset is connected (spec §0.5 "
            "decision 3). Use 'lcc' (the default) or 'mask'."
        )
    if disconnected not in ("lcc", "mask"):
        raise ValueError(f"disconnected must be 'lcc' or 'mask', got {disconnected!r}")

    lengths = np.array(
        [w for _, _, w in G.edges(data=weight, default=1.0)], dtype=float
    )
    if lengths.size and lengths.min() <= 0:
        # A zero-weight edge vanishes from a CSR matrix, so csgraph reads it as *absent* rather
        # than as length 0 — the pair would come back at its detour distance, or at inf. Spec
        # §0.5 is explicit that a zero ideal distance is legitimate under `probability` and
        # `kernel`, so this is refused loudly rather than papered over with a masked-array path
        # for a case no dataset currently produces.
        raise ValueError(
            f"edge lengths must be positive; {int((lengths <= 0).sum())} of {lengths.size} are "
            "not. A zero-length edge disappears from the sparse matrix and would be read as a "
            "missing edge, not as a coincident pair."
        )

    A = nx.to_scipy_sparse_array(G, nodelist=range(n), weight=weight, format="csr")
    A = sp.csr_array(A, dtype=float)
    if transform_name != "identity":
        A.data = WEIGHT_TRANSFORMS[transform_name](A.data)
        if A.data.size and A.data.min() <= 0:
            raise ValueError(
                f"weight_transform={transform_name!r} produced non-positive edge lengths, which "
                "shortest paths cannot use."
            )

    D = csgraph.dijkstra(A, directed=False)

    if disconnected == "lcc":
        n_components, labels = csgraph.connected_components(A, directed=False)
        if n_components > 1:
            counts = np.bincount(labels)
            keep = labels == counts.argmax()
            D = np.where(keep[:, None] & keep[None, :], D, np.inf)

    np.fill_diagonal(D, 0.0)
    return D


def condensed(D: np.ndarray) -> np.ndarray:
    """The upper triangle of a square matrix in :data:`CONDENSED_ORDERING`.

    The same ordering ``scipy.spatial.distance.pdist`` returns, so an ideal-distance vector and
    a layout's pairwise distances line up elementwise with no bookkeeping.
    """
    D = np.asarray(D)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {D.shape}")
    return D[np.triu_indices(D.shape[0], k=1)]


def finite_pair_mask(D: np.ndarray) -> np.ndarray:
    """Which pairs of ``D`` were actually scored, **condensed**.

    False marks a pair the disconnection policy excluded or the graph cannot reach. Every
    consumer of :func:`ideal_distances` masks with this instead of growing its own branch for
    the disconnected case.
    """
    return np.isfinite(condensed(D))
