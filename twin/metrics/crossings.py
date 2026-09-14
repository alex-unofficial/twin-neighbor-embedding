r"""Edge crossings, computed exactly (spec §2.4, thesis eq. 73–81).

This module is what makes the straight-versus-curved experiment of §4.3 possible. It is built in
three layers:

* **Layer 1** — :func:`edge_curves`: the drawing of each edge, as three Bézier control points.
* **Layer 2** — ``crossing_pairs``: which pairs of those curves meet, and where.
* **Layer 3** — ``crossing_number``, ``crossings_per_edge_pair``, ``length_weighted_crossings``.

**The method is exact and algebraic — no sampling, no subdivision, no geometric tolerance**
(DECISIONS (28)). The drawn edge is exactly a quadratic Bézier, two quadratic Béziers meet in at
most four points, and those points are the roots of one quartic obtained by eliminating one
curve's parameter. The derivation, the measurements and the degeneracy analysis are in
`docs/CROSSING_GEOMETRY.md`, which is the specification for this module; spec §2.4 is its summary.

That exactness is the point of the design, not a flourish. "Your crossing count is an artifact of
how finely you sampled" is the obvious objection to a crossing metric, and there is no sampling
parameter here to aim it at: `n_samples` was **removed, not defaulted** (DECISIONS (27)).

**The curve geometry lives in `twin/geometry.py`**, shared with `plots.draw_curved_edges`, so the
picture and the measurement cannot disagree about what an edge looks like.

**Conventions.** Arguments are ``(G, V, E=None, *, ...)`` throughout — graph, vertex points, edge
points, everything else keyword-only (DECISIONS (5)). ``V`` is ``(n, d)`` and ``E`` is ``(m, d)``
in ``G.edges()`` order, which is the per-edge order of record for the whole package (CLAUDE.md);
ADR's CSR ordering is a different one and the two must not be mixed.
"""

from dataclasses import dataclass
from math import comb
from typing import Iterator

import networkx as nx
import numpy as np
import shapely

from twin import geometry
from twin.metrics.common import edge_index_array, require_n_by_d

__all__ = [
    "CrossingReport",
    "candidate_pairs",
    "crossing_number",
    "crossings_per_edge_pair",
    "detour_ratio",
    "length_weighted_crossings",
    "count_candidate_pairs",
    "crossing_pairs",
    "crossing_pairs_report",
    "edge_curves",
    "normalize_to_unit_box",
]

#: How many edges are queried against the R-tree at once. **Not a tuning knob for speed** — it is
#: what bounds the pipeline's memory, and it is measured. `shapely.STRtree.query` materialises
#: every pair it finds before returning, so a bulk query over all of tabula's 49368 edges is an
#: 11.4 M pair array (365 MB) in one allocation, and blocking the *solve* does not help: the
#: array already exists by then (DECISIONS (31a)). Chunking the query is what caps it, and the
#: same chunk then sizes every downstream pass.
#:
#: Measured on a tabula twin layout, γ=0.5, identical crossing counts at every setting:
#: 4096 → 40.2 s / +768 MB, **1024 → 39.4 s / +428 MB**, 256 → 39.4 s / +492 MB. Below ~1024 the
#: per-query overhead starts to show and the allocator stops giving the memory back.
QUERY_CHUNK_EDGES = 1024

#: How far the exact boxes are grown before the overlap query, in unit-box units (~4500 ulp).
#: **This is not a geometric tolerance** — nothing downstream of the broad phase uses it, and it
#: cannot create a crossing. It exists because two edges that share a vertex touch *exactly*
#: there, and when that shared point is extremal in the same coordinate for both curves, the two
#: box coordinates are computed by different expressions — one an endpoint, the other the
#: closed-form stationary value — and can differ by a couple of ulp. Unpadded, the query then
#: reports the boxes as disjoint. On a karate layout that loses 9 of 2154 candidate pairs;
#: harmless for the shared vertex itself (excluded by identity anyway), not harmless for a
#: tangential contact at a box boundary. Padding restores "never discards a crossing pair" at a
#: cost of 0.4 % more candidates on karate and 0.03 % on `digits`.
BOX_PAD = 1e-12


def edge_curves(
    G: nx.Graph, V: np.ndarray, E: np.ndarray | None = None, *, gamma: float = 0.5
) -> np.ndarray:
    r"""The drawn shape of every edge, as ``(m, 3, d)`` Bézier control points.

    One row per edge, **in ``G.edges()`` order**, each holding $(B_0, B_1, B_2)$ of the curve
    `plots.draw_curved_edges` would draw for that edge at this γ. Three points describe a
    quadratic completely, which is what lets layer 2 be algebraic.

    **Both variants of §4.3 come through here.** ``E is None`` puts $B_1$ at the chord midpoint,
    which is an exact straight segment — as does ``gamma=0`` with any ``E``. So the straight
    variant is not a separate code path with its own bugs; it is this one, with a bulge of exactly
    zero (DECISIONS (27), superseding (15)'s two-sample short-circuit).

    Args:
        G: the graph. Nodes must be ``0..n-1``.
        V: ``(n, d)`` vertex points, in the original vertex indexing.
        E: ``(m, d)`` edge points in ``G.edges()`` order, or None for straight edges.
        gamma: curvature. 0 is the chord, 0.5 the classic Bézier, 1 interpolates the edge point.

    Returns:
        ``(m, 3, d)`` Bernstein control points. ``(0, 3, d)`` for an edgeless graph.

    Raises:
        ValueError: if ``V`` or ``E`` has the wrong shape. A ``(d, n)`` layout is rejected, never
            transposed — CLAUDE.md's one conversion point is `LayoutResult.from_raw`.
    """
    n = G.number_of_nodes()
    V = require_n_by_d(V, n, "V").astype(float, copy=False)
    m = G.number_of_edges()
    edges = edge_index_array(G, n)

    if E is None:
        P1 = 0.5 * (V[edges[:, 0]] + V[edges[:, 1]])
    else:
        P1 = require_n_by_d(E, m, "E").astype(float, copy=False)
        if P1.shape[1] != V.shape[1]:
            raise ValueError(f"V and E disagree on dimension: {V.shape[1]} vs {P1.shape[1]}")

    controls = np.stack([V[edges[:, 0]], P1, V[edges[:, 1]]], axis=1)
    return geometry.bezier_control_points(controls, gamma)


# --------------------------------------------------------------------------- #
#                        Layer 2a — the broad phase                            #
# --------------------------------------------------------------------------- #


def normalize_to_unit_box(controls: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    r"""Rescale a set of curves into the unit box, isotropically.

    **Why this is not cosmetic.** The pipeline has no tolerance on the *geometry*, but it does
    have tolerances on root reality, on $[0,1]$ membership and on the two degeneracy guards
    ($\lVert D\rVert \approx 0$, $|Q_4| \approx 0$). Those are absolute numbers, and a layout
    whose coordinates run to $10^{3}$ would meet them at a different physical scale than one
    running to $10^{-2}$ — which would make the method tolerance-free on geometry but
    dataset-dependent in practice, the worse of the two failures because it is invisible.

    The map is a translation and a **single** scale factor for both axes, so it is a similarity:
    it moves no crossing, changes no parameter value, and is exactly invertible.

    Args:
        controls: ``(m, 3, 2)`` Bézier control points.

    Returns:
        ``(scaled, scale, origin)`` with ``scaled = (controls - origin) * scale``. A degenerate
        input — every point coincident — gets ``scale = 1``.
    """
    controls = np.asarray(controls, dtype=float)
    if controls.size == 0:
        return controls, 1.0, np.zeros(controls.shape[-1])

    boxes = geometry.exact_bounding_boxes(controls)
    origin = boxes[:, 0, :].min(axis=0)
    extent = (boxes[:, 1, :].max(axis=0) - origin).max()
    scale = 1.0 / extent if extent > 0 else 1.0
    return (controls - origin) * scale, float(scale), origin


def candidate_pairs(
    controls: np.ndarray, *, chunk_edges: int = QUERY_CHUNK_EDGES
) -> Iterator[np.ndarray]:
    r"""Yield blocks of ``(2, k)`` edge-index pairs whose **exact** boxes overlap.

    The broad phase: an `shapely.STRtree` over the exact bounding boxes of `twin.geometry`,
    $O(m\log m)$ to build, queried without enumerating the $\binom{m}{2}$ pairs it rejects. Every
    crossing pair is a box-overlapping pair, so this discards nothing real.

    The exact box rather than the box on the control points is worth the 123 ns per edge: it is
    a median 1.9× smaller in area and candidate counts scale with area.

    Pairs come out with ``a < b`` and each pair once. Blocks are yielded rather than
    concatenated, and the *query itself* is chunked over edges — see :data:`QUERY_CHUNK_EDGES`
    for why that is the memory bound and blocking the downstream passes is not.

    Args:
        controls: ``(m, 3, 2)`` Bézier control points, normally already normalised.
        chunk_edges: how many edges to query at a time.

    Yields:
        ``(2, k)`` arrays of edge indices. Possibly zero blocks, for an edgeless graph.
    """
    controls = np.asarray(controls, dtype=float)
    m = controls.shape[0]
    if m < 2:
        return

    boxes = geometry.exact_bounding_boxes(controls)
    lo, hi = boxes[:, 0] - BOX_PAD, boxes[:, 1] + BOX_PAD
    rectangles = shapely.box(lo[:, 0], lo[:, 1], hi[:, 0], hi[:, 1])
    tree = shapely.STRtree(rectangles)

    for start in range(0, m, chunk_edges):
        stop = min(start + chunk_edges, m)
        hits = tree.query(rectangles[start:stop], predicate="intersects")
        a = hits[0] + start
        b = hits[1]
        keep = a < b
        if keep.any():
            yield np.stack([a[keep], b[keep]])


def count_candidate_pairs(controls: np.ndarray, **kwargs) -> int:
    """$K$ — how many pairs survive the broad phase. The cost model's one free parameter.

    Measured before the solver was written (DECISIONS (31a)): 27058 on `football_2023`, 1380736
    on `digits`, 11375494 on `tabula`, for SG-*t*-SNE twin layouts at γ=0.5.
    """
    return int(sum(block.shape[1] for block in candidate_pairs(controls, **kwargs)))


# --------------------------------------------------------------------------- #
#                        Layer 2b — the exact solve                            #
# --------------------------------------------------------------------------- #


#: Power-to-Bernstein basis map on $[0,1]$ for a quartic: $\beta_j = \sum_{k\le j}
#: \binom{j}{k}/\binom{4}{k}\, Q_k$. Row $j$ is that combination, so ``Q @ BERNSTEIN.T`` converts
#: a whole block at once. Note $\beta_0 = Q(0)$ and $\beta_4 = Q(1)$ **exactly**, which is what
#: makes the endpoint-root exception below free.
BERNSTEIN = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1 / 4, 0.0, 0.0, 0.0],
        [1.0, 2 / 4, 1 / 6, 0.0, 0.0],
        [1.0, 3 / 4, 3 / 6, 1 / 4, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ]
)

#: A curve counts as a straight segment when $\lVert D\rVert$ falls below this. **Absolute, in
#: unit-box units** — which is the entire reason :func:`normalize_to_unit_box` runs first; on a
#: raw layout the same number would mean a different amount of bend per dataset.
#:
#: DECISIONS (30) as corrected warns of a **coupled band**: a pair whose bulge sits just *above*
#: this cutoff — so no segment dispatch happens — while $Q_4 = -16\det(D_A,D_B)^2$, which falls
#: as the *square* of either bulge, is already small enough to wreck the companion matrix. The
#: band was looked for with constructed pairs sweeping $\lVert D\rVert$ across twelve decades,
#: since a real layout never populates it (min $\lVert D\rVert$ on a football twin layout is
#: $5.5\times10^{-4}$).
#:
#: **It is empty, and not because of this constant.** :data:`EPS_DEFLATE` closes it: by the time
#: the conditioning would matter, $|Q_4|/\max|Q|$ is already below the deflation threshold and a
#: well-conditioned cubic is solved instead. Across $\lVert D\rVert \in [10^{-12}, 10^{-1}]$ the
#: count stays exactly right and the reported point stays within $2.5\times10^{-15}$ of both
#: curves, including at $\lVert D\rVert = 10^{-9}$ where the undeflated companion matrix would
#: have had a condition number of $2.5\times10^{17}$. So the two thresholds are coupled in
#: principle and the coupling is *already broken* by the deflation; this value's only real job is
#: to route an exactly-zero bulge (γ=0, or A_V's layouts) to the segment path.
EPS_LIN = 1e-9

#: A curve with a non-negligible bulge is still degenerate if its control points are collinear:
#: the parabola collapses onto a line and $\beta$, the denominator of the parameter recovery,
#: vanishes. Relative, since $\beta = \det(C_1, C_2)$ scales with both.
EPS_COLLINEAR = 1e-12

#: The quartic is deflated to its true degree before the companion matrix is formed, because
#: forming it divides by the leading coefficient. Relative to $\max_k|Q_k|$: below this, the
#: discarded root is far outside $[0,1]$ and cannot be the one being looked for.
EPS_DEFLATE = 1e-12

#: Every coefficient below this multiple of the magnitudes that built them means the substituted
#: curve lies *on* the implicitised parabola — identical or overlapping curves. Counted as
#: exactly 1 crossing (spec §2.4), and detected before any root-finding.
EPS_IDENTICAL = 1e-13

#: A root counts as real when its imaginary part is within this. It is deliberately loose: a
#: **double** root of a quartic is only determined to $\sqrt{\varepsilon}\approx10^{-8}$ in double
#: arithmetic, so a true tangency surfaces as a conjugate pair at $\pm10^{-8}i$ and a tighter
#: test would report 0 crossings where the convention says 1. The cost is that a genuine miss by
#: less than about $10^{-14}$ of a unit box counts as a crossing; such pairs are reported
#: separately as `n_near_tangential` rather than being silently folded in.
IMAG_TOL = 1e-7

#: Slack on $s, t \in [0,1]$. A crossing exactly at a curve's endpoint is a real configuration —
#: a vertex lying on another edge — so the interval is closed and slightly padded.
PARAM_TOL = 1e-9

#: Two accepted roots of the same pair closer than this in $(s,t)$ are one crossing — the
#: tangency convention of DECISIONS (29), and the resolution limit noted at :data:`IMAG_TOL`.
DUPLICATE_TOL = 1e-6

#: Reflection $Q(s) \mapsto Q(1-u)$ on the quartic's coefficients: $\tilde q_j = (-1)^j
#: \sum_{k\ge j}\binom{k}{j} q_k$. Used to move a known root at $s=1$ to $u=0$, where deflating
#: it is exactly "drop the constant term".
REFLECT = np.array([[(-1.0) ** j * float(comb(k, j)) for k in range(5)] for j in range(5)])


def _pair_coefficients(Ca: np.ndarray, Cb: np.ndarray):
    r"""The quartic in $s$ whose roots are curve B's parameters at the intersections.

    ``Ca``/``Cb`` are ``(P, 3, 2)`` power-form coefficients; A is implicitised and B substituted.
    Returns ``(Q, P02, beta, scale)``: ``Q`` is ``(P, 5)`` in ascending powers of $s$, ``P02`` is
    the ``(P, 3)`` quadratic that recovers $t = -P_{02}(s)/\beta$ for free, and ``scale`` is the
    magnitude of the terms ``Q`` was built from — the reference an "all coefficients vanish" test
    needs. Derivation: `docs/CROSSING_GEOMETRY.md` §3.
    """
    a0, a1, a2 = Ca[:, 0, 0], Ca[:, 1, 0], Ca[:, 2, 0]
    b0, b1, b2 = Ca[:, 0, 1], Ca[:, 1, 1], Ca[:, 2, 1]
    c0, c1, c2 = Cb[:, 0, 0], Cb[:, 1, 0], Cb[:, 2, 0]
    d0, d1, d2 = Cb[:, 0, 1], Cb[:, 1, 1], Cb[:, 2, 1]

    beta = a1 * b2 - a2 * b1
    k01 = a0 * b1 - a1 * b0
    k02 = a0 * b2 - a2 * b0

    p01 = np.stack([k01 - b1 * c0 + a1 * d0, -b1 * c1 + a1 * d1, -b1 * c2 + a1 * d2], axis=1)
    p02 = np.stack([k02 - b2 * c0 + a2 * d0, -b2 * c1 + a2 * d1, -b2 * c2 + a2 * d2], axis=1)

    Q = np.stack(
        [
            beta * p01[:, 0] - p02[:, 0] ** 2,
            beta * p01[:, 1] - 2 * p02[:, 0] * p02[:, 1],
            beta * p01[:, 2] - p02[:, 1] ** 2 - 2 * p02[:, 0] * p02[:, 2],
            -2 * p02[:, 1] * p02[:, 2],
            -(p02[:, 2] ** 2),
        ],
        axis=1,
    )
    scale = np.abs(beta) * np.abs(p01).max(axis=1) + np.abs(p02).max(axis=1) ** 2
    return Q, p02, beta, scale


def _quartic_roots(Q: np.ndarray) -> np.ndarray:
    r"""Roots of each row of ``Q`` (ascending powers), as ``(P, 4)`` complex with NaN padding.

    **Companion-matrix eigenvalues, batched — not Ferrari's closed form.** Ferrari routes through
    a resolvent cubic whose cancellation is worst when two roots are close, which is exactly
    near-tangency, which is exactly where the answer matters. A balanced companion plus QR is
    backward stable and reports a near-tangency legibly, as two nearly coincident roots.

    Rows are grouped by **deflated** degree first: forming the companion matrix divides by the
    leading coefficient, and $Q_4 = -16\det(D_A,D_B)^2$ vanishes as the *square* of either
    curve's bulge, so a single near-straight edge degrades it quadratically (DECISIONS (30) as
    corrected — the trigger is a near-straight curve, not parallel bulges).
    """
    n_pairs = len(Q)
    roots = np.full((n_pairs, 4), np.nan, dtype=complex)
    if n_pairs == 0:
        return roots

    largest = np.abs(Q).max(axis=1)
    degree = np.zeros(n_pairs, dtype=int)
    for k in range(1, 5):
        degree = np.where(np.abs(Q[:, k]) > EPS_DEFLATE * largest, k, degree)

    for d in (1, 2, 3, 4):
        rows = np.flatnonzero(degree == d)
        if rows.size == 0:
            continue
        monic = Q[rows, :d] / Q[rows, d][:, None]
        companion = np.zeros((rows.size, d, d))
        companion[:, 0, :] = -monic[:, ::-1]
        if d > 1:
            companion[:, 1:, :-1] = np.eye(d - 1)
        roots[rows, :d] = np.linalg.eigvals(companion)
    return roots


def _parameter_at_point(C: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Curve parameter of a point known to lie on the curve. ``C`` power form, ``points`` (P, 2).

    Solves the scalar quadratic in whichever coordinate the curve varies more, so it is exact for
    a straight curve — where the quadratic term is zero — and still correct for a
    collinear-control one, where the curve doubles back and the *segment* parameter is not the
    curve parameter.
    """
    c0, c1, c2 = C[:, 0], C[:, 1], C[:, 2]
    rows = np.arange(len(C))
    axis = np.argmax(np.abs(c1) + np.abs(c2), axis=1)
    a = c2[rows, axis]
    b = c1[rows, axis]
    rhs = points[rows, axis] - c0[rows, axis]

    linear = np.abs(a) <= EPS_LIN * (np.abs(b) + 1.0)
    t_linear = rhs / np.where(np.abs(b) > 0, b, 1.0)

    root = np.sqrt(np.maximum(b * b + 4.0 * a * rhs, 0.0))
    safe_a = np.where(linear, 1.0, a)
    candidates = np.stack([(-b + root) / (2.0 * safe_a), (-b - root) / (2.0 * safe_a)], axis=1)
    outside = np.abs(candidates - np.clip(candidates, 0.0, 1.0))
    t_quadratic = candidates[rows, np.argmin(outside, axis=1)]

    return np.clip(np.where(linear, t_linear, t_quadratic), 0.0, 1.0)


def _swept_segments(C: np.ndarray) -> np.ndarray:
    """``(P, 2, 2)`` endpoints of the segment a degenerate curve sweeps.

    For a straight curve that is $(B_0, B_2)$. For a curve whose control points are collinear the
    parabola collapses onto a line and may double back, so the swept set runs from one end to the
    turning point — hence the three candidates and the widest-apart pair.
    """
    c0, c1, c2 = C[:, 0], C[:, 1], C[:, 2]
    denom = 2.0 * np.sum(c2 * c2, axis=1)
    turn = np.clip(
        np.where(denom > 0, -np.sum(c1 * c2, axis=1) / np.where(denom > 0, denom, 1.0), 0.0),
        0.0,
        1.0,
    )[:, None]
    points = np.stack([c0, c0 + c1 + c2, c0 + c1 * turn + c2 * turn * turn], axis=1)

    spans = np.stack(
        [
            np.linalg.norm(points[:, 0] - points[:, 1], axis=1),
            np.linalg.norm(points[:, 0] - points[:, 2], axis=1),
            np.linalg.norm(points[:, 1] - points[:, 2], axis=1),
        ],
        axis=1,
    )
    ends = np.array([[0, 1], [0, 2], [1, 2]])[np.argmax(spans, axis=1)]
    rows = np.arange(len(C))
    return np.stack([points[rows, ends[:, 0]], points[rows, ends[:, 1]]], axis=1)


def _cross2(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]


def _segment_intersections(Ca: np.ndarray, Cb: np.ndarray):
    r"""Intersection point of two degenerate (segment-shaped) curves, if any.

    Returns ``(hit, point, overlapping)``: a boolean mask, the ``(P, 2)`` point (meaningful where
    ``hit``), and a mask of the pairs that are **collinear and overlapping along a positive
    stretch**, which spec §2.4 counts as exactly one crossing rather than the infinite set eq.
    (74) would give.

    Collinear segments that meet at a single point are *not* an overlap — they are a point
    intersection, and for an adjacent pair that point is the shared vertex. Keeping the two
    apart is what stops a path drawn along a straight line from reporting a crossing at every
    interior vertex.

    This branch carries the whole straight variant of §4.3 (γ=0 builds no quartic at all), so it
    is a vectorised pass and not a fallback.
    """
    sa, sb = _swept_segments(Ca), _swept_segments(Cb)
    p, r = sa[:, 0], sa[:, 1] - sa[:, 0]
    q, u = sb[:, 0], sb[:, 1] - sb[:, 0]
    w = q - p

    denom = _cross2(r, u)
    magnitude = np.linalg.norm(r, axis=1) * np.linalg.norm(u, axis=1)
    proper = np.abs(denom) > EPS_COLLINEAR * np.maximum(magnitude, np.finfo(float).tiny)

    safe = np.where(proper, denom, 1.0)
    t = _cross2(w, u) / safe
    s = _cross2(w, r) / safe
    hit = proper & (t >= -PARAM_TOL) & (t <= 1 + PARAM_TOL)
    hit &= (s >= -PARAM_TOL) & (s <= 1 + PARAM_TOL)
    point = p + np.clip(t, 0.0, 1.0)[:, None] * r

    # Parallel pairs: collinear ones may still overlap, in which case the convention is one
    # crossing, reported at the middle of the shared stretch.
    rr = np.sum(r * r, axis=1)
    collinear = ~proper & (np.abs(_cross2(w, r)) <= EPS_COLLINEAR * np.maximum(magnitude, 1e-300))
    with np.errstate(invalid="ignore", divide="ignore"):
        k0 = np.where(rr > 0, np.sum(w * r, axis=1) / np.where(rr > 0, rr, 1.0), 0.0)
        k1 = np.where(rr > 0, np.sum((q + u - p) * r, axis=1) / np.where(rr > 0, rr, 1.0), 0.0)
    lo = np.maximum(np.minimum(k0, k1), 0.0)
    hi = np.minimum(np.maximum(k0, k1), 1.0)
    overlapping = collinear & (hi > lo + PARAM_TOL)
    touching = collinear & (hi >= lo - PARAM_TOL) & ~overlapping

    mid = np.clip(0.5 * (lo + hi), 0.0, 1.0)
    shared_stretch = overlapping | touching
    point = np.where(shared_stretch[:, None], p + mid[:, None] * r, point)
    return hit | shared_stretch, point, overlapping


def _rows(a: np.ndarray, b: np.ndarray, t: np.ndarray, s: np.ndarray, powers: np.ndarray):
    """Assemble ``(k, 6)``. The point is evaluated on ``edge_a`` at its own parameter."""
    points = _evaluate(powers[a], t)
    return np.stack([a.astype(float), b.astype(float), t, s, points[:, 0], points[:, 1]], axis=1)


def _solve_segment_block(
    a: np.ndarray,
    b: np.ndarray,
    shared: np.ndarray,
    powers: np.ndarray,
    edges: np.ndarray,
    rank: np.ndarray,
    report: "CrossingReport",
) -> np.ndarray:
    """Pairs where both curves have collapsed to a segment (DECISIONS (30), first two guards).

    **The whole straight variant of §4.3 comes through here**, so the shared-vertex rule matters
    as much as it does on the quartic path — and here it is exact with nothing to tune. Two
    distinct lines meet in at most one point; both curves pass through the vertex they share by
    index; so for an adjacent pair *any* proper intersection **is** that vertex. What survives is
    only the collinear-overlap case, which the convention counts as one crossing.
    """
    hit, point, overlapping = _segment_intersections(powers[a], powers[b])
    at_shared_vertex = hit & (shared >= 0) & ~overlapping
    report.n_shared_endpoint += int(at_shared_vertex.sum())
    hit &= ~at_shared_vertex
    report.n_degenerate += int((hit & overlapping).sum())
    if not hit.any():
        return np.empty((0, 6))

    a, b, point = a[hit], b[hit], point[hit]
    return _rows(
        a, b, _parameter_at_point(powers[a], point), _parameter_at_point(powers[b], point), powers
    )


def _deflate_shared_root(Q: np.ndarray, at_one: np.ndarray) -> np.ndarray:
    r"""Divide out the root the two edges' shared vertex is known to put at $s \in \{0, 1\}$.

    **This is the endpoint exclusion of DECISIONS (13), done exactly.** Both curves pass through
    the vertex they share by index, so the contact there is not a crossing and its parameter is
    known in advance — no coordinate comparison, no distance tolerance, and (unlike a proximity
    test) a genuine second crossing arbitrarily close to the vertex survives.

    A tolerance would have had nowhere safe to sit: measured on a `digits` twin layout, the
    distance from a root to its pair's shared vertex is a **continuum** from $10^{-14}$ to
    $10^{-3}$ with no gap — 208 roots in $[10^{-12}, 10^{-9})$, 54 in $[10^{-9}, 10^{-7})$, 326 in
    $[10^{-7}, 10^{-6})$, 1488 in $[10^{-6}, 10^{-5})$. Any cutoff would have been arbitrary, and
    a tight one demonstrably mislabels: a `football_2023` pair whose root sat $9.7\times10^{-8}$
    from the shared vertex was the vertex.

    Roots at $s=1$ are reflected to $u = 1-s$ first, so deflation is always "drop the constant
    term" — exact, rather than a synthetic division whose stability depends on where the other
    roots are.

    Returns the deflated coefficients in ascending powers of $s$ (or of $u$ where reflected), one
    degree lower, zero-padded back to five.
    """
    shifted = np.where(at_one[:, None], Q @ REFLECT.T, Q)
    deflated = np.zeros_like(shifted)
    deflated[:, :4] = shifted[:, 1:]
    return deflated


def _solve_quartic_block(
    a: np.ndarray,
    b: np.ndarray,
    shared: np.ndarray,
    powers: np.ndarray,
    edges: np.ndarray,
    rank: np.ndarray,
    report: "CrossingReport",
) -> np.ndarray:
    """Build the quartic, reject on Bernstein, solve what survives, keep the roots that qualify."""
    swap = rank[b] > rank[a]
    implicit = np.where(swap, b, a)
    substituted = np.where(swap, a, b)
    Q, p02, beta, scale = _pair_coefficients(powers[implicit], powers[substituted])

    # Every coefficient vanishing means the substituted curve lies on the implicitised parabola.
    identical = np.abs(Q).max(axis=1) <= EPS_IDENTICAL * np.maximum(scale, np.finfo(float).tiny)
    report.n_degenerate += int(identical.sum())

    # The Bernstein hull property: with non-negative weights summing to 1, no sign change among
    # the five coefficients proves no root in the OPEN interval (0, 1).
    bernstein = Q @ BERNSTEIN.T
    changes_sign = (bernstein > 0).any(axis=1) & (bernstein < 0).any(axis=1)
    # ...which leaves one real false negative: a root exactly at an endpoint. That is a large
    # share of rejections on real data, because adjacent edges meet exactly at their shared
    # vertex — where it is harmless. For a NON-adjacent pair it is a lost crossing: a vertex
    # lying exactly on another edge's curve. Since beta_0 = Q(0) and beta_4 = Q(1) exactly the
    # fix is free, and the non-adjacency gate is load-bearing — without it the adjacent
    # rejections all re-enter the solve and the filter stops paying for itself (guide §4).
    endpoint_tol = EPS_DEFLATE * np.abs(Q).max(axis=1)
    adjacent = shared >= 0
    at_endpoint = ~adjacent & (
        (np.abs(bernstein[:, 0]) <= endpoint_tol) | (np.abs(bernstein[:, 4]) <= endpoint_tol)
    )
    live = (changes_sign | at_endpoint) & ~identical
    report.n_filtered += int((~live & ~identical).sum())

    rows = []
    if identical.any():
        # The overlap is a set, not a point; spec §2.4 counts it once, reported at mid-curve.
        half = np.full(int(identical.sum()), 0.5)
        rows.append(_rows(a[identical], b[identical], half, half, powers))

    if live.any():
        a_l, b_l = a[live], b[live]
        swap_l, beta_l, p02_l = swap[live], beta[live], p02[live]
        adjacent_l = adjacent[live]

        # The shared vertex is a known root: divide it out rather than filter it back out later.
        reflected = np.zeros(int(live.sum()), dtype=bool)
        coefficients = Q[live]
        if adjacent_l.any():
            reflected = adjacent_l & (edges[substituted[live], 0] != shared[live])
            report.n_shared_endpoint += int(adjacent_l.sum())
            coefficients = np.where(
                adjacent_l[:, None], _deflate_shared_root(coefficients, reflected), coefficients
            )

        roots = _quartic_roots(coefficients)
        roots = np.where(reflected[:, None], 1.0 - roots, roots)

        s_sub = roots.real
        ok = np.abs(roots.imag) <= IMAG_TOL * (1.0 + np.abs(s_sub))
        ok &= (s_sub >= -PARAM_TOL) & (s_sub <= 1.0 + PARAM_TOL)
        s_sub = np.clip(np.where(ok, s_sub, 0.0), 0.0, 1.0)

        # t comes free from the quadratic already built: t = -P02(s) / beta.
        t_imp = -(p02_l[:, [0]] + p02_l[:, [1]] * s_sub + p02_l[:, [2]] * s_sub**2) / beta_l[:, None]
        ok &= (t_imp >= -PARAM_TOL) & (t_imp <= 1.0 + PARAM_TOL)
        t_imp = np.clip(t_imp, 0.0, 1.0)

        # A tangency is a double root: count it once, and say how often that happened.
        for j in range(1, 4):
            for i in range(j):
                duplicate = (
                    ok[:, i]
                    & ok[:, j]
                    & (np.abs(s_sub[:, i] - s_sub[:, j]) <= DUPLICATE_TOL)
                    & (np.abs(t_imp[:, i] - t_imp[:, j]) <= DUPLICATE_TOL)
                )
                ok[:, j] &= ~duplicate
                report.n_near_tangential += int(duplicate.sum())

        pair_of, root_of = np.nonzero(ok)
        if pair_of.size:
            on_implicit = t_imp[pair_of, root_of]
            on_substituted = s_sub[pair_of, root_of]
            swapped = swap_l[pair_of]
            rows.append(
                _rows(
                    a_l[pair_of],
                    b_l[pair_of],
                    np.where(swapped, on_substituted, on_implicit),
                    np.where(swapped, on_implicit, on_substituted),
                    powers,
                )
            )

    return np.concatenate(rows, axis=0) if rows else np.empty((0, 6))


@dataclass
class CrossingReport:
    """The crossings of one drawing, with the diagnostics DECISIONS (18) keeps out of the value.

    Attributes:
        crossings: ``(k, 6)`` of ``(edge_a, edge_b, t, s, x, y)``, one row per crossing, with
            ``edge_a < edge_b`` as indices into ``G.edges()``. ``t`` is the parameter on
            ``edge_a`` and ``s`` on ``edge_b``; ``x, y`` are in the **original** layout
            coordinates, not the normalised ones.
        n_candidates: $K$, the pairs that survived the broad phase.
        n_filtered: of those, how many the Bernstein sign test rejected without a solve.
        n_near_tangential: accepted roots that coalesced — a tangency counted once (DECISIONS
            (29)). A large count means the drawing has curves grazing each other and the crossing
            number is correspondingly delicate.
        n_degenerate: pairs whose curves coincide or overlap along a stretch, each counted as
            exactly 1 crossing rather than eq. (74)'s infinite set.
        n_shared_endpoint: contacts removed because they were at a vertex the two edges share
            **by index**. Not a defect — it is how often adjacent edges were in play. It counts
            contacts *divided out*, so an adjacent pair the Bernstein test rejects before the
            solve does not appear here even though its curves do meet at the shared vertex.
    """

    crossings: np.ndarray
    n_candidates: int = 0
    n_filtered: int = 0
    n_near_tangential: int = 0
    n_degenerate: int = 0
    n_shared_endpoint: int = 0


def _shared_vertices(edges_a: np.ndarray, edges_b: np.ndarray) -> np.ndarray:
    """``(P,)`` index of the vertex each pair shares, or -1. A simple graph shares at most one."""
    same = edges_a[:, :, None] == edges_b[:, None, :]
    flat = same.reshape(len(edges_a), 4)
    which = np.argmax(flat, axis=1) // 2
    return np.where(flat.any(axis=1), edges_a[np.arange(len(edges_a)), which], -1)


def _evaluate(C: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Points on power-form curves ``C`` at parameters ``t``, both ``(P, ...)``-shaped."""
    t = t[:, None]
    return C[:, 0] + C[:, 1] * t + C[:, 2] * t * t


def crossing_pairs_report(
    G: nx.Graph,
    V: np.ndarray,
    E: np.ndarray | None = None,
    *,
    gamma: float = 0.5,
    include_adjacent: bool = True,
) -> CrossingReport:
    r"""Every crossing of the drawing, exactly — layer 2 of spec §2.4.

    Broad phase, Bernstein rejection, then one quartic solve per surviving pair, run as **passes
    over arrays** in blocks: per-pair Python control flow costs about 1 µs/pair in interpreter
    overhead alone, which would dominate everything the algebra saves.

    Four things that are easy to get wrong, all of them load-bearing:

    * **Filtering $s\in[0,1]$ is not enough.** The implicit equation describes the whole parabola,
      not the arc between the endpoints, so $t$ is recovered and tested too — about a third of
      in-range $s$ roots lie on the other curve's extension.
    * **The curve with the larger control-triangle area is the one implicitised**, because
      $\beta = 2\det(B_1-B_0, B_2-B_0)$ is the denominator of the parameter recovery. A curve
      flagged as degenerate is never the one implicitised.
    * **The quartic is deflated before the companion matrix is formed** — see
      :func:`_quartic_roots`.
    * **The contact at a shared vertex is divided out, not filtered out.** Exclusion is by vertex
      identity (DECISIONS (13)) and here it is *exact*: both curves pass through the vertex they
      share by index, so the root's parameter is known in advance and
      :func:`_deflate_shared_root` removes it from the quartic. No coordinates are compared, so
      two distinct vertices landing on one point — the crowding case the rule exists for — is
      untouched, and a genuine second crossing arbitrarily close to the shared vertex survives.

    Args:
        G: the graph. Nodes must be ``0..n-1``.
        V: ``(n, 2)`` vertex points.
        E: ``(m, 2)`` edge points in ``G.edges()`` order, or None for straight edges.
        gamma: curvature, as in :func:`edge_curves`.
        include_adjacent: keep pairs of edges that share a vertex. **True is the thesis
            convention** (§3.1.3: curved edges incident to a shared vertex may still cross away
            from it); the shared endpoint itself is excluded either way. `plots.count_intersections`
            passes False, which is a standing intentional divergence for the paper's Figure 3.

    Returns:
        A :class:`CrossingReport`.
    """
    n = G.number_of_nodes()
    controls = edge_curves(G, V, E, gamma=gamma)
    if controls.shape[-1] != 2:
        raise ValueError(
            f"crossing detection is planar; got a {controls.shape[-1]}-dimensional layout"
        )
    edges = edge_index_array(G, n)
    normalised, scale, origin = normalize_to_unit_box(controls)
    powers = geometry.polynomial_from_bezier(normalised)

    # Per-edge conditioning, computed once: a curve is "a segment" when it has no bulge, or when
    # its control points are collinear so the parabola has collapsed onto a line (DECISIONS (30)).
    beta_curve = powers[:, 1, 0] * powers[:, 2, 1] - powers[:, 2, 0] * powers[:, 1, 1]
    spread = np.linalg.norm(powers[:, 1], axis=1) * np.linalg.norm(powers[:, 2], axis=1)
    degenerate_curve = (np.linalg.norm(geometry.bulge(normalised), axis=1) <= EPS_LIN) | (
        np.abs(beta_curve) <= EPS_COLLINEAR * spread
    )
    #: -1 keeps a degenerate curve from ever being the one implicitised.
    rank = np.where(degenerate_curve, -1.0, np.abs(beta_curve))

    report = CrossingReport(crossings=np.empty((0, 6)))
    found: list[np.ndarray] = []

    for block in candidate_pairs(normalised):
        a, b = block
        report.n_candidates += a.size
        shared = _shared_vertices(edges[a], edges[b])
        if not include_adjacent:
            keep = shared < 0
            a, b, shared = a[keep], b[keep], shared[keep]
        if a.size == 0:
            continue

        # Both curves degenerate is the segment path — and it is the *whole* straight variant of
        # §4.3, not a fallback. Everything else builds a quartic.
        as_segment = degenerate_curve[a] & degenerate_curve[b]
        for solve, subset in (
            (_solve_segment_block, np.flatnonzero(as_segment)),
            (_solve_quartic_block, np.flatnonzero(~as_segment)),
        ):
            if subset.size == 0:
                continue
            rows = solve(a[subset], b[subset], shared[subset], powers, edges, rank, report)
            if rows.size:
                found.append(rows)

    crossings = np.concatenate(found, axis=0) if found else np.empty((0, 6))
    if crossings.size:
        order = np.lexsort((crossings[:, 1], crossings[:, 0]))
        crossings = crossings[order]
        crossings[:, 4:] = crossings[:, 4:] / scale + origin
    report.crossings = crossings
    return report


def crossing_pairs(
    G: nx.Graph,
    V: np.ndarray,
    E: np.ndarray | None = None,
    *,
    gamma: float = 0.5,
    include_adjacent: bool = True,
) -> np.ndarray:
    """The ``(k, 6)`` crossing array alone. See :func:`crossing_pairs_report` for everything."""
    return crossing_pairs_report(
        G, V, E, gamma=gamma, include_adjacent=include_adjacent
    ).crossings


# --------------------------------------------------------------------------- #
#                          Layer 3 — the metrics                               #
# --------------------------------------------------------------------------- #


def _crossings_for(
    G: nx.Graph,
    V: np.ndarray,
    E: np.ndarray | None,
    gamma: float,
    include_adjacent: bool,
    crossings: np.ndarray | None,
) -> np.ndarray:
    """Use a caller's crossing array if it has one, otherwise compute it.

    ⚠️ **The escape hatch exists for cost, not for convenience.** Layer 2 is ~40 s on a tabula
    layout, and the three metrics below are the same measurement summarised three ways; running
    layer 2 once per metric would treble the most expensive thing in the suite. It is the
    caller's job to have produced ``crossings`` with the same ``gamma`` and ``include_adjacent``
    — nothing here can check that, which is why step 7's registry, not a metric, should own it.
    """
    if crossings is not None:
        return np.asarray(crossings, dtype=float).reshape(-1, 6)
    return crossing_pairs(G, V, E, gamma=gamma, include_adjacent=include_adjacent)


def crossing_number(
    G: nx.Graph,
    V: np.ndarray,
    E: np.ndarray | None = None,
    *,
    gamma: float = 0.5,
    include_adjacent: bool = True,
    crossings: np.ndarray | None = None,
) -> int:
    r"""$\mathrm{Cr}(X)$ — the number of edge crossings in the drawing (eq. 78).

    Every degeneracy is already resolved to a count by layer 2: an overlap is 1, a tangency is 1,
    and a contact at a shared vertex is not a crossing at all (DECISIONS (29)). So this is
    exactly the number of rows layer 2 returns.

    Args:
        G, V, E, gamma: as in :func:`edge_curves`.
        include_adjacent: as in :func:`crossing_pairs_report`.
        crossings: a precomputed layer-2 array, to avoid solving three times for three metrics.

    Returns:
        A non-negative integer. 0 for a graph with fewer than two edges.
    """
    return int(len(_crossings_for(G, V, E, gamma, include_adjacent, crossings)))


def crossings_per_edge_pair(
    G: nx.Graph,
    V: np.ndarray,
    E: np.ndarray | None = None,
    *,
    gamma: float = 0.5,
    include_adjacent: bool = True,
    crossings: np.ndarray | None = None,
) -> float:
    r"""$\mathrm{Cr}(X)/\binom{m}{2}$ — the cross-graph-comparable form.

    ⚠️ **Deliberately not normalised by Purchase's** $c_{\max} = \binom{m}{2} -
    \frac12\sum_v \deg(v)(\deg(v)-1)$. That upper bound assumes at most one crossing per pair and
    that adjacent edges never cross, and this codebase discards both: two quadratic Béziers meet
    in up to four points, and `include_adjacent=True` is the thesis convention. Dividing by a
    bound the measurement can exceed would produce values above 1 and a metric nobody can read.

    Returns:
        A non-negative float, or ``nan`` for a graph with fewer than two edges.
    """
    m = G.number_of_edges()
    if m < 2:
        return float("nan")
    total = crossing_number(
        G, V, E, gamma=gamma, include_adjacent=include_adjacent, crossings=crossings
    )
    return 2.0 * total / (m * (m - 1))


def length_weighted_crossings(
    G: nx.Graph,
    V: np.ndarray,
    E: np.ndarray | None = None,
    *,
    gamma: float = 0.5,
    include_adjacent: bool = True,
    crossings: np.ndarray | None = None,
) -> float:
    r"""$\mathrm{LWCr}(X)$ — crossings weighted by the lengths of the edges involved (eq. 81).

    $$\mathrm{LWCr}(X) = \frac{1}{\bar L(X)}\sum_{a<b}
        \frac{L(e_a) + L(e_b)}{2}\,\mathrm{cr}(e_a, e_b)$$

    A crossing between two long edges disrupts more of the drawing than one between two short
    ones, so this is Cr with that weighting and eq. (80)'s mean edge length as the normaliser.

    ⚠️ **$L(e)$ is the chord $\lVert x_i - x_j\rVert$ (eq. 79), never the arc length, and that is
    load-bearing rather than stylistic.** The chord is identical for the straight and the curved
    rendering of one layout, and so is $\bar L$ — so the entire weight vector is held fixed
    across the §4.3 comparison and the difference between the two variants isolates the change in
    *crossings*. Weighting by arc length would move the crossings and the weights together and
    make the comparison uninterpretable: a curved drawing would score worse partly because its
    edges are longer, which is not a fact about its readability. The arc-over-chord ratio is a
    metric in its own right — :func:`detour_ratio` — and is deliberately not folded in here.

    The lengths come from :func:`~twin.metrics.visual.edge_lengths`, in ``G.edges()`` order, which
    is the order layer 2's edge indices are in. ADR's CSR ordering is a *different* one and the
    two are not interchangeable (CLAUDE.md).

    Returns:
        A non-negative float, or ``nan`` when the graph has no edges or every edge has collapsed
        to zero length.
    """
    from twin.metrics.visual import edge_lengths

    lengths = edge_lengths(V, G)
    if lengths.size == 0:
        return float("nan")
    mean_length = float(lengths.mean())
    if mean_length <= 0:
        return float("nan")

    rows = _crossings_for(G, V, E, gamma, include_adjacent, crossings)
    if len(rows) == 0:
        return 0.0
    a = rows[:, 0].astype(np.int64)
    b = rows[:, 1].astype(np.int64)
    return float(0.5 * (lengths[a] + lengths[b]).sum() / mean_length)


def detour_ratio(
    G: nx.Graph, V: np.ndarray, E: np.ndarray | None = None, *, gamma: float = 0.5
) -> np.ndarray:
    r"""Arc length over chord length, per edge, in ``G.edges()`` order (spec §2.2).

    What the curved formulation costs in ink. Bounded below by 1, and **1 at γ=0** — the curve is
    then its chord rather than a nearly-flat approximation of it, to within rounding
    (~$10^{-16}$; the two sides are computed by different expressions, not compared bitwise).

    ⚠️ **No `n_samples`.** The arc length is closed-form (:func:`twin.geometry.arc_length`), and
    approximating it by summing a polyline would make the one metric in the suite that is *about*
    arc length depend on a sampling parameter — the objection this whole module exists to answer
    (DECISIONS (27)).

    It is **deliberately not folded into LWCr**, which weights by the chord: see
    :func:`length_weighted_crossings` for why holding the weights fixed across the straight/curved
    comparison is what makes that comparison mean anything.

    Args:
        G, V, E, gamma: as in :func:`edge_curves`. The signature is `(G, V, E)` like everything
            else here; spec §2.2 writes `(X, E, G)`, reordered by DECISIONS (5).

    Returns:
        ``(m,)`` ratios in ``G.edges()`` order, ``nan`` for an edge whose endpoints coincide —
        a zero chord has no ratio to report, and a layout with coincident vertices is the
        crowding case, not an error.
    """
    controls = edge_curves(G, V, E, gamma=gamma)
    if controls.shape[0] == 0:
        return np.empty(0)

    chord = np.linalg.norm(controls[:, 2] - controls[:, 0], axis=-1)
    arc = geometry.arc_length(controls)
    return np.where(chord > 0, arc / np.where(chord > 0, chord, 1.0), np.nan)
