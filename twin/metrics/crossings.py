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

from typing import Iterator

import networkx as nx
import numpy as np
import shapely

from twin import geometry
from twin.metrics.common import edge_index_array, require_n_by_d

__all__ = ["candidate_pairs", "count_candidate_pairs", "edge_curves", "normalize_to_unit_box"]

#: How many edges are queried against the R-tree at once. **Not a tuning knob for speed** — it
#: is what bounds the broad phase's memory. `shapely.STRtree.query` materialises every pair it
#: finds before returning, so the bulk query over all of tabula's 49368 edges returns an 11.4 M
#: pair array (365 MB) in one allocation, and blocking the *solve* does not help: the array
#: already exists by then (DECISIONS (31a)). Chunking the query is what caps it.
QUERY_CHUNK_EDGES = 4096

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
