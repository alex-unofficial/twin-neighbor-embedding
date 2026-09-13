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

import networkx as nx
import numpy as np

from twin import geometry
from twin.metrics.common import edge_index_array, require_n_by_d

__all__ = ["edge_curves"]


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
