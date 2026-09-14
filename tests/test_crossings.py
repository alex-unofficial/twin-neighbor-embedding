r"""`twin/metrics/crossings.py` — the exact crossing pipeline (spec §2.4, step 5).

`docs/CROSSING_GEOMETRY.md` is the specification; this file is what checks it. Two rules govern
what is written here, and both come from failures this project has already had:

* **The oracle must not share code with the thing it checks.** `twin_adjacency` went three
  sessions untested because every test compared two callers of the same function. So the
  reference implementation below is a dense polyline built in *this module*, and never
  `plots.count_intersections` — which is built on layer 2 and is documented to disagree with it
  on shared-endpoint pairs anyway.
* **Constructed cases, not random sweeps, for the degeneracies.** No-false-negatives for a
  transversal crossing is a *theorem* — the Bernstein hull property forces a sign-changing
  quartic to straddle zero in its coefficients — so a random-quartic sweep confirms something
  that cannot fail. The gap is the exact interior tangency and the exact-endpoint root, and those
  do not occur under random generation. They are constructed here.
"""

import networkx as nx
import numpy as np
import pytest

from twin import geometry
from twin.metrics.crossings import (
    candidate_pairs,
    crossing_number,
    crossing_pairs,
    crossing_pairs_report,
    crossings_per_edge_pair,
    edge_curves,
    length_weighted_crossings,
    normalize_to_unit_box,
)


def _sample(B: np.ndarray, n: int) -> np.ndarray:
    """de Casteljau evaluation, written out here so the oracle shares nothing with the module."""
    t = np.linspace(0.0, 1.0, n).reshape(-1, *([1] * (B.ndim - 2)), 1)
    b0, b1, b2 = B[..., 0, :], B[..., 1, :], B[..., 2, :]
    return (1 - t) ** 2 * b0 + 2 * t * (1 - t) * b1 + t**2 * b2


# --------------------------------------------------------------------------- #
#                        Layer 1 — edge_curves                                 #
# --------------------------------------------------------------------------- #


def test_edge_curves_shape_and_endpoints(karate, karate_random_layout):
    G, V = karate, karate_random_layout
    m = G.number_of_edges()
    E = np.random.default_rng(1).normal(size=(m, 2))

    B = edge_curves(G, V, E, gamma=0.5)

    assert B.shape == (m, 3, 2)
    for e, (u, v) in enumerate(G.edges()):
        np.testing.assert_allclose(B[e, 0], V[u])
        np.testing.assert_allclose(B[e, 2], V[v])


def test_edge_curves_is_in_G_edges_order(karate, karate_random_layout):
    """The order of record. A CSR-ordered implementation would pass on karate — hence the
    endpoint check above rather than a comparison against a sorted edge list."""
    G, V = karate, karate_random_layout
    B = edge_curves(G, V)
    endpoints = np.array([[V[u], V[v]] for u, v in G.edges()])

    np.testing.assert_allclose(B[:, [0, 2]], endpoints)


@pytest.mark.parametrize("gamma", [0.0, 0.25, 0.5, 1.0, 1.5])
def test_no_edge_points_gives_an_exact_straight_segment(karate, karate_random_layout, gamma):
    """``E is None`` is the straight variant of §4.3, at any γ, through the curved code path."""
    G, V = karate, karate_random_layout
    B = edge_curves(G, V, None, gamma=gamma)

    np.testing.assert_allclose(geometry.bulge(B), 0.0, atol=1e-15)
    midpoints = np.array([(V[u] + V[v]) / 2 for u, v in G.edges()])
    np.testing.assert_allclose(B[:, 1], midpoints, atol=1e-15)


def test_gamma_zero_ignores_the_edge_points(karate, karate_random_layout):
    """γ=0 is the chord whatever the edge points do — the other route to the straight variant."""
    G, V = karate, karate_random_layout
    E = np.random.default_rng(2).normal(size=(G.number_of_edges(), 2)) * 100

    np.testing.assert_allclose(
        edge_curves(G, V, E, gamma=0.0), edge_curves(G, V, None), atol=1e-15
    )


def test_edge_curves_matches_what_draw_curved_edges_draws(karate, karate_random_layout):
    """Layer 1 and the picture are the same curve — the reason `twin/geometry.py` exists.

    Compared against the *plotted* polyline, not against `geometry`, so a conversion applied in
    one place and not the other shows up here.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from twin.plots import draw_curved_edges

    G, V = karate, karate_random_layout
    E = np.random.default_rng(3).normal(size=(G.number_of_edges(), 2))

    figure, ax = plt.subplots()
    try:
        # plots.py is the legacy (d, n) surface (CLAUDE.md): it takes the transposes.
        draw_curved_edges(G, 2, V.T, E.T, ax, n_samples=9, alpha=0.5, linewidth=1.0, gamma=0.75)
        drawn = np.array([line.get_xydata() for line in ax.lines])
    finally:
        plt.close(figure)

    measured = np.moveaxis(_sample(edge_curves(G, V, E, gamma=0.75), 9), 0, 1)
    np.testing.assert_allclose(measured, drawn, atol=1e-13)


def test_edge_curves_rejects_a_transposed_layout(karate):
    G = karate
    with pytest.raises(ValueError, match="probably a"):
        edge_curves(G, np.zeros((2, G.number_of_nodes())))
    with pytest.raises(ValueError, match="E must be"):
        edge_curves(G, np.zeros((G.number_of_nodes(), 2)), np.zeros((3, 2)))


def test_edge_curves_on_an_edgeless_graph():
    G = nx.empty_graph(4)
    B = edge_curves(G, np.zeros((4, 2)))
    assert B.shape == (0, 3, 2)


def test_edge_curves_is_dimension_agnostic(karate, karate_random_layout):
    """Layer 1 does not care about d; only the detector is planar."""
    G = karate
    V = np.random.default_rng(4).normal(size=(G.number_of_nodes(), 3))
    assert edge_curves(G, V).shape == (G.number_of_edges(), 3, 3)


# --------------------------------------------------------------------------- #
#                     Layer 2a — normalisation and broad phase                 #
# --------------------------------------------------------------------------- #


def test_normalisation_fills_the_unit_box_isotropically(karate, karate_random_layout):
    G = karate
    E = np.random.default_rng(5).normal(size=(G.number_of_edges(), 2)) * 7 + 1000
    B = edge_curves(G, karate_random_layout * 3 + 1000, E)

    scaled, scale, origin = normalize_to_unit_box(B)
    boxes = geometry.exact_bounding_boxes(scaled)
    lo, hi = boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)

    assert lo == pytest.approx([0.0, 0.0], abs=1e-12)
    assert max(hi) == pytest.approx(1.0, abs=1e-12)
    assert min(hi) <= 1.0 + 1e-12
    # one scale for both axes: the map is a similarity, so it moves no crossing
    np.testing.assert_allclose(scaled, (B - origin) * scale, atol=0)


def test_normalisation_survives_a_fully_collapsed_layout():
    """Every point coincident: the extent is 0 and the scale must not be an infinity."""
    B = np.zeros((5, 3, 2))
    scaled, scale, _ = normalize_to_unit_box(B)
    assert scale == 1.0
    assert np.isfinite(scaled).all()

    empty, scale, origin = normalize_to_unit_box(np.empty((0, 3, 2)))
    assert empty.shape == (0, 3, 2) and scale == 1.0 and origin.shape == (2,)


def _brute_force_overlaps(B: np.ndarray, tol: float = 0.0) -> set:
    """Every box-overlapping pair, by sampling the curve rather than by the closed-form box."""
    samples = np.moveaxis(_sample(B, 2001), 0, 1)  # (m, n_samples, 2)
    lo, hi = samples.min(axis=1), samples.max(axis=1)
    m = len(B)
    return {
        (a, b)
        for a in range(m)
        for b in range(a + 1, m)
        if np.all(lo[a] <= hi[b] + tol) and np.all(lo[b] <= hi[a] + tol)
    }


def test_the_broad_phase_finds_every_box_overlapping_pair(karate, karate_random_layout):
    """Conservative in the direction that matters: it may pass a non-crossing pair, never drop
    a crossing one. Checked against sampled extents, not against the module's own boxes."""
    G = karate
    E = np.random.default_rng(6).normal(size=(G.number_of_edges(), 2))
    B, _, _ = normalize_to_unit_box(edge_curves(G, karate_random_layout, E))

    found = {(int(a), int(b)) for block in candidate_pairs(B) for a, b in block.T}
    assert found >= _brute_force_overlaps(B)
    # BOX_PAD's reason to exist: pairs that touch to within a few ulp are still offered.
    touching = _brute_force_overlaps(B, tol=1e-13) - _brute_force_overlaps(B)
    assert touching and found >= touching


def test_candidate_pairs_are_unique_ordered_and_chunk_independent(karate, karate_random_layout):
    G = karate
    E = np.random.default_rng(7).normal(size=(G.number_of_edges(), 2))
    B, _, _ = normalize_to_unit_box(edge_curves(G, karate_random_layout, E))

    blocks = list(candidate_pairs(B))
    pairs = np.concatenate(blocks, axis=1)
    assert (pairs[0] < pairs[1]).all()
    assert len({tuple(p) for p in pairs.T}) == pairs.shape[1]

    for chunk in (1, 3, 10_000):
        other = np.concatenate(list(candidate_pairs(B, chunk_edges=chunk)), axis=1)
        assert {tuple(p) for p in other.T} == {tuple(p) for p in pairs.T}


def test_chunking_bounds_the_block_size(karate, karate_random_layout):
    """The memory bound of the broad phase is the *query* chunk, not a downstream block loop."""
    G = karate
    E = np.random.default_rng(8).normal(size=(G.number_of_edges(), 2))
    B, _, _ = normalize_to_unit_box(edge_curves(G, karate_random_layout, E))

    blocks = list(candidate_pairs(B, chunk_edges=4))
    assert len(blocks) > 1
    assert all(block.shape[1] <= 4 * G.number_of_edges() for block in blocks)


def test_the_broad_phase_on_degenerate_inputs():
    assert list(candidate_pairs(np.empty((0, 3, 2)))) == []
    assert list(candidate_pairs(np.zeros((1, 3, 2)))) == []
    # Two identical degenerate curves still overlap and must be offered to the solver.
    assert sum(b.shape[1] for b in candidate_pairs(np.zeros((2, 3, 2)))) == 1


# --------------------------------------------------------------------------- #
#                     Layer 2b — the oracle, and the constructions             #
# --------------------------------------------------------------------------- #


def _pair(Ba: np.ndarray, Bb: np.ndarray):
    """Two non-adjacent edges drawn as exactly these Bézier curves.

    At γ=0.5 the conversion is the identity — $B_1 = 2\\gamma P_1 + (1-2\\gamma)(P_0+P_2)/2 = P_1$
    — so a curve can be specified by its control points and fed straight through `edge_curves`.
    """
    G = nx.Graph([(0, 1), (2, 3)])
    V = np.array([Ba[0], Ba[2], Bb[0], Bb[2]])
    E = np.array([Ba[1], Bb[1]])
    return G, V, E


def _count(Ba, Bb, **kw) -> int:
    return len(crossing_pairs(*_pair(Ba, Bb), gamma=0.5, **kw))


def _polyline_oracle(Ba: np.ndarray, Bb: np.ndarray) -> int | None:
    """Crossings by dense polyline intersection. Shares no code with `crossings.py`.

    None when the two polylines overlap along a stretch, which shapely reports as a `LineString`
    and which the convention counts differently — those pairs are checked by construction
    instead.
    """
    from shapely.geometry import LineString, MultiPoint, Point

    inter = LineString(_sample(Ba, 1200)).intersection(LineString(_sample(Bb, 1200)))
    if inter.is_empty:
        return 0
    if isinstance(inter, Point):
        return 1
    if isinstance(inter, MultiPoint):
        return len(inter.geoms)
    return None


def test_the_exact_method_agrees_with_a_dense_polyline_oracle():
    """1200 random non-adjacent pairs, counts must agree. The oracle is built here, from
    `_sample`, and never from `plots.count_intersections` — which is about to be built on top of
    the thing under test and is documented to disagree on shared-endpoint pairs."""
    rng = np.random.default_rng(0)
    disagreements, checked, worst = [], 0, 0.0

    for gamma in (0.5, 0.9, 1.3):
        for _ in range(400):
            P = rng.normal(size=(2, 3, 2))
            B = geometry.bezier_control_points(P, gamma)
            G = nx.Graph([(0, 1), (2, 3)])
            V = np.array([B[0, 0], B[0, 2], B[1, 0], B[1, 2]])
            rows = crossing_pairs(G, V, np.array([P[0, 1], P[1, 1]]), gamma=gamma)

            want = _polyline_oracle(B[0], B[1])
            if want is None:
                continue
            checked += 1
            if len(rows) != want:
                disagreements.append((gamma, len(rows), want))
            for row in rows:
                t, s = row[2], row[3]
                pa = (1 - t) ** 2 * B[0, 0] + 2 * t * (1 - t) * B[0, 1] + t**2 * B[0, 2]
                pb = (1 - s) ** 2 * B[1, 0] + 2 * s * (1 - s) * B[1, 1] + s**2 * B[1, 2]
                worst = max(worst, np.linalg.norm(pa - pb), np.linalg.norm(pa - row[4:]))

    assert checked >= 1000
    assert disagreements == []
    assert worst < 1e-7, "every reported point must lie on both curves"


# --- the five intersection counts, from one family --------------------------- #
#
# y = x^2 - 1 and x = y^2 - 1 meet where x(x+1)(x^2-x-1) = 0, i.e. at four real points
# x = -1, -0.618, 0, 1.618. Shortening the second parabola's arc from below drops them one at a
# time, so 0/1/2/3/4 intersections come out of a single construction with one parameter.
#
# The arc of y = x^2 for x in [-a, a] is the Bézier (-a, a^2), (0, -a^2), (a, a^2); a sub-arc of
# x = y^2 - 1 for y in [y0, y1] is (y0^2-1, y0), (y0*y1-1, (y0+y1)/2), (y1^2-1, y1).

PARABOLA_A = np.array([[-2.0, 3.0], [0.0, -5.0], [2.0, 3.0]])  # y = x^2 - 1, x in [-2, 2]


def _sideways_parabola(y0: float, y1: float = 2.0) -> np.ndarray:
    return np.array([[y0 * y0 - 1, y0], [y0 * y1 - 1, (y0 + y1) / 2], [y1 * y1 - 1, y1]])


@pytest.mark.parametrize(
    "y0, expected",
    [(-2.0, 4), (-0.9, 3), (-0.5, 2), (0.5, 1), (1.9, 0)],
)
def test_exactly_k_intersections(y0, expected):
    """Bézout's bound is 4 and every count below it is reachable; all five are constructed."""
    B = _sideways_parabola(y0)
    assert _count(PARABOLA_A, B) == expected
    assert _polyline_oracle(PARABOLA_A, B) == expected


def test_interior_tangency_counts_once_and_is_reported():
    """$y = x^2$ and $y = -x^2$ touch at the origin without crossing: a double root.

    This is the one case the Bernstein filter's hull property does not cover — Q touches zero
    without changing sign — so it is constructed rather than hoped for.
    """
    up = np.array([[-1.0, 1.0], [0.0, -1.0], [1.0, 1.0]])
    down = np.array([[-1.0, -1.0], [0.0, 1.0], [1.0, -1.0]])

    report = crossing_pairs_report(*_pair(up, down), gamma=0.5)
    assert len(report.crossings) == 1
    assert report.n_near_tangential >= 1
    np.testing.assert_allclose(report.crossings[0, 4:], [0.0, 0.0], atol=1e-6)


def test_identical_curves_count_as_exactly_one():
    """Eq. (74)'s cardinality is infinite here; spec §2.4 fixes the convention at 1."""
    curve = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]])

    report = crossing_pairs_report(*_pair(curve, curve.copy()), gamma=0.5)
    assert len(report.crossings) == 1
    assert report.n_degenerate == 1


def test_collinear_control_points_do_not_break_the_solve():
    """β = 2·det(B1-B0, B2-B0) vanishes, so this curve can never be the one implicitised.

    The control points are collinear but B1 is not the midpoint, so the curve runs out to x=1.8
    and doubles back — the parameter of a point on it is *not* the parameter of the segment it
    sweeps, which is why `_parameter_at_point` solves the quadratic rather than interpolating.
    """
    folding = np.array([[0.0, 0.0], [3.0, 0.0], [1.0, 0.0]])
    crossing = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])

    rows = crossing_pairs(*_pair(folding, crossing), gamma=0.5)
    assert len(rows) == 1
    np.testing.assert_allclose(rows[0, 4:], [1.0, 0.0], atol=1e-9)


def test_parallel_bulges_the_q4_equals_zero_path():
    """$Q_4 = -16\\det(D_A, D_B)^2$ is exactly zero when the two bulges are parallel.

    Building the companion matrix divides by $Q_4$, so without deflation this is a division by
    zero rather than a hard case (DECISIONS (30)).
    """
    a = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    b = np.array([[0.5, -1.0], [1.0, 1.5], [1.5, 2.0]])
    Ba = geometry.bezier_control_points(a, 0.5)
    Bb = geometry.bezier_control_points(b, 0.5)
    da, db = geometry.bulge(Ba), geometry.bulge(Bb)
    assert abs(da[0] * db[1] - da[1] * db[0]) == 0.0, "the construction must give Q4 == 0 exactly"

    expected = _polyline_oracle(Ba, Bb)
    assert expected == 1
    assert _count(a, b) == expected


def test_a_vertex_lying_exactly_on_another_edges_curve_is_found():
    """The Bernstein filter's one real false negative: a root exactly at s=0 or s=1.

    Edge (2,3) starts *on* edge (0,1)'s curve. The pair is non-adjacent, so this is a genuine
    crossing — the coincident-vertex crowding case — and the plain sign test would drop it.
    """
    arc = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]])
    on_curve = 0.25 * (arc[0] + 2 * arc[1] + arc[2])  # the curve at t = 1/2

    G = nx.Graph([(0, 1), (2, 3)])
    V = np.array([arc[0], arc[2], on_curve, on_curve + np.array([0.0, -2.0])])
    E = np.array([arc[1], on_curve + np.array([0.0, -1.0])])

    rows = crossing_pairs(G, V, E, gamma=0.5)
    assert len(rows) == 1
    np.testing.assert_allclose(rows[0, 4:], on_curve, atol=1e-9)


def test_the_bernstein_endpoint_case_directly():
    """The guide's example: ``Q = [0, 1, 1, 1, 1]`` is ``s(s+1)(s^2+1)``, a root at 0 exactly,
    and its Bernstein coefficients never change sign. β0 = Q(0) and β4 = Q(1) exactly, which is
    what makes the fix free."""
    from twin.metrics.crossings import BERNSTEIN

    Q = np.array([[0.0, 1.0, 1.0, 1.0, 1.0]])
    beta = Q @ BERNSTEIN.T

    np.testing.assert_allclose(beta[0], [0.0, 0.25, 2 / 3, 1.5, 4.0], atol=1e-15)
    assert not ((beta > 0).any() and (beta < 0).any()), "no sign change — the false negative"
    assert beta[0, 0] == Q[0, 0]  # = Q(0)
    assert beta[0, 4] == pytest.approx(Q.sum())  # = Q(1)


# --- adjacent edges ---------------------------------------------------------- #


def _adjacent(second_end, second_mid):
    """Edges (0,1) and (0,2): an arc and a second edge, sharing vertex 0."""
    G = nx.Graph([(0, 1), (0, 2)])
    V = np.array([[0.0, 0.0], [4.0, 0.0], second_end])
    E = np.array([[2.0, 4.0], second_mid])
    return G, V, E


def test_adjacent_edges_meeting_only_at_the_shared_vertex_do_not_cross():
    """The contact at the shared vertex is excluded by identity, and nothing else is there.

    Here the Bernstein test rejects the pair before the solve, so `n_shared_endpoint` — which
    counts contacts *divided out*, not contacts that exist — stays 0. The pair below, which has a
    genuine second crossing and therefore reaches the solve, is where that counter moves.
    """
    report = crossing_pairs_report(*_adjacent([4.0, -3.0], [2.0, -1.5]), gamma=0.5)
    assert len(report.crossings) == 0


def test_adjacent_edges_keep_a_genuine_second_crossing():
    """§3.1.3's reason for ``include_adjacent=True``: curved edges incident to a shared vertex
    may still cross away from it. The arc is $y = 2x - x^2/2$ and the second edge is $y = 0.75x$,
    which meet again at exactly $(2.5, 1.875)$."""
    report = crossing_pairs_report(*_adjacent([4.0, 3.0], [2.0, 1.5]), gamma=0.5)

    assert len(report.crossings) == 1
    np.testing.assert_allclose(report.crossings[0, 4:], [2.5, 1.875], atol=1e-9)
    assert report.n_shared_endpoint == 1


def test_include_adjacent_false_drops_the_pair_entirely():
    args = _adjacent([4.0, 3.0], [2.0, 1.5])
    assert len(crossing_pairs(*args, gamma=0.5, include_adjacent=True)) == 1
    assert len(crossing_pairs(*args, gamma=0.5, include_adjacent=False)) == 0


def test_straight_adjacent_edges_meet_only_at_their_shared_vertex():
    """The segment path gets the same rule for free, and exactly: two distinct lines meet once."""
    G = nx.Graph([(0, 1), (0, 2)])
    V = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    assert len(crossing_pairs(G, V, None)) == 0


def test_collinear_overlapping_straight_edges_count_as_one():
    """Two edges drawn along the same line, overlapping: cardinality is infinite, `cr` is 1."""
    G = nx.Graph([(0, 1), (2, 3)])
    V = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0], [3.0, 0.0]])

    report = crossing_pairs_report(G, V, None)
    assert len(report.crossings) == 1
    assert report.n_degenerate == 1


# --- invariance and plumbing -------------------------------------------------- #


@pytest.mark.parametrize("gamma", [0.0, 0.5])
def test_the_count_is_invariant_under_similarity(karate, karate_random_layout, gamma):
    """Rotation, translation and uniform scaling move no crossing — and the unit-box
    normalisation is what makes that true of the *implementation* as well as the geometry."""
    G, V = karate, karate_random_layout
    E = np.random.default_rng(11).normal(size=(G.number_of_edges(), 2))
    base = len(crossing_pairs(G, V, E, gamma=gamma))
    assert base > 0

    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    for transform in (
        lambda X: X + np.array([13.0, -7.0]),
        lambda X: X @ rotation.T,
        lambda X: X * 1e3,
        lambda X: X * 1e-3,
        lambda X: (X @ rotation.T) * 250 + np.array([-4.0, 9.0]),
    ):
        assert len(crossing_pairs(G, transform(V), transform(E), gamma=gamma)) == base


def test_the_count_does_not_depend_on_the_query_chunk(karate, karate_random_layout):
    """`QUERY_CHUNK_EDGES` bounds memory; it must not be able to change an answer."""
    from twin.metrics import crossings as module

    G, V = karate, karate_random_layout
    E = np.random.default_rng(12).normal(size=(G.number_of_edges(), 2))
    saved = module.QUERY_CHUNK_EDGES
    try:
        counts = []
        for chunk in (1, 7, 4096):
            module.QUERY_CHUNK_EDGES = chunk
            counts.append(len(crossing_pairs(G, V, E)))
    finally:
        module.QUERY_CHUNK_EDGES = saved
    assert len(set(counts)) == 1


def test_the_output_columns_are_what_they_claim(karate, karate_random_layout):
    G, V = karate, karate_random_layout
    E = np.random.default_rng(13).normal(size=(G.number_of_edges(), 2))
    rows = crossing_pairs(G, V, E)

    a, b, t, s = rows[:, 0].astype(int), rows[:, 1].astype(int), rows[:, 2], rows[:, 3]
    assert (a < b).all()
    assert (0 <= t).all() and (t <= 1).all() and (0 <= s).all() and (s <= 1).all()

    B = edge_curves(G, V, E)
    for row, ea, eb, tt, ss in zip(rows, a, b, t, s):
        ca = (1 - tt) ** 2 * B[ea, 0] + 2 * tt * (1 - tt) * B[ea, 1] + tt**2 * B[ea, 2]
        cb = (1 - ss) ** 2 * B[eb, 0] + 2 * ss * (1 - ss) * B[eb, 1] + ss**2 * B[eb, 2]
        np.testing.assert_allclose(ca, cb, atol=1e-7)
        np.testing.assert_allclose(ca, row[4:], atol=1e-7)


def test_a_three_dimensional_layout_is_refused(karate):
    G = karate
    V = np.random.default_rng(14).normal(size=(G.number_of_nodes(), 3))
    with pytest.raises(ValueError, match="planar"):
        crossing_pairs(G, V)


def test_an_edgeless_graph_has_no_crossings():
    report = crossing_pairs_report(nx.empty_graph(4), np.zeros((4, 2)))
    assert report.crossings.shape == (0, 6)
    assert report.n_candidates == 0


# --------------------------------------------------------------------------- #
#                          Layer 3 — the metrics                               #
# --------------------------------------------------------------------------- #


def test_crossing_number_is_the_row_count(karate, karate_random_layout):
    G, V = karate, karate_random_layout
    E = np.random.default_rng(20).normal(size=(G.number_of_edges(), 2))
    rows = crossing_pairs(G, V, E)

    assert crossing_number(G, V, E) == len(rows)
    assert crossing_number(G, V, E, crossings=rows) == len(rows)


def test_crossings_per_edge_pair_is_cr_over_m_choose_2(karate, karate_random_layout):
    G, V = karate, karate_random_layout
    E = np.random.default_rng(21).normal(size=(G.number_of_edges(), 2))
    m = G.number_of_edges()

    assert crossings_per_edge_pair(G, V, E) == pytest.approx(
        crossing_number(G, V, E) / (m * (m - 1) / 2)
    )
    assert np.isnan(crossings_per_edge_pair(nx.path_graph(2), np.zeros((2, 2))))


def test_lwcr_uses_the_chord_and_the_G_edges_order(karate, karate_random_layout):
    """Recomputed here from `edge_lengths`, the order of record, against the metric."""
    from twin.metrics.visual import edge_lengths

    G, V = karate, karate_random_layout
    E = np.random.default_rng(22).normal(size=(G.number_of_edges(), 2))
    rows = crossing_pairs(G, V, E)
    lengths = edge_lengths(V, G)

    expected = sum(
        (lengths[int(row[0])] + lengths[int(row[1])]) / 2 for row in rows
    ) / lengths.mean()
    assert length_weighted_crossings(G, V, E, crossings=rows) == pytest.approx(expected)


def test_lwcr_weights_do_not_move_between_the_straight_and_curved_variants(
    karate, karate_random_layout
):
    """The reason $L(e)$ is the chord (eq. 79). The arc length of the curved variant is strictly
    longer, so an arc-weighted LWCr would differ between the variants even if they had exactly
    the same crossings — confounding "fewer crossings" with "longer curves"."""
    from twin.metrics.visual import edge_lengths

    G, V = karate, karate_random_layout
    E = np.random.default_rng(23).normal(size=(G.number_of_edges(), 2))

    straight = edge_lengths(V, G)
    # `edge_lengths` reads only the vertex points, so γ cannot enter it -- which is the point.
    curved_controls = edge_curves(G, V, E, gamma=0.9)
    np.testing.assert_allclose(
        straight, np.linalg.norm(curved_controls[:, 2] - curved_controls[:, 0], axis=1)
    )


def test_lwcr_reduces_to_the_crossing_number_when_every_edge_is_the_same_length():
    """With all $L(e) = \\bar L$, every weight is 1 and LWCr must equal Cr exactly."""
    G = nx.cycle_graph(8)
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    V = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    # A long chord through the middle of every edge forces plenty of crossings.
    E = -2.0 * np.stack([np.cos(angles + np.pi / 8), np.sin(angles + np.pi / 8)], axis=1)

    cr = crossing_number(G, V, E)
    assert cr > 0
    assert length_weighted_crossings(G, V, E) == pytest.approx(float(cr))


def test_the_metrics_on_a_drawing_with_no_crossings():
    """A path drawn on a line: no crossings, and LWCr is 0 rather than nan."""
    G = nx.path_graph(4)
    V = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    assert crossing_number(G, V, None) == 0
    assert crossings_per_edge_pair(G, V, None) == 0.0
    assert length_weighted_crossings(G, V, None) == 0.0


def test_lwcr_is_nan_when_there_is_nothing_to_weight():
    assert np.isnan(length_weighted_crossings(nx.empty_graph(3), np.zeros((3, 2))))
    assert np.isnan(length_weighted_crossings(nx.path_graph(3), np.zeros((3, 2))))


def test_collinear_edges_touching_at_a_point_are_not_an_overlap():
    """A path drawn along a straight line must not report a crossing at every interior vertex —
    the consecutive edges are collinear and meet at exactly one point, which is the shared
    vertex. Two *non-adjacent* collinear edges meeting at one point is the coincident-vertex
    case, and that does count."""
    path = nx.path_graph(4)
    on_a_line = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert crossing_number(path, on_a_line, None) == 0

    disjoint = nx.Graph([(0, 1), (2, 3)])
    touching = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    report = crossing_pairs_report(disjoint, touching, None)
    assert len(report.crossings) == 1
    assert report.n_degenerate == 0  # a point, not a stretch
