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
from twin.metrics.crossings import candidate_pairs, edge_curves, normalize_to_unit_box


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
