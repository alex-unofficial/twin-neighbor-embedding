r"""Spec §2.2 — the visual-quality metrics: point crowding, distributions, edge lengths.

These read the layout and nothing else (except `edge_lengths`, which reads an edge set), so
almost every property here is a statement about geometry that can be hand-computed. Where a
number is pinned it is pinned against a closed form, not against a previous run — the golden
file (`test_metric_golden.py`) is what guards against a silent move on a realistic layout.
"""

import numpy as np
import pytest

from twin.metrics.common import clear_loo_cache
from twin.metrics.visual import (
    nn_distance_distribution,
    node_resolution,
    node_resolution_vertexwise,
    normalized_nn_distance,
    pairwise_distance_distribution,
)


# --------------------------------------------------------------------------- #
#                                  Fixtures                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def unit_square() -> np.ndarray:
    """Four corners of a unit square: four sides at 1, two diagonals at $\\sqrt2$."""
    return np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


@pytest.fixture
def two_tight_clusters() -> np.ndarray:
    """Two groups of four exactly coincident points, far apart.

    The extreme of point crowding: every vertex has a neighbour at distance 0 while the layout
    as a whole is not collapsed, so $\\bar d > 0$ and the crowding metrics are 0 rather than
    ``nan``.
    """
    return np.repeat(np.array([[0.0, 0.0], [100.0, 0.0]]), 4, axis=0)


@pytest.fixture
def grid9() -> np.ndarray:
    """A 3x3 unit grid — an uncrowded layout, for the direction-of-effect tests."""
    xs, ys = np.meshgrid(np.arange(3.0), np.arange(3.0))
    return np.column_stack([xs.ravel(), ys.ravel()])


def similarity(X: np.ndarray, *, scale=2.5, angle=0.7, reflect=True, shift=(3.0, -4.0)):
    """Scale, rotate, optionally reflect, and translate — the transformations a layout is
    defined only up to."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    if reflect:
        R = R @ np.array([[1.0, 0.0], [0.0, -1.0]])
    return scale * (X @ R.T) + np.asarray(shift)


# --------------------------------------------------------------------------- #
#                             NND (eqs. 65-67)                                 #
# --------------------------------------------------------------------------- #


def test_nnd_on_the_unit_square_is_its_closed_form(unit_square):
    r"""$d_1 = 1$ (every corner has two neighbours at 1) and
    $\bar d = (4 + 2\sqrt2)/6$, so $\mathrm{NND}_1 = 6/(4+2\sqrt2)$."""
    expected = 6.0 / (4.0 + 2.0 * np.sqrt(2.0))
    assert normalized_nn_distance(unit_square, 1) == pytest.approx(expected)
    # k=2 too: the two nearest of every corner are both at distance 1.
    assert normalized_nn_distance(unit_square, 2) == pytest.approx(expected)


def test_nnd_at_k_equals_n_minus_1_is_exactly_one(unit_square, grid9):
    r"""With every other vertex in the neighbourhood, $d_k$ *is* $\bar d$.

    A universal identity rather than a fixture value, and the cheapest possible check that the
    numerator and the denominator average over the same distances.
    """
    for X in (unit_square, grid9):
        assert normalized_nn_distance(X, X.shape[0] - 1) == pytest.approx(1.0)


def test_nnd_is_one_when_every_distance_is_equal():
    """An equilateral triangle has one distance, so local spacing is the global scale."""
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    assert normalized_nn_distance(X, 1) == pytest.approx(1.0)


def test_nnd_is_invariant_under_similarity(grid9):
    """Scale, rotation, reflection and translation: a layout is defined only up to these."""
    for k in (1, 3, 5):
        assert normalized_nn_distance(similarity(grid9), k) == pytest.approx(
            normalized_nn_distance(grid9, k)
        )


def test_nnd_sees_crowding(grid9, two_tight_clusters):
    """Direction of effect: smaller means more crowded. Without this the metric could be
    inverted and every invariance above would still hold."""
    assert normalized_nn_distance(two_tight_clusters, 1) == 0.0
    assert normalized_nn_distance(grid9, 1) > 0.5


def test_nnd_is_zero_on_a_collapsed_neighbourhood(two_tight_clusters):
    """Three coincident neighbours each, so the numerator is 0 while the layout is not."""
    assert normalized_nn_distance(two_tight_clusters, 3) == 0.0


def test_nnd_is_nan_on_a_fully_collapsed_layout():
    r"""$\bar d = 0$. The limit exists; reporting a limit as a score is how an undefined
    quantity reaches a table (as in SNS)."""
    assert np.isnan(normalized_nn_distance(np.zeros((6, 2)), 2))
    assert np.isnan(normalized_nn_distance(np.zeros((6, 2)), [1, 2])).all()


# --------------------------------------------------------------------------- #
#                         Sweeping k, and the slicing rule                     #
# --------------------------------------------------------------------------- #


def test_swept_k_agrees_with_scalar_k(grid9):
    swept = normalized_nn_distance(grid9, [1, 3, 5])
    assert swept.shape == (3,)
    for value, k in zip(swept, [1, 3, 5]):
        assert value == pytest.approx(normalized_nn_distance(grid9, k))


def test_a_scalar_k_returns_a_float_and_a_tuple_an_array(grid9):
    """`np.ndim(k)`, not `isinstance`: a tuple used to fall into the scalar path."""
    assert isinstance(normalized_nn_distance(grid9, 3), float)
    assert isinstance(normalized_nn_distance(grid9, (1, 2)), np.ndarray)


def test_swept_k_agrees_with_scalar_k_under_exact_duplicates(karate_coincident_layout):
    """The one place slicing is safe, and why it is safe.

    The kNN family's neighbour *identities* are not invariant to ``k_max`` under duplicates —
    that is what the leave-one-out cache is forbidden from slicing across. NND reads only the
    neighbour *distances*, whose sorted multiset is unique, so querying $\\max(k)$ once and
    slicing is exact here. This asserts it on the fixture where the identities do move.
    """
    clear_loo_cache()
    swept = normalized_nn_distance(karate_coincident_layout, [3, 5, 16])
    for value, k in zip(swept, [3, 5, 16]):
        clear_loo_cache()
        assert value == pytest.approx(normalized_nn_distance(karate_coincident_layout, k))


def test_nnd_refuses_a_k_with_no_neighbours(grid9):
    with pytest.raises(ValueError, match="exceeds n-1"):
        normalized_nn_distance(grid9, grid9.shape[0])
    with pytest.raises(ValueError, match="k must be >= 1"):
        normalized_nn_distance(grid9, 0)


def test_nnd_refuses_a_non_2d_array():
    with pytest.raises(ValueError, match="must be 2-D"):
        normalized_nn_distance(np.zeros(10), 1)


# --------------------------------------------------------------------------- #
#                          NR and VNR (eqs. 68, 69)                            #
# --------------------------------------------------------------------------- #


@pytest.fixture
def grid16() -> np.ndarray:
    """A 4x4 unit grid: no crowding and no outlier — the reference readable layout."""
    xs, ys = np.meshgrid(np.arange(4.0), np.arange(4.0))
    return np.column_stack([xs.ravel(), ys.ravel()])


@pytest.fixture
def grid16_plus_outlier(grid16) -> np.ndarray:
    """`grid16` and one vertex 50 units away. Seventeen vertices, no near-overlap at all."""
    return np.vstack([grid16, [50.0, 0.0]])


@pytest.fixture
def grid16_plus_overlap(grid16) -> np.ndarray:
    """`grid16` and one vertex at a fifth of the grid spacing from a corner.

    Seventeen vertices and a genuine near-overlap — the artifact VNR exists to detect.
    """
    return np.vstack([grid16, grid16[0] + 0.2])


def test_nr_on_the_unit_square(unit_square):
    r"""$1/\sqrt2$: sides against diagonals."""
    assert node_resolution(unit_square) == pytest.approx(1 / np.sqrt(2))


def test_nr_is_one_when_every_distance_is_equal():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    assert node_resolution(X) == pytest.approx(1.0)


def test_nr_is_zero_on_any_coincident_pair(grid16):
    """A worst-case detector: one duplicated vertex, and the rest of the layout is ignored."""
    X = np.vstack([grid16, grid16[0]])
    assert node_resolution(X) == 0.0


@pytest.mark.parametrize("normalize", ["mean", "max"])
def test_vnr_and_nr_are_invariant_under_similarity(grid16, normalize):
    assert node_resolution_vertexwise(similarity(grid16), normalize) == pytest.approx(
        node_resolution_vertexwise(grid16, normalize)
    )
    assert node_resolution(similarity(grid16)) == pytest.approx(node_resolution(grid16))


@pytest.mark.parametrize("normalize", ["mean", "max"])
def test_vnr_is_nan_on_a_collapsed_layout(normalize):
    assert np.isnan(node_resolution_vertexwise(np.zeros((5, 2)), normalize))


def test_nr_is_nan_on_a_collapsed_layout():
    assert np.isnan(node_resolution(np.zeros((5, 2))))


def test_vnr_refuses_an_unknown_normaliser(grid16):
    with pytest.raises(ValueError, match="normalize must be"):
        node_resolution_vertexwise(grid16, "median")


def test_the_max_normaliser_is_chunked_but_not_changed_by_chunking(grid16, monkeypatch):
    """A chunk boundary must not move the answer; the chunking is a memory bound, not a model."""
    import twin.metrics.visual as visual

    reference = node_resolution_vertexwise(grid16, "max")
    monkeypatch.setattr(visual, "MAX_NORMALISER_CHUNK_BYTES", 8 * 4)  # a few rows at a time
    assert node_resolution_vertexwise(grid16, "max") == pytest.approx(reference)


# --------------------------------------------------------------------------- #
#          The [THESIS EDIT]: what "mean" buys, and what it costs              #
# --------------------------------------------------------------------------- #


def test_the_two_normalisers_agree_on_ordering_without_an_outlier(grid16, grid16_plus_overlap):
    """Where the thesis form is well behaved, the edit changes nothing that matters."""
    for normalize in ("mean", "max"):
        clean = node_resolution_vertexwise(grid16, normalize)
        crowded = node_resolution_vertexwise(grid16_plus_overlap, normalize)
        assert clean > crowded


def test_the_two_normalisers_disagree_when_one_vertex_is_pushed_far_away(
    grid16_plus_outlier, grid16_plus_overlap
):
    """The justification for the edit, as a test rather than as a paragraph (spec §0.5).

    Two seventeen-vertex layouts. One is a clean grid plus a distant vertex — nothing overlaps
    anything. The other is a clean grid plus a near-overlap, which is precisely the artifact VNR
    is for. The ``"mean"`` normaliser ranks them correctly. The ``"max"`` normaliser ranks the
    near-overlap as the *more readable* of the two, by a factor of three, because a single
    distant vertex makes every per-vertex maximum approximately the diameter — at which point it
    is no longer a per-vertex quantity at all.
    """
    outlier_mean = node_resolution_vertexwise(grid16_plus_outlier, "mean")
    overlap_mean = node_resolution_vertexwise(grid16_plus_overlap, "mean")
    outlier_max = node_resolution_vertexwise(grid16_plus_outlier, "max")
    overlap_max = node_resolution_vertexwise(grid16_plus_overlap, "max")

    assert outlier_mean > overlap_mean
    assert outlier_max < overlap_max
    assert overlap_max > 3 * outlier_max


def test_an_outlier_makes_the_max_normaliser_stop_being_per_vertex(grid16, grid16_plus_outlier):
    """The mechanism behind the edit, stated as what it actually is.

    §0.5 says the ``"max"`` form "collapses back toward the global NR". That is a claim about
    the *normaliser*, not about the value: with one distant vertex, every vertex's maximum is
    approximately the diameter, so VNR stops dividing by a per-vertex quantity and starts
    dividing by a global one — which is NR's normaliser. Measured here as the spread of the
    per-vertex maxima, over the sixteen vertices that are not themselves the outlier.

    Read as a *value*, the phrase does not hold and is not asserted: VNR$_{\\max}$/NR rises from
    1.2 to 3.7 here, because NR falls off as the diameter while VNR$_\\max$ is held up by the
    outlier's own well-separated row. `DECISIONS.md` records the correction.
    """
    from twin.metrics.visual import _rowwise_max_distance

    def spread(X):
        widest = _rowwise_max_distance(X)[:16]
        return widest.std() / widest.mean()

    # 0.140 -> 0.023: a genuinely per-vertex normaliser becomes an almost constant one.
    assert spread(grid16) > 0.1
    assert spread(grid16_plus_outlier) < spread(grid16) / 5


def test_an_outlier_barely_moves_the_mean_normaliser(grid16, grid16_plus_outlier):
    """Why: the outlier inflates the numerator — its own nearest neighbour is far away — about
    as much as it inflates $\\bar d$, and the two largely cancel."""
    assert node_resolution_vertexwise(grid16_plus_outlier, "mean") == pytest.approx(
        node_resolution_vertexwise(grid16, "mean"), rel=0.1
    )
    assert node_resolution_vertexwise(grid16_plus_outlier, "max") < 0.4 * (
        node_resolution_vertexwise(grid16, "max")
    )


def test_vnr_with_the_mean_normaliser_is_exactly_nnd_at_k_equals_one(grid16, karate_random_layout):
    r"""Not an approximation — the same expression with the sum and the division commuted.

    Worth pinning because it is a consequence of the edit that §0.5 does not state: the
    ``"mean"`` form does not merely join NND's family, it *is* NND$_1$. The thesis's ``"max"``
    form is the one that carries information NND$_k$ does not.
    """
    for X in (grid16, karate_random_layout):
        assert node_resolution_vertexwise(X, "mean") == normalized_nn_distance(X, 1)


# --------------------------------------------------------------------------- #
#                    Distance distributions (eqs. 70, 71)                      #
# --------------------------------------------------------------------------- #


def test_the_pairwise_distribution_is_one_value_per_pair(grid16):
    n = grid16.shape[0]
    assert pairwise_distance_distribution(grid16).shape == (n * (n - 1) // 2,)


def test_the_nn_distribution_is_one_value_per_vertex(grid16):
    assert nn_distance_distribution(grid16).shape == (grid16.shape[0],)


def test_the_pairwise_distribution_is_the_condensed_vector_over_its_mean(grid16):
    """The ordering is `CONDENSED_ORDERING`, so it lines up with an ideal-distance vector."""
    from twin.metrics.common import pairwise_distances

    distances = pairwise_distances(grid16)
    assert pairwise_distance_distribution(grid16) == pytest.approx(distances / distances.mean())


def test_the_pairwise_distribution_has_mean_one(grid16, karate_random_layout):
    """It is normalised by its own mean, which is what makes two layouts comparable."""
    for X in (grid16, karate_random_layout):
        assert pairwise_distance_distribution(X).mean() == pytest.approx(1.0)


def test_the_nn_distributions_mean_is_vnr(grid16, karate_random_layout):
    """Eq. (71) averages to eq. (69) under the `"mean"` normaliser — one family, one
    normaliser."""
    for X in (grid16, karate_random_layout):
        assert nn_distance_distribution(X).mean() == pytest.approx(
            node_resolution_vertexwise(X, "mean")
        )


def test_the_distributions_minimum_is_nrs_numerator_over_the_mean(grid16_plus_overlap):
    """The lower tail is what the scalar crowding metrics are read against."""
    from twin.metrics.common import pairwise_distances

    distribution = pairwise_distance_distribution(grid16_plus_overlap)
    distances = pairwise_distances(grid16_plus_overlap)
    assert distribution.min() == pytest.approx(distances.min() / distances.mean())
    assert nn_distance_distribution(grid16_plus_overlap).min() == pytest.approx(
        distribution.min()
    )


@pytest.mark.parametrize(
    "distribution", [pairwise_distance_distribution, nn_distance_distribution]
)
def test_the_distributions_are_invariant_under_similarity(grid16, distribution):
    assert distribution(similarity(grid16)) == pytest.approx(distribution(grid16))


@pytest.mark.parametrize(
    "distribution", [pairwise_distance_distribution, nn_distance_distribution]
)
def test_the_distributions_are_nan_on_a_collapsed_layout(distribution):
    assert np.isnan(distribution(np.zeros((5, 2)))).all()


def test_the_pairwise_distribution_does_not_hand_back_the_cache(grid16):
    """It divides by the mean, so a view would let a consumer scale the cached vector."""
    from twin.metrics.common import clear_pairwise_cache, pairwise_distances

    clear_pairwise_cache()
    cached = pairwise_distances(grid16)
    before = cached.copy()
    pairwise_distance_distribution(grid16)[:] = 0.0
    assert np.array_equal(pairwise_distances(grid16), before)


def test_the_distributions_are_returned_whole(karate_random_layout):
    """§3.1.3: diagnostics, not scores. Summarising here would pre-empt step 11's choice."""
    assert pairwise_distance_distribution(karate_random_layout).ndim == 1
    assert nn_distance_distribution(karate_random_layout).ndim == 1
