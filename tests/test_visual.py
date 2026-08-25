r"""Spec §2.2 — the visual-quality metrics: point crowding, distributions, edge lengths.

These read the layout and nothing else (except `edge_lengths`, which reads an edge set), so
almost every property here is a statement about geometry that can be hand-computed. Where a
number is pinned it is pinned against a closed form, not against a previous run — the golden
file (`test_metric_golden.py`) is what guards against a silent move on a realistic layout.
"""

import numpy as np
import pytest

from twin.metrics.common import clear_loo_cache
from twin.metrics.visual import normalized_nn_distance


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
