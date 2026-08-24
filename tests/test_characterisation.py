"""Characterisation tests — pinned values of the metric code *as it stands today*.

These exist to make "behaviour-preserving" checkable rather than asserted. The neighbourhood
metrics get split into four named functions in spec §2.1 and the label metric is reimplemented
in §2.3; both changes must reproduce these numbers.

A value here changing is not automatically a failure — but it must be a *deliberate* change,
declared in the commit message, with the pin updated in the same commit (CLAUDE.md, "Commit
granularity"). Silent movement is the thing this file exists to catch.

All layouts are ``(n, d)``; all fixtures are deterministic without running a layout algorithm.
"""

import numpy as np
import pytest

from twin.metrics import knn_neighbor_preservation_accuracy

# (mode, k) -> value, computed on this repository at the parent of this commit.
NEIGHBORHOOD_PINS = {
    "random": {
        ("recall", 5): 0.0955353710111496,
        ("recall", 10): 0.249567474048443,
        ("precision", 5): 0.0882352941176471,
        ("precision", 10): 0.117647058823529,
        ("jaccard", 5): 0.0472077866234213,
        ("jaccard", 10): 0.0829988436168608,
        ("weighted recall", 5): 0.0846939702676598,
        ("weighted recall", 10): 0.237212842877873,
    },
    "structured": {
        ("recall", 5): 0.131427816224529,
        ("recall", 10): 0.26596261053441,
        ("precision", 5): 0.135294117647059,
        ("precision", 10): 0.152941176470588,
        ("jaccard", 5): 0.0734593837535014,
        ("jaccard", 10): 0.110025794942021,
        ("weighted recall", 5): 0.146775859233823,
        ("weighted recall", 10): 0.276014975558586,
    },
}

# The two edges the karate pins above cannot see, pinned before the §2.1 split shares a core
# between the neighbourhood family and the label family.
#
# (mode, k) -> value on `isolated_vertex_graph`: a weighted path plus a zero-degree vertex.
# Every registered dataset is connected, so the NaN-exclusion branch is unreachable from the
# roster. Note that it is not uniform across modes — see the test below.
ISOLATED_VERTEX_PINS = {
    ("recall", 2): 0.3,
    ("recall", 3): 0.5,
    ("precision", 2): 0.25,
    ("precision", 3): 0.2222222222222222,
    ("jaccard", 2): 0.16666666666666666,
    ("jaccard", 3): 0.18055555555555555,
    ("weighted recall", 2): 0.30095238095238097,
    ("weighted recall", 3): 0.5009523809523809,
}

# (mode, k) -> value on `karate_coincident_layout`: nine coincident points, which is the only
# regime in which sklearn's leave-one-out convention (mask column 0) and `_loo_neighbors`' one
# (mask the last column) disagree. karate itself has no coincident points, so all sixteen pins
# above reproduce whichever convention the split adopts; these do not.
COINCIDENT_PINS = {
    ("recall", 5): 0.13370098039215686,
    ("recall", 10): 0.26898308342945015,
    ("precision", 5): 0.1588235294117647,
    ("precision", 10): 0.15588235294117647,
    ("jaccard", 5): 0.08954566675154911,
    ("jaccard", 10): 0.11880437759694724,
    ("weighted recall", 5): 0.1460194654084813,
    ("weighted recall", 10): 0.27718496005728566,
}

# (layout, k) -> value produced by the original sklearn-backed implementation, which used
# `KNeighborsClassifier(weights="distance").predict(None)`. `predict(None)` leaves each point
# out of its own neighbour set, and sklearn breaks ties toward the lowest class label — so these
# are the values the reimplementation must reproduce at
# `weighting="distance", tie_rule="first"`. See spec §0.5.
LABEL_PINS = {
    ("random", 5): 0.411764705882353,
    ("random", 10): 0.411764705882353,
    ("structured", 5): 0.852941176470588,
    ("structured", 10): 0.852941176470588,
}


@pytest.fixture
def layouts(karate_random_layout, karate_structured_layout):
    return {"random": karate_random_layout, "structured": karate_structured_layout}


@pytest.mark.parametrize("which", ["random", "structured"])
@pytest.mark.parametrize("mode", ["recall", "precision", "jaccard", "weighted recall"])
@pytest.mark.parametrize("k", [5, 10])
def test_neighborhood_preservation_pinned(layouts, karate, which, mode, k):
    got = knn_neighbor_preservation_accuracy(layouts[which], karate, k, mode=mode)
    assert got == pytest.approx(NEIGHBORHOOD_PINS[which][(mode, k)], rel=1e-12)


def test_neighborhood_preservation_vectorised_matches_scalar(karate_random_layout, karate):
    """Passing a list of k must agree elementwise with the scalar calls.

    Spec §2.1 replaces the recursive re-fit with a single query at max(k) plus slicing; this is
    the invariant that change has to hold.
    """
    swept = knn_neighbor_preservation_accuracy(karate_random_layout, karate, [5, 10])
    scalar = [
        knn_neighbor_preservation_accuracy(karate_random_layout, karate, k) for k in (5, 10)
    ]
    assert np.asarray(swept) == pytest.approx(np.asarray(scalar), rel=1e-12)


def test_neighborhood_preservation_rejects_unknown_mode(karate_random_layout, karate):
    with pytest.raises(ValueError):
        knn_neighbor_preservation_accuracy(karate_random_layout, karate, 5, mode="nonsense")


@pytest.mark.parametrize("mode", ["recall", "precision", "jaccard", "weighted recall"])
@pytest.mark.parametrize("k", [2, 3])
def test_zero_degree_vertex_pinned(isolated_vertex_layout, isolated_vertex_graph, mode, k):
    got = knn_neighbor_preservation_accuracy(
        isolated_vertex_layout, isolated_vertex_graph, k, mode=mode
    )
    assert got == pytest.approx(ISOLATED_VERTEX_PINS[(mode, k)], rel=1e-12)


def test_zero_degree_exclusion_is_not_uniform_across_modes(
    isolated_vertex_layout, isolated_vertex_graph
):
    """recall / jaccard / weighted recall exclude the zero-degree vertex; precision does not.

    The exclusion is driven by a zero *normalisation*, and precision's normalisation is ``k``,
    which is never zero — so the isolated vertex is scored at 0 rather than dropped, and
    precision averages over 6 vertices where the others average over 5. Spec §2.1 keeps the
    NaN handling and asks for the excluded count to be *reported*; this pins what that count
    is per mode before there is a field to put it in.
    """
    n = isolated_vertex_graph.number_of_nodes()
    k = 2

    def scored_sum(mode, denominator):
        value = knn_neighbor_preservation_accuracy(
            isolated_vertex_layout, isolated_vertex_graph, k, mode=mode
        )
        return value * denominator

    # recall over 5 scored vertices, precision over all 6, both from the same matches.
    assert scored_sum("recall", n - 1) == pytest.approx(1.5, rel=1e-12)
    assert scored_sum("precision", n) == pytest.approx(3.0 / k, rel=1e-12)


@pytest.mark.parametrize("mode", ["recall", "precision", "jaccard", "weighted recall"])
@pytest.mark.parametrize("k", [5, 10])
def test_coincident_points_pinned(karate_coincident_layout, karate, mode, k):
    got = knn_neighbor_preservation_accuracy(karate_coincident_layout, karate, k, mode=mode)
    assert got == pytest.approx(COINCIDENT_PINS[(mode, k)], rel=1e-12)


def test_coincident_vectorised_matches_scalar(karate_coincident_layout, karate):
    """The single-query-at-max(k)-and-slice rewrite (§2.1) must hold here too.

    On a layout with no duplicates, slicing a max(k) query and re-querying at k are trivially
    the same. With nine coincident points the tie ordering is what makes them agree or not, so
    this is the case that has something to say.
    """
    swept = knn_neighbor_preservation_accuracy(karate_coincident_layout, karate, [5, 10])
    scalar = [
        knn_neighbor_preservation_accuracy(karate_coincident_layout, karate, k) for k in (5, 10)
    ]
    assert np.asarray(swept) == pytest.approx(np.asarray(scalar), rel=1e-12)
