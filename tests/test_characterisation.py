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
