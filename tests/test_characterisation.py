"""Characterisation tests — pinned values of the metric code, from before the §2 rewrite.

These exist to make "behaviour-preserving" checkable rather than asserted. They were written
ahead of spec §2.1's split of the neighbourhood metrics into four named functions and §2.3's
reimplementation of the label metric; **both landed at step 3, and reproduced these numbers**
except where the commit said otherwise (`DECISIONS.md`, step-3 departures: two coincident-point
values moved when the two metric families were put on one leave-one-out convention). The pins
stay because their job now is the same one going forwards — steps 5-11 must not move them
silently either.

A value here changing is not automatically a failure — but it must be a *deliberate* change,
declared in the commit message, with the pin updated in the same commit (CLAUDE.md, "Commit
granularity"). Silent movement is the thing this file exists to catch.

All layouts are ``(n, d)``; all fixtures are deterministic without running a layout algorithm.
"""

import numpy as np
import pytest

from twin.metrics import knn_jaccard, knn_precision, knn_recall, weighted_recall

# The four named functions of spec §2.1, keyed by the `mode=` string they replaced. The keys
# below became function names at step 3 commit 7; the values did not move, except where the
# commit message says they did.
FUNCTIONS = {
    "recall": knn_recall,
    "precision": knn_precision,
    "jaccard": knn_jaccard,
    "weighted recall": weighted_recall,
}

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
#
# Two of these moved at step 3 commit 7, which adopted `_loo_neighbors` for both families —
# declared there with the reason. The neighbour *sets* change for the nine coincident rows;
# recall and precision happen to match the same number of graph neighbours either way, while
# jaccard's union and weighted recall's affinities do not:
#
#     ("jaccard", 5)          0.08954566675154911 -> 0.09206667515491045
#     ("weighted recall", 5)  0.1460194654084813  -> 0.1482819088473953
COINCIDENT_PINS = {
    ("recall", 5): 0.13370098039215686,
    ("recall", 10): 0.26898308342945015,
    ("precision", 5): 0.1588235294117647,
    ("precision", 10): 0.15588235294117647,
    ("jaccard", 5): 0.09206667515491045,
    ("jaccard", 10): 0.11880437759694724,
    ("weighted recall", 5): 0.1482819088473953,
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
    got = FUNCTIONS[mode](layouts[which], karate, k)
    assert got == pytest.approx(NEIGHBORHOOD_PINS[which][(mode, k)], rel=1e-12)


def test_neighborhood_preservation_vectorised_matches_scalar(karate_random_layout, karate):
    """Passing a list of k must agree elementwise with the scalar calls.

    Spec §2.1 replaces the recursive re-fit with a single query at max(k) plus slicing; this is
    the invariant that change has to hold.
    """
    swept = knn_recall(karate_random_layout, karate, [5, 10])
    scalar = [knn_recall(karate_random_layout, karate, k) for k in (5, 10)]
    assert np.asarray(swept) == pytest.approx(np.asarray(scalar), rel=1e-12)


def test_there_is_no_mode_string_left_to_reject(karate_random_layout, karate):
    """The `mode=` string is gone; a wrong criterion is now a wrong *name*, caught at import."""
    with pytest.raises(TypeError):
        knn_recall(karate_random_layout, karate, 5, mode="nonsense")


@pytest.mark.parametrize("mode", ["recall", "precision", "jaccard", "weighted recall"])
@pytest.mark.parametrize("k", [2, 3])
def test_zero_degree_vertex_pinned(isolated_vertex_layout, isolated_vertex_graph, mode, k):
    got = FUNCTIONS[mode](isolated_vertex_layout, isolated_vertex_graph, k)
    assert got == pytest.approx(ISOLATED_VERTEX_PINS[(mode, k)], rel=1e-12)


def test_zero_degree_exclusion_is_not_uniform_across_modes(
    isolated_vertex_layout, isolated_vertex_graph
):
    """recall and weighted recall exclude the zero-degree vertex; precision and jaccard do not.

    The exclusion is driven by a zero *normalisation*. Precision normalises by ``k`` and
    jaccard by the union with the embedded neighbourhood, and neither is ever zero — so the
    isolated vertex is scored at 0 there rather than dropped, and those two average over 6
    vertices where recall averages over 5. Spec §2.1 keeps the NaN handling and asks for the
    excluded count to be *reported*; this pins what that count is per criterion.
    """
    n = isolated_vertex_graph.number_of_nodes()
    k = 2

    def scored_sum(mode, denominator):
        return FUNCTIONS[mode](isolated_vertex_layout, isolated_vertex_graph, k) * denominator

    # recall over 5 scored vertices, precision over all 6, both from the same matches.
    assert scored_sum("recall", n - 1) == pytest.approx(1.5, rel=1e-12)
    assert scored_sum("precision", n) == pytest.approx(3.0 / k, rel=1e-12)


@pytest.mark.parametrize("mode", ["recall", "precision", "jaccard", "weighted recall"])
@pytest.mark.parametrize("k", [5, 10])
def test_coincident_points_pinned(karate_coincident_layout, karate, mode, k):
    got = FUNCTIONS[mode](karate_coincident_layout, karate, k)
    assert got == pytest.approx(COINCIDENT_PINS[(mode, k)], rel=1e-12)


def test_a_swept_call_and_a_scalar_call_disagree_on_a_tied_layout(karate_coincident_layout, karate):
    """The one place the single-query-and-slice rewrite is *not* equivalent to re-querying.

    Which of nine coincident points count as "the five nearest" depends on how many neighbours
    were asked for, and there is nothing to break the tie with — so slicing a k=10 query at 5
    need not give the k=5 query's answer. This is a property of exact duplicates, not of the
    implementation: no LOO convention fixes it, because the arbitrariness is in the data.

    Pinned rather than asserted away, so that a change to the tie handling is visible. On a
    layout without duplicates the two agree exactly, which is the test above.
    """
    swept = knn_recall(karate_coincident_layout, karate, [5, 10])
    scalar = [knn_recall(karate_coincident_layout, karate, k) for k in (5, 10)]

    assert swept[1] == pytest.approx(scalar[1], rel=1e-12)
    assert swept[0] != pytest.approx(scalar[0], rel=1e-12)
    assert swept[0] == pytest.approx(0.0975898692810458, rel=1e-12)
