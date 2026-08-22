"""Spec §1.1 / §2.3 — the label-accuracy reimplementation."""

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from twin.metrics import knn_accuracy, knn_accuracy_report, majority_class_rate
from tests.test_characterisation import LABEL_PINS


@pytest.fixture
def layouts(karate_random_layout, karate_structured_layout):
    return {"random": karate_random_layout, "structured": karate_structured_layout}


# ------------------------------------------------ equivalence with the old path


@pytest.mark.parametrize(("which", "k"), sorted(LABEL_PINS))
def test_reproduces_the_sklearn_backed_values(layouts, karate_labels, which, k):
    """Under (distance, first) this must reproduce what `predict(None)` produced.

    That is the whole point of pinning them: `predict(None)` was valid sklearn and already
    leave-one-out, so a reimplementation that disagrees here has changed the metric, not fixed
    it (spec §0.5).
    """
    got = knn_accuracy(
        layouts[which], karate_labels, k, weighting="distance", tie_rule="first"
    )
    assert got == pytest.approx(LABEL_PINS[(which, k)], rel=1e-12)


@pytest.mark.parametrize("weighting", ["uniform", "distance"])
def test_agrees_with_sklearn_on_a_random_problem(weighting):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 3))
    labels = rng.integers(0, 4, size=120)

    clf = KNeighborsClassifier(n_neighbors=7, weights=weighting).fit(X, labels)
    expected = float(np.mean(clf.predict(None) == labels))

    got = knn_accuracy(X, labels, 7, weighting=weighting, tie_rule="first")
    assert got == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------- leave-one-out


def test_is_leave_one_out_not_self_prediction():
    """Fitting and predicting the same points makes each point its own neighbour.

    A layout where every point's nearest *other* neighbour has the opposite label must score 0,
    not 1. This is what would silently pass if self were included.
    """
    X = np.array([[0.0], [0.1], [1.0], [1.1]])
    labels = np.array([0, 1, 0, 1])
    assert knn_accuracy(X, labels, 1, tie_rule="first") == pytest.approx(0.0)


def test_self_removed_by_identity_under_exact_duplicates():
    """With coincident points the self entry need not sort first; it must still go."""
    X = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [5.0, 5.0]])
    labels = np.array([0, 0, 0, 1])
    # Each of the three coincident points sees the other two (label 0) plus the far point.
    assert knn_accuracy(X, labels, 2, tie_rule="first") == pytest.approx(0.75)


# ------------------------------------------------------------------- tie rules


def test_fractional_tie_rule_is_the_expectation_over_tie_breakings():
    """Three collinear points, k = 2, so every vertex sees both others.

    labels = [0, 0, 1]:
      v0 (true 0): neighbours labelled {0, 1} -> tie, p = 2, true label tied  -> 1/2
      v1 (true 0): neighbours labelled {0, 1} -> tie, p = 2, true label tied  -> 1/2
      v2 (true 1): neighbours labelled {0, 0} -> no tie, predicts 0           -> 0
    """
    X = np.array([[-1.0], [0.0], [1.0]])
    labels = np.array([0, 0, 1])

    assert knn_accuracy(X, labels, 2, tie_rule="fractional") == pytest.approx(1 / 3)
    # "first" resolves both ties toward label 0, which happens to be correct for v0 and v1.
    assert knn_accuracy(X, labels, 2, tie_rule="first") == pytest.approx(2 / 3)

    report = knn_accuracy_report(X, labels, 2, tie_rule="fractional")
    assert report.n_ties[0] == 2


def test_first_tie_rule_breaks_toward_the_lowest_label():
    """Same geometry, labels = [0, 1, 1], so the ties now resolve *away* from the truth.

      v0 (true 0): neighbours {1, 1} -> no tie, predicts 1  -> wrong
      v1 (true 1): neighbours {0, 1} -> tie, "first" -> 0   -> wrong
      v2 (true 1): neighbours {0, 1} -> tie, "first" -> 0   -> wrong
    """
    X = np.array([[-1.0], [0.0], [1.0]])
    labels = np.array([0, 1, 1])

    assert knn_accuracy(X, labels, 2, tie_rule="first") == pytest.approx(0.0)
    # Fractional credits the two tied vertices 1/2 each rather than zero.
    assert knn_accuracy(X, labels, 2, tie_rule="fractional") == pytest.approx(1 / 3)
    assert knn_accuracy_report(X, labels, 2).n_ties[0] == 2


# ---------------------------------------------------------- degenerate inputs


def test_distance_weighting_does_not_divide_by_zero():
    """Coincident points are what crowding produces; they must not yield inf or nan."""
    X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    labels = np.array([0, 0, 1, 1])
    report = knn_accuracy_report(X, labels, 1, weighting="distance")
    assert np.isfinite(report.accuracy).all()
    assert report.accuracy[0] == pytest.approx(1.0)
    assert report.n_zero_distance[0] == 4, "every vertex has a coincident nearest neighbour"


def test_zero_distance_count_is_zero_when_points_are_distinct(
    karate_random_layout, karate_labels
):
    report = knn_accuracy_report(karate_random_layout, karate_labels, 5, weighting="distance")
    assert report.n_zero_distance[0] == 0


# --------------------------------------------------------------- chance floor


def test_majority_class_rate_is_reported():
    labels = np.array([0] * 90 + [1] * 10)
    assert majority_class_rate(labels) == pytest.approx(0.9)

    X = np.random.default_rng(0).normal(size=(100, 2))
    report = knn_accuracy_report(X, labels, 5)
    assert report.majority_class_rate == pytest.approx(0.9)


def test_karate_chance_floor(karate_labels, karate_structured_layout):
    report = knn_accuracy_report(karate_structured_layout, karate_labels, 5)
    assert report.majority_class_rate == pytest.approx(17 / 34)
    assert report.accuracy[0] > report.majority_class_rate


# ------------------------------------------------------------------ sweeping k


def test_swept_k_matches_scalar_calls(karate_structured_layout, karate_labels):
    ks = [3, 5, 10]
    swept = knn_accuracy(karate_structured_layout, karate_labels, ks)
    scalar = [knn_accuracy(karate_structured_layout, karate_labels, k) for k in ks]
    assert swept == pytest.approx(scalar, rel=1e-12)


def test_scalar_k_returns_a_float(karate_structured_layout, karate_labels):
    assert isinstance(knn_accuracy(karate_structured_layout, karate_labels, 5), float)
    assert knn_accuracy(karate_structured_layout, karate_labels, [5]).shape == (1,)


# ------------------------------------------------------------------ validation


def test_rejects_k_larger_than_available_neighbours(karate_structured_layout, karate_labels):
    with pytest.raises(ValueError, match="not that many neighbours"):
        knn_accuracy(karate_structured_layout, karate_labels, 34)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"weighting": "gaussian"}, "weighting"),
        ({"tie_rule": "random"}, "tie_rule"),
        ({"k": 0}, "k must be"),
    ],
)
def test_rejects_bad_arguments(karate_structured_layout, karate_labels, kwargs, match):
    with pytest.raises(ValueError, match=match):
        knn_accuracy(karate_structured_layout, karate_labels, **{"k": 5, **kwargs})
