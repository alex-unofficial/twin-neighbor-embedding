r"""Spec §2.3 eqs. (49)-(52) — the layout-side silhouette, against an independent oracle.

The implementation is ``sklearn``-backed, so the whole validation is the oracle below: a
transcription of thesis §3.1.2's $a_i$ and $b_i$ in pure Python that **shares no code with
what it checks**. Session 4's audit found `twin_adjacency` untested through three sessions
because every test compared two callers of one function; an sklearn-versus-sklearn check here
would repeat that exactly.
"""

import math

import numpy as np
import pytest

from twin.datasets import load_dataset
from twin.metrics import silhouette, silhouette_report


# --------------------------------------------------------------------------- #
#                            The independent oracle                            #
# --------------------------------------------------------------------------- #


def euclidean_matrix(X) -> list[list[float]]:
    """``math.dist`` in a double loop. Not ``pdist``, not ``cdist``, not the package's cache."""
    return [[math.dist(xi, xj) for xj in X] for xi in X]


def rousseeuw_silhouette(D, labels) -> float:
    r"""Eqs. (49)-(52) transcribed from the definitions, over a precomputed distance matrix.

    One oracle for both sides of the metric: ``D`` is Euclidean for Sil and the graph closure
    for Sil\*, which is the only difference between the two equations.

    Note $b_i$'s normaliser is $1/|C|$ and $a_i$'s is $1/(|C_i| - 1)$ — vertex $i$ belongs to
    $C_i$ and not to $C$. Getting that wrong is the classic silhouette bug and it is invisible on
    balanced classes, so it is written out here rather than shared with the implementation.
    """
    labels = list(labels)
    n = len(labels)
    classes = sorted(set(labels))
    per_vertex = []
    for i in range(n):
        own = [j for j in range(n) if labels[j] == labels[i] and j != i]
        if not own:  # Rousseeuw's singleton convention
            per_vertex.append(0.0)
            continue
        a = sum(D[i][j] for j in own) / len(own)
        b = min(
            sum(D[i][j] for j in range(n) if labels[j] == c)
            / sum(1 for j in range(n) if labels[j] == c)
            for c in classes
            if c != labels[i]
        )
        per_vertex.append((b - a) / max(a, b))
    return sum(per_vertex) / n


def test_the_oracle_shares_no_import_with_the_implementation():
    """The point of the oracle, asserted so a later tidy-up cannot quietly undo it."""
    source = open(__file__).read()
    body = source.split("def rousseeuw_silhouette")[1].split("def test_")[0]
    for forbidden in ("silhouette_samples", "sklearn", "pdist", "cdist", "np."):
        assert forbidden not in body, f"the oracle reaches for {forbidden}"


# --------------------------------------------------------------------------- #
#                            Agreement with the oracle                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(("n", "d", "n_classes"), [(40, 2, 3), (25, 5, 2), (60, 3, 7)])
def test_agrees_with_the_oracle_on_random_layouts(seed, n, d, n_classes):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    labels = rng.integers(0, n_classes, size=n)
    if np.unique(labels).size < 2:  # pragma: no cover - guard on the generated fixture
        pytest.skip("degenerate draw")

    expected = rousseeuw_silhouette(euclidean_matrix(X), labels)
    assert silhouette(X, labels) == pytest.approx(expected, rel=1e-12)


def test_agrees_with_the_oracle_on_a_structured_layout(
    karate_structured_layout, karate_labels
):
    """A layout that partially separates the labels, so the score is away from 0 and from 1."""
    expected = rousseeuw_silhouette(euclidean_matrix(karate_structured_layout), karate_labels)
    got = silhouette(karate_structured_layout, karate_labels)
    assert got == pytest.approx(expected, rel=1e-12)
    assert 0.05 < got < 0.95


def test_agrees_with_the_oracle_on_unbalanced_classes():
    r"""$b_i$'s $1/|C|$ versus $a_i$'s $1/(|C_i|-1)$ is invisible on balanced classes.

    16 vertices in one class against 2 in each of two others: if the implementation used
    $1/(|C|-1)$ for $b_i$ the two would disagree, and on equal-sized classes they would not.
    """
    rng = np.random.default_rng(7)
    X = rng.normal(size=(20, 2))
    labels = np.array([0] * 16 + [1, 1] + [2, 2])
    expected = rousseeuw_silhouette(euclidean_matrix(X), labels)
    assert silhouette(X, labels) == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------- #
#                      Singletons and the two degenerate ends                  #
# --------------------------------------------------------------------------- #


def test_a_singleton_class_contributes_exactly_zero():
    """Rousseeuw's convention (thesis §3.1.2), on a constructed fixture.

    Spec §6's golden dataset cannot exercise this: `karate_weighted`'s only label set has two
    classes of 17, and no dataset in the roster has a singleton class at all.
    """
    X = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0], [1.1, 1.0], [9.0, 9.0]])
    labels = np.array([0, 0, 1, 1, 2])  # class 2 is a singleton

    report = silhouette_report(X, labels)
    assert report.n_singletons == 1
    assert report.min_class_size == 1

    # The mean is over all five vertices, with the singleton entering as a 0 — so dropping it
    # from the denominator would raise the score. Both readings are checked against the oracle,
    # which takes the same convention from the definition.
    assert report.value == pytest.approx(
        rousseeuw_silhouette(euclidean_matrix(X), labels), rel=1e-12
    )
    without = silhouette(X[:4], labels[:4])
    assert report.value < without


def test_a_single_class_is_nan_not_an_exception(karate_structured_layout):
    """A one-class label vector is a property of the data, not a caller error.

    ``sklearn`` raises; the package's convention is ``nan`` for an undefined score (as
    ``edge_length_cv`` on an edgeless graph) and a raise for a caller mistake
    (``require_n_by_d``). ``n_classes`` in the report is what makes it visible.
    """
    labels = np.zeros(karate_structured_layout.shape[0], dtype=int)
    report = silhouette_report(karate_structured_layout, labels)
    assert np.isnan(report.value)
    assert report.n_classes == 1


def test_every_class_a_singleton_scores_exactly_zero():
    r"""$|\mathcal{L}| = n$ is the other end ``sklearn`` refuses, and it is not ``nan``.

    Every $s_i$ is 0 by the convention, so the mean is 0. One pair plus $n-2$ singletons goes
    through ``sklearn`` normally and scores near 0 too, so returning ``nan`` one step further
    along would be a discontinuity of ours rather than of the metric.
    """
    rng = np.random.default_rng(11)
    X = rng.normal(size=(6, 2))

    report = silhouette_report(X, np.arange(6))
    assert report.value == 0.0
    assert report.n_classes == report.n_singletons == 6

    # The neighbouring case, computed by sklearn rather than short-circuited, agrees with the
    # oracle and is small — which is what makes 0.0 the continuous answer.
    almost = np.array([0, 0, 1, 2, 3, 4])
    assert silhouette(X, almost) == pytest.approx(
        rousseeuw_silhouette(euclidean_matrix(X), almost), rel=1e-12
    )


# --------------------------------------------------------------------------- #
#                            Bounds and invariances                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(5))
def test_bounded_in_minus_one_to_one(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(30, 2))
    labels = rng.integers(0, 3, size=30)
    assert -1.0 <= silhouette(X, labels) <= 1.0


def test_a_layout_anti_correlated_with_the_labels_scores_negative():
    """Interleaving two classes along a line puts every vertex nearer the other class."""
    X = np.arange(12, dtype=float)[:, None]
    labels = np.tile([0, 1], 6)
    assert silhouette(X, labels) < 0


def test_invariant_under_translation_rotation_and_uniform_scaling(
    karate_structured_layout, karate_labels
):
    r"""$s_i$ is a ratio of distances, so a similarity transform cannot move it.

    Scaling is the one that matters for the results tables: layout methods do not agree on a
    unit, and a metric that moved with the unit would rank backends by their units (the argument
    ``scale_normalized_stress`` is built on).

    The tolerance is measured rather than guessed, and on `digits` and `tabula` rather than here
    (CLAUDE.md): at these offsets the worst relative deviation is $7\times10^{-14}$ on this
    fixture and $3\times10^{-12}$ at tabula's $n = 5007$. See the next test for why it is not
    $10^{-12}$.
    """
    X = karate_structured_layout
    base = silhouette(X, karate_labels)

    theta = 0.7
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    for transformed in (X + 17.0, X @ rotation.T, X * 1e3, X * 1e-3):
        assert silhouette(transformed, karate_labels) == pytest.approx(base, rel=1e-11)


def test_translation_invariance_is_only_exact_to_about_eight_digits():
    r"""⚠️ ``sklearn``'s Euclidean path is the $\lVert x\rVert^2+\lVert y\rVert^2-2x\cdot y$
    expansion, which loses digits when the layout sits far from the origin relative to its own
    extent. Measured on real configurations at an offset/extent ratio of 4000:

    ==========================  =================  =====================
    layout                      ``sklearn`` route  exact-distance route
    ==========================  =================  =====================
    `digits`, PCA 2-D, n=1797   4.4e-9             1.0e-13
    `tabula`, ``X_umap``, n=5007 1.8e-8            1.2e-14
    ==========================  =================  =====================

    So the metric is invariant and the *implementation* is invariant to about eight digits.
    ``metric="precomputed"`` from ``pdist`` would remove it, and is **deliberately not used**:
    ``silhouette_samples`` chunks its Euclidean reduction and never materialises $(n, n)$, where
    a precomputed matrix is 200 MB at tabula's size — paid per label set, of which tabula has
    four. Real layouts sit within a few multiples of their own extent of the origin, where the
    deviation is $10^{-14}$ or below; eight digits is far below any effect the results report.

    The second assertion is what identifies the cause: handed exact distances, the same
    ``sklearn`` code is three orders tighter, so the loss is in the distance expansion and not
    in the silhouette.
    """
    from scipy.spatial.distance import pdist, squareform
    from sklearn.metrics import silhouette_samples

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2)) * 1e-3
    labels = rng.integers(0, 4, size=200)
    far = X - 4.0

    base = silhouette(X, labels)
    assert silhouette(far, labels) == pytest.approx(base, rel=1e-7)
    assert silhouette(far, labels) != pytest.approx(base, rel=1e-12)

    def exact(Y):
        return float(
            silhouette_samples(squareform(pdist(Y)), labels, metric="precomputed").mean()
        )

    assert exact(far) == pytest.approx(exact(X), rel=1e-12)


def test_coincident_points_inflate_the_score(karate_coincident_layout, karate_labels):
    r"""⚠️ Recorded, not repaired — spec §0.5 decision 1 predicts exactly this.

    Nine vertices on one point give those of them that share a label $a_i = 0$ and hence
    $s_i = 1$: the strongest possible score for the least informative layout. No registered
    dataset has a coincident vertex pair, but crowding at large λ can produce one, and this is
    what it would do to the number. There is no defensible repair inside the metric — an epsilon
    moves the threshold rather than removing it — so the behaviour is pinned instead.
    """
    from twin.metrics.label import _silhouette_report

    per_vertex = _silhouette_report(karate_coincident_layout, karate_labels, metric="euclidean")
    assert np.isfinite(per_vertex.value)

    collapsed = karate_coincident_layout[:9]
    same_label = karate_labels[:9] == karate_labels[0]
    assert same_label.sum() >= 2, "the fixture must collapse at least two same-label vertices"
    assert np.allclose(collapsed[same_label], collapsed[0])

    # The oracle agrees, so this is the metric's definition and not an sklearn artifact.
    assert per_vertex.value == pytest.approx(
        rousseeuw_silhouette(
            euclidean_matrix(karate_coincident_layout), karate_labels
        ),
        rel=1e-12,
    )


# --------------------------------------------------------------------------- #
#                       The class-size diagnostics (§0.5 dec. 9)              #
# --------------------------------------------------------------------------- #


def test_the_report_carries_the_class_size_diagnostics():
    X = np.random.default_rng(3).normal(size=(10, 2))
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3])

    report = silhouette_report(X, labels)
    assert report.n_classes == 4
    assert report.min_class_size == 1
    assert report.median_class_size == 2.5
    assert report.n_singletons == 1
    assert report.n_excluded == 0


def test_the_footballs_composite_has_no_class_size_collapse():
    """⚠️ §0.5 decision 9's premise, checked rather than assumed.

    It cites "14 conferences × 3 divisions over 132 teams" producing many tiny classes. The
    roster says otherwise: 11 conferences, a 15-class composite with a minimum size of 4 and no
    singletons. That is why the guard is these diagnostics and not a minimum-size filter.
    """
    dataset = load_dataset("football_2023")
    X = np.random.default_rng(5).normal(size=(dataset.n, 2))

    conference = silhouette_report(X, dataset.labels("conference"))
    assert (conference.n_classes, conference.min_class_size, conference.n_singletons) == (
        11,
        4,
        0,
    )

    composite = silhouette_report(X, dataset.labels("conference_division"))
    assert (composite.n_classes, composite.min_class_size, composite.n_singletons) == (15, 4, 0)


def test_tabulas_composite_is_redundant():
    """`compartment` is a function of `cell_ontology_class`, so the composite adds nothing.

    Scoring it would produce a duplicate column, which is worth knowing at step 7 when the
    label-set enumeration is written. Asserted on the labels rather than on a score, so it does
    not need a layout.
    """
    dataset = load_dataset("tabula")
    primary = dataset.labels("cell_ontology_class")
    composite = dataset.labels("compartment_cell_type")

    assert np.unique(primary).size == np.unique(composite).size == 13
    # Same partition, possibly renumbered: each composite class lies inside one primary class.
    for code in np.unique(composite):
        assert np.unique(primary[composite == code]).size == 1


# --------------------------------------------------------------------------- #
#                                   Refusals                                   #
# --------------------------------------------------------------------------- #


def test_refuses_a_transposed_layout(karate_structured_layout, karate_labels):
    with pytest.raises(ValueError, match="would fit"):
        silhouette(karate_structured_layout.T, karate_labels)


def test_refuses_a_two_dimensional_label_vector(karate_structured_layout):
    """`silhouette_graph` takes labels first, so the two argument orders must not blur."""
    with pytest.raises(ValueError, match="1-D"):
        silhouette(karate_structured_layout, karate_structured_layout)
