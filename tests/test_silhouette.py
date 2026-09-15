r"""Spec §2.3 — the silhouette family, against an independent oracle.

Eqs. (49)-(52) are the layout side and (53)-(55) the graph side, and the two differ only in
which $d$ they read — so **one** oracle checks both: a transcription of thesis §3.1.2's $a_i$
and $b_i$ in pure Python that **shares no code with what it checks**. Session 4's audit found
`twin_adjacency` untested through three sessions because every test compared two callers of one
function; an sklearn-versus-sklearn check here would repeat that exactly.

Spec §4.4's disconnected policy is load-bearing for Sil\* and is exercised at the bottom: the
`mask` refusal on all three paths a policy can arrive by, and the `lcc` subset that
`ideal_distances` deliberately does not perform for its caller.
"""

import math

import networkx as nx
import numpy as np
import pytest

from twin.datasets import load_dataset
from twin.metrics import (
    ideal_distances,
    silhouette,
    silhouette_graph,
    silhouette_graph_report,
    silhouette_report,
)
from twin.metrics.context import EvaluationContext


# --------------------------------------------------------------------------- #
#                            The independent oracle                            #
# --------------------------------------------------------------------------- #


def vertex_graph_distance(dataset):
    """The least-derived distance form — what `ideal_distances` is meant to be called on."""
    from twin.layoutgraph import vertex_graph

    return vertex_graph(dataset).distance


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
    (CLAUDE.md): over ``pdist`` distances the worst relative deviation at these offsets is
    $2\times10^{-15}$ on this fixture, $8\times10^{-16}$ on `digits` and $1\times10^{-16}$ at
    tabula's $n = 5007$. On ``sklearn``'s ``metric="euclidean"`` the same three are
    $7\times10^{-14}$, $2\times10^{-14}$ and $3\times10^{-12}$ — a decade short of this
    tolerance at the largest $n$, which is the size the guess would have been wrong at. See the
    next test.
    """
    X = karate_structured_layout
    base = silhouette(X, karate_labels)

    theta = 0.7
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    for transformed in (X + 17.0, X @ rotation.T, X * 1e3, X * 1e-3):
        assert silhouette(transformed, karate_labels) == pytest.approx(base, rel=1e-12)


def test_translation_invariance_survives_an_adversarial_offset():
    r"""⚠️ It would not, on ``sklearn``'s ``metric="euclidean"`` route.

    That route computes distances by the $\lVert x\rVert^2+\lVert y\rVert^2-2x\cdot y$
    expansion, which loses digits when the layout sits far from the origin relative to its own
    extent. At an offset/extent ratio of 4000 on real configurations:

    ============================  ======================  ======================
    layout                        ``metric="euclidean"``  ``pdist`` + squareform
    ============================  ======================  ======================
    `digits`, PCA 2-D, n=1797     4.4e-9                  1.0e-13
    `tabula`, ``X_umap``, n=5007  1.8e-8                  1.2e-14
    ============================  ======================  ======================

    Eight digits is far below anything the results report, so accuracy alone did not decide
    it — the exact route turned out to be **faster and leaner too**, against the obvious guess
    that an $(n, n)$ matrix must cost more than a chunked reduction (0.60 s / +273 MB against
    0.91 s / +373 MB at tabula's $n = 5007$; 4.64 s / +1634 MB against 4.71 s / +2039 MB at
    $n = 12000$). ``working_memory`` defaults to 1024 MB, so at these sizes the chunked
    reduction takes the matrix in one or two blocks and allocates more than the matrix itself.

    Both halves are asserted: the implementation is tight, and the route it does not take
    would not have been.
    """
    from sklearn.metrics import silhouette_samples

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2)) * 1e-3
    labels = rng.integers(0, 4, size=200)
    far = X - 4.0

    assert silhouette(far, labels) == pytest.approx(silhouette(X, labels), rel=1e-12)

    def sklearn_euclidean(Y):
        return float(silhouette_samples(Y, labels, metric="euclidean").mean())

    assert sklearn_euclidean(far) == pytest.approx(sklearn_euclidean(X), rel=1e-7)
    assert sklearn_euclidean(far) != pytest.approx(sklearn_euclidean(X), rel=1e-12)


def test_the_layout_side_goes_through_the_shared_pairwise_cache():
    """Why the exact route costs nothing extra during a metric sweep.

    ADR, NND and the two distance distributions all read ``common.pairwise_distances``, so Sil
    reading it too makes its marginal cost the ``squareform`` copy alone.
    """
    from twin.metrics.common import _pairwise_cache, clear_pairwise_cache

    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    labels = rng.integers(0, 3, size=50)

    clear_pairwise_cache()
    assert not _pairwise_cache
    silhouette(X, labels)
    assert _pairwise_cache, "Sil did not reach the shared pdist cache"


def test_coincident_points_inflate_the_score(karate_coincident_layout, karate_labels):
    r"""⚠️ Recorded, not repaired — spec §0.5 decision 1 predicts exactly this.

    Nine vertices on one point give those of them that share a label $a_i = 0$ and hence
    $s_i = 1$: the strongest possible score for the least informative layout. No registered
    dataset has a coincident vertex pair, but crowding at large λ can produce one, and this is
    what it would do to the number. There is no defensible repair inside the metric — an epsilon
    moves the threshold rather than removing it — so the behaviour is pinned instead.
    """
    per_vertex = silhouette_report(karate_coincident_layout, karate_labels)
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


def test_tabulas_composite_is_degenerate_by_the_partition_lattice():
    """⚠️ The redundancy is structural, and the earlier version of this test proved nothing.

    It walked the label vectors and asserted each composite class lies inside one primary class
    — which is true of *any* product partition and its parts, so it would have passed against
    any dataset. The property that makes a composite redundant is that the product **equals** a
    part, and since a product always refines its parts that is exactly a class-count identity:
    13 = 13 here.
    """
    dataset = load_dataset("tabula")

    assert dataset.degenerate_composites() == {
        "compartment_cell_type": "cell_ontology_class"
    }

    composite = dataset.labels("compartment_cell_type")
    primary = dataset.labels("cell_ontology_class")

    # ⚠️ The two arrays are **not** equal: `composite_labels` renumbers by sorted
    # `(compartment, cell_type)` tuple while `cell_ontology_class` is numbered by cell-type
    # name. It is the same partition under a different naming, which is precisely why every
    # metric agrees to the digit and why an `array_equal` check would have missed the
    # redundancy altogether.
    assert not np.array_equal(composite, primary)
    assert {frozenset(np.flatnonzero(composite == c)) for c in np.unique(composite)} == {
        frozenset(np.flatnonzero(primary == c)) for c in np.unique(primary)
    }

    ctx = EvaluationContext.for_dataset(dataset)
    assert silhouette_graph(composite, ideal=ctx) == pytest.approx(
        silhouette_graph(primary, ideal=ctx), rel=1e-12
    )


def test_the_declared_parent_cannot_be_the_criterion():
    """⚠️ The obvious guard — "a composite of a child and its own parent" — is wrong.

    Both datasets that set ``parent`` compose it with its child, and only one is degenerate:

    * `tabula`: `cell_ontology_class.parent = "compartment"`, and it **does** refine it, so the
      composite collapses to 13 classes;
    * `football_2023`: `division.parent = "conference"`, and it **does not** — West/East/none
      is applied across conferences — so the composite is a genuine 15-class product.

    ``parent`` is documented as "the label set this one subdivides" and has no consumer in the
    tree, so it carries no refinement guarantee. This test exists to keep the wrong criterion
    from being reintroduced as a simplification.
    """
    tabula = load_dataset("tabula")
    football = load_dataset("football_2023")

    # Identical declaration shape...
    assert tabula.label_set("cell_ontology_class").parent == "compartment"
    assert football.label_set("division").parent == "conference"
    assert "compartment" in tabula.composite_of["compartment_cell_type"]
    assert "conference" in football.composite_of["conference_division"]

    # ...opposite refinement, and opposite verdicts.
    def refines(dataset, child, parent):
        c, p = dataset.labels(child), dataset.labels(parent)
        return all(np.unique(p[c == k]).size == 1 for k in np.unique(c))

    assert refines(tabula, "cell_ontology_class", "compartment")
    assert not refines(football, "division", "conference")
    assert football.degenerate_composites() == {}
    assert np.unique(football.labels("conference_division")).size == 15


def test_only_tabula_has_a_degenerate_composite():
    """Across the roster, so a future dataset is caught here and not in a results table."""
    from twin.datasets import DATASET_REGISTRY

    found = {name: load_dataset(name).degenerate_composites() for name in DATASET_REGISTRY}
    assert {k: v for k, v in found.items() if v} == {
        "tabula": {"compartment_cell_type": "cell_ontology_class"}
    }


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


# --------------------------------------------------------------------------- #
#                  Sil* — the graph side (eqs. 53-55, spec §4.4)               #
# --------------------------------------------------------------------------- #


def test_sil_star_agrees_with_the_oracle_on_karate(karate_labels):
    """The same oracle, handed graph geodesics instead of Euclidean distances.

    That is the whole content of eqs. (53)-(55): $d_{ij}$ replaces
    $\\lVert x_i - x_j\\rVert$ and nothing else changes. Using one oracle for both is what makes
    the claim checkable.
    """
    ctx = EvaluationContext.for_dataset(load_dataset("karate_weighted"))
    expected = rousseeuw_silhouette(ctx.ideal.tolist(), karate_labels)
    assert silhouette_graph(karate_labels, ideal=ctx) == pytest.approx(expected, rel=1e-12)


def test_the_context_and_the_graph_paths_agree(karate_labels):
    """`ideal=ctx` must be an optimisation and not a second definition."""
    dataset = load_dataset("karate_weighted")
    ctx = EvaluationContext.for_dataset(dataset)
    assert silhouette_graph(karate_labels, vertex_graph_distance(dataset)) == pytest.approx(
        silhouette_graph(karate_labels, ideal=ctx), rel=1e-12
    )


def test_labels_matching_the_topology_score_far_above_a_shuffled_partition():
    r"""What makes Sil\* a label-validity check rather than a layout score.

    `ring_of_cliques` is ten 4-cliques in a ring and its `clique` labels *are* its topology, so
    the labels are as recoverable from the graph as labels get. Shuffling them destroys that
    while leaving every class size intact — so the comparison isolates label-topology agreement
    from the class-size effect the diagnostics track.

    ⚠️ **Two measured facts about the scale, both of which contradict the obvious reading:**

    * The aligned score is **0.532**, not near 1. Sil\* has a structural ceiling well below 1:
      $a^*_i = 1$ inside a clique, but the nearest other clique is 2-3 hops away, so
      $b^*_i = 1.75$ for a vertex holding a ring edge and $s^*_i = 0.43$. Graph distances are
      small integers, and $b^*$ is a *mean* over the nearest class; a Sil\* of 0.5 is therefore
      strong agreement, not mediocre.
    * A shuffled partition scores **−0.32**, not 0 (−0.346 to −0.296 over ten permutations).
      Chance is negative, not zero, whenever there are several classes: $b^*_i$ is a min over
      nine class means and so sits below the overall mean, while $a^*_i$ is at it. "Low means
      the labels are not in the topology" therefore needs a per-dataset chance level to be read
      against — the same role ``majority_class_rate`` plays for the accuracy metrics.
    """
    dataset = load_dataset("ring_of_cliques")
    ctx = EvaluationContext.for_dataset(dataset)
    labels = dataset.labels("clique")

    aligned = silhouette_graph(labels, ideal=ctx)
    shuffled = [
        silhouette_graph(np.random.default_rng(seed).permutation(labels), ideal=ctx)
        for seed in range(10)
    ]

    assert aligned == pytest.approx(0.5324675324675324, rel=1e-12)
    assert max(shuffled) < -0.25
    assert aligned > max(shuffled) + 0.8


# ---------------------------------------------------------- the `mask` refusal


def test_refuses_the_mask_policy_as_a_keyword(karate_labels, karate_dataset):
    with pytest.raises(ValueError, match="refuses disconnected='mask'"):
        silhouette_graph(karate_labels, vertex_graph_distance(karate_dataset), disconnected="mask")


def test_refuses_a_context_built_under_the_mask_policy(karate_labels, karate_dataset):
    """§4.4 permits `mask` for SNS only, so a context carrying it must still be refused here.

    A runner that sets `mask` once for SNS would otherwise get an inflated Sil\\* for free.
    """
    ctx = EvaluationContext.for_dataset(karate_dataset, disconnected="mask")
    with pytest.raises(ValueError, match="refuses disconnected='mask'"):
        silhouette_graph(karate_labels, ideal=ctx)


def test_refuses_the_mask_footprint_in_a_bare_array(disjoint_paths):
    """The backstop: a raw array where neither the kwarg nor the context is visible.

    Under `mask` the smaller component keeps its own internal distances, so an excluded vertex
    has more than one finite entry in its row — which is the footprint checked for. Without it
    the array would be scored as if it had come from `lcc`, silently changing the policy.
    """
    labels = np.array([0, 0, 1, 1, 0, 1, 1])
    masked = ideal_distances(disjoint_paths, disconnected="mask")
    with pytest.raises(ValueError, match="footprint of disconnected='mask'"):
        silhouette_graph(labels, ideal=masked)


def test_the_refusal_is_unconditional_not_conditional_on_disconnection(karate_dataset):
    """`karate_weighted` is connected, so `lcc` and `mask` would give the identical array.

    The guard still fires. A guard that only triggers on a disconnected graph would be inert on
    the whole current roster — dormant until the first dataset that needs it, which is the one
    case where it must not be.
    """
    assert karate_dataset.is_connected()
    lcc = ideal_distances(vertex_graph_distance(karate_dataset), disconnected="lcc")
    masked = ideal_distances(vertex_graph_distance(karate_dataset), disconnected="mask")
    assert np.array_equal(lcc, masked)

    with pytest.raises(ValueError, match="refuses disconnected='mask'"):
        silhouette_graph(karate_dataset.labels("club"), ideal=masked, disconnected="mask")



# ------------------------------------------------ the `lcc` subset and n_excluded


def test_excludes_vertices_outside_the_largest_component_and_reports_the_count(
    disjoint_paths,
):
    """⚠️ `ideal_distances` returns the full (n, n) with `inf` outside the component.

    It does not subset; the consumer does. And the alternative is not an inflated number:
    `silhouette_samples` rejects a precomputed matrix containing `inf` outright, so forgetting
    the subset fails with an sklearn message about input validation. `n_excluded` is what makes
    the discard reportable.
    """
    labels = np.array([0, 0, 1, 1, 0, 1, 1])  # 4-path is 0-3; triangle is 4-6
    report = silhouette_graph_report(labels, disjoint_paths)

    assert report.n_excluded == 3
    assert report.n_classes == 2
    assert report.min_class_size == 2  # the scored subset's classes, not the whole graph's

    # The score is exactly the 4-path's own, computed on a graph that has nothing else in it.
    path_only = silhouette_graph_report(labels[:4], nx.path_graph(4))
    assert report.value == pytest.approx(path_only.value, rel=1e-12)
    assert path_only.n_excluded == 0


def test_a_connected_graph_excludes_nothing(karate_labels, karate_dataset):
    report = silhouette_graph_report(karate_labels, ideal=EvaluationContext.for_dataset(
        karate_dataset
    ))
    assert report.n_excluded == 0


def test_sklearn_would_have_refused_the_unsubsetted_matrix(disjoint_paths):
    """Why the subset is mandatory rather than tidy — asserted, so the reason cannot rot."""
    from sklearn.metrics import silhouette_samples

    labels = np.array([0, 0, 1, 1, 0, 1, 1])
    D = ideal_distances(disjoint_paths)
    assert np.isinf(D).any()
    with pytest.raises(ValueError, match="infinity"):
        silhouette_samples(D, labels, metric="precomputed")


# ----------------------------------------------------------------- the contract


def test_refuses_a_condensed_ideal(karate_labels, karate_dataset):
    """Sil* needs the square form; the stresses take the condensed one. Mixing them is silent."""
    ctx = EvaluationContext.for_dataset(karate_dataset)
    with pytest.raises(ValueError, match="square"):
        silhouette_graph(karate_labels, ideal=ctx.ideal_condensed)


def test_refuses_ideal_together_with_ideal_distance_kwargs(karate_labels, karate_dataset):
    ctx = EvaluationContext.for_dataset(karate_dataset)
    with pytest.raises(ValueError, match="would be ignored"):
        silhouette_graph(karate_labels, ideal=ctx, weight_transform="inverse")


def test_refuses_neither_graph_nor_ideal(karate_labels):
    with pytest.raises(ValueError, match="either a graph or precomputed"):
        silhouette_graph(karate_labels)


def test_refuses_a_size_mismatch(karate_labels, karate_dataset):
    ctx = EvaluationContext.for_dataset(karate_dataset)
    with pytest.raises(ValueError, match="entries"):
        silhouette_graph(karate_labels[:10], ideal=ctx)
    with pytest.raises(ValueError, match="entries"):
        silhouette_graph(karate_labels[:10], ideal=ctx.ideal)


def test_refuses_an_affinity_valued_graph(karate_labels, karate_dataset):
    """Owned by `ideal_distances`, exercised here so the path is known to reach it.

    §1.6's bug class: an affinity read as a length inverts the target, and on `digits` the
    stored `weight` attribute genuinely is a distance while `karate_weighted`'s is an affinity.
    """
    from twin.layoutgraph import vertex_graph

    affinity = vertex_graph(karate_dataset).affinity_graph()
    with pytest.raises(ValueError, match="stamped 'affinity'"):
        silhouette_graph(karate_labels, affinity)


def test_both_graph_side_docstrings_say_label_validity_check_not_a_baseline():
    """§5.3 guard 2 is a schema rule at step 11; the sentence has to exist before anyone reads
    a number. The exporter cannot be written from a docstring, but a docstring that never said
    it is how the rule gets forgotten."""
    from twin.metrics.label import silhouette_graph, silhouette_graph_report

    for function in (silhouette_graph, silhouette_graph_report):
        text = " ".join(function.__doc__.split())
        assert "not a baseline" in text.lower()
        assert "incomparable" in text.lower()
