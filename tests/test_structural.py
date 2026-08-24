"""Spec §2.1 — ADR, the $k$-NN neighbourhood family, SNS.

The analytic cases build their graphs by hand and check against values computed in the test
file, per spec §6. That is deliberate: a metric checked only against its own past output pins
whatever it does, and a metric checked against a hand computation pins what it is *for*.
"""

import networkx as nx
import numpy as np
import pytest

from twin.datasets import load_dataset
from twin.forms import stamp_form
from twin.layoutgraph import vertex_graph
from twin.metrics.common import (
    condensed,
    default_k,
    ideal_distances,
    mean_pairwise_distance,
    pairwise_distances,
)
from twin.metrics.structural import (
    NeighborPreservationReport,
    adjacency_distance_ratio,
    knn_jaccard,
    knn_jaccard_report,
    knn_precision,
    knn_precision_report,
    knn_recall,
    knn_recall_report,
    normalized_stress,
    optimal_scale,
    raw_stress,
    scale_normalized_stress,
    weighted_recall,
)


# --------------------------------------------------------------------------- #
#                      ADR — hand-computed analytic cases                      #
# --------------------------------------------------------------------------- #


def test_adr_of_a_complete_graph_is_exactly_one(k4):
    """Both averages run over the same pairs, so the ratio is 1 for *any* layout.

    The strongest analytic case available: it holds for a square, for a random cloud, and for
    a degenerate line, so it pins the normalisation rather than one layout.
    """
    square = np.array([[0.0, 0], [1, 0], [1, 1], [0, 1]])
    random = np.random.default_rng(7).normal(size=(4, 3))
    for X in (square, random):
        assert adjacency_distance_ratio(X, k4) == pytest.approx(1.0)


def test_adr_of_a_collinear_path(path5):
    """Edges are the four unit steps; all ten pairs average 2. ADR = 1/2."""
    X = np.column_stack([np.arange(5.0), np.zeros(5)])
    assert adjacency_distance_ratio(X, path5) == pytest.approx(0.5)


def test_adr_of_a_star(star5):
    """Four unit spokes over the mean of ten pairs: 4 at 1, 4 at sqrt(2), 2 at 2."""
    X = np.array([[0.0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]])
    expected = 1.0 / ((4 * 1 + 4 * np.sqrt(2) + 2 * 2) / 10)
    assert adjacency_distance_ratio(X, star5) == pytest.approx(expected)
    assert adjacency_distance_ratio(X, star5) == pytest.approx(10 / (8 + 4 * np.sqrt(2)))


def test_adr_of_a_triangle_is_one_because_it_is_complete(triangle):
    X = np.array([[0.0, 0], [1, 0], [0.5, 3.0]])
    assert adjacency_distance_ratio(X, triangle) == pytest.approx(1.0)


def test_adr_falls_when_adjacent_vertices_are_drawn_closer(path5):
    """Low is good: the direction of the metric, pinned."""
    spread = np.column_stack([np.arange(5.0), np.zeros(5)])
    clustered = np.column_stack([np.array([0.0, 0.1, 0.2, 4.0, 4.1]), np.zeros(5)])
    assert adjacency_distance_ratio(clustered, path5) < adjacency_distance_ratio(spread, path5)


# --------------------------------------------------------------------------- #
#                          ADR — weights, and eq. (29)                         #
# --------------------------------------------------------------------------- #


def eq29(X: np.ndarray, G: nx.Graph) -> float:
    """Eq. (29) written out independently: mean edge length over mean pairwise distance."""
    lengths = [float(np.linalg.norm(X[u] - X[v])) for u, v in G.edges()]
    return float(np.mean(lengths) / mean_pairwise_distance(X))


def test_eq27_reduces_to_eq29_on_an_unweighted_graph(karate_structured_layout):
    """No unweighted branch exists in the source; this is the test that replaces one."""
    dataset = load_dataset("karate")
    X = karate_structured_layout
    assert adjacency_distance_ratio(X, dataset.graph) == pytest.approx(eq29(X, dataset.graph))


def test_the_weighted_case_is_not_the_unweighted_one(karate_structured_layout):
    """Otherwise the eq.-(29) agreement above would be vacuous."""
    dataset = load_dataset("karate_weighted")
    X = karate_structured_layout
    weighted = adjacency_distance_ratio(X, dataset.graph)
    assert weighted != pytest.approx(eq29(X, dataset.graph))


def test_the_source_has_no_unweighted_branch():
    """Eq. (29) is eq. (27) with w in {0, 1}; `nx.adjacency_matrix` already fills the 1s."""
    import inspect

    source = inspect.getsource(adjacency_distance_ratio)
    body = source[source.index('"""', source.index('"""') + 3) :]
    assert "number_of_edges" not in body or "weight is None" not in body
    assert body.count("nx.adjacency_matrix") == 1


def test_adr_reads_the_weights_it_is_given():
    """Doubling one edge's affinity moves the number in the direction its weight implies."""
    G = nx.path_graph(3)
    X = np.array([[0.0, 0], [1, 0], [4, 0]])
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    even = adjacency_distance_ratio(X, G)

    G[0][1]["weight"] = 9.0  # nearly all the mass on the short edge
    assert adjacency_distance_ratio(X, G) < even


# --------------------------------------------------------------------------- #
#                              ADR — invariance                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("transform", ["scale", "rotate", "reflect", "translate"])
def test_adr_is_invariant_under_similarity_transforms(karate, karate_structured_layout, transform):
    """Numerator and denominator are both lengths, so a uniform scaling cancels exactly."""
    X = karate_structured_layout
    theta = 0.7
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    moved = {
        "scale": 7.0 * X,
        "rotate": X @ rotation.T,
        "reflect": X * np.array([1.0, -1.0]),
        "translate": X + np.array([3.0, -2.0]),
    }[transform]

    assert adjacency_distance_ratio(moved, karate) == pytest.approx(
        adjacency_distance_ratio(X, karate), rel=1e-12
    )


# --------------------------------------------------------------------------- #
#                    ADR — what it refuses, and what it accepts                #
# --------------------------------------------------------------------------- #


def test_adr_refuses_the_digits_graph():
    """digits' `weight` is a stored Euclidean distance, 5.3-37.8 — bug §1.6 one layer up.

    Silently accepting it would make ADR reward putting graph-*distant* pairs close together,
    and unlike §1.6 there is no Spearman of -1 to make it visible afterwards.
    """
    dataset = load_dataset("digits")
    X = np.random.default_rng(8).normal(size=(dataset.n, 2))
    with pytest.raises(ValueError, match="affinity-valued") as excinfo:
        adjacency_distance_ratio(X, dataset.graph)
    assert "'distance'" in str(excinfo.value)


def test_adr_accepts_the_affinity_form_of_the_same_dataset():
    """The correct call: the affinity graph, where a large weight means *coupled*."""
    dataset = load_dataset("digits")
    X = np.random.default_rng(8).normal(size=(dataset.n, 2))
    affinity = vertex_graph(dataset).affinity_graph()
    assert np.isfinite(adjacency_distance_ratio(X, affinity))


def test_adr_refuses_a_layout_graph_distance_form(karate_dataset, karate_structured_layout):
    with pytest.raises(ValueError, match="affinity-valued"):
        adjacency_distance_ratio(
            karate_structured_layout, vertex_graph(karate_dataset).distance
        )


def test_adr_accepts_a_bare_unstamped_graph(k4):
    """§6's analytic cases build a K_4 by hand and must not need a Dataset."""
    assert adjacency_distance_ratio(np.eye(4)[:, :2], k4) == pytest.approx(1.0)


def test_adr_accepts_a_combinatorial_graph(karate_structured_layout):
    dataset = load_dataset("karate")
    assert np.isfinite(adjacency_distance_ratio(karate_structured_layout, dataset.graph))


def test_adr_checks_orientation(karate, karate_structured_layout):
    with pytest.raises(ValueError, match="n_points"):
        adjacency_distance_ratio(karate_structured_layout.T, karate)


def test_adr_of_an_edgeless_graph_is_nan():
    G = nx.empty_graph(4)
    assert np.isnan(adjacency_distance_ratio(np.random.default_rng(9).normal(size=(4, 2)), G))


def test_adr_of_a_collapsed_layout_is_nan(k4):
    """Every pairwise distance is zero, so the denominator is 0 — say so rather than divide."""
    assert np.isnan(adjacency_distance_ratio(np.zeros((4, 2)), k4))


def test_adr_uses_the_original_vertex_indexing():
    """`nx.adjacency_matrix` follows `G.nodes()` order; X rows are 0..n-1.

    Every current graph happens to be inserted in order, so `nodelist=range(n)` is invisible
    today and stays invisible until a dataset relabels. This is the test that sees it.
    """
    G = nx.Graph()
    G.add_nodes_from([2, 0, 1])  # insertion order is not vertex order
    G.add_edge(0, 1)
    X = np.array([[0.0, 0], [1, 0], [5, 0]])

    # Edge 0-1 has length 1; the three pairs average 10/3. Under `G.nodes()` order the edge
    # would land on rows 1 and 2 instead, giving length 4 and an ADR of 1.2.
    assert adjacency_distance_ratio(X, G) == pytest.approx(1.0 / (10 / 3))


def test_the_docstring_states_the_comparability_rule():
    """§5.3 enforces it at step 11; the docstring is what makes that read as a rule."""
    doc = adjacency_distance_ratio.__doc__
    assert "fixed graph under a fixed weighting" in doc
    assert "across nothing" in doc


# --------------------------------------------------------------------------- #
#                    The kNN neighbourhood family (eqs. 30-39)                 #
# --------------------------------------------------------------------------- #


def test_a_perfect_layout_recovers_everything(path5):
    """A 5-path drawn as a line: at k = 1 each interior vertex finds one of its two."""
    X = np.column_stack([np.arange(5.0), np.zeros(5)])
    assert knn_recall(X, path5, 2) == pytest.approx(1.0)
    assert knn_precision(X, path5, 2) == pytest.approx((1 + 1 + 1 + 1 + 1) / 5 * (2 / 2) - 0.2)


def test_recall_precision_and_jaccard_agree_where_they_must(k4):
    """On K_4 at k = 3 every vertex recovers all three neighbours, so all three metrics are 1."""
    X = np.random.default_rng(11).normal(size=(4, 2))
    assert knn_recall(X, k4, 3) == pytest.approx(1.0)
    assert knn_precision(X, k4, 3) == pytest.approx(1.0)
    assert knn_jaccard(X, k4, 3) == pytest.approx(1.0)


def test_a_scalar_k_gives_a_float_and_an_array_k_gives_an_array(karate, karate_random_layout):
    assert isinstance(knn_recall(karate_random_layout, karate, 5), float)
    assert knn_recall(karate_random_layout, karate, [5]).shape == (1,)


def test_a_tuple_of_k_is_swept_like_a_list(karate, karate_random_layout):
    """`np.atleast_1d`, not `isinstance(k, (list, np.ndarray))` — a tuple used to fall through
    the latter into the scalar path and then into an obscure sklearn error."""
    assert knn_recall(karate_random_layout, karate, (5, 10)) == pytest.approx(
        knn_recall(karate_random_layout, karate, [5, 10])
    )


def test_the_neighbours_are_fitted_once_for_a_swept_call(karate, karate_random_layout, monkeypatch):
    """Query at max(k) once and slice, rather than re-fitting per k."""
    import twin.metrics.common as common

    calls = []
    original = common._loo_neighbors

    def counting(X, k_max):
        calls.append(k_max)
        return original(X, k_max)

    monkeypatch.setattr(common, "_loo_neighbors", counting)
    monkeypatch.setattr("twin.metrics.structural._loo_neighbors", counting)

    knn_recall(karate_random_layout, karate, [5, 10])
    assert calls == [10]


@pytest.mark.parametrize(
    "metric", [knn_recall, knn_precision, knn_jaccard, weighted_recall], ids=lambda f: f.__name__
)
def test_every_criterion_refuses_a_transposed_layout(karate, karate_structured_layout, metric):
    with pytest.raises(ValueError, match=r"\(n_points, d\)"):
        metric(karate_structured_layout.T, karate, 5)


@pytest.mark.parametrize(
    "metric", [knn_recall, knn_precision, knn_jaccard, weighted_recall], ids=lambda f: f.__name__
)
def test_k_is_bounded_by_the_vertex_count(karate, karate_structured_layout, metric):
    with pytest.raises(ValueError, match="not that many neighbours"):
        metric(karate_structured_layout, karate, 34)
    with pytest.raises(ValueError, match="k must be >= 1"):
        metric(karate_structured_layout, karate, 0)


# --------------------------------------------------------------------------- #
#                        Eq. (39): weighted recall = recall                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["karate", "football_2023"])
def test_weighted_recall_equals_recall_on_an_unweighted_graph(name, karate_structured_layout):
    """Eq. (39). Cheap, and it says the weighting is wired up rather than merely present."""
    dataset = load_dataset(name)
    X = np.random.default_rng(12).normal(size=(dataset.n, 2))
    for k in (5, 10):
        assert weighted_recall(X, dataset.graph, k) == pytest.approx(
            knn_recall(X, dataset.graph, k), rel=1e-12
        )


def test_a_declared_unweighted_graph_beats_a_stray_weight_attribute():
    """football_2023 declares `Unweighted()` and *still* carries weights: 732 at 1, three at 2.

    The declaration is what the representation is built from, so it is what weighted recall
    must read. Reading the stray attribute instead would break eq. (39) on a graph the dataset
    says is unweighted — quietly, since the difference is three edges in 735.
    """
    graph = load_dataset("football_2023").graph
    stray = {w for _, _, w in graph.edges(data="weight", default=1.0)}
    assert stray == {1.0, 2.0}


def test_weighted_recall_differs_on_a_weighted_graph(karate_structured_layout):
    """The half that proves the half above is not vacuous."""
    dataset = load_dataset("karate_weighted")
    X = karate_structured_layout
    assert weighted_recall(X, dataset.graph, 5) != pytest.approx(
        knn_recall(X, dataset.graph, 5), rel=1e-9
    )


def test_weighted_recall_refuses_a_distance_valued_graph():
    """digits again: its `weight` attribute is a stored Euclidean distance."""
    dataset = load_dataset("digits")
    X = np.random.default_rng(13).normal(size=(dataset.n, 2))
    with pytest.raises(ValueError, match="stamped 'distance'"):
        weighted_recall(X, dataset.graph, 5)


def test_weighted_recall_accepts_the_affinity_form():
    dataset = load_dataset("digits")
    X = np.random.default_rng(13).normal(size=(dataset.n, 2))
    affinity = vertex_graph(dataset).affinity_graph()
    assert 0.0 <= weighted_recall(X, affinity, 5) <= 1.0


def test_the_unweighted_criteria_do_not_care_about_the_stamp():
    """They read the support, not the valuation, so a distance-valued graph is fine for them."""
    dataset = load_dataset("digits")
    X = np.random.default_rng(13).normal(size=(dataset.n, 2))
    assert 0.0 <= knn_recall(X, dataset.graph, 5) <= 1.0


# --------------------------------------------------------------------------- #
#                        The report and its exclusions                         #
# --------------------------------------------------------------------------- #


def test_the_report_counts_zero_degree_exclusions(isolated_vertex_layout, isolated_vertex_graph):
    report = knn_recall_report(isolated_vertex_layout, isolated_vertex_graph, [2, 3])
    assert list(report.n_excluded) == [1, 1]
    assert list(report.n_scored) == [5, 5]


def test_precision_and_jaccard_exclude_nothing(isolated_vertex_layout, isolated_vertex_graph):
    for report_fn in (knn_precision_report, knn_jaccard_report):
        report = report_fn(isolated_vertex_layout, isolated_vertex_graph, 2)
        assert list(report.n_excluded) == [0]


def test_the_report_refuses_to_let_the_two_counts_disagree():
    """`np.nanmean` over an all-NaN row warns and returns NaN, which reaches a table as a blank."""
    with pytest.raises(ValueError, match="n_excluded"):
        NeighborPreservationReport(
            k=np.array([5, 10]),
            value=np.array([0.5, 0.5]),
            n_excluded=np.array([1, 2]),
            n_scored=np.array([5, 5]),
        )


def test_a_graph_of_isolated_vertices_scores_nan_rather_than_warning():
    G = nx.empty_graph(5)
    X = np.random.default_rng(14).normal(size=(5, 2))
    report = knn_recall_report(X, G, 2)
    assert np.isnan(report.value[0])
    assert report.n_excluded[0] == 5


def test_the_report_and_the_score_agree(karate, karate_structured_layout):
    report = knn_recall_report(karate_structured_layout, karate, [5, 10])
    assert list(report.value) == pytest.approx(knn_recall(karate_structured_layout, karate, [5, 10]))


# --------------------------------------------------------------------------- #
#                                  default_k                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "expected"), [("karate", 5), ("digits", 16), ("football_2023", 11)]
)
def test_default_k_is_the_input_graphs_mean_degree(name, expected):
    assert default_k(load_dataset(name).graph) == expected


def test_default_k_rounds_to_nearest_not_to_even():
    """`round(16.5)` is 16 — banker's rounding. §0.5 decision 6 says round to nearest.

    A 4-cycle plus a pendant edge: n = 5, m = 5, mean degree 2.0. Scaled to land on a half:
    a graph with n = 4 and m = 5 has mean degree 2.5, which must give 3.
    """
    G = nx.complete_graph(4)
    G.remove_edge(0, 1)  # n = 4, m = 5 -> mean degree 2.5
    assert G.number_of_edges() == 5
    assert default_k(G) == 3


def test_default_k_has_a_floor_of_one():
    assert default_k(nx.empty_graph(10)) == 1
    assert default_k(nx.empty_graph(0)) == 1


# --------------------------------------------------------------------------- #
#                 SNS and the two stresses it is contrasted with               #
# --------------------------------------------------------------------------- #


@pytest.fixture
def collinear_path():
    """A 5-path drawn on a line at unit spacing: an *exact* realisation of its hop counts."""
    return nx.path_graph(5), np.column_stack([np.arange(5.0), np.zeros(5)])


def test_an_exact_embedding_has_zero_stress_of_every_kind(collinear_path):
    G, X = collinear_path
    assert scale_normalized_stress(X, G) == pytest.approx(0.0)
    assert normalized_stress(X, G) == pytest.approx(0.0)
    assert raw_stress(X, G) == pytest.approx(0.0)


def test_sns_is_invariant_under_scaling_the_layout(karate, karate_structured_layout):
    X = karate_structured_layout
    assert scale_normalized_stress(7 * X, karate) == pytest.approx(
        scale_normalized_stress(X, karate), rel=1e-12
    )


def test_sns_is_invariant_under_scaling_the_ideal_distances(karate, karate_structured_layout):
    D = ideal_distances(karate)
    assert scale_normalized_stress(karate_structured_layout, ideal=7 * D) == pytest.approx(
        scale_normalized_stress(karate_structured_layout, ideal=D), rel=1e-12
    )


def test_raw_stress_moves_under_both_scalings(karate, karate_structured_layout):
    """The asymmetry is the point: it is why all three functions exist (§2.1)."""
    X, D = karate_structured_layout, ideal_distances(karate)
    base = raw_stress(X, ideal=D)
    assert raw_stress(7 * X, ideal=D) != pytest.approx(base, rel=1e-6)
    assert raw_stress(X, ideal=7 * D) != pytest.approx(base, rel=1e-6)


def test_normalized_stress_moves_under_both_scalings_separately(karate, karate_structured_layout):
    """Pinned because the plan predicted "under one" and it is under both.

    Eq. (41) fixes alpha at 1, so it is invariant only when X and d are scaled *together* —
    see the test below. Neither scaling alone leaves it fixed, and that is exactly the
    scale-dependence SNS removes.
    """
    X, D = karate_structured_layout, ideal_distances(karate)
    base = normalized_stress(X, ideal=D)
    assert normalized_stress(7 * X, ideal=D) != pytest.approx(base, rel=1e-6)
    assert normalized_stress(X, ideal=7 * D) != pytest.approx(base, rel=1e-6)


def test_normalized_stress_is_invariant_under_the_joint_scaling(karate, karate_structured_layout):
    """And raw stress is not, which is the difference between eqs. (40) and (41)."""
    X, D = karate_structured_layout, ideal_distances(karate)
    assert normalized_stress(7 * X, ideal=7 * D) == pytest.approx(
        normalized_stress(X, ideal=D), rel=1e-12
    )
    assert raw_stress(7 * X, ideal=7 * D) != pytest.approx(raw_stress(X, ideal=D), rel=1e-6)


def test_alpha_star_is_the_minimiser(karate, karate_structured_layout):
    """Closed form, checked against a coarse search: no optimiser, and none needed."""
    from twin.metrics.common import condensed as to_condensed
    from twin.metrics.common import pairwise_distances as pdist_of

    d = to_condensed(ideal_distances(karate))
    x = pdist_of(karate_structured_layout)
    alpha = optimal_scale(d, x)

    def objective(a):
        residual = d - a * x
        return residual @ residual

    grid = alpha * np.array([0.5, 0.9, 0.99, 1.01, 1.1, 1.5])
    assert objective(alpha) < min(objective(a) for a in grid)


def test_sns_at_alpha_one_is_normalized_stress(karate, karate_structured_layout):
    """The two differ by exactly the rescaling, which is the §3 argument in one line."""
    X = karate_structured_layout
    D = ideal_distances(karate)
    alpha = optimal_scale(condensed(D), pairwise_distances(X))
    assert scale_normalized_stress(X, ideal=D) == pytest.approx(
        normalized_stress(alpha * X, ideal=D), rel=1e-12
    )


def test_a_collapsed_layout_is_nan_rather_than_a_number(k4):
    """alpha* is 0/0 there. The limit is 1, but reporting it would present it as a score."""
    assert np.isnan(scale_normalized_stress(np.zeros((4, 2)), k4))


def test_the_disconnected_mask_needs_no_second_code_path(disjoint_paths):
    """Infinite pairs are dropped by `finite_pair_mask`, in all three functions."""
    X = np.random.default_rng(15).normal(size=(disjoint_paths.number_of_nodes(), 2))
    for policy in ("lcc", "mask"):
        D = ideal_distances(disjoint_paths, disconnected=policy)
        assert np.isfinite(scale_normalized_stress(X, ideal=D))
        assert np.isfinite(raw_stress(X, ideal=D))

    # ... and the two policies score different pair sets, so they give different numbers.
    assert scale_normalized_stress(
        X, ideal=ideal_distances(disjoint_paths)
    ) != pytest.approx(
        scale_normalized_stress(X, ideal=ideal_distances(disjoint_paths, disconnected="mask"))
    )


def test_a_condensed_ideal_vector_is_accepted(karate, karate_structured_layout):
    """The runner holds a square array; a caller that already condensed one need not square it."""
    D = ideal_distances(karate)
    assert scale_normalized_stress(karate_structured_layout, ideal=condensed(D)) == pytest.approx(
        scale_normalized_stress(karate_structured_layout, ideal=D), rel=1e-12
    )


def test_supplying_both_ideal_distances_and_their_parameters_raises(karate, karate_structured_layout):
    """They would be silently ignored: the closure they configure has already happened."""
    with pytest.raises(ValueError, match="already happened"):
        scale_normalized_stress(
            karate_structured_layout, ideal=ideal_distances(karate), disconnected="mask"
        )


def test_neither_a_graph_nor_ideal_distances_raises(karate_structured_layout):
    with pytest.raises(ValueError, match="either a graph or"):
        scale_normalized_stress(karate_structured_layout)


def test_sns_checks_orientation(karate, karate_structured_layout):
    with pytest.raises(ValueError, match=r"\(n_points, d\)"):
        scale_normalized_stress(karate_structured_layout.T, karate)


def test_the_sns_docstring_carries_the_kamada_kawai_caveat():
    """It has to exist before anyone reads a number; the reporting flag is step 11."""
    doc = scale_normalized_stress.__doc__
    assert "Kamada-Kawai" in doc
    assert "optimiser success" in doc
