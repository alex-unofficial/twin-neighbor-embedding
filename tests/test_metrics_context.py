"""The per-dataset evaluation context — `DECISIONS.md` (5), "Ideal distances: one source".

The load-bearing check of step 3 lives here: **one** ideal-distance array, built once from
``vertex_graph(dataset).distance``, scores an $A_V$, an $A_E$ and an $A_{tne}$ layout of the
same dataset. If any representation reaches for its own closure, or a metric recomputes one per
call, the central ablation is measuring three different things and the comparison between the
three means nothing.
"""

import networkx as nx
import numpy as np
import pytest

from twin.datasets import load_dataset
from twin.forms import graph_side
from twin.layoutgraph import edge_graph, twin_graph, vertex_graph
from twin.metrics.common import default_k
from twin.metrics.context import EvaluationContext
from twin.metrics.structural import (
    adjacency_distance_ratio,
    knn_recall,
    scale_normalized_stress,
    weighted_recall,
)


@pytest.fixture
def context(karate_dataset) -> EvaluationContext:
    return EvaluationContext.for_dataset(karate_dataset)


# --------------------------------------------------------------------------- #
#                          The load-bearing check                              #
# --------------------------------------------------------------------------- #


def test_one_target_scores_all_three_representations(karate_dataset, monkeypatch):
    """Three representations, one target, one closure. The point of the whole object."""
    import twin.metrics.common as common
    import twin.metrics.context as context_module

    calls = []
    original = common.ideal_distances

    def counting(G, *args, **kw):
        calls.append(G.graph.get("twin_source"))
        return original(G, *args, **kw)

    monkeypatch.setattr(context_module, "ideal_distances", counting)

    ctx = EvaluationContext.for_dataset(karate_dataset)
    vertex = vertex_graph(karate_dataset)
    representations = {
        "A_V": vertex,
        "A_E": edge_graph(vertex),
        "A_tne": twin_graph(vertex),
    }

    rng = np.random.default_rng(20)
    scores = {}
    for name, layout_graph in representations.items():
        raw = rng.normal(size=(2, layout_graph.n_points))  # (d, n_points), as the embedder gives
        X_v, _, _ = layout_graph.vertex_and_edge_points(raw)
        scores[name] = scale_normalized_stress(X_v.T, ideal=ctx.ideal)

    assert set(scores) == {"A_V", "A_E", "A_tne"}
    assert all(np.isfinite(v) for v in scores.values())

    # One closure, from the vertex graph's distance form — not one per representation.
    assert len(calls) == 1
    assert calls[0].startswith("A_V of karate_weighted")


def test_the_ideal_array_is_computed_once_and_reused(context):
    assert context.ideal is context.ideal


def test_the_ideal_array_is_the_vertex_graphs_closure(context, karate_dataset):
    from twin.metrics.common import ideal_distances

    expected = ideal_distances(vertex_graph(karate_dataset).distance)
    assert context.ideal == pytest.approx(expected)


def test_the_derived_representations_own_closure_is_a_different_target(karate_dataset):
    """Why the rule exists, stated as a measurement rather than as prose.

    `edge_graph`'s distance form is a generic normalise-symmetrise-log fallback over a matrix
    in nobody's units, and it is indexed by *edges*. Scoring A_E against it would answer a
    different question from the one A_V is scored on.
    """
    from twin.metrics.common import ideal_distances

    vertex = vertex_graph(karate_dataset)
    edge = edge_graph(vertex)

    vertex_target = ideal_distances(vertex.distance)
    edge_target = ideal_distances(edge.distance)
    assert vertex_target.shape != edge_target.shape


# --------------------------------------------------------------------------- #
#                                   Laziness                                   #
# --------------------------------------------------------------------------- #


def test_building_a_context_does_not_compute_the_closure(karate_dataset):
    """A metric set of ADR and kNN recall must not pay for a closure it never reads."""
    ctx = EvaluationContext.for_dataset(karate_dataset)
    assert not ctx.ideal_is_computed

    X = np.random.default_rng(21).normal(size=(ctx.n, 2))
    knn_recall(X, ctx.dataset.graph, ctx.k)
    adjacency_distance_ratio(X, ctx.affinity)
    assert not ctx.ideal_is_computed

    scale_normalized_stress(X, ideal=ctx.ideal)
    assert ctx.ideal_is_computed


def test_the_closure_can_be_released(context):
    assert context.ideal.shape == (context.n, context.n)
    context.release()
    assert not context.ideal_is_computed


def test_the_guard_is_carried_into_the_closure(karate_dataset):
    ctx = EvaluationContext.for_dataset(karate_dataset, max_n=10)
    with pytest.raises(ValueError, match="max_n=10"):
        ctx.ideal


def test_the_disconnection_policy_is_carried_into_the_closure(disjoint_paths):
    from twin.datasets import Dataset, Unweighted
    from twin.metrics.common import finite_pair_mask

    dataset = Dataset(
        name="two_components",
        graph=disjoint_paths,
        affinity=Unweighted(),
        distance=Unweighted(),
    )
    lcc = EvaluationContext.for_dataset(dataset)
    masked = EvaluationContext.for_dataset(dataset, disconnected="mask")
    assert finite_pair_mask(lcc.ideal).sum() == 6
    assert finite_pair_mask(masked.ideal).sum() == 9


# --------------------------------------------------------------------------- #
#                            What it hands to metrics                          #
# --------------------------------------------------------------------------- #


def test_the_affinity_form_is_stamped_and_accepted_by_both_readers(context):
    assert graph_side(context.affinity) == "affinity"

    X = np.random.default_rng(22).normal(size=(context.n, 2))
    assert np.isfinite(adjacency_distance_ratio(X, context.affinity))
    assert np.isfinite(weighted_recall(X, context.affinity, 5))


def test_the_affinity_form_is_not_the_datasets_graph():
    """On digits they differ by an inversion, which is the entire reason this property exists."""
    dataset = load_dataset("digits")
    ctx = EvaluationContext.for_dataset(dataset)

    # Paired by endpoint, not by iteration order: the affinity graph comes off a matrix and
    # does not enumerate its edges in `dataset.graph.edges()` order.
    edges = list(dataset.graph.edges())
    stored = np.array([dataset.graph[u][v]["weight"] for u, v in edges])
    derived = np.array([ctx.affinity[u][v]["weight"] for u, v in edges])
    from scipy.stats import spearmanr

    assert spearmanr(stored, derived).statistic == pytest.approx(-1.0)


def test_k_is_the_input_graphs_mean_degree(context, karate_dataset):
    assert context.k == default_k(karate_dataset.graph) == 5


def test_the_context_refuses_a_derived_representation(karate_dataset):
    vertex = vertex_graph(karate_dataset)
    with pytest.raises(ValueError, match="takes the vertex graph"):
        EvaluationContext(dataset=karate_dataset, vertex=edge_graph(vertex))


def test_the_context_refuses_a_normalised_vertex_graph(karate_dataset, monkeypatch):
    """`normalize=True` perturbs A_V, so it is not the dataset's own target."""
    import twin.metrics.context as context_module

    monkeypatch.setattr(
        context_module, "vertex_graph", lambda d, **kw: vertex_graph(d, normalize=True)
    )
    with pytest.raises(ValueError, match="degree-normalised"):
        EvaluationContext.for_dataset(karate_dataset)


def test_the_context_holds_no_layout_and_no_labels_of_its_own(context):
    """It is the meeting point of the two layers, not a merge of them."""
    assert not hasattr(context, "X")
    assert not hasattr(context, "labels")
    assert context.dataset.label_sets is not None
