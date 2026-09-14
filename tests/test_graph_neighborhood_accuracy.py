r"""Spec §2.3 eqs. (61)-(64) — label recovery from the 1-hop graph neighbourhood.

The graph twin of `test_knn_accuracy.py`, and the same discipline: the oracle below walks
``G.neighbors`` in a Python loop and shares no code with the sparse product the implementation
uses. Acc\* is a **label-validity check, not a baseline** (§5.3 guard 2).

The valuation guard is the point of the module. `digits`' stored ``weight`` is a Euclidean
*distance* and `tabula`'s graph is unstamped, so a metric reading ``dataset.graph`` would vote
with an inverted sign on one dataset and an unchecked quantity on another — bug §1.6 one layer
up, which this project has already shipped once.
"""

import networkx as nx
import numpy as np
import pytest

from twin.datasets import load_dataset
from twin.metrics import (
    graph_neighborhood_accuracy,
    graph_neighborhood_accuracy_report,
    majority_class_rate,
)
from twin.metrics.context import EvaluationContext


# --------------------------------------------------------------------------- #
#                            The independent oracle                            #
# --------------------------------------------------------------------------- #


def graph_vote_accuracy(G, labels, *, weighted: bool, tie_rule: str) -> float:
    """Eqs. (61)-(64) as a dictionary count over ``G.neighbors``, one vertex at a time.

    Deliberately nothing in common with the implementation's ``A @ onehot``: no adjacency
    matrix, no class codes, no vectorised argmax. ``labels`` must be dense ``0..|L|-1`` for the
    ``"first"`` rule to line up, since the implementation breaks ties toward the lowest class
    *code*.
    """
    labels = list(labels)
    per_vertex = []
    for i in G.nodes():
        votes: dict = {}
        for j in G.neighbors(i):
            w = float(G[i][j].get("weight", 1.0)) if weighted else 1.0
            votes[labels[j]] = votes.get(labels[j], 0.0) + w
        if not votes or max(votes.values()) <= 0:
            continue  # isolated, or no vote mass at all
        best = max(votes.values())
        tied = sorted(label for label, mass in votes.items() if mass == best)
        if tie_rule == "fractional":
            per_vertex.append(1.0 / len(tied) if labels[i] in tied else 0.0)
        else:
            per_vertex.append(1.0 if tied[0] == labels[i] else 0.0)
    return sum(per_vertex) / len(per_vertex)


# --------------------------------------------------------------------------- #
#                           Agreement with the oracle                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tie_rule", ["fractional", "first"])
def test_agrees_with_the_oracle_on_karate(karate, karate_labels, tie_rule):
    for weighting, weighted in (("uniform", False), ("affinity", True)):
        expected = graph_vote_accuracy(
            karate, karate_labels, weighted=weighted, tie_rule=tie_rule
        )
        got = graph_neighborhood_accuracy(
            karate_labels, karate, weighting=weighting, tie_rule=tie_rule
        )
        assert got == pytest.approx(expected, rel=1e-12), (weighting, tie_rule)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("tie_rule", ["fractional", "first"])
def test_agrees_with_the_oracle_on_random_weighted_graphs(seed, tie_rule):
    rng = np.random.default_rng(seed)
    G = nx.gnp_random_graph(40, 0.12, seed=seed)
    for u, v in G.edges():
        G[u][v]["weight"] = float(rng.uniform(0.1, 5.0))
    labels = rng.integers(0, 3, size=40)

    for weighting, weighted in (("uniform", False), ("affinity", True)):
        expected = graph_vote_accuracy(G, labels, weighted=weighted, tie_rule=tie_rule)
        got = graph_neighborhood_accuracy(
            labels, G, weighting=weighting, tie_rule=tie_rule
        )
        assert got == pytest.approx(expected, rel=1e-12), weighting


def test_hand_computable_case():
    r"""A star whose leaves split 2-1, so every count can be written down.

    Centre 0 (label 0) has neighbours 1, 2 (label 1) and 3 (label 0): $s^*_0(1) = 2$ beats
    $s^*_0(0) = 1$, so the centre is wrong. Each leaf's only neighbour is the centre, label 0 —
    so leaves 1 and 2 are wrong and leaf 3 is right. One of four: 0.25.
    """
    G = nx.star_graph(3)
    labels = np.array([0, 1, 1, 0])
    assert graph_neighborhood_accuracy(labels, G) == pytest.approx(0.25)


def test_affinity_weighting_can_overturn_a_uniform_vote():
    """Eq. (62) is a different metric from eq. (61), not a refinement of it.

    Centre 0 has two neighbours of class 1 at weight 0.1 and one of class 0 at weight 10:
    uniform voting calls it class 1 and wrong, affinity voting calls it class 0 and right.
    """
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 0.1), (0, 2, 0.1), (0, 3, 10.0)])
    labels = np.array([0, 1, 1, 0])

    assert graph_neighborhood_accuracy(labels, G, weighting="uniform") == pytest.approx(0.25)
    assert graph_neighborhood_accuracy(labels, G, weighting="affinity") == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
#                      Isolated vertices (§2.3's requirement)                  #
# --------------------------------------------------------------------------- #


def test_isolated_vertices_are_excluded_and_counted(isolated_vertex_graph):
    """§2.3: "Isolated vertices excluded, and the excluded count returned."

    Vertex 5 has no neighbourhood, so there is no information to recover its label from —
    including it would score it by the fractional tie rule over every class, which is noise.
    """
    labels = np.array([0, 0, 1, 1, 1, 0])
    report = graph_neighborhood_accuracy_report(labels, isolated_vertex_graph)

    assert report.n_excluded == 1
    assert report.n_isolated == 1
    assert report.n_scored == 5

    # The score is the mean over the five connected vertices, which the oracle also skips.
    assert report.accuracy == pytest.approx(
        graph_vote_accuracy(
            isolated_vertex_graph, labels, weighted=False, tie_rule="fractional"
        ),
        rel=1e-12,
    )

    # Scoring vertex 5 as well would have moved the number, so the exclusion is not cosmetic.
    connected_only = graph_neighborhood_accuracy(labels[:5], nx.path_graph(5))
    assert report.accuracy == pytest.approx(connected_only, rel=1e-12)


def test_a_graph_with_no_edges_at_all_is_nan():
    """Not 0, and not a division by zero: nothing was scored, so there is no score."""
    G = nx.empty_graph(5)
    report = graph_neighborhood_accuracy_report(np.array([0, 0, 1, 1, 1]), G)

    assert np.isnan(report.accuracy)
    assert np.isnan(report.majority_class_rate)
    assert (report.n_scored, report.n_excluded, report.n_isolated) == (0, 5, 5)


def test_the_chance_floor_is_over_the_scored_vertices(isolated_vertex_graph):
    """The score and its floor must share a denominator, or the comparison is invalid.

    Vertex 5, the isolated one, carries the minority label here — so a floor computed over all
    six vertices differs from one computed over the five that were scored.
    """
    labels = np.array([0, 0, 0, 0, 0, 1])
    report = graph_neighborhood_accuracy_report(labels, isolated_vertex_graph)

    assert report.majority_class_rate == pytest.approx(1.0)  # all five scored ones are class 0
    assert majority_class_rate(labels) == pytest.approx(5 / 6)


# --------------------------------------------------------------------------- #
#                  The valuation guard — bug §1.6's shape (⚠️)                 #
# --------------------------------------------------------------------------- #


def test_refuses_a_distance_stamped_graph_under_affinity_weighting():
    """⚠️ The trap this metric is most likely to fall into, on the dataset that carries it.

    `digits`' graph ``weight`` attribute is a stored Euclidean distance (5.3-37.8), so eq. (62)
    over it would count the *least* coupled neighbours most. The stamp is what makes the refusal
    possible; the numbers themselves cannot say which side they are.
    """
    dataset = load_dataset("digits")
    labels = dataset.labels("digit")

    assert dataset.graph.graph["twin_side"] == "distance"
    with pytest.raises(ValueError, match="stamped 'distance'"):
        graph_neighborhood_accuracy(labels, dataset.graph, weighting="affinity")

    # The context's affinity graph is the right argument and goes through.
    ctx = EvaluationContext.for_dataset(dataset)
    assert 0.0 < graph_neighborhood_accuracy(labels, ctx.affinity, weighting="affinity") <= 1.0


def test_uniform_weighting_reads_no_valuation_and_so_needs_no_guard():
    """Eq. (61) counts neighbours, so a distance-stamped graph is harmless there.

    Same reasoning as `twin/metrics/visual.py`'s deliberate absence of a stamp guard: the
    guard belongs where a valuation is read, and nowhere else.
    """
    dataset = load_dataset("digits")
    labels = dataset.labels("digit")
    assert 0.0 < graph_neighborhood_accuracy(labels, dataset.graph) <= 1.0


def test_a_combinatorial_declaration_wins_over_a_stray_weight_attribute():
    """`football_2023` declares `Unweighted()` and still carries a `weight` from its .mat.

    732 edges at 1 and three at 2. Reading those would make eq. (62) disagree with eq. (61) on
    a graph the dataset says is unweighted. `weighted_recall` takes the same position.
    """
    dataset = load_dataset("football_2023")
    labels = dataset.labels("conference")

    # It is `dataset.graph` that carries both the declaration and the stray attribute. The
    # context's affinity graph is already all-ones -- the `Unweighted()` declaration is honoured
    # one layer earlier -- so that path cannot exercise the branch.
    assert dataset.graph.graph["twin_side"] == "combinatorial"
    assert {w for _, _, w in dataset.graph.edges(data="weight")} == {1.0, 2.0}

    assert graph_neighborhood_accuracy(
        labels, dataset.graph, weighting="affinity"
    ) == pytest.approx(graph_neighborhood_accuracy(labels, dataset.graph, weighting="uniform"))

    ctx = EvaluationContext.for_dataset(dataset)
    assert {w for _, _, w in ctx.affinity.edges(data="weight")} == {1.0}

    # ⚠️ Measured, and it cuts against the branch mattering *here*: three edges at 2 out of 735
    # do not move Acc* at all -- an unstamped copy of the same graph scores identically under
    # both weightings. So football cannot demonstrate the branch, only that the declaration is
    # read. The next test constructs a graph where it does change the answer.
    unstamped = nx.Graph()
    unstamped.add_weighted_edges_from(dataset.graph.edges(data="weight"))
    assert graph_neighborhood_accuracy(
        labels, unstamped, weighting="affinity"
    ) == pytest.approx(graph_neighborhood_accuracy(labels, unstamped, weighting="uniform"))


def test_the_combinatorial_branch_changes_the_answer_when_the_weights_would():
    """The branch, on a graph built so that honouring the declaration is visible.

    Centre 0 has two class-1 neighbours at weight 1 and one class-0 neighbour at weight 10.
    Counted, class 1 wins and the centre is wrong; weighted, class 0 wins and it is right. A
    graph stamped `"combinatorial"` must give the *counted* answer whatever its edges say.
    """
    from twin.forms import stamp_form

    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 1.0), (0, 2, 1.0), (0, 3, 10.0)])
    labels = np.array([0, 1, 1, 0])

    unstamped = graph_neighborhood_accuracy(labels, G, weighting="affinity")
    stamp_form(G, side="combinatorial", source="test: a declared-unweighted graph")
    stamped = graph_neighborhood_accuracy(labels, G, weighting="affinity")

    assert unstamped == pytest.approx(0.5)
    assert stamped == pytest.approx(0.25)
    assert stamped == pytest.approx(graph_neighborhood_accuracy(labels, G, weighting="uniform"))


def test_an_unstamped_graph_is_accepted(karate, karate_labels):
    """The analytic graphs of §6 are built by hand and carry no stamp; the guard is a backstop
    for a *declared* wrong side, not a requirement that everything be declared."""
    assert "twin_side" not in karate.graph
    assert graph_neighborhood_accuracy(karate_labels, karate, weighting="affinity") > 0.5


# --------------------------------------------------------------------------- #
#                            Ties and the contract                             #
# --------------------------------------------------------------------------- #


def test_a_variable_size_neighbourhood_ties_readily(karate, karate_labels):
    """Unlike a fixed $k$, a degree-2 vertex with one neighbour of each class is a tie.

    The count is reported for exactly that reason: the fractional rule turns ties into
    fractional credit, so a large `n_ties` means the score is largely an expectation over
    tie-breakings rather than a count of correct vertices.
    """
    report = graph_neighborhood_accuracy_report(karate_labels, karate)
    assert report.n_ties > 0

    strict = graph_neighborhood_accuracy(karate_labels, karate, tie_rule="first")
    assert strict != pytest.approx(report.accuracy)


def test_the_two_tie_rules_agree_when_nothing_ties():
    """A star with a unanimous neighbourhood has no ties, so the rule cannot matter."""
    G = nx.star_graph(4)
    labels = np.array([0, 1, 1, 1, 1])
    assert graph_neighborhood_accuracy(labels, G, tie_rule="fractional") == pytest.approx(
        graph_neighborhood_accuracy(labels, G, tie_rule="first")
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"weighting": "distance"}, "opposite quantity"),
        ({"weighting": "nonsense"}, "must be 'uniform'"),
        ({"tie_rule": "random"}, "tie_rule must be"),
    ],
)
def test_refuses_bad_arguments(karate, karate_labels, kwargs, match):
    with pytest.raises(ValueError, match=match):
        graph_neighborhood_accuracy(karate_labels, karate, **kwargs)


def test_refuses_a_size_mismatch(karate, karate_labels):
    with pytest.raises(ValueError, match="per original vertex"):
        graph_neighborhood_accuracy(karate_labels[:10], karate)


def test_refuses_a_two_dimensional_label_vector(karate, karate_random_layout):
    """`labels` comes first here, so a layout in that slot must be refused loudly."""
    with pytest.raises(ValueError, match="1-D"):
        graph_neighborhood_accuracy(karate_random_layout, karate)


def test_both_docstrings_say_label_validity_check_not_a_baseline():
    """§5.3 guard 2 is enforced in the exporter at step 11; the sentence must exist first."""
    for function in (graph_neighborhood_accuracy, graph_neighborhood_accuracy_report):
        text = " ".join(function.__doc__.split()).lower()
        assert "not a baseline" in text
        assert "incomparable" in text
