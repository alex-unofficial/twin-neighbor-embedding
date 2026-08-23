"""Spec §3.4 — the dataset connectivity / weight / label report."""

import networkx as nx
import numpy as np
import pytest

from twin.datasets import DATASET_REGISTRY, Dataset, Stored, StoredMatrix, Unweighted
from twin.layoutgraph import vertex_graph
from twin.report import build_report, describe_dataset, main, to_markdown, zero_distance_pair_fraction


# --------------------------------------------------------------------------- #
#                            Pairs at zero distance                            #
# --------------------------------------------------------------------------- #


def test_zero_distance_pairs_compose_through_zero_weight_paths():
    """The reason it is a *pair* count and not an edge count: zero-weight edges chain.

    Three vertices joined by two zero-weight edges are mutually distance-zero, so that is three
    pairs, not two edges. Computed from the components of the zero-weight subgraph, which is
    exact for non-negative weights and avoids an all-pairs shortest path.
    """
    G = nx.Graph()
    G.add_nodes_from(range(4))
    G.add_weighted_edges_from([(0, 1, 0.0), (1, 2, 0.0), (2, 3, 5.0)])

    assert zero_distance_pair_fraction(G, 4) == pytest.approx(3 / 6)


def test_no_zero_distances_means_zero_fraction():
    G = nx.Graph()
    G.add_nodes_from(range(3))
    G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 2.0)])
    assert zero_distance_pair_fraction(G, 3) == 0.0


def test_negative_distances_are_refused():
    G = nx.Graph()
    G.add_nodes_from(range(2))
    G.add_weighted_edges_from([(0, 1, -1.0)])
    with pytest.raises(ValueError, match="negative edge weights"):
        zero_distance_pair_fraction(G, 2)


def test_the_current_roster_has_no_coincident_vertex_pairs():
    """Measured, and it settles §0.5's concern for the roster as it stands.

    Zero ideal distances distort silhouette and collapse shortest-path derivations, so the
    question was whether they occur. They do not: tabula's smallest *measured* distance is 0.2,
    and the only transform that would have produced zeros — applying the `kernel` rule to its
    adaptive affinities — is the one that is refused.
    """
    for name in ("tabula", "digits", "karate_weighted", "football_2023"):
        row = describe_dataset(name)
        assert row.zero_distance_pair_fraction == 0.0, name


# --------------------------------------------------------------------------- #
#                                  The rows                                    #
# --------------------------------------------------------------------------- #


def test_a_row_records_the_lambda_criterion_input():
    """§4.2 defines the lambda criterion at k = mean degree, so it must be readable here."""
    row = describe_dataset("football_2023")
    assert row.mean_degree == pytest.approx(2 * 735 / 132)
    assert (row.n, row.m, row.connected, row.n_components) == (132, 735, True, 1)


def test_a_row_names_the_derived_side_and_its_rule():
    digits = describe_dataset("digits")
    assert digits.affinity_source == "derived"
    assert "exp(-d/d_scale)" in digits.affinity_derivation
    assert digits.distance_derivation.startswith("stored")

    karate = describe_dataset("karate_weighted")
    assert "1/w" in karate.distance_derivation


def test_a_row_carries_both_label_roles():
    roles = {name: (scoring, display) for name, _, scoring, display, _ in describe_dataset("tabula").labels}
    assert roles["cell_ontology_class"] == ("primary", "shade")
    assert roles["compartment"] == ("sub", "hue")
    assert roles["donor"] == ("auxiliary", "none")


def test_weight_consistency_appears_only_where_both_sides_are_stored():
    assert describe_dataset("tabula").consistency is not None
    assert describe_dataset("digits").consistency is None


# --------------------------------------------------------------------------- #
#                                  Rendering                                   #
# --------------------------------------------------------------------------- #


def test_the_report_covers_every_registered_dataset_and_nothing_else():
    markdown = to_markdown(build_report(["karate", "digits"]))
    assert "`karate`" in markdown and "`digits`" in markdown
    assert "`tabula`" not in markdown


def test_no_column_branches_on_a_dataset_name():
    """CLAUDE.md: nothing outside `twin/datasets/` may branch on a dataset name."""
    import inspect

    import twin.report

    source = inspect.getsource(twin.report)
    for name in DATASET_REGISTRY:
        assert f'== "{name}"' not in source
        assert f'"{name}":' not in source


def test_the_cli_writes_markdown(tmp_path):
    out = tmp_path / "report.md"
    assert main(["karate", "-o", str(out)]) == 0
    assert out.read_text().startswith("# Dataset report")


def test_the_cli_rejects_an_unknown_dataset():
    with pytest.raises(SystemExit):
        main(["mnist"])
