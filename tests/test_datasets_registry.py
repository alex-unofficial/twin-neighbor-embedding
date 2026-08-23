"""Spec §3.4 — the registered datasets, and what they declare.

These are assertions about the *roster*: that every entry builds, that its stored sides are
declared consistently with `docs/DATASET_PROVENANCE.md`, and that the label vectors are the ones
pinned in ``test_step2_characterisation.py`` before they moved out of the figures.
"""

import networkx as nx
import numpy as np
import pytest

from twin.config import REAL_WORLD_DIR, SYNTHETIC_NET_DIR
from twin.datasets import (
    DATASET_REGISTRY,
    Stored,
    StoredMatrix,
    Unweighted,
    check_weight_consistency,
    load_dataset,
)
from tests.test_step2_characterisation import FIXTURE, GRAPH_PINS, NEEDS_FILE, weight_summary

TABULA = REAL_WORLD_DIR / "tabula-liver.mat"


@pytest.fixture(scope="module")
def pins():
    if not FIXTURE.exists():
        pytest.skip("step2_labels.npz not present; run tests/fixtures/make_step2_pins.py")
    return np.load(FIXTURE, allow_pickle=True)


def needs_data(name):
    if name in NEEDS_FILE and not (REAL_WORLD_DIR.exists() and SYNTHETIC_NET_DIR.exists()):
        pytest.skip("data/networks not present")


# --------------------------------------------------------------------------- #
#                                 The roster                                   #
# --------------------------------------------------------------------------- #


def test_the_registry_is_exactly_the_pinned_roster():
    assert set(DATASET_REGISTRY) == set(GRAPH_PINS)


@pytest.mark.parametrize("name", sorted(GRAPH_PINS))
def test_every_registered_dataset_builds_and_matches_its_graph_pins(name):
    """Same invariants as the pre-restructure pins, now reached through the registry.

    This is the load-bearing half of the move: the registry defaults must reproduce the graphs
    the batch was actually running, not merely graphs of the same kind.
    """
    needs_data(name)
    ds = load_dataset(name)
    expected = GRAPH_PINS[name]

    assert ds.name == name
    assert (ds.n, ds.m) == (expected["n"], expected["m"])
    assert ds.is_connected() is expected["connected"]

    got = weight_summary(ds.graph)
    assert got[3] == expected["weights"][3]
    np.testing.assert_allclose(got[:3], expected["weights"][:3], rtol=1e-5)


@pytest.mark.parametrize("name", sorted(GRAPH_PINS))
def test_every_dataset_declares_a_stored_side_and_a_description(name):
    needs_data(name)
    ds = load_dataset(name)
    assert ds.affinity is not None or ds.distance is not None
    assert len(ds.description) > 40, "the description is the generated dataset table's text"
    assert ds.style is not None


@pytest.mark.parametrize("name", sorted(GRAPH_PINS))
def test_no_label_vector_is_the_wrong_length(name):
    """The failure mode the layering exists to prevent, checked at construction."""
    needs_data(name)
    ds = load_dataset(name)
    for label_name, ls in ds.label_sets.items():
        assert ls.values.shape == (ds.n,), label_name


def test_unknown_dataset_lists_the_registered_ones():
    with pytest.raises(KeyError, match="unknown dataset"):
        load_dataset("mnist")


def test_factory_parameters_are_recorded_for_the_manifest():
    ds = load_dataset("small_world", n=60, k=6, p=0.05)
    assert ds.params == {"n": 60, "k": 6, "p": 0.05}
    assert ds.n == 60


# --------------------------------------------------------------------------- #
#                              football_2023                                   #
# --------------------------------------------------------------------------- #


def test_football_labels_are_the_pinned_ones(pins):
    needs_data("football_2023")
    ds = load_dataset("football_2023")

    np.testing.assert_array_equal(ds.labels("conference"), pins["football_conference"])
    np.testing.assert_array_equal(ds.labels("division"), pins["football_division"])

    conference = ds.label_set("conference")
    assert conference.noise_label == -1
    assert conference.display_name(-1) == "NCAA Division I FBS independents"
    assert conference.n_classes == 10  # 11 distinct strings, one of which is the noise class


def test_football_division_subdivides_the_conference_and_is_ordered():
    needs_data("football_2023")
    division = load_dataset("football_2023").label_set("division")
    assert division.parent == "conference" and division.ordered
    assert (division.scoring_role, division.display_role) == ("sub", "shade")
    assert division.derivation.startswith("derived:")


def test_football_composite_is_declared_not_improvised():
    """Spec §3.4: the composite is a direct test of the sub-conference claim, so it is named."""
    needs_data("football_2023")
    ds = load_dataset("football_2023")
    assert ds.composite_of == {"conference_division": ("conference", "division")}
    assert np.unique(ds.labels("conference_division")).size == 15


def test_football_is_declared_unweighted_on_both_sides():
    """732 of 735 edges weigh 1 (spec §0.5), so the stored weights carry no information."""
    needs_data("football_2023")
    ds = load_dataset("football_2023")
    assert isinstance(ds.affinity, Unweighted) and isinstance(ds.distance, Unweighted)
    assert set(np.unique(ds.stored_matrix("affinity").data)) == {1.0}


def test_football_style_reproduces_the_figures_palette(pins):
    """`DatasetStyle` is a faithful record of what `paper_figures` hardcodes today.

    The figure builds `3 * n_hues + 3` rows — dark / normal / light per hue, then three blacks —
    and indexes them as `3*l + 1 + s`. The unshaded row of each triple is the base hue, so the
    style's table must reproduce exactly that.
    """
    needs_data("football_2023")
    style = load_dataset("football_2023").style
    palette = pins["football_palette"]

    np.testing.assert_allclose(style.hue_table(11), palette[1::3][:11])
    assert style.shade_factors == (0.7, 1.0, 1.3)
    np.testing.assert_array_equal(palette[-3:], np.zeros((3, 3)))


# --------------------------------------------------------------------------- #
#                                   karate                                     #
# --------------------------------------------------------------------------- #


def test_the_two_karates_are_different_datasets():
    """The registry used to expose only the weighted one, under the other one's description."""
    needs_data("karate")
    plain, weighted = load_dataset("karate"), load_dataset("karate_weighted")

    assert isinstance(plain.affinity, Unweighted) and isinstance(plain.distance, Unweighted)
    assert isinstance(weighted.affinity, Stored) and weighted.affinity.meaning == "strength"
    assert weighted.distance is None, "the distance side is derived, 1/w"

    assert set(np.unique(plain.stored_matrix("affinity").data)) == {1.0}
    assert np.unique(weighted.stored_matrix("affinity").data).size == 7


def test_karate_weighted_carries_the_club_split():
    ds = load_dataset("karate_weighted")
    club = ds.labels("club")
    assert club.shape == (34,) and set(np.unique(club)) == {0, 1}
    assert ds.label_set("club").display_names == {0: "Mr. Hi", 1: "Officer"}


# --------------------------------------------------------------------------- #
#                                   digits                                     #
# --------------------------------------------------------------------------- #


def test_digits_stores_the_distance_side(pins):
    """The §1.6 fact: the weights are Euclidean distances, so the affinity is the derived one."""
    ds = load_dataset("digits")
    assert ds.affinity is None
    assert isinstance(ds.distance, Stored) and ds.distance.meaning == "distance"

    weights = ds.stored_matrix("distance").data
    assert weights.min() > 5 and weights.max() < 38

    np.testing.assert_array_equal(ds.labels("digit"), pins["digits_target"])


def test_digits_defaults_to_the_k_that_was_actually_run():
    """`dataset.mnist` defaulted to 100 neighbours; nothing ever ran it that way."""
    assert (load_dataset("digits").n, load_dataset("digits").m) == (1797, 14732)
    assert load_dataset("digits").params == {"n_neighbors": 12}
    assert load_dataset("digits", n_neighbors=5).m < 14732


# --------------------------------------------------------------------------- #
#                                   tabula                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not TABULA.exists(), reason="tabula-liver.mat not present")
def test_tabula_stores_both_sides_and_refuses_to_transform_either():
    ds = load_dataset("tabula")
    assert isinstance(ds.affinity, StoredMatrix) and ds.affinity.meaning == "adaptive_kernel"
    assert isinstance(ds.distance, StoredMatrix) and ds.distance.meaning == "distance"
    assert set(ds.matrices) == {"A", "D"}

    D = ds.stored_matrix("distance")
    assert (D != D.T).nnz == 0, "the distance side must reach the layout layer symmetric"
    assert D.data.min() > 0.2 and D.data.max() < 6.4  # measured kNN distances, not memberships


@pytest.mark.skipif(not TABULA.exists(), reason="tabula-liver.mat not present")
def test_tabula_labels_are_the_pinned_ones_plus_the_two_covariates(pins):
    ds = load_dataset("tabula")
    np.testing.assert_array_equal(ds.labels("cell_ontology_class"), pins["tabula_cell_type"])
    assert [
        ds.label_set("cell_ontology_class").display_names[i] for i in range(13)
    ] == list(pins["tabula_cell_type_names"])

    # The roles invert football's: the fine label scores, the coarse one supplies hue.
    assert ds.label_set("cell_ontology_class").scoring_role == "primary"
    assert ds.label_set("cell_ontology_class").display_role == "shade"
    assert ds.label_set("compartment").display_role == "hue"
    assert ds.label_set("cell_ontology_class").parent == "compartment"

    # Scored deliberately: separation by donor is an artifact, so a low score is the good one.
    assert {ls.name for ls in ds.label_sets_by_scoring_role("auxiliary")} == {"donor", "method"}
    assert "gender" not in ds.label_sets, "constant fields carry no information"


@pytest.mark.skipif(not TABULA.exists(), reason="tabula-liver.mat not present")
def test_tabula_exposes_its_reference_layout_and_provenance():
    """`X_umap` was reached by a figure re-opening the `.mat`; it is dataset state."""
    ds = load_dataset("tabula")
    assert ds.reference_layouts["X_umap"].shape == (5007, 2)
    assert ds.provenance["distance_space"] == "X_scvi"
    assert ds.provenance["knn_k"] == 14


# --------------------------------------------------------------------------- #
#                          Weight-source consistency                           #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not TABULA.exists(), reason="tabula-liver.mat not present")
def test_tabula_weight_consistency_measured_values():
    """Measured, then pinned — the spec's argument said -1 and the data says otherwise.

    Global rank correlation is only -0.24, far from "strongly negative", because sigma_i varies
    by row and decouples membership from distance *across* rows. Per-row it is -0.78 on the
    symmetric forms the layout consumes. Restricted to the pairs recorded in one direction only
    — where fuzzy-union symmetrisation cannot have mixed row j's parameters into row i — it is
    **exactly -1 in every scored row**.

    So the relationship is exact where it can be seen unmixed, and symmetrisation is the cause
    of the degradation, not a disagreement between the two exports. That is the conclusion the
    spec anticipated but could not distinguish with the first two statistics alone. Do not
    replace these with -1 and loosen the tolerance.
    """
    report = check_weight_consistency(load_dataset("tabula"))

    assert report.comparable and report.spans_same_edge_set
    assert report.n_shared == 49368

    assert report.global_spearman == pytest.approx(-0.236351, abs=1e-5)
    assert report.median_row_spearman == pytest.approx(-0.782037, abs=1e-5)
    assert report.n_rows_scored == 5007

    assert report.median_row_spearman_unmixed == -1.0
    assert report.n_rows_scored_unmixed == 3819


def test_consistency_is_not_run_when_only_one_side_is_stored():
    assert check_weight_consistency(load_dataset("digits")) is None
    assert check_weight_consistency(load_dataset("karate_weighted")) is None


def test_two_indicator_sides_are_reported_as_not_comparable():
    """`Unweighted` on both sides spans the same edge set by construction and carries no ranks."""
    needs_data("football_2023")
    report = check_weight_consistency(load_dataset("football_2023"))
    assert report is not None and not report.comparable
    assert report.spans_same_edge_set
    assert np.isnan(report.global_spearman)


def test_a_disagreeing_pair_of_sides_is_detected():
    """The failure the check exists for: sides that rank the same pairs in the same direction."""
    from twin.datasets import Dataset

    G = nx.path_graph(6)
    for k, (u, v) in enumerate(G.edges()):
        G[u][v]["weight"] = float(k + 1)

    agreeing = Dataset(
        name="agreeing",
        graph=G,
        affinity=Stored(meaning="strength"),
        distance=StoredMatrix("D"),
        matrices={"D": nx.to_scipy_sparse_array(G, weight="weight").astype(float) ** -1.0},
    )
    disagreeing = Dataset(
        name="disagreeing",
        graph=G,
        affinity=Stored(meaning="strength"),
        distance=StoredMatrix("D"),
        matrices={"D": nx.to_scipy_sparse_array(G, weight="weight").astype(float)},
    )

    assert check_weight_consistency(agreeing).global_spearman == pytest.approx(-1.0)
    assert check_weight_consistency(disagreeing).global_spearman == pytest.approx(+1.0)
