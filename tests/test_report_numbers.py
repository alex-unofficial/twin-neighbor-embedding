r"""Executable form of the graph-side label scores in `docs/DATASET_REPORT.md`.

The same pattern as `test_dataset_provenance.py`: the report is **generated**, and the numbers it
generates are **pinned**, so a change to Sil\*, Acc\*, `ideal_distances`, the affinity derivation
or the Leiden ceiling cannot move a documented number without a test saying so. `DECISIONS.md`
(42) and (44) quote these values, and a design record that quotes numbers nothing checks is a
design record that goes stale silently.

**Split by cost, not by importance.** The scores are deterministic and cheap; the permutation
nulls are not — one evaluation is 0.001 s at karate's size and 0.87 s at tabula's, where four
label sets multiply it. So the three small datasets are pinned in full here, and `digits` and
`tabula` carry their own `slow` module-scoped run. The two numbers from tabula that `DECISIONS.md`
leans on hardest — Cramér's V against the primary, and `method`'s point-mass null — need no
closure and are pinned in the default suite.
"""

import numpy as np
import pytest

from twin.datasets import load_dataset
from twin.report import PermutationNull, _permutation_count, cramers_v, describe_dataset

#: Sil\* and Acc\* per (dataset, label set). Deterministic: no layout, no seed, no permutation.
#: Regenerate with `python -m twin.report` and only in a commit that says a metric moved.
PINNED = {
    ("ring_of_cliques", "clique"): dict(sil=0.5325, acc=1.0000, acc_w=1.0000, floor=0.1000),
    ("karate_weighted", "club"): dict(sil=0.3774, acc=0.9412, acc_w=0.9706, floor=0.5000),
    ("football_2023", "conference"): dict(sil=0.3304, acc=0.9697, acc_w=0.9697, floor=0.1061),
    ("football_2023", "division"): dict(sil=0.0203, acc=0.9886, acc_w=0.9886, floor=0.5909),
    ("football_2023", "conference_division"): dict(
        sil=0.3266, acc=0.9722, acc_w=0.9722, floor=0.1061
    ),
}

PINNED_SLOW = {
    ("digits", "digit"): dict(sil=0.3670, acc=0.9853, acc_w=0.9855, floor=0.1018),
    ("tabula", "cell_ontology_class"): dict(
        sil=0.2839, acc=0.9615, acc_w=0.9683, floor=0.3112
    ),
    ("tabula", "compartment"): dict(sil=0.2975, acc=0.9897, acc_w=0.9930, floor=0.5279),
    ("tabula", "donor"): dict(sil=0.3102, acc=0.9873, acc_w=0.9918, floor=0.5712),
    ("tabula", "method"): dict(sil=0.0849, acc=0.9868, acc_w=0.9918, floor=0.9575),
}

#: The best Sil\* a Leiden partition of each graph reaches — the reference the Sil\* column is
#: read against, and the answer to "why is 0.33 not low".
PINNED_CEILING = {
    "ring_of_cliques": 0.5325,
    "karate_weighted": 0.3890,
    "football_2023": 0.3433,
}
PINNED_CEILING_SLOW = {"digits": 0.3950, "tabula": 0.3890}


@pytest.fixture(scope="module")
def expensive():
    """`digits` and `tabula` computed **once** for the whole module.

    Not per parametrised case: `tabula`'s four label sets at 39 permutations each are ~145 s, so
    a function-scoped fixture ran them five times over and took the `slow` module from 1.5
    minutes to nearly eight.
    """
    return {name: describe_dataset(name) for name in ("digits", "tabula")}


@pytest.fixture(scope="module")
def cheap():
    """One report run over the three datasets whose permutation nulls are free."""
    return {
        name: describe_dataset(name)
        for name in ("ring_of_cliques", "karate_weighted", "football_2023")
    }


def scored(row, label_set):
    return next(s for s in row.label_scores if s.label_set == label_set)


# --------------------------------------------------------------------------- #
#                            The scores, as documented                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("key", "expected"), sorted(PINNED.items()))
def test_the_documented_scores_reproduce(cheap, key, expected):
    name, label_set = key
    s = scored(cheap[name], label_set)

    assert s.silhouette_graph == pytest.approx(expected["sil"], abs=5e-5)
    assert s.graph_accuracy == pytest.approx(expected["acc"], abs=5e-5)
    assert s.graph_accuracy_affinity == pytest.approx(expected["acc_w"], abs=5e-5)
    assert s.majority_class_rate == pytest.approx(expected["floor"], abs=5e-5)


@pytest.mark.parametrize(("name", "expected"), sorted(PINNED_CEILING.items()))
def test_the_compactness_ceiling_reproduces(cheap, name, expected):
    """Leiden is seeded, so this is reproducible — and it is what makes Sil\\* readable."""
    assert cheap[name].compactness_ceiling == pytest.approx(expected, abs=5e-5)


def test_the_ground_truth_sits_just_below_the_ceiling(cheap):
    """The finding: 0.33 is 96 % of what the graph admits, not a third of what it should be."""
    fractions = {
        name: scored(row, row.label_scores[0].label_set).silhouette_graph
        / row.compactness_ceiling
        for name, row in cheap.items()
    }
    assert fractions["ring_of_cliques"] == pytest.approx(1.0, abs=1e-3)
    assert fractions["karate_weighted"] > 0.95
    assert fractions["football_2023"] > 0.95


@pytest.mark.slow
@pytest.mark.parametrize(("key", "expected"), sorted(PINNED_SLOW.items()))
def test_the_documented_scores_reproduce_on_the_large_datasets(expensive, key, expected):
    """⚠️ `digits` and `tabula`, opt-in: their nulls are 39-111 evaluations each."""
    name, label_set = key
    row = expensive[name]
    s = scored(row, label_set)

    assert s.silhouette_graph == pytest.approx(expected["sil"], abs=5e-5)
    assert s.graph_accuracy == pytest.approx(expected["acc"], abs=5e-5)
    assert s.graph_accuracy_affinity == pytest.approx(expected["acc_w"], abs=5e-5)
    assert s.majority_class_rate == pytest.approx(expected["floor"], abs=5e-5)
    assert row.compactness_ceiling == pytest.approx(
        PINNED_CEILING_SLOW[name], abs=5e-5
    )


# --------------------------------------------------------------------------- #
#             (44) The local/global contrast, which the thesis wants           #
# --------------------------------------------------------------------------- #


def test_footballs_division_is_locally_clean_and_globally_diffuse(cheap):
    r"""The numbers `DECISIONS.md` (44)'s suggested thesis wording quotes.

    Acc\* 0.9886 against a 0.5909 floor, Sil\* 0.0203 at 6 % of the ceiling. Pinned as a pair,
    because either alone supports a confident and opposite conclusion about whether divisions
    are "in" the graph.
    """
    division = scored(cheap["football_2023"], "division")
    conference = scored(cheap["football_2023"], "conference")
    ceiling = cheap["football_2023"].compactness_ceiling

    assert division.graph_accuracy == pytest.approx(0.9886, abs=5e-5)
    assert division.majority_class_rate == pytest.approx(0.5909, abs=5e-5)
    assert division.graph_accuracy > conference.graph_accuracy

    assert division.silhouette_graph == pytest.approx(0.0203, abs=5e-5)
    assert division.silhouette_graph / ceiling < 0.07
    assert conference.silhouette_graph / ceiling > 0.95

    # Above its permutation null, but barely, where Acc* is far above its floor.
    assert 5 < division.silhouette_graph_null.z < 15
    assert division.graph_accuracy_null.z > 15


def test_the_b_over_a_ratio_is_what_caps_sil_star():
    r"""Why the ceiling is low at all: $b^*/a^*$ near 1.5, and 1.02 for `division`.

    Computed here from the definitions rather than read off the metric, so the explanation in
    the report's prose is checked rather than asserted.
    """
    from twin.metrics.context import EvaluationContext

    dataset = load_dataset("football_2023")
    D = EvaluationContext.for_dataset(dataset).ideal

    ratios = {}
    for name in ("conference", "division"):
        labels = dataset.labels(name)
        a, b = [], []
        for i in range(labels.size):
            own = labels == labels[i]
            own[i] = False
            a.append(D[i][own].mean())
            b.append(
                min(D[i][labels == c].mean() for c in np.unique(labels) if c != labels[i])
            )
        ratios[name] = float(np.mean(b) / np.mean(a))

    assert ratios["conference"] == pytest.approx(1.503, abs=5e-4)
    assert ratios["division"] == pytest.approx(1.022, abs=5e-4)
    # And Sil* tracks 1 - a/b, which is the cap the prose claims.
    assert 1 - 1 / ratios["conference"] == pytest.approx(0.335, abs=5e-3)


# --------------------------------------------------------------------------- #
#          (42) The confound column, and the null that does not exist          #
# --------------------------------------------------------------------------- #


def test_donor_nearly_determines_cell_type():
    r"""⚠️ The number (42) turns on, and it needs no closure — labels only.

    $V = 0.917$ is what makes tabula's high Sil\*(`donor`) unattributable to batch effect
    without conditioning. Pinned alongside NMI, because NMI is the measure that *understated* it
    — 0.413 reads as moderate association where $V$ correctly reads as near-determination, and
    an earlier version of the record concluded the wrong thing from the NMI alone.
    """
    from sklearn.metrics import normalized_mutual_info_score

    dataset = load_dataset("tabula")
    donor = dataset.labels("donor")
    method = dataset.labels("method")
    cell = dataset.labels("cell_ontology_class")

    assert cramers_v(donor, cell) == pytest.approx(0.9174, abs=5e-4)
    assert cramers_v(donor, dataset.labels("compartment")) == pytest.approx(0.7719, abs=5e-4)
    assert cramers_v(method, cell) == pytest.approx(0.5967, abs=5e-4)
    assert cramers_v(method, dataset.labels("compartment")) == pytest.approx(0.1507, abs=5e-4)

    assert normalized_mutual_info_score(donor, cell) == pytest.approx(0.4132, abs=5e-4)
    assert cramers_v(donor, cell) > 2 * normalized_mutual_info_score(donor, cell)


def test_the_donor_confound_is_a_disjoint_sampling_of_cell_types():
    """The mechanism, so the V is interpretable: the two donors are different tissue samples."""
    dataset = load_dataset("tabula")
    donor = dataset.labels("donor")
    cell = dataset.labels("cell_ontology_class")
    names = {str(v): k for k, v in dataset.label_set("cell_ontology_class").display_names.items()}

    tsp14, tsp6 = donor == 0, donor == 1
    assert (tsp14.sum(), tsp6.sum()) == (2147, 2860)
    assert (cell[tsp14] == names["hepatocyte"]).sum() == 1491   # 69 % of TSP14
    assert (cell[tsp6] == names["macrophage"]).sum() == 1379    # 48 % of TSP6


def test_methods_accuracy_null_is_a_point_mass_and_reports_no_z():
    r"""⚠️ The case that produced $z = 8.7\times10^{13}$ before the fix.

    213 cells against 4794, so every relabelling leaves almost every 1-hop neighbourhood
    majority-`10X` and Acc\* is the majority rate on **every** permutation — one distinct value.
    ``np.std`` still returns $2.3\times10^{-16}$ on that array, because it subtracts a rounded
    mean from a constant; dividing by it reported a point-mass null as the most significant
    result in the table. Degeneracy is decided by counting distinct values, which needs no
    tolerance.

    Ten permutations are enough to demonstrate it, so this does not need the `slow` mark.
    """
    from twin.metrics import graph_neighborhood_accuracy, majority_class_rate
    from twin.metrics.context import EvaluationContext

    dataset = load_dataset("tabula")
    ctx = EvaluationContext.for_dataset(dataset)
    labels = dataset.labels("method")

    rng = np.random.default_rng(0)
    samples = np.array(
        [graph_neighborhood_accuracy(rng.permutation(labels), ctx.affinity) for _ in range(10)]
    )

    assert np.unique(samples).size == 1
    assert samples[0] == majority_class_rate(labels)
    assert 0 < samples.std(ddof=1) < 1e-15  # the trap: not zero, but a point mass

    null = PermutationNull.from_samples(0.9868, samples)
    assert null.degenerate
    assert np.isnan(null.z)
    assert null.sd == 0.0


def test_a_non_degenerate_null_reports_a_finite_z():
    """The other branch, so the degeneracy test cannot be satisfied by always returning nan."""
    null = PermutationNull.from_samples(1.0, np.array([0.0, 0.1, 0.2, 0.3]))
    assert not null.degenerate
    assert null.z == pytest.approx((1.0 - 0.15) / np.std([0.0, 0.1, 0.2, 0.3], ddof=1))


# --------------------------------------------------------------------------- #
#                   The z-score's scope, and the permutation budget            #
# --------------------------------------------------------------------------- #


def test_the_z_score_does_not_make_magnitudes_comparable(cheap):
    r"""⚠️ Asserted because it is the natural thing to assume about a z-score.

    $\sigma_0$ depends on the class count and balance, so a 2-class split reaches a large $z$ at
    a modest Sil\*. On football, `conference_division` (15 classes) and `conference` (11) have
    nearly equal Sil\* — 0.3266 and 0.3304 — and their nulls differ in both mean and spread, so
    the z-scores cannot be read as effect sizes.
    """
    composite = scored(cheap["football_2023"], "conference_division")
    conference = scored(cheap["football_2023"], "conference")

    assert composite.silhouette_graph == pytest.approx(conference.silhouette_graph, abs=0.01)
    assert composite.silhouette_graph_null.mean < conference.silhouette_graph_null.mean
    assert composite.silhouette_graph_null.sd != pytest.approx(
        conference.silhouette_graph_null.sd, rel=0.05
    )


def test_the_permutation_count_is_large_where_it_is_free():
    """1000 on the small graphs, where 20 draws visibly straddled zero; 30 floor on tabula."""
    assert _permutation_count(34) == 1000
    assert _permutation_count(132) == 1000
    assert _permutation_count(1797) == 111
    assert _permutation_count(5007) == 39
    assert _permutation_count(10**9) == 30      # clamped, never below a usable sd
    assert _permutation_count(0) == 1000        # and never a division by zero


def test_the_null_is_estimated_from_the_documented_number_of_draws(cheap):
    for name, row in cheap.items():
        expected = _permutation_count(load_dataset(name).n)
        for s in row.label_scores:
            assert s.silhouette_graph_null.n_permutations == expected
            assert s.graph_accuracy_null.n_permutations == expected


# --------------------------------------------------------------------------- #
#                      The degenerate composite is scored once                 #
# --------------------------------------------------------------------------- #


def test_the_degenerate_composite_is_omitted_from_the_scores():
    """`compartment_cell_type` *is* `cell_ontology_class`, so scoring both is one column twice.

    Asserted without computing a closure: the omission is decided from the declarations.
    """
    dataset = load_dataset("tabula")
    assert dataset.degenerate_composites() == {
        "compartment_cell_type": "cell_ontology_class"
    }

    row = describe_dataset("tabula", label_scores=False)
    assert row.label_scores == []  # skipped entirely, so nothing to check here


@pytest.mark.slow
def test_tabulas_score_rows_exclude_the_degenerate_composite(expensive):
    names = [s.label_set for s in expensive["tabula"].label_scores]

    assert "compartment_cell_type" not in names
    assert names == ["cell_ontology_class", "compartment", "donor", "method"]


def test_footballs_composite_is_scored_because_it_is_not_degenerate(cheap):
    names = [s.label_set for s in cheap["football_2023"].label_scores]
    assert names == ["conference", "division", "conference_division"]
