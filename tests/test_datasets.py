"""Spec §3.4 — the `Dataset` layer: stored sources, labels, style.

The layering rule these tests exist to hold: a `Dataset` is stored information, it is never
laid out, and it holds no derived information. The other half — that a `LayoutGraph` is never
labelled — lives in ``test_layout_graph.py``.
"""

import networkx as nx
import numpy as np
import pytest

from twin.datasets import (
    Dataset,
    DatasetStyle,
    LabelSet,
    Stored,
    StoredMatrix,
    Unweighted,
)


@pytest.fixture
def toy() -> nx.Graph:
    """A 4-path with distinct weights, so a transform of them is visible."""
    G = nx.path_graph(4)
    for k, (u, v) in enumerate(G.edges()):
        G[u][v]["weight"] = float(k + 1)
    return G


def make(toy, **kw) -> Dataset:
    kw.setdefault("affinity", Stored(meaning="strength"))
    return Dataset(name="toy", graph=toy, **kw)


# --------------------------------------------------------------------------- #
#                              Stored sources                                  #
# --------------------------------------------------------------------------- #


def test_a_dataset_must_store_at_least_one_side(toy):
    with pytest.raises(ValueError, match="neither side is stored"):
        Dataset(name="toy", graph=toy, affinity=None, distance=None)


def test_stored_reads_the_graphs_own_weights(toy):
    A = make(toy).stored_matrix("affinity").toarray()
    assert A[0, 1] == 1.0 and A[1, 2] == 2.0 and A[2, 3] == 3.0
    assert np.array_equal(A, A.T)


def test_unweighted_is_the_indicator(toy):
    A = make(toy, affinity=Unweighted()).stored_matrix("affinity").toarray()
    assert set(np.unique(A)) == {0.0, 1.0}


def test_stored_matrix_is_looked_up_and_symmetrised(toy):
    """A directed kNN distance matrix folds onto the undirected edge set with ``max``."""
    import scipy.sparse as sp

    D = sp.csr_array(np.array([[0, 5, 0, 0], [0, 0, 0, 0], [0, 7, 0, 0], [0, 0, 0, 0]], float))
    ds = make(toy, distance=StoredMatrix("D"), matrices={"D": D})

    got = ds.stored_matrix("distance").toarray()
    assert got[0, 1] == got[1, 0] == 5.0
    assert got[1, 2] == got[2, 1] == 7.0


def test_a_missing_stored_matrix_says_which_keys_exist(toy):
    ds = make(toy, distance=StoredMatrix("nope"), matrices={"D": np.eye(4)})
    with pytest.raises(KeyError, match="declares \\['D'\\]"):
        ds.stored_matrix("distance")


def test_unknown_meaning_is_refused():
    with pytest.raises(ValueError, match="unknown meaning"):
        Stored(meaning="affinity")  # retired as a kind — see spec §0.5 decision 1


# --------------------------------------------------------------------------- #
#                                   Labels                                     #
# --------------------------------------------------------------------------- #


def test_label_sets_are_per_original_vertex(toy):
    with pytest.raises(ValueError, match="per original vertex"):
        make(toy, label_sets={"x": LabelSet(name="x", values=np.zeros(3))})


def test_noise_class_is_excluded_from_the_class_count():
    ls = LabelSet(name="conference", values=np.array([-1, -1, 0, 1, 2]), noise_label=-1)
    assert ls.n_classes == 3
    assert ls.class_sizes() == {-1: 2, 0: 1, 1: 1, 2: 1}


def test_the_two_roles_are_independent(toy):
    """football and tabula invert the arrangement; one `role` field cannot express both."""
    ds = make(
        toy,
        label_sets={
            "coarse": LabelSet(
                name="coarse", values=np.zeros(4, int), scoring_role="sub", display_role="hue"
            ),
            "fine": LabelSet(
                name="fine",
                values=np.arange(4),
                scoring_role="primary",
                display_role="shade",
                parent="coarse",
            ),
        },
    )
    assert [ls.name for ls in ds.label_sets_by_scoring_role("primary")] == ["fine"]
    assert ds.label_set_by_display_role("hue").name == "coarse"


def test_a_label_set_may_not_name_a_parent_that_does_not_exist(toy):
    with pytest.raises(ValueError, match="unknown parent"):
        make(
            toy,
            label_sets={"fine": LabelSet(name="fine", values=np.arange(4), parent="coarse")},
        )


def test_composite_labels_are_declared_and_densely_coded(toy):
    ds = make(
        toy,
        label_sets={
            "conf": LabelSet(name="conf", values=np.array([0, 0, 1, 1])),
            "div": LabelSet(name="div", values=np.array([-1, 1, -1, -1])),
        },
        composite_of={"conf_div": ("conf", "div")},
    )
    composite = ds.labels("conf_div")
    assert composite.tolist() == [0, 1, 2, 2]
    assert set(np.unique(composite)) == {0, 1, 2}


def test_a_composite_may_not_name_an_unknown_label_set(toy):
    with pytest.raises(ValueError, match="unknown label sets"):
        make(toy, composite_of={"x": ("nope",)})


def test_labels_says_what_is_available(toy):
    with pytest.raises(KeyError, match="no label set"):
        make(toy).labels("conference")


# --------------------------------------------------------------------------- #
#                             Shape and connectivity                           #
# --------------------------------------------------------------------------- #


def test_graph_must_be_indexed_from_zero():
    G = nx.Graph()
    G.add_edge("a", "b")
    with pytest.raises(ValueError, match="0..n-1"):
        Dataset(name="lettered", graph=G, affinity=Unweighted())


def test_connectivity_summary(toy):
    ds = make(toy)
    assert ds.is_connected() and ds.n_components() == 1
    assert ds.largest_component_fraction() == 1.0
    assert ds.mean_degree() == pytest.approx(2 * 3 / 4)


def test_largest_component_is_deliberately_absent(toy):
    """Dropped from the interface (spec §3.4, `DECISIONS.md`): no registered dataset needs it.

    Untested code for a case that does not occur is worse than an honest absence, and a method
    that exists will get written in a hurry at step 3 when SNS turns out to want it.
    """
    assert not hasattr(make(toy), "largest_component")


# --------------------------------------------------------------------------- #
#                                   Style                                      #
# --------------------------------------------------------------------------- #


def test_hue_table_shapes_and_swaps():
    style = DatasetStyle(palette="husl", palette_seed=1)
    plain = style.hue_table(11)
    assert plain.shape == (11, 3)

    swapped = DatasetStyle(palette="husl", palette_seed=1, palette_swaps=((0, 5),)).hue_table(11)
    np.testing.assert_array_equal(swapped[0], plain[5])
    np.testing.assert_array_equal(swapped[5], plain[0])
    np.testing.assert_array_equal(swapped[1:5], plain[1:5])


def test_distinctipy_palette_keeps_the_trailing_black_row():
    """The tabula figure indexes it by wrapping past the end; dropping it recolours edges."""
    table = DatasetStyle(palette="distinctipy", palette_seed=7).hue_table(13)
    assert table.shape == (14, 3)
    np.testing.assert_array_equal(table[-1], np.zeros(3))
