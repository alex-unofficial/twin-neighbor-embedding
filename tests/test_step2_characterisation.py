"""Characterisation pins for spec §7 step 2 — the derived labels, colours and graphs.

The restructure in spec §3.4/§3.5 moves the label derivations out of ``paper_figures.py`` and
``metadata.py`` into ``twin.datasets``, and moves the representation algebra behind
``LayoutGraph``. Both are meant to be behaviour-preserving. This file is what makes that
checkable rather than asserted; the equivalent pins in session 1 caught an inverted sort in the
leave-one-out neighbour selection that would otherwise have read as a plausible metric.

Two layers of pin, deliberately:

* the arrays themselves live in ``fixtures/step2_labels.npz``, regenerated only by
  ``fixtures/make_step2_pins.py``;
* a **digest of every array** is written out below, so that regenerating the fixture shows up in
  the diff of this file and cannot happen quietly.

A pin changing is not automatically a failure — but it must be a *deliberate* change, declared
in the commit message, with the pin updated in the same commit (CLAUDE.md, "Commit
granularity").
"""

import hashlib
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import twin.dataset as legacy
from twin.config import REAL_WORLD_DIR
from twin.datasets import load_dataset

FIXTURE = Path(__file__).parent / "fixtures" / "step2_labels.npz"
TABULA = REAL_WORLD_DIR / "tabula-liver.mat"

#: sha256 (first 16 hex) of every pinned array. Recomputed by the test, so a fixture that is
#: regenerated without updating this table fails loudly.
DIGESTS = {
    "digits_target": "a3c91c262eddcf7b",
    "football_conference": "786217d874029c7b",
    "football_conference_codes": "004862bbccddb6b2",
    "football_conference_names": "6b173228d7fc8556",
    "football_division": "f355633fac46b700",
    "football_edge_alpha": "9dae570382d4ee21",
    "football_edge_colors": "4c24c8e02c4b1951",
    "football_edge_labels": "f7dd98c29480ad2e",
    "football_edge_standing": "71c8ad1f8604a15e",
    "football_palette": "4a27a61529384ed0",
    "football_vertex_colors": "f0a94f23bf9c9465",
    "tabula_alpha_edge_points": "a262317a4d1b4df8",
    "tabula_cell_type": "1658130feab5bf49",
    "tabula_cell_type_names": "872d28af199606b1",
    "tabula_edge_alpha": "a13bb3ea2c4de502",
    "tabula_edge_colors": "69a6f657a2a732fc",
    "tabula_edge_labels": "3a9e48d0122736a8",
    "tabula_edge_marker_size": "216fdde341521268",
    "tabula_palette": "bf5c56e6a809a735",
    "tabula_vertex_colors": "271e34e52f4a6d35",
}

#: Per-dataset graph invariants, keyed by the name each dataset gets in ``DATASET_REGISTRY``
#: (spec §3.4). ``weights`` is (min, median, max, n_distinct) over edges, taking a missing
#: ``weight`` attribute as 1.0.
GRAPH_PINS = {
    "karate": dict(n=34, m=78, connected=True, weights=(1.0, 1.0, 1.0, 1)),
    "karate_weighted": dict(n=34, m=78, connected=True, weights=(1.0, 3.0, 7.0, 7)),
    "football_2023": dict(n=132, m=735, connected=True, weights=(1.0, 1.0, 2.0, 2)),
    "sbm_er_3modes": dict(
        n=210, m=5753, connected=True, weights=(8.6911177e-05, 8.6911177e-05, 8.6911177e-05, 1)
    ),
    "digits": dict(
        n=1797, m=14732, connected=True, weights=(5.29150262, 21.84032967, 37.84177586, 1079)
    ),
    "tabula": dict(
        n=5007, m=49368, connected=True, weights=(2.57161492e-08, 0.211888939, 1.0, 45354)
    ),
    "ring_of_cliques": dict(n=40, m=70, connected=True, weights=(1.0, 1.0, 1.0, 1)),
    "tree_of_cliques": dict(n=157, m=273, connected=True, weights=(1.0, 1.0, 1.0, 1)),
    "small_world": dict(n=150, m=900, connected=True, weights=(1.0, 1.0, 1.0, 1)),
    "barabasi_albert": dict(n=1000, m=6951, connected=True, weights=(1.0, 1.0, 1.0, 1)),
}

#: How each pinned graph is built *before* ``DATASET_REGISTRY`` exists, with the parameters the
#: registry will use as its defaults. Replaced by the registry itself once it lands.
LEGACY_FACTORIES = {
    "karate": (legacy.karate, {}),
    "karate_weighted": (nx.karate_club_graph, {}),
    "football_2023": (legacy.football_2023, {}),
    "sbm_er_3modes": (legacy.sbmer3modes, {}),
    "digits": (legacy.mnist, {"n_neighbors": 12}),
    "tabula": (legacy.tabula, {"organ": "liver"}),
    "ring_of_cliques": (legacy.ring_of_cliques, {"n": 10, "k": 4}),
    "tree_of_cliques": (legacy.tree_of_cliques, {"h": 3, "k": 3, "n": 4}),
    "small_world": (legacy.small_world, {"n": 150, "k": 12, "p": 0.01}),
    "barabasi_albert": (legacy.barabasi_albert, {"n": 1000, "k": 7}),
}

NEEDS_FILE = {"karate", "football_2023", "sbm_er_3modes", "tabula"}


def digest(a: np.ndarray) -> str:
    """Stable digest of an array, including the object-dtype label-name vectors."""
    if a.dtype == object:
        return hashlib.sha256("\x00".join(map(str, a.ravel())).encode()).hexdigest()[:16]
    return hashlib.sha256(np.ascontiguousarray(a)).hexdigest()[:16]


def weight_summary(G: nx.Graph) -> tuple[float, float, float, int]:
    w = np.array([d.get("weight", 1.0) for _, _, d in G.edges(data=True)], dtype=float)
    return float(w.min()), float(np.median(w)), float(w.max()), int(np.unique(w).size)


@pytest.fixture(scope="module")
def pins():
    if not FIXTURE.exists():
        pytest.skip("step2_labels.npz not present; run tests/fixtures/make_step2_pins.py")
    return np.load(FIXTURE, allow_pickle=True)


# --------------------------------------------------------------------------- #
#                    The fixture itself has not been regenerated               #
# --------------------------------------------------------------------------- #


def test_every_pinned_array_is_covered_by_a_digest(pins):
    assert set(pins.files) == set(DIGESTS), (
        "the fixture and the digest table disagree about which arrays are pinned"
    )


@pytest.mark.parametrize("key", sorted(DIGESTS))
def test_pinned_array_digest(pins, key):
    assert digest(pins[key]) == DIGESTS[key], (
        f"{key} changed. If that was intended, say so in the commit message and update both "
        "the fixture and this table in the same commit."
    )


# --------------------------------------------------------------------------- #
#           The derived label vectors and colour arrays, as produced today     #
# --------------------------------------------------------------------------- #


def test_football_label_arrays_match_the_pins(pins):
    """Guards the move of the conference / division derivation into ``twin.datasets``.

    The division code comes from the ``stand[-5:-1]`` string slice buried in the figure; the
    palette from the ``3*l + 1 + s`` index arithmetic next to it. Both move; neither may change.
    """
    from twin.paper_figures import football_label_arrays

    got = football_label_arrays(load_dataset("football_2023"))

    np.testing.assert_array_equal(got["vertex_labels"], pins["football_conference"])
    np.testing.assert_array_equal(got["stand_labels"], pins["football_division"])
    np.testing.assert_array_equal(got["edge_labels"], pins["football_edge_labels"])
    np.testing.assert_array_equal(got["edge_standing"], pins["football_edge_standing"])
    np.testing.assert_array_equal(got["edge_alpha"], pins["football_edge_alpha"])
    np.testing.assert_array_equal(got["palette"], pins["football_palette"])
    np.testing.assert_array_equal(got["vertex_colors"], pins["football_vertex_colors"])
    np.testing.assert_array_equal(got["edge_colors"], pins["football_edge_colors"])

    assert list(got["conf_labels"]) == list(pins["football_conference_names"])
    assert list(got["conf_labels"].values()) == list(pins["football_conference_codes"])


def test_football_division_is_east_none_west_and_ordered(pins):
    """The three-way ordered sub-label the shade rule depends on (spec §3.4)."""
    division = pins["football_division"]
    assert set(np.unique(division)) == {-1, 0, 1}
    # 132 teams; the conferences without an East/West split sit at 0.
    assert dict(zip(*np.unique(division, return_counts=True))) == {-1: 27, 0: 78, 1: 27}


def test_football_independents_are_the_noise_class(pins):
    """Independents are coded -1, which indexes the three black rows at the palette's tail."""
    conference = pins["football_conference"]
    names = list(pins["football_conference_names"])
    codes = list(pins["football_conference_codes"])

    assert codes[names.index("NCAA Division I FBS independents")] == -1
    assert (conference == -1).sum() == 4

    palette = pins["football_palette"]
    for s in (-1, 0, 1):
        np.testing.assert_array_equal(palette[3 * -1 + 1 + s], np.zeros(3))


@pytest.mark.skipif(not TABULA.exists(), reason="tabula-liver.mat not present")
def test_tabula_label_arrays_match_the_pins(pins):
    """Guards the move of the second ad-hoc ``loadmat`` into the dataset definition."""
    from twin.paper_figures import tabula_label_arrays

    got = tabula_label_arrays(load_dataset("tabula"))

    np.testing.assert_array_equal(got["vertex_labels"], pins["tabula_cell_type"])
    np.testing.assert_array_equal(got["edge_labels"], pins["tabula_edge_labels"])
    np.testing.assert_array_equal(got["palette"], pins["tabula_palette"])
    np.testing.assert_array_equal(got["vertex_colors"], pins["tabula_vertex_colors"])
    np.testing.assert_array_equal(got["edge_colors"], pins["tabula_edge_colors"])
    np.testing.assert_array_equal(got["edge_alpha"], pins["tabula_edge_alpha"])
    np.testing.assert_array_equal(got["edge_marker_size"], pins["tabula_edge_marker_size"])
    np.testing.assert_array_equal(got["alpha_edge_points"], pins["tabula_alpha_edge_points"])

    assert list(got["label2id"]) == list(pins["tabula_cell_type_names"])


@pytest.mark.skipif(not TABULA.exists(), reason="tabula-liver.mat not present")
def test_tabula_cell_types_are_13_alphabetically_coded_classes(pins):
    names = list(pins["tabula_cell_type_names"])
    assert len(names) == 13 and names == sorted(names)
    assert pins["tabula_cell_type"].max() == 12

    # ``custom_cmap`` appends black as a 14th row, and the inter-cluster edge label -1 indexes
    # it by wrapping round the end of the array. Pinned because it is load-bearing and
    # accidental-looking: any palette that stops emitting the trailing black silently recolours
    # every inter-cluster edge.
    assert pins["tabula_palette"].shape == (14, 3)
    np.testing.assert_array_equal(pins["tabula_palette"][13], np.zeros(3))
    inter = pins["tabula_edge_labels"] == -1
    np.testing.assert_array_equal(pins["tabula_edge_colors"][inter][0], np.zeros(3))


def test_digits_target_is_the_sklearn_digits_label(pins):
    target = pins["digits_target"]
    assert target.shape == (1797,) and set(np.unique(target)) == set(range(10))


# --------------------------------------------------------------------------- #
#                        Per-dataset graph invariants                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(GRAPH_PINS))
def test_graph_invariants(name):
    """n, m, connectivity and the weight distribution, before the registry moves them.

    ``weights`` is what settles each dataset's declared stored sources: football is effectively
    unweighted (732 of 735 edges at 1), sbm-er-3modes carries one constant weight, and digits'
    5.3-37.8 range is what identifies its stored side as the *distance* one.
    """
    if name in NEEDS_FILE and not REAL_WORLD_DIR.exists():
        pytest.skip("data/networks not present")

    factory, kwargs = LEGACY_FACTORIES[name]
    G = factory(**kwargs)
    expected = GRAPH_PINS[name]

    assert (G.number_of_nodes(), G.number_of_edges()) == (expected["n"], expected["m"])
    assert nx.is_connected(G) is expected["connected"]
    assert nx.number_of_selfloops(G) == 0

    got = weight_summary(G)
    assert got[3] == expected["weights"][3]
    np.testing.assert_allclose(got[:3], expected["weights"][:3], rtol=1e-5)


def test_football_is_effectively_unweighted():
    """The measurement behind declaring football ``Unweighted()`` on both sides (spec §0.5)."""
    G = legacy.football_2023()
    w = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    assert (w == 1).sum() == 732 and (w == 2).sum() == 3
