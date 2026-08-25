"""Spec §2.0 — the ideal-distance model, the pair ordering, and the O(n²) guard.

The load-bearing property is not any single number here but *one target per dataset*: ideal
distances come from ``vertex_graph(dataset).distance`` and are shared by every representation
(`DECISIONS.md`, "Ideal distances: one source, one owner"). That is asserted where the
evaluation context lives; what this file pins is the primitive underneath it.
"""

import networkx as nx
import numpy as np
import pytest

from twin.datasets import load_dataset
from twin.layoutgraph import stamp_form, vertex_graph
from twin.metrics.common import (
    _loo_neighbors,
    clear_loo_cache,
    clear_pairwise_cache,
    condensed,
    finite_pair_mask,
    ideal_distances,
    mean_pairwise_distance,
    pairwise_distances,
)


# --------------------------------------------------------------------------- #
#                              The distances                                   #
# --------------------------------------------------------------------------- #


def test_unweighted_edge_lengths_are_hop_counts(path5):
    D = ideal_distances(path5)
    assert D == pytest.approx(np.abs(np.subtract.outer(np.arange(5), np.arange(5))))


def test_the_diagonal_is_zero_and_the_matrix_is_symmetric(karate):
    D = ideal_distances(karate)
    assert np.all(np.diag(D) == 0)
    assert D == pytest.approx(D.T)


def test_weighted_shortest_paths_beat_direct_edges():
    """A long direct edge must lose to a short detour — the point of a shortest-path closure."""
    G = nx.Graph()
    G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 1.0), (0, 2, 10.0)])
    assert ideal_distances(G)[0, 2] == pytest.approx(2.0)


def test_non_adjacent_pairs_are_present_and_finite(triangle):
    G = nx.path_graph(4)
    D = ideal_distances(G)
    assert np.isfinite(D).all()
    assert D[0, 3] == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
#                          The disconnection policies                          #
# --------------------------------------------------------------------------- #


def test_lcc_scores_the_largest_component_only(disjoint_paths):
    """Six finite pairs: the 4-path's C(4, 2). The triangle is excluded by policy."""
    assert finite_pair_mask(ideal_distances(disjoint_paths)).sum() == 6


def test_mask_scores_every_reachable_pair(disjoint_paths):
    """Nine: the 4-path's six plus the triangle's three. Only cross-component pairs are inf."""
    assert finite_pair_mask(ideal_distances(disjoint_paths, disconnected="mask")).sum() == 9


def test_both_policies_return_the_full_square_array(disjoint_paths):
    n = disjoint_paths.number_of_nodes()
    for policy in ("lcc", "mask"):
        assert ideal_distances(disjoint_paths, disconnected=policy).shape == (n, n)


def test_lcc_is_a_no_op_on_a_connected_graph(karate):
    assert ideal_distances(karate) == pytest.approx(
        ideal_distances(karate, disconnected="mask")
    )


def test_components_raises_not_implemented(disjoint_paths):
    """NotImplementedError, not ValueError: it is a deliberate absence, not a bad argument."""
    with pytest.raises(NotImplementedError, match="does not occur"):
        ideal_distances(disjoint_paths, disconnected="components")


def test_an_unknown_policy_is_a_value_error(disjoint_paths):
    with pytest.raises(ValueError, match="disconnected"):
        ideal_distances(disjoint_paths, disconnected="lcc_but_nicer")


# --------------------------------------------------------------------------- #
#                                 The guard                                    #
# --------------------------------------------------------------------------- #


def test_the_guard_fires_at_an_explicit_max_n(karate):
    """The mechanism."""
    with pytest.raises(ValueError, match="max_n"):
        ideal_distances(karate, max_n=10)


def test_the_guard_fires_at_the_default_max_n():
    """The default. A test that only passes max_n=10 still passes if the default becomes None.

    A 20001-node path is cheap to build and never gets as far as the O(n²) allocation, because
    the guard is the first statement in the function.
    """
    with pytest.raises(ValueError, match="max_n=20000"):
        ideal_distances(nx.path_graph(20001))


def test_the_guard_message_quotes_the_projected_allocation():
    with pytest.raises(ValueError, match=r"3\.2\d? GB"):
        ideal_distances(nx.path_graph(20001))


# --------------------------------------------------------------------------- #
#                    weight_transform as an escape hatch only                  #
# --------------------------------------------------------------------------- #


def test_transforms_apply_to_edge_lengths():
    G = nx.path_graph(3)
    for u, v in G.edges():
        G[u][v]["weight"] = 4.0
    assert ideal_distances(G)[0, 2] == pytest.approx(8.0)
    assert ideal_distances(G, "inverse")[0, 2] == pytest.approx(0.5)

    # `neg_log` is the surprisal model, so it only has a meaning for w <= 1 — above that it
    # produces negative lengths, and the guard below is what says so.
    P = nx.path_graph(3)
    for u, v in P.edges():
        P[u][v]["weight"] = 0.5
    assert ideal_distances(P, "neg_log")[0, 2] == pytest.approx(-2 * np.log(0.5))
    with pytest.raises(ValueError, match="non-positive"):
        ideal_distances(G, "neg_log")


def test_an_unknown_transform_raises():
    with pytest.raises(ValueError, match="unknown weight_transform"):
        ideal_distances(nx.path_graph(3), "sqrt")


def test_a_transform_producing_non_positive_lengths_raises():
    G = nx.path_graph(3)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    with pytest.raises(ValueError, match="non-positive"):
        ideal_distances(G, "one_minus")


def test_a_stamped_graph_plus_an_explicit_transform_raises(karate_dataset):
    """DECISIONS.md (10): the escape hatch may not override a declared source."""
    D = vertex_graph(karate_dataset).distance
    with pytest.raises(ValueError, match="already the declared"):
        ideal_distances(D, "inverse")


def test_the_same_stamped_graph_without_a_transform_is_accepted(karate_dataset):
    """The other half — otherwise the guard would forbid the one correct call site."""
    assert np.isfinite(ideal_distances(vertex_graph(karate_dataset).distance)).all()


def test_an_affinity_stamped_graph_is_refused():
    """Reading affinities as lengths is bug §1.6 one layer up, so it is refused outright."""
    G = stamp_form(nx.path_graph(3), side="affinity", source="a test")
    with pytest.raises(ValueError, match="distance-valued"):
        ideal_distances(G)


def test_a_bare_unstamped_graph_stays_acceptable(k4):
    """§6's analytic cases build graphs by hand and must not need a Dataset."""
    assert ideal_distances(k4, "identity").shape == (4, 4)


# --------------------------------------------------------------------------- #
#                            Zero-length edges                                 #
# --------------------------------------------------------------------------- #


def test_a_zero_length_edge_is_refused_with_its_reason():
    """It would vanish from the CSR matrix and be read as a missing edge, not a coincidence."""
    G = nx.path_graph(3)
    G[0][1]["weight"] = 0.0
    G[1][2]["weight"] = 1.0
    with pytest.raises(ValueError, match="disappears from the sparse matrix"):
        ideal_distances(G)


def test_a_negative_length_is_refused():
    G = nx.path_graph(2)
    G[0][1]["weight"] = -1.0
    with pytest.raises(ValueError, match="must be positive"):
        ideal_distances(G)


# --------------------------------------------------------------------------- #
#                             The pair ordering                                #
# --------------------------------------------------------------------------- #


def test_condensed_matches_scipy_pdist_ordering():
    """One ordering for the whole package. `pdist` is the definition; `condensed` follows it."""
    from scipy.spatial.distance import pdist, squareform

    X = np.random.default_rng(3).normal(size=(9, 3))
    assert condensed(squareform(pdist(X))) == pytest.approx(pdist(X))


def test_condensed_excludes_the_diagonal():
    n = 6
    assert condensed(np.zeros((n, n))).size == n * (n - 1) // 2


def test_condensed_refuses_a_non_square_matrix():
    with pytest.raises(ValueError, match="square"):
        condensed(np.zeros((3, 4)))


def test_condensed_is_bit_identical_to_the_index_form_on_a_non_symmetric_matrix():
    """The row-concatenation form must equal ``D[np.triu_indices(n, 1)]`` **exactly**.

    Deliberately non-symmetric: on a symmetric matrix the lower triangle equals the upper one,
    so reading the wrong triangle — or reading the right one transposed — produces the same
    numbers and the substitution would be untestable. Here $D_{ij} = 100i + j$, so every entry
    names its own position and both a reordering and a transposition are visible.

    ``array_equal``, not ``approx``: this is a pure re-indexing and there is no arithmetic in it
    to lose a bit to.
    """
    n = 7
    D = np.arange(n)[:, None] * 100 + np.arange(n)[None, :]
    assert not np.array_equal(D, D.T)
    assert np.array_equal(condensed(D), D[np.triu_indices(n, k=1)])
    # ... and the first few entries are row 0's tail, which is what "row-major upper triangle"
    # means. A column-major reading would start 1, 2, 102.
    assert list(condensed(D)[:3]) == [1, 2, 3]


@pytest.mark.parametrize("n", [0, 1, 2, 3, 8, 33])
def test_condensed_is_bit_identical_at_every_small_size(n):
    """Including the degenerate sizes: `np.concatenate` refuses an empty sequence."""
    D = np.random.default_rng(n).normal(size=(n, n))
    assert np.array_equal(condensed(D), D[np.triu_indices(n, k=1)])


def test_finite_pair_mask_is_condensed_and_not_square(disjoint_paths):
    n = disjoint_paths.number_of_nodes()
    assert finite_pair_mask(ideal_distances(disjoint_paths)).shape == (n * (n - 1) // 2,)


# --------------------------------------------------------------------------- #
#                        On the roster, not just fixtures                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["karate", "karate_weighted", "football_2023"])
def test_the_vertex_graphs_distance_form_closes_to_a_finite_target(name):
    """The one correct call: `vertex_graph(dataset).distance`, unstamped-transform-free."""
    dataset = load_dataset(name)
    D = ideal_distances(vertex_graph(dataset).distance)
    assert D.shape == (dataset.n, dataset.n)
    assert np.isfinite(D).all()
    assert (D[~np.eye(dataset.n, dtype=bool)] > 0).all()


# --------------------------------------------------------------------------- #
#                      Pairwise layout distances and the cache                 #
# --------------------------------------------------------------------------- #


def test_pairwise_distances_are_scipy_pdist(karate_structured_layout):
    from scipy.spatial.distance import pdist

    assert pairwise_distances(karate_structured_layout) == pytest.approx(
        pdist(karate_structured_layout)
    )


def test_pairwise_distances_align_with_condensed_ideal_distances(karate, karate_random_layout):
    """The two vectors a stress metric subtracts must be the same length and ordering."""
    D = condensed(ideal_distances(karate))
    assert pairwise_distances(karate_random_layout).shape == D.shape


def test_the_mean_is_the_mean_of_the_condensed_vector(karate_random_layout):
    assert mean_pairwise_distance(karate_random_layout) == pytest.approx(
        pairwise_distances(karate_random_layout).mean()
    )


def test_the_cache_is_keyed_on_content_not_identity():
    """An id-keyed cache would serve one layout's distances for another's after a gc."""
    clear_pairwise_cache()
    X = np.random.default_rng(4).normal(size=(20, 2))
    first = pairwise_distances(X).copy()

    X[0] += 5.0  # same object, different content
    assert not np.allclose(pairwise_distances(X), first)

    # ... and an equal-content copy is a hit, not a recomputation.
    assert pairwise_distances(X.copy()) is pairwise_distances(X)


def test_the_cache_is_capped(karate_random_layout):
    """Tabula's condensed vector is 100 MB; an lru_cache(maxsize=128) here is a 12 GB footgun."""
    from twin.metrics.common import PAIRWISE_CACHE_MAXSIZE, _pairwise_cache

    clear_pairwise_cache()
    rng = np.random.default_rng(5)
    for _ in range(PAIRWISE_CACHE_MAXSIZE + 3):
        pairwise_distances(rng.normal(size=(15, 2)))
    assert len(_pairwise_cache) == PAIRWISE_CACHE_MAXSIZE
    assert PAIRWISE_CACHE_MAXSIZE <= 2


def test_the_cache_can_be_bypassed(karate_random_layout):
    from twin.metrics.common import _pairwise_cache

    clear_pairwise_cache()
    pairwise_distances(karate_random_layout, cache=False)
    assert not _pairwise_cache


# --------------------------------------------------------------------------- #
#                  The leave-one-out cache (spec §7 step 4)                    #
# --------------------------------------------------------------------------- #


def test_the_loo_cache_never_serves_a_small_k_out_of_a_large_one(karate_coincident_layout):
    """The rule the whole cache is built around, on the fixture that can see it break.

    Step 3 established that slicing a $k = 16$ query down to 5 is **not** the same as querying
    5, when the layout has exact duplicates: which of nine coincident points count as the five
    nearest depends on how many were asked for. So a cache keyed on the digest alone — serving
    ``[:, :5]`` of a stored 16 — would make ``knn_recall(X, G, 5)`` depend on which metric ran
    first. This asserts the value is the standalone one, in either order.
    """
    clear_loo_cache()
    standalone = _loo_neighbors(karate_coincident_layout, 5)

    clear_loo_cache()
    _loo_neighbors(karate_coincident_layout, 16)
    after_16 = _loo_neighbors(karate_coincident_layout, 5)

    assert np.array_equal(after_16[0], standalone[0])
    assert np.array_equal(after_16[1], standalone[1])


def test_slicing_would_actually_have_changed_the_answer(karate_coincident_layout):
    """Without this, the test above passes on a fixture where the hazard cannot occur.

    Eleven of karate's 34 rows disagree: the nine coincident points, and the two whose own
    neighbourhoods reach into the coincident block. The *distances* agree — they are zero either
    way — so only the neighbour identities move, which is exactly what a recall computation
    reads.
    """
    clear_loo_cache()
    _, idx16 = _loo_neighbors(karate_coincident_layout, 16)
    _, idx5 = _loo_neighbors(karate_coincident_layout, 5)
    assert not np.array_equal(idx16[:, :5], idx5)


def test_the_loo_cache_returns_the_same_arrays_on_a_hit(karate_random_layout):
    clear_loo_cache()
    first = _loo_neighbors(karate_random_layout, 5)
    assert _loo_neighbors(karate_random_layout, 5) is first


def test_the_loo_cache_is_keyed_on_content_not_identity():
    """As the pairwise cache is: ids are reused after a gc, digests are not."""
    clear_loo_cache()
    X = np.random.default_rng(6).normal(size=(20, 2))
    first = _loo_neighbors(X, 3)[0].copy()

    X[0] += 5.0  # same object, different content
    assert not np.allclose(_loo_neighbors(X, 3)[0], first)
    assert _loo_neighbors(X.copy(), 3) is _loo_neighbors(X, 3)


def test_the_loo_cache_keeps_k_max_in_the_key(karate_random_layout):
    """Two entries for two k, not one entry reused — that is the no-slicing rule in the key."""
    from twin.metrics.common import _loo_cache

    clear_loo_cache()
    _loo_neighbors(karate_random_layout, 5)
    _loo_neighbors(karate_random_layout, 16)
    assert len(_loo_cache) == 2


def test_the_loo_cache_has_its_own_capacity():
    """Not `PAIRWISE_CACHE_MAXSIZE`: 1.6 MB per entry is not 100 MB per entry, and one layout
    is queried at several `k_max` within a single case."""
    from twin.metrics.common import (
        LOO_CACHE_MAXSIZE,
        PAIRWISE_CACHE_MAXSIZE,
        _loo_cache,
    )

    assert LOO_CACHE_MAXSIZE > PAIRWISE_CACHE_MAXSIZE

    clear_loo_cache()
    rng = np.random.default_rng(7)
    for _ in range(LOO_CACHE_MAXSIZE + 3):
        _loo_neighbors(rng.normal(size=(15, 2)), 3)
    assert len(_loo_cache) == LOO_CACHE_MAXSIZE


def test_the_loo_cache_can_be_bypassed(karate_random_layout):
    from twin.metrics.common import _loo_cache

    clear_loo_cache()
    _loo_neighbors(karate_random_layout, 5, cache=False)
    assert not _loo_cache


def test_caching_does_not_change_the_neighbourhood(karate_coincident_layout):
    """Including on the duplicate-heavy layout, where the query itself is order-sensitive."""
    clear_loo_cache()
    cached = _loo_neighbors(karate_coincident_layout, 7)
    uncached = _loo_neighbors(karate_coincident_layout, 7, cache=False)
    assert np.array_equal(cached[0], uncached[0])
    assert np.array_equal(cached[1], uncached[1])
