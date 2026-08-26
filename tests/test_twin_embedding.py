r"""The TNE transform itself — `twin_adjacency`, the method under study.

**Why this file exists.** Until it did, `twin_adjacency` had no test at all, and the gap was
invisible: every test that touched the twin representation compared `twin.layoutgraph` against
`twin.graphmatrix`, and *both call this same function*. That comparison verifies the wrapper and
is blind to the algebra inside it. A mutation audit confirmed the consequence — forcing
``alpha`` to 0.5, or the polynomial order to 1, regardless of what the caller asked for, left the
entire suite green while changing $A_{tne}$ by max\|diff\| 22.7 and 34.7 on `karate_weighted`.

That is the worst place in the package to have a blind spot. $\alpha$ and the polynomial order
are the method's own knobs; §4.2's $\lambda$ sweep varies a *third* parameter on top of them;
and spec §0.5 records that thesis §4-§7 are placeholders, so there is no prose to check the
implementation against either. CLAUDE.md's "treat `twin_embedding.py` as given" is a reason to
**pin** it rather than to leave it unpinned: pinning does not change given code, it detects
whether given code has changed.

**What is pinned, and why not a digest.** The recursion

.. code-block:: text

    twin_adj = alpha * A                      # k = 1
    twin_adj = alpha * A @ (twin_adj + I)     # each further k

telescopes to a closed form,

.. math::

    A_{tne}(\alpha, n) = \sum_{k=1}^{n} \alpha^{k} A^{k},
    \qquad A = \begin{bmatrix} 0 & B^\top \\ B & 0\end{bmatrix}

so the tests compare against an **independent dense reimplementation** rather than against a
stored digest. That is strictly stronger: a digest says "this is what it did last time", whereas
this says "this is the matrix polynomial it is supposed to be", and it fails on any change to
either parameter's role rather than only on a change of output.
"""

import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

from twin.datasets import load_dataset
from twin.layoutgraph import vertex_graph
from twin.twin_embedding import twin_adjacency


# --------------------------------------------------------------------------- #
#                        An independent implementation                         #
# --------------------------------------------------------------------------- #


def blocks(B) -> np.ndarray:
    r"""$A = [[0, B^\top], [B, 0]]$, dense, rows **[edges (m); vertices (n)]**."""
    B = np.asarray(B.todense() if sp.issparse(B) else B, dtype=float)
    n_vertices, n_edges = B.shape
    return np.block(
        [
            [np.zeros((n_edges, n_edges)), B.T],
            [B, np.zeros((n_vertices, n_vertices))],
        ]
    )


def polynomial(B, alpha: float, n: int) -> np.ndarray:
    r"""$\sum_{k=1}^{n} \alpha^{k} A^{k}$ — the closed form of the recursion, written out.

    Deliberately not a refactor of the implementation: dense, direct, and slow. Its only job is
    to be a second opinion, so sharing code with the thing it checks would defeat it.
    """
    A = blocks(B)
    return sum(alpha**k * np.linalg.matrix_power(A, k) for k in range(1, n + 1))


@pytest.fixture
def path3_incidence() -> np.ndarray:
    """Path 0-1-2: three vertices, two edges, entirely hand-computable."""
    return nx.incidence_matrix(nx.path_graph(3)).toarray()


@pytest.fixture
def karate_incidence():
    """The weighted incidence the representation layer actually feeds the transform."""
    return vertex_graph(load_dataset("karate_weighted")).incidence_weighted


# --------------------------------------------------------------------------- #
#                            The block structure                               #
# --------------------------------------------------------------------------- #


def test_at_order_one_the_matrix_is_alpha_times_the_incidence_blocks(path3_incidence):
    r"""$n = 1$ is exactly $\alpha A$, written out by hand.

    Path 0-1-2 has edges $e_0 = (0,1)$ and $e_1 = (1,2)$, so
    $B = [[1,0],[1,1],[0,1]]$ and the twin matrix is 5x5 with rows
    $[e_0, e_1, v_0, v_1, v_2]$.
    """
    expected = 0.5 * np.array(
        [
            #  e0   e1   v0   v1   v2
            [0.0, 0.0, 1.0, 1.0, 0.0],  # e0 touches v0, v1
            [0.0, 0.0, 0.0, 1.0, 1.0],  # e1 touches v1, v2
            [1.0, 0.0, 0.0, 0.0, 0.0],  # v0 touches e0
            [1.0, 1.0, 0.0, 0.0, 0.0],  # v1 touches e0, e1
            [0.0, 1.0, 0.0, 0.0, 0.0],  # v2 touches e1
        ]
    )
    got = twin_adjacency(sp.csr_array(path3_incidence), alpha=0.5, n=1).toarray()
    np.testing.assert_allclose(got, expected, atol=1e-15)


def test_the_row_ordering_is_edges_then_vertices(path3_incidence):
    """CLAUDE.md's block-ordering convention, asserted on the matrix rather than on a slice.

    `test_layout_graph.py` checks the same thing through `vertex_and_edge_points`, which is the
    *consumer* of this convention. If both the matrix and the slicing were transposed together
    that test would still pass, so the convention is pinned at its source here.
    """
    A = twin_adjacency(sp.csr_array(path3_incidence), alpha=0.9, n=1).toarray()
    n_vertices, n_edges = path3_incidence.shape

    assert A.shape == (n_edges + n_vertices, n_edges + n_vertices)
    # The two diagonal blocks are empty: edges never touch edges, vertices never vertices.
    assert not A[:n_edges, :n_edges].any()
    assert not A[n_edges:, n_edges:].any()
    # ... and the off-diagonal blocks are transposes of one another.
    np.testing.assert_allclose(A[:n_edges, n_edges:], A[n_edges:, :n_edges].T)


def test_the_matrix_is_symmetric(karate_incidence):
    """It is handed to SG-*t*-SNE as an affinity, which is undirected."""
    A = twin_adjacency(karate_incidence, alpha=0.85, n=2).toarray()
    np.testing.assert_allclose(A, A.T, atol=1e-12)


# --------------------------------------------------------------------------- #
#                    Against the closed form (the real pin)                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.85, 1.0])
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_the_recursion_equals_the_matrix_polynomial(karate_incidence, alpha, order):
    r"""$A_{tne} = \sum_{k=1}^{n}\alpha^k A^k$, against a dense reimplementation.

    Sixteen combinations, so neither parameter can be ignored, clamped or swapped without a
    failure — which is precisely what the mutation audit found the suite could not detect.
    """
    got = twin_adjacency(karate_incidence, alpha, n=order).toarray()
    np.testing.assert_allclose(got, polynomial(karate_incidence, alpha, order), atol=1e-12)


def test_the_polynomial_holds_on_an_unweighted_graph(path3_incidence):
    """The weighted incidence is the production path; the unweighted one is the readable case."""
    B = sp.csr_array(path3_incidence)
    for alpha, order in ((0.85, 2), (0.4, 3)):
        np.testing.assert_allclose(
            twin_adjacency(B, alpha, n=order).toarray(),
            polynomial(path3_incidence, alpha, order),
            atol=1e-14,
        )


# --------------------------------------------------------------------------- #
#              Both parameters are read (the mutations that escaped)           #
# --------------------------------------------------------------------------- #


def test_alpha_is_actually_read(karate_incidence):
    """A mutant that ignored `alpha` and always used 0.5 passed all 609 tests.

    At order 1 the dependence is exact — the matrix *is* $\\alpha A$ — so this pins the
    coefficient rather than merely observing that something moved.
    """
    at_half = twin_adjacency(karate_incidence, 0.5, n=1).toarray()
    at_quarter = twin_adjacency(karate_incidence, 0.25, n=1).toarray()

    assert not np.allclose(at_half, at_quarter)
    np.testing.assert_allclose(at_quarter * 2.0, at_half, atol=1e-12)

    # At higher order the dependence is polynomial, so only the inequality is asserted.
    assert not np.allclose(
        twin_adjacency(karate_incidence, 0.85, n=2).toarray(),
        twin_adjacency(karate_incidence, 0.5, n=2).toarray(),
    )


def test_the_polynomial_order_is_actually_read(karate_incidence):
    r"""A mutant that ignored the order and always used 1 passed all 609 tests.

    Successive orders differ by exactly one term, $\alpha^{n} A^{n}$, so this pins *what* the
    extra iteration adds and not just that the answer changed.
    """
    alpha = 0.6
    A = blocks(karate_incidence)
    for order in (1, 2, 3):
        lower = twin_adjacency(karate_incidence, alpha, n=order).toarray()
        higher = twin_adjacency(karate_incidence, alpha, n=order + 1).toarray()
        assert not np.allclose(lower, higher)
        np.testing.assert_allclose(
            higher - lower,
            alpha ** (order + 1) * np.linalg.matrix_power(A, order + 1),
            atol=1e-12,
        )


def test_the_order_is_a_reach_in_the_twin_graph_not_in_the_original(path3_incidence):
    r"""What the order *means*, which is not what it looks like it means.

    On the path 0-1-2, $v_0$ and $v_2$ are two hops apart **in $G$** — so the tempting reading
    is that order 2 relates them. It does not. The twin graph is bipartite between edges and
    vertices, so a vertex-to-vertex walk alternates $v \to e \to v$ and the shortest $v_0$-$v_2$
    walk is *four* steps: $v_0, e_0, v_1, e_1, v_2$. Measured: the entry is 0 at orders 1, 2 and
    3 and first non-zero at order 4.

    Worth a test of its own because it is the parameter's actual semantics, and because getting
    it wrong is silent — every closed-form test above passes under either reading. A default of
    ``k = 2`` therefore couples each vertex to its **incident edges' other endpoints via one
    intermediate edge**, i.e. one hop of the line graph, not two hops of $G$.
    """
    n_edges = path3_incidence.shape[1]
    v0, v2 = n_edges + 0, n_edges + 2
    B = sp.csr_array(path3_incidence)

    for order in (1, 2, 3):
        assert twin_adjacency(B, 0.85, n=order).toarray()[v0, v2] == 0.0, order
    assert twin_adjacency(B, 0.85, n=4).toarray()[v0, v2] > 0.0

    # The bipartite parity, stated directly: odd orders relate edges to vertices, even orders
    # relate like to like. At order 1 no vertex-vertex pair is related at all.
    order1 = twin_adjacency(B, 0.85, n=1).toarray()
    assert not order1[n_edges:, n_edges:].any()
    assert not order1[:n_edges, :n_edges].any()


def test_an_order_below_one_is_refused(path3_incidence):
    """``assert n >= 1`` in the implementation — and asserts vanish under ``python -O``.

    Pinned so that the refusal is a tested behaviour rather than a debugging aid, since a
    silently-accepted ``n = 0`` would return an all-zero affinity and lay out noise.
    """
    with pytest.raises(AssertionError):
        twin_adjacency(sp.csr_array(path3_incidence), 0.85, n=0)


def test_the_defaults_are_the_documented_ones(karate_incidence):
    """`alpha=0.85`, `n=2` — what every call site relies on and what the figures were made with."""
    np.testing.assert_allclose(
        twin_adjacency(karate_incidence).toarray(),
        twin_adjacency(karate_incidence, 0.85, n=2).toarray(),
        atol=1e-15,
    )


# --------------------------------------------------------------------------- #
#                       The wrapper reaches the transform                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("alpha", "order"), [(0.85, 2), (0.4, 3)])
def test_twin_graph_passes_its_parameters_through(alpha, order):
    """`twin_graph(vg, alpha=..., k=...)` must reach `twin_adjacency`'s two arguments.

    The layer above was the one that *was* tested, but only against `graphmatrix`, which calls
    the same function — so a parameter dropped in either wrapper was invisible. Here the
    reference is the closed form, so this catches a drop in the wrapper too.
    """
    from twin.layoutgraph import twin_graph

    vertex = vertex_graph(load_dataset("karate_weighted"))
    got = twin_graph(vertex, alpha=alpha, k=order).affinity

    expected = polynomial(vertex.incidence_weighted, alpha, order)
    np.fill_diagonal(expected, 0.0)  # `twin_graph` strips it — see the test below
    np.testing.assert_allclose(sp.csr_array(got).toarray(), expected, atol=1e-12)


def test_twin_graph_strips_the_diagonal_that_the_transform_produces():
    r"""$A^k$ has a non-zero diagonal for even $k$ — a node's own return walks — and
    `twin_graph` removes it before the matrix becomes an affinity.

    Not cosmetic: the diagonal is large (1.44 against off-diagonal entries around 0.7 on the
    3-vertex path at $\alpha = 0.85$, $k = 2$), and SG-*t*-SNE would read a self-affinity as a
    point being strongly attracted to itself. Pinned because the stripping happens in the
    wrapper while the diagonal is produced in the transform, so neither file's tests would
    otherwise cover the seam.
    """
    from twin.layoutgraph import twin_graph

    vertex = vertex_graph(load_dataset("karate_weighted"))
    raw = twin_adjacency(vertex.incidence_weighted, 0.85, n=2)
    assert raw.diagonal().any(), "the transform is expected to produce a diagonal at even order"

    assert not sp.csr_array(twin_graph(vertex).affinity).diagonal().any()
