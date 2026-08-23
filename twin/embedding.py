"""Layout methods (spec §3.5).

Every method takes a `LayoutGraph` and **asks for the form it needs** rather than being handed
one by a runner that guessed::

    C = layout_graph.combinatorial   # Yifan Hu (sfdp)
    A = layout_graph.affinity        # SG-t-SNE, spectral, spring (FR), ForceAtlas2
    G = layout_graph.distance        # Kamada-Kawai, neato

The `mode="distance"|"similarity"` parameters are gone. That decision never belonged to the
layout class, and it is not a runner decision either — it is a property of the graph being laid
out. Half of bug §1.6 closes here; the other half is that the graph now knows which of its
sides is stored.

**Reading the right form is not enough — it has to arrive meaning the right thing.** The two
graphviz backends go through `nx.nx_pydot`, which writes every edge attribute into the dot
source, and graphviz reads `weight` as a spring-strength multiplier and `len` as a desired edge
length. So a distance passed as `weight` is read as "far apart means pull harder". Both
graphviz methods therefore build their own input graph here rather than handing over
`.distance` directly.

`EmbeddingSpec.expects` is kept as declared metadata for the manifest and the report, but the
runner no longer branches on it — so nothing keeps it honest except
``tests/test_embedding.py::test_every_method_touches_exactly_the_form_it_declares``, which hands
each method a graph that records which form was read.

Original author: Dimitris Floros (2025-01-31).
"""

from dataclasses import dataclass
from typing import Literal

import networkx as nx
import numpy as np
import scipy.sparse as sp
from fa2 import ForceAtlas2
from sgtsnepi import sgtsnepi


@dataclass
class GraphEmbeddingMethod:
    """Base class for graph embedding methods.

    Attributes:
        method_name (str): The name of the embedding method.
    """

    method_name: str

    def embed(self, layout_graph, y0=None):
        """Lay out one `LayoutGraph`, returning ``(d, n_points)``.

        Args:
            layout_graph: a :class:`~twin.layoutgraph.LayoutGraph`, from which the method reads
                whichever of ``.affinity`` / ``.distance`` its `EmbeddingSpec` declares.
            y0: initial positions, ``(d, n_points)``, or None.
        """
        raise NotImplementedError("Each embedding method must implement its own 'embed' function.")

    def init_embedding(self, n, d):
        """Initial positions, ``(d, n)``. Zeros unless a subclass overrides."""
        return np.zeros((d, n))


def _positions(y0, n):
    """networkx's ``pos`` mapping from a ``(d, n)`` array, or None."""
    return None if y0 is None else {i: y0[:, i] for i in range(n)}


def _structure_only(layout_graph) -> nx.Graph:
    """The combinatorial form as a graph with **no edge attributes at all**.

    Deliberately attribute-free. `nx.nx_pydot` writes every edge attribute into the dot source,
    and graphviz reads `weight` as a spring-strength multiplier — so a graph that carries any
    valuation would silently hand one to a method that has no use for it.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(layout_graph.n_points))
    graph.add_edges_from(zip(*sp.triu(sp.csr_array(layout_graph.combinatorial), k=1).nonzero()))
    return graph


def _desired_lengths(layout_graph) -> nx.Graph:
    """The distance form as a graph carrying graphviz's ``len``, not ``weight``.

    This distinction is the whole of bug §1.6 in a second place. For `neato` and `sfdp`,
    graphviz's ``len`` is the *desired edge length* and ``weight`` is how strongly the edge is
    held to that length — so passing a distance as ``weight`` says "far apart means pull
    harder", which inverts it. `nx.nx_pydot` writes `weight=` for the `weight` attribute and
    nothing else, so the attribute has to be renamed on the way in.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(layout_graph.n_points))
    graph.add_edges_from(
        (u, v, {"len": float(d)}) for u, v, d in layout_graph.distance.edges(data="weight")
    )
    return graph


def _as_array(positions, n, d):
    """A networkx position mapping back into a ``(d, n)`` array."""
    Y = np.zeros((d, n))
    for node, position in positions.items():
        Y[:, node] = position
    return Y


@dataclass
class SpringLayout(GraphEmbeddingMethod):
    """Fruchterman-Reingold. Consumes the affinity: an edge weight is an attraction strength."""

    method_name: str = "Spring Layout"
    seed: int = 0
    d: int = 2

    def init_embedding(self, n, d):
        return np.random.default_rng(self.seed).uniform(size=(d, n))

    def embed(self, layout_graph, y0=None):
        n = layout_graph.n_points
        graph = nx.from_scipy_sparse_array(layout_graph.affinity)
        positions = nx.spring_layout(graph, seed=self.seed, pos=_positions(y0, n), dim=self.d)
        return _as_array(positions, n, self.d)


@dataclass
class KamadaKawaiLayout(GraphEmbeddingMethod):
    """Stress minimisation against the graph's **distance** form."""

    method_name: str = "Kamada-Kawai"
    seed: int = 0
    d: int = 2

    def init_embedding(self, n, d):
        return np.random.default_rng(self.seed).uniform(size=(d, n))

    def embed(self, layout_graph, y0=None):
        n = layout_graph.n_points
        positions = nx.kamada_kawai_layout(
            layout_graph.distance, pos=_positions(y0, n), dim=self.d
        )
        return _as_array(positions, n, self.d)


@dataclass
class YifanHuLayout(GraphEmbeddingMethod):
    """Graphviz `sfdp`, a multilevel force method, over the **combinatorial** form.

    Hu's method is combinatorial: attractive $d^2/K$ along edges, repulsive $-CK^2/d$ between
    all pairs, with $K$ a **global** constant. There is no per-edge target length in the
    formulation, so it is handed the structure and nothing else.

    Note:
        `y0` and `d` are ignored — spec §1.4, deferred to step 9.
    """

    method_name: str = "Yifan Hu"
    seed: int = 0
    d: int = 2

    def embed(self, layout_graph, y0=None):
        positions = nx.nx_pydot.graphviz_layout(_structure_only(layout_graph), prog="sfdp")
        return _as_array(positions, layout_graph.n_points, self.d)


@dataclass
class NeatoLayout(GraphEmbeddingMethod):
    """Graphviz `neato` — stress majorization (MDS) against the **distance** form.

    The distances go in as graphviz's ``len`` (desired edge length), not ``weight`` (how
    strongly the edge is held to that length). See :func:`_desired_lengths`.
    """

    method_name: str = "NEATO"
    seed: int = 0
    d: int = 2

    def embed(self, layout_graph, y0=None):
        positions = nx.nx_pydot.graphviz_layout(_desired_lengths(layout_graph), prog="neato")
        return _as_array(positions, layout_graph.n_points, self.d)


@dataclass
class ForceAtlas2Layout(GraphEmbeddingMethod):
    """ForceAtlas2 over the affinity matrix.

    Attributes:
        linlog (bool): Enable ForceAtlas2 LinLog mode.
    """

    method_name: str = "ForceAtlas2"
    seed: int = 0
    d: int = 2
    linlog: bool = False
    verbose: bool = False

    def init_embedding(self, n, d):
        return np.random.default_rng(self.seed).normal(size=(d, n), loc=0, scale=1e-4)

    def embed(self, layout_graph, y0=None):
        fa2 = ForceAtlas2(
            seed=self.seed, dim=self.d, linLogMode=self.linlog, verbose=self.verbose
        )
        Y = fa2.forceatlas2(layout_graph.affinity, None if y0 is None else y0.T)
        return np.array(Y).T


@dataclass
class SGtSNELayout(GraphEmbeddingMethod):
    """SG-t-SNE-Pi. The primary backend; consumes the affinity.

    Attributes:
        lambda_par (float): the rescaling parameter; per-row exponents are found so that each
            row's rescaled affinities sum to it.
    """

    method_name: str = "SGtSNE Layout"
    lambda_par: float = 1.0
    seed: int = 0
    d: int = 2
    silent: bool = True
    run_exact: bool = False

    def init_embedding(self, n, d):
        return np.random.default_rng(self.seed).normal(size=(d, n), loc=0, scale=1e-4)

    def embed(self, layout_graph, y0=None):
        affinity = layout_graph.affinity
        if y0 is None:
            y0 = np.random.default_rng(self.seed).normal(
                size=(self.d, layout_graph.n_points), loc=0, scale=1e-3
            )

        return sgtsnepi(
            affinity,
            d=self.d,
            lambda_par=self.lambda_par,
            silent=self.silent,
            y0=y0,
            run_exact=self.run_exact,
        )


@dataclass
class SpectralLayout(GraphEmbeddingMethod):
    """Eigenvectors of the graph Laplacian of the affinity matrix."""

    method_name: str = "Spectral Layout"
    d: int = 2
    normed: bool = True
    seed: int = 0

    def embed(self, layout_graph, y0=None):
        L = sp.csgraph.laplacian(layout_graph.affinity, normed=self.normed)
        _, eigvecs = sp.linalg.eigsh(L, k=self.d + 1, which="SM")
        return eigvecs[:, 1:].T


@dataclass(frozen=True)
class EmbeddingSpec:
    """What a layout method is, declaratively (spec §3.5).

    Attributes:
        cls: the method class.
        expects: which of the graph's three forms it reads — ``combinatorial`` (the structure
            alone), ``affinity`` (larger means more strongly coupled) or ``distance`` (larger
            means further apart). Declared metadata for the manifest and the report — the
            runner does **not** branch on it, because nothing converts at the boundary any
            more. Since nothing depends on it, nothing keeps it honest either, so it is
            asserted by a test rather than trusted.
        stochastic: whether repeated runs at the same seed and `y0` can differ, which is what
            decides whether a case needs the seed axis (spec §4.1).
        has_lambda: whether it takes the SG-t-SNE rescaling parameter, i.e. whether the lambda
            sweep applies to it (spec §4.2).
    """

    cls: type
    expects: Literal["combinatorial", "affinity", "distance"]
    stochastic: bool
    has_lambda: bool


EMBEDDING_REGISTRY: dict[str, EmbeddingSpec] = {
    "spring": EmbeddingSpec(SpringLayout, "affinity", stochastic=True, has_lambda=False),
    "yifanhu": EmbeddingSpec(
        YifanHuLayout, "combinatorial", stochastic=False, has_lambda=False
    ),
    "forceatlas2": EmbeddingSpec(
        ForceAtlas2Layout, "affinity", stochastic=True, has_lambda=False
    ),
    "kamada-kawai": EmbeddingSpec(
        KamadaKawaiLayout, "distance", stochastic=False, has_lambda=False
    ),
    "neato": EmbeddingSpec(NeatoLayout, "distance", stochastic=False, has_lambda=False),
    "spectral": EmbeddingSpec(SpectralLayout, "affinity", stochastic=False, has_lambda=False),
    "sgtsne": EmbeddingSpec(SGtSNELayout, "affinity", stochastic=True, has_lambda=True),
}
