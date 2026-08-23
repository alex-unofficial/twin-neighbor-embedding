"""Layout methods (spec §3.5).

Every method takes a `LayoutGraph` and **asks for the side it needs** rather than being handed
one by a runner that guessed::

    A = layout_graph.affinity     # SG-t-SNE, spectral, spring (FR), ForceAtlas2
    G = layout_graph.distance     # Kamada-Kawai, neato, Yifan Hu

The `mode="distance"|"similarity"` parameters are gone. That decision never belonged to the
layout class, and it is not a runner decision either — it is a property of the graph being laid
out. Half of bug §1.6 closes here; the other half is that the graph now knows which of its
sides is stored.

`EmbeddingSpec.expects` is kept as declared metadata for the manifest and the report, but the
runner no longer branches on it — so nothing keeps it honest except
``tests/test_embedding.py::test_every_method_touches_exactly_the_side_it_declares``, which hands
each method a graph that records which side was read.

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
    """Graphviz `sfdp`, a multilevel force method, over the **distance** form.

    Note:
        `y0` and `d` are ignored — spec §1.4, deferred to step 9.
    """

    method_name: str = "Yifan Hu"
    seed: int = 0
    d: int = 2

    def embed(self, layout_graph, y0=None):
        positions = nx.nx_pydot.graphviz_layout(layout_graph.distance, prog="sfdp")
        return _as_array(positions, layout_graph.n_points, self.d)


@dataclass
class NeatoLayout(GraphEmbeddingMethod):
    """Graphviz `neato` — stress majorization (MDS) over the **distance** form."""

    method_name: str = "NEATO"
    seed: int = 0
    d: int = 2

    def embed(self, layout_graph, y0=None):
        positions = nx.nx_pydot.graphviz_layout(layout_graph.distance, prog="neato")
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
        expects: which side it reads. Declared metadata for the manifest and the report — the
            runner does **not** branch on it, because nothing converts at the boundary any
            more. Since nothing depends on it, nothing keeps it honest either, so it is
            asserted by a test rather than trusted.
        stochastic: whether repeated runs at the same seed and `y0` can differ, which is what
            decides whether a case needs the seed axis (spec §4.1).
        has_lambda: whether it takes the SG-t-SNE rescaling parameter, i.e. whether the lambda
            sweep applies to it (spec §4.2).
    """

    cls: type
    expects: Literal["affinity", "distance"]
    stochastic: bool
    has_lambda: bool


EMBEDDING_REGISTRY: dict[str, EmbeddingSpec] = {
    "spring": EmbeddingSpec(SpringLayout, "affinity", stochastic=True, has_lambda=False),
    "yifanhu": EmbeddingSpec(YifanHuLayout, "distance", stochastic=False, has_lambda=False),
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
