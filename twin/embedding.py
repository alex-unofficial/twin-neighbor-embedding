"""
Author: Dimitris Floros
Date: 2025-01-31
Description: This module contains classes for graph embedding methods.
"""

from dataclasses import dataclass
import networkx as nx
import numpy as np
import scipy.sparse as sp
from sgtsnepi import sgtsnepi
from fa2 import ForceAtlas2

import twin.graphmatrix as gm

@dataclass
class GraphEmbeddingMethod:
    """Base class for graph embedding methods.

    Attributes:
        method_name (str): The name of the embedding method.
    """

    method_name: str

    def embed(self, graph, y0=None):
        """Placeholder method for embedding. Subclasses should override this.

        Args:
            graph (networkx.Graph): The input graph to embed.

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError("Each embedding method must implement its own 'embed' function.")

    def init_embedding(self, n, d):
        """Placeholder method for initializing the embedding. Subclasses should override this.

        Args:
            n (int): The number of nodes in the graph.
            d (int): The dimensionality of the embedding.

        Returns:
            np.ndarray: The initial embedding (all zeros if not overridden).
        """
        return np.zeros((d, n))


@dataclass
class SpringLayout(GraphEmbeddingMethod):
    """Spring layout embedding.

    Attributes:
        method_name (str): The name of the embedding method.
        seed (int): Seed for the random number generator.
    """

    method_name: str = "Spring Layout"  # Override the method name
    seed: int = 0  # Seed for the random number generator
    d: int = 2  # Dimensionality of the embedding
    # --- add any additional parameters here ---

    def init_embedding(self, n, d):
        """Initialize the embedding with random values.

        Args:
            n (int): The number of nodes in the graph.
            d (int): The dimensionality of the embedding.

        Returns:
            np.ndarray: The initial embedding.
        """
        rng = np.random.default_rng(self.seed)
        return rng.uniform(size=(d, n))

    def embed(self, G: gm.GraphMatrixMethod, y0=None):
        """Embed the graph using a spring layout.

        Args:
            graph (networkx.Graph): The input graph to embed.

        Returns:
            dict: A dictionary of positions keyed by node.
        """
        # form graph from adjacency matrix sp
        adj = G.get_adj()
        graph = nx.from_scipy_sparse_array(adj)

        if y0 is None:
            pos = None
        else:
            pos = {i: y0[:, i] for i in range(adj.shape[0])}

        Y_spring = nx.spring_layout(graph, seed=self.seed, pos=pos, dim=self.d)

        Y = np.zeros((self.d, adj.shape[0]))
        for node, pos in Y_spring.items():
            Y[:, node] = pos

        return Y


@dataclass
class KamadaKawaiLayout(GraphEmbeddingMethod):
    """Kamada-Kawai embedding.

    Attributes:
        method_name (str): The name of the embedding method.
        mode (str): The type of edge weights. Can be 'distance' or 'similarity'.
        seed (int): Seed for the random number generator.
    """

    method_name: str = "Kamada-Kawai"  # Override the method name
    mode: str = 'distance' # Edge weight mode ('distance' or 'similarity')
    seed: int = 0  # Seed for the random number generator
    d: int = 2  # Dimensionality of the embedding
    # --- add any additional parameters here ---

    def init_embedding(self, n, d):
        """Initialize the embedding with random values.

        Args:
            n (int): The number of nodes in the graph.
            d (int): The dimensionality of the embedding.

        Returns:
            np.ndarray: The initial embedding.
        """
        rng = np.random.default_rng(self.seed)
        return rng.uniform(size=(d, n))

    def embed(self, G: gm.GraphMatrixMethod, y0=None):
        """Embed the graph using Kamada-Kawai layout.

        Args:
            graph (networkx.Graph): The input graph to embed.

        Returns:
            dict: A dictionary of positions keyed by node.
        """
        # form graph from adjacency matrix sp
        adj = G.get_adj()

        if self.mode == 'distance':
            dist_adj = adj
        elif self.mode == 'similarity':
            # Normalize weights for row-stochastic input matrix
            row_sums = np.array(adj.sum(axis=1)).ravel()

            inv_row_sums = np.reciprocal(row_sums, out=None, where=row_sums != 0)
            inv_row_sums[row_sums == 0] = 0.0

            D_inv = sp.diags(inv_row_sums)
            norm_adj = D_inv @ adj

            # Transform to distance via information function
            dist_adj = norm_adj.copy()
            dist_adj.data = -np.log(dist_adj.data)

        else:
            raise ValueError("mode must be 'distance' or 'similarity'")

        graph = nx.from_scipy_sparse_array(dist_adj)

        if y0 is None:
            pos = None
        else:
            pos = {i: y0[:, i] for i in range(adj.shape[0])}

        Y_kk = nx.kamada_kawai_layout(graph, pos=pos, dim=self.d)

        Y = np.zeros((self.d, adj.shape[0]))
        for node, pos in Y_kk.items():
            Y[:, node] = pos

        return Y


@dataclass
class YifanHuLayout(GraphEmbeddingMethod):
    """Yifan Hu embedding.

    Attributes:
        method_name (str): The name of the embedding method.
        seed (int): Seed for the random number generator.
    """

    method_name: str = "Yifan Hu"  # Override the method name
    seed: int = 0  # Seed for the random number generator
    d: int = 2  # Dimensionality of the embedding
    # --- add any additional parameters here ---

    def embed(self, G: gm.GraphMatrixMethod, y0=None):
        """Embed the graph using Yifan Hu layout.

        Args:
            graph (networkx.Graph): The input graph to embed.

        Returns:
            dict: A dictionary of positions keyed by node.
        """
        # form graph from adjacency matrix sp
        adj = G.get_adj()
        graph = nx.from_scipy_sparse_array(adj)

        if y0 is None:
            pos = None
        else:
            pos = {i: y0[:, i] for i in range(adj.shape[0])}

        Y_sfdp = nx.nx_pydot.graphviz_layout(graph, prog='sfdp')

        Y = np.zeros((self.d, adj.shape[0]))
        for node, pos in Y_sfdp.items():
            Y[:, node] = pos

        return Y


@dataclass
class NeatoLayout(GraphEmbeddingMethod):
    """Stress Majorization (MDS based) embedding.

    Attributes:
        method_name (str): The name of the embedding method.
        mode (str): The type of edge weights. Can be 'distance' or 'similarity'.
    """

    method_name: str = "NEATO"  # Override the method name
    seed: int = 0  # Seed for the random number generator
    d: int = 2  # Dimensionality of the embedding
    mode: str = 'distance' # Edge weight mode ('distance' or 'similarity')
    # --- add any additional parameters here ---

    def embed(self, G: gm.GraphMatrixMethod, y0=None):
        """Embed the graph using NEATO layout.

        Args:
            graph (networkx.Graph): The input graph to embed.

        Returns:
            dict: A dictionary of positions keyed by node.
        """
        # form graph from adjacency matrix sp
        adj = G.get_adj()

        if self.mode == 'distance':
            dist_adj = adj
        elif self.mode == 'similarity':
            # Normalize weights for row-stochastic input matrix
            row_sums = np.array(adj.sum(axis=1)).ravel()

            inv_row_sums = np.reciprocal(row_sums, out=None, where=row_sums != 0)
            inv_row_sums[row_sums == 0] = 0.0

            D_inv = sp.diags(inv_row_sums)
            norm_adj = D_inv @ adj

            # Transform to distance via information function
            dist_adj = norm_adj.copy()
            dist_adj.data = -np.log(dist_adj.data)
        else:
            raise ValueError("mode must be 'distance' or 'similarity'")

        graph = nx.from_scipy_sparse_array(dist_adj)

        if y0 is None:
            pos = None
        else:
            pos = {i: y0[:, i] for i in range(adj.shape[0])}

        Y_neato = nx.nx_pydot.graphviz_layout(graph, prog='neato')

        Y = np.zeros((self.d, adj.shape[0]))
        for node, pos in Y_neato.items():
            Y[:, node] = pos

        return Y


@dataclass
class ForceAtlas2Layout(GraphEmbeddingMethod):
    """ForceAtlas2 embedding.

    Attributes:
        method_name (str): The name of the embedding method.
        seed (int): Seed for the random number generator.
        d (int): Dimensionality of the embedding.
        linlog (bool): Enable ForceAtlas2 LinLog mode
    """

    method_name: str = "ForceAtlas2"  # Override the method name
    seed: int = 0  # Seed for the random number generator
    d: int = 2  # Dimensionality of the embedding
    linlog: bool = False # ForceAtlas2 LinLog mode
    verbose: bool = False # Verbose layout

    def init_embedding(self, n, d):
        """Initialize the embedding with random values.

        Args:
            n (int): The number of nodes in the graph.
            d (int): The dimensionality of the embedding.

        Returns:
            np.ndarray: The initial embedding.
        """
        rng = np.random.default_rng(self.seed)
        return rng.normal(size=(d, n), loc=0, scale=1e-4)

    def embed(self, G: gm.GraphMatrixMethod, y0=None):
        """Embed the graph using ForceAtlas2.

        Args:
            graph (networkx.Graph): The input graph to embed.

        Returns:
            np.ndarray: The embedded coordinates.
        """

        fa2 = ForceAtlas2(
            seed=self.seed,
            dim=self.d,
            linLogMode=self.linlog,
            verbose=self.verbose,
        )

        adj = G.get_adj()

        if y0 is None:
            pos = None
        else:
            pos = y0.T

        Y_fa2 = fa2.forceatlas2(adj, pos)

        return np.array(Y_fa2).T


@dataclass
class SGtSNELayout(GraphEmbeddingMethod):
    """SGtSNE layout embedding.

    Attributes:
        method_name (str): The name of the embedding method.
        lambda_par (float): Lambda parameter for SGtSNE.
        seed (int): Seed for the random number generator.
        d (int): Dimensionality of the embedding.
    """

    method_name: str = "SGtSNE Layout"  # Override the method name
    lambda_par: float = 1.0  # Lambda parameter for SGtSNE
    seed: int = 0  # Seed for the random number generator
    d: int = 2  # Dimensionality of the embedding
    silent: bool = True  # Whether to suppress output from SGtSNE
    run_exact: bool = False  # Whether to run the exact version of SGtSNE

    def init_embedding(self, n, d):
        """Initialize the embedding with random values.

        Args:
            n (int): The number of nodes in the graph.
            d (int): The dimensionality of the embedding.

        Returns:
            np.ndarray: The initial embedding.
        """
        rng = np.random.default_rng(self.seed)
        return rng.normal(size=(d, n), loc=0, scale=1e-4)

    def embed(self, G: gm.GraphMatrixMethod, y0=None):
        """Embed the graph using SGtSNE.

        Args:
            graph (networkx.Graph): The input graph to embed.

        Returns:
            np.ndarray: The embedded coordinates.
        """
        adj = G.get_adj()
        rng = np.random.default_rng(self.seed)
        if y0 is None:
            y0 = rng.normal(size=(self.d, adj.shape[0]), loc=0, scale=1e-3)

        return sgtsnepi(
            adj,
            d=self.d,
            lambda_par=self.lambda_par,
            silent=self.silent,
            y0=y0,
            run_exact=self.run_exact,
        )


@dataclass
class SpectralLayout(GraphEmbeddingMethod):
    """Spectral layout embedding.

    Attributes:
        method_name (str): The name of the embedding method.
        d (int): Dimensionality of the embedding.
    """

    method_name: str = "Spectral Layout"
    d: int = 2
    normed: bool = True
    seed: int = 0

    def embed(self, G: gm.GraphMatrixMethod, y0=None):
        """Embed the graph using spectral layout.

        Args:
            graph (networkx.Graph): The input graph to embed.

        Returns:
            np.ndarray: The embedded coordinates.
        """
        adj = G.get_adj()
        L = sp.csgraph.laplacian(adj, normed=self.normed)
        eigvals, eigvecs = sp.linalg.eigsh(L, k=self.d + 1, which="SM")
        eigvecs = eigvecs[:, 1:].T

        return eigvecs
