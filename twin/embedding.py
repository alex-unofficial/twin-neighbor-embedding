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

import twin.graphmatrix as gm

@dataclass
class GraphEmbeddingMethod:
    """Base class for graph embedding methods.

    Attributes:
        method_name (str): The name of the embedding method.
    """
    method_name: str

    def embed(self, graph, y0 = None):
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
    method_name: str = "Spring Layout" # Override the method name
    seed: int = 0 # Seed for the random number generator
    d: int = 2 # Dimensionality of the embedding
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

    def embed(self, G: gm.GraphMatrixMethod, y0 = None):
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

        Y_spring = nx.spring_layout(graph, seed=self.seed, pos=pos, dim = self.d)

        Y = np.zeros( (self.d, adj.shape[0]) )
        for node, pos in Y_spring.items():
            Y[:, node] = pos

        return Y

@dataclass
class SGtSNELayout(GraphEmbeddingMethod):
    """SGtSNE layout embedding.

    Attributes:
        method_name (str): The name of the embedding method.
        lambda_par (float): Lambda parameter for SGtSNE.
        seed (int): Seed for the random number generator.
        d (int): Dimensionality of the embedding.
    """
    method_name: str = "SGtSNE Layout" # Override the method name
    lambda_par: float = 1.0 # Lambda parameter for SGtSNE
    seed: int = 0 # Seed for the random number generator
    d: int = 2 # Dimensionality of the embedding
    silent: bool = True # Whether to suppress output from SGtSNE
    run_exact: bool = False # Whether to run the exact version of SGtSNE

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

    def embed(self, G: gm.GraphMatrixMethod, y0 = None):
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

    def embed(self, G: gm.GraphMatrixMethod, y0 = None):
        """Embed the graph using spectral layout.

        Args:
            graph (networkx.Graph): The input graph to embed.

        Returns:
            np.ndarray: The embedded coordinates.
        """
        adj = G.get_adj()
        L = sp.csgraph.laplacian(adj, normed=self.normed)
        eigvals, eigvecs = sp.linalg.eigsh(L, k=self.d+1, which='SM')
        eigvecs = eigvecs[:, 1:].T

        return eigvecs
