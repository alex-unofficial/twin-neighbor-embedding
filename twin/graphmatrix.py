"""
Author: Dimitris Floros
Date: 2025-01-31
Description: This module contains classes for extracting graph matrices.
"""

import networkx as nx
from dataclasses import dataclass
import rasterio
import scipy.sparse as sp
import numpy as np
from twin.twin_embedding import twin_adjacency

def compute_slopes(B, Xv):
    """Compute the slopes of the lines connecting the vertices of the graph.

    Args:
        B (scipy.sparse.cscmatrix): The incidence matrix of the graph
        Xv (numpy.ndarray): The vertex embeddings of the graph.

    Returns:
        numpy.ndarray: The slopes of the lines connecting the vertices.
    """
    
    s = np.zeros(B.shape[1])
    for i in range(B.shape[1]):
        # get the indices of the vertices connected by the edge
        idx = B[:,i].nonzero()[0]
        # get the coordinates of the vertices
        x1, y1 = Xv[:,idx[0]]
        x2, y2 = Xv[:,idx[1]]
        # compute the slope
        s[i] = (y2 - y1) / (x2 - x1)

    return s

class GraphMatrixMethod:
    """Base class for graph preprocessing."""
    adj: sp.csc_matrix = None    
    incidence: sp.csc_matrix = None
    incidence_weighted: sp.csc_matrix = None
    normalize: bool = False

    def __init__(self, graph: nx.Graph):
        """Initialize the GraphMatrixMethod with a graph."""
        self.adj = self.preprocess(graph)
    

    def get_adj(self):
        return self.adj
    
    def get_incidence(self):
        return self.incidence

    def preprocess(self, graph: nx.Graph):
        """Placeholder method. Each subclass must implement this.

        Args:
            graph (networkx.Graph): The input graph to preprocess.

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError("Each preprocessing method must implement 'preprocess'.")
    
    def vertex_and_edge_embeddings(self, adj: sp.csc_matrix):
        """Placeholder method. Each subclass must implement this.

        Args:
            adj (scipy.sparse.csc_matrix): The adjacency matrix of the graph.

        Raises:
            NotImplementedError: If the method is not overridden by a subclass.
        """
        raise NotImplementedError("Each preprocessing method must implement 'vertex_and_edge_embeddings'.")

@dataclass
class VertexMatrix(GraphMatrixMethod):
    """Preprocessing that operates on vertices (e.g., feature extraction)."""
    adj: sp.csc_matrix = None
    incidence: sp.csc_matrix = None
    incidence_weighted: sp.csc_matrix = None
    normalize: bool = False

    def __init__(self, graph: nx.Graph, normalize = False):
        """Initialize the GraphMatrixMethod with a graph."""
        # form the incidence matrix
        self.incidence = nx.incidence_matrix(graph, oriented=False).tocsc()
        self.incidence_weighted = nx.incidence_matrix(graph, oriented=False, weight='weight').tocsc().sqrt()
        self.normalize = normalize

        # normalize
        if self.normalize:
            self.incidence_weighted = nx.incidence_matrix(graph, oriented=False, weight='weight').tocsc()
            vertex_degrees_sqrt = np.sqrt( self.incidence_weighted.sum(axis=1) ).reshape(-1,1)
            self.incidence_weighted = self.incidence_weighted.sqrt() / vertex_degrees_sqrt
        else:
            self.incidence_weighted = nx.incidence_matrix(graph, oriented=False, weight='weight').tocsc().sqrt()
        
        # build the adjacency matrix as the product of the incidence matrix with its transpose
        adj_vertex = self.incidence_weighted @ self.incidence_weighted.T

        # remove self-loops
        adj_vertex = adj_vertex - sp.diags(adj_vertex.diagonal())

        self.adj = adj_vertex
    
    
    def vertex_and_edge_embeddings(self, X: np.ndarray):
        """Extract vertex and edge embeddings from the graph.

        Args:
            adj (scipy.sparse.csc_matrix): The adjacency matrix of the graph.

        Returns:
            tuple: Vertex and edge embeddings.
        """

        # make sure adj is defined
        if self.adj is None:
            raise ValueError("Adjacency matrix not defined. Please preprocess the graph first.")

        # Placeholder: extract vertex and edge embeddings
        ne = self.incidence.shape[1]

        X_v = X
        X_e = np.zeros((X.shape[0], ne))

        # degree vector
        d = np.sum(self.incidence, axis=1).reshape(-1,1)

        # scale incidence by degree vector
        inc_deg = self.incidence * d

        # make sure column sum is 1
        inc_deg = inc_deg / np.sum(inc_deg, axis=0).reshape(1,-1)

        # loop over the edges of the graph and for each edge, get the
        # edge location by finding the point along the line that connections
        # the vertices (from the locations in X_v) as a convex combination.
        # The weights of each vertex should be relative degree of the vertex.
        # we do this using matrix multiplication
        X_e = inc_deg.T @ X_v.T
        X_e = X_e.T

        return X_v, X_e, compute_slopes(self.incidence, X_v)
        
        # raise NotImplementedError("Vertex and edge embeddings not implemented for VertexMatrix.")

@dataclass
class EdgeMatrix(GraphMatrixMethod):
    """Preprocessing that operates on edges (e.g., edge sampling)."""
    adj: sp.csc_matrix = None
    incidence: sp.csc_matrix = None
    incidence_weighted: sp.csc_matrix = None
    normalize: bool = False

    def __init__(self, graph: nx.Graph, normalize = False):
        """Initialize the GraphMatrixMethod with a graph."""
        
        # form the incidence matrix
        self.incidence = nx.incidence_matrix(graph, oriented=False).tocsc()
        self.incidence_weighted = nx.incidence_matrix(graph, oriented=False, weight='weight').tocsc().sqrt()
        self.normalize = normalize

        # normalize
        if self.normalize:
            self.incidence_weighted = nx.incidence_matrix(graph, oriented=False, weight='weight').tocsc()
            vertex_degrees_sqrt = np.sqrt( self.incidence_weighted.sum(axis=1) ).reshape(-1,1)
            self.incidence_weighted = self.incidence_weighted.sqrt() / vertex_degrees_sqrt
        else:
            self.incidence_weighted = nx.incidence_matrix(graph, oriented=False, weight='weight').tocsc().sqrt()

        # build the line graph as the product of the incidence matrix with its transpose
        adj_line = self.incidence_weighted.T @ self.incidence_weighted

        # remove self-loops
        adj_line = adj_line - sp.diags(adj_line.diagonal())

        self.adj = adj_line
    
    
    def vertex_and_edge_embeddings(self, X: np.ndarray):
        """Extract vertex and edge embeddings from the graph.

        Args:
            adj (scipy.sparse.csc_matrix): The adjacency matrix of the graph.

        Returns:
            tuple: Vertex and edge embeddings.
        """
        X_e = X

        # edge-degree vector from self.adj
        d = np.sum(self.adj, axis=0).reshape(1,-1)

        # scale incidence by edge-degree vector
        inc_deg = d * self.incidence

        # make sure row sum is 1
        inc_deg = inc_deg / np.sum(inc_deg, axis=1).reshape(-1,1)

        # for each vertex of the original graph, get the vertex location as the centroid
        # of the edge locations of the line graph that are connected to the vertex
        # use matrix multiplication to sum all incident edges
        X_v = inc_deg @ X_e.T
        X_v = X_v.T

        return X_v, X_e, compute_slopes(self.incidence, X_v)

@dataclass
class TwinEmbeddingMatrix(GraphMatrixMethod):
    """Preprocessing using a twin embedding technique (e.g., creating two separate graphs)."""
    adj: sp.csc_matrix = None
    incidence: sp.csc_matrix = None
    incidence_weighted: sp.csc_matrix = None
    normalize: bool = False

    alpha: float = 0.85
    k: int = 2

    def __init__(self, graph: nx.Graph, alpha = 0.85, k = 2, normalize = False):
        """Initialize the GraphMatrixMethod with a graph."""

        # form the incidence matrix
        self.incidence = nx.incidence_matrix(graph, oriented=False).tocsc()
        self.normalize = normalize

        # normalize
        if self.normalize:
            self.incidence_weighted = nx.incidence_matrix(graph, oriented=False, weight='weight').tocsc()
            vertex_degrees_sqrt = np.sqrt( self.incidence_weighted.sum(axis=1) ).reshape(-1,1)
            self.incidence_weighted = self.incidence_weighted.sqrt() / vertex_degrees_sqrt
        else:
            self.incidence_weighted = nx.incidence_matrix(graph, oriented=False, weight='weight').tocsc().sqrt()


        # build the line graph as the product of the incidence matrix with its transpose
        adj_line = twin_adjacency( self.incidence_weighted, alpha, n = k )

        # remove self-loops
        adj_line = adj_line - sp.diags(adj_line.diagonal())

        self.alpha = alpha
        self.k = k

        self.adj = adj_line
    
    
    def vertex_and_edge_embeddings(self, X: np.ndarray):
        """Extract vertex and edge embeddings from the graph.

        Args:
            adj (scipy.sparse.csc_matrix): The adjacency matrix of the graph.

        Returns:
            tuple: Vertex and edge embeddings.
        """
        # Placeholder: extract vertex and edge embeddings
        nv = self.incidence.shape[0]
        ne = self.incidence.shape[1]

        X_v = X[:, ne:]
        X_e = X[:, :ne]

        return X_v, X_e, compute_slopes(self.incidence, X_v)
