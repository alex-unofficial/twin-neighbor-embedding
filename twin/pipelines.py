
import networkx as nx
import scipy.sparse as sp
import numpy as np
import pandas as pd

from twin import graphmatrix as gm
from twin import embedding as emb
from twin.metrics import *

from typeguard import typechecked


def graph_embedding_all(
    G: nx.Graph, embedding = emb.SpringLayout,
    seed: int = 0,
    d: int = 2,
    embedding_kw: dict = {},
    twinmatrix_kw: dict = {},
    common_init: bool = True,
    normalize: bool = False
):
    """ Runs the full pipeline for all graph matrix methods. """

    E_in = embedding(seed=seed, d=d, **embedding_kw)
    Gv_in = gm.VertexMatrix(G, normalize=False)
    Ge_in = gm.EdgeMatrix(G, normalize=False)
    Gtw_in = gm.TwinEmbeddingMatrix(G, normalize=normalize, **twinmatrix_kw)

    # number of nodes + edges
    nv = Gtw_in.incidence.shape[0]
    ne = Gtw_in.incidence.shape[1]

    # Step 1: Create initial positions for the embeddings
    y0 = E_in.init_embedding(nv+ne, d)

    y0e  = y0[:, :ne] if common_init else None
    y0v  = y0[:, ne:] if common_init else None
    y0tw = y0 if common_init else None

    # Step 2: Embed using all three approaches
    X = E_in.embed(Gv_in, y0 = y0v)
    Y = E_in.embed(Ge_in, y0 = y0e)
    Z = E_in.embed(Gtw_in, y0 = y0tw)

    # Step 3: Extract vertex and edge embeddings for each approach
    X_v, X_e, X_s = Gv_in.vertex_and_edge_embeddings(X)
    Y_v, Y_e, Y_s = Ge_in.vertex_and_edge_embeddings(Y)
    Z_v, Z_e, Z_s = Gtw_in.vertex_and_edge_embeddings(Z)

    # Optional: form line graph & incidence for postprocess
    L = Ge_in.incidence.T @ Ge_in.incidence
    L = L - sp.diags(L.diagonal())

    inc_emb = sp.block_array(
        [[None, Gv_in.incidence.T],
         [Gv_in.incidence, None]],
        format='csc', dtype=Gv_in.incidence.dtype
    )

    dict_result = {
        "VertexEmbedding": {"V": X_v, "E": X_e, "S": X_s},
        "EdgeEmbedding": {"V": Y_v, "E": Y_e, "S": Y_s},
        "TwinEmbedding": {"V": Z_v, "E": Z_e, "S": Z_s},
        "OriginalGraph": G,
        "Gv_in": Gv_in,
        "Ge_in": Ge_in,
        "Gtw_in": Gtw_in,
        "LineGraph": nx.from_scipy_sparse_array(L),
        "IncidenceGraph": nx.from_scipy_sparse_array(inc_emb)
    }
    return dict_result


def embedding_metrics_all(
    G: nx.Graph, embedding = emb.SpringLayout,
    d: int = 2,
    n_samples : int = 1,
    n_neighbors: int = 10,
    target = None,
    embedding_kw: dict = {},
    twinmatrix_kw: dict = {},
    common_init: bool = True,
    normalize: bool = False
):

    Gv_in = gm.VertexMatrix(G, normalize=False)
    Ge_in = gm.EdgeMatrix(G, normalize=False)
    Gtw_in = gm.TwinEmbeddingMatrix(G, normalize=normalize, **twinmatrix_kw)

    # number of nodes + edges
    nv = Gtw_in.incidence.shape[0]
    ne = Gtw_in.incidence.shape[1]

    vertex_point_dist_metric_v = np.float64(0)
    vertex_point_dist_metric_e = np.float64(0)
    vertex_point_dist_metric_tw = np.float64(0)

    edge_point_dist_metric_v = np.float64(0)
    edge_point_dist_metric_e = np.float64(0)
    edge_point_dist_metric_tw = np.float64(0)

    neighbor_preservation_metric_v = np.zeros_like(n_neighbors, dtype=np.float64)
    neighbor_preservation_metric_e = np.zeros_like(n_neighbors, dtype=np.float64)
    neighbor_preservation_metric_tw = np.zeros_like(n_neighbors, dtype=np.float64)

    classification_metric_v = np.zeros_like(n_neighbors, dtype=np.float64)
    classification_metric_e = np.zeros_like(n_neighbors, dtype=np.float64)
    classification_metric_tw = np.zeros_like(n_neighbors, dtype=np.float64)

    for seed in range(n_samples):
        E_in = embedding(seed=seed, d=d, **embedding_kw)

        # Step 1: Create initial positions for the embeddings
        y0 = E_in.init_embedding(nv+ne, d)

        y0e  = y0[:, :ne] if common_init else None
        y0v  = y0[:, ne:] if common_init else None
        y0tw = y0 if common_init else None

        # Step 2: Embed using all three approaches
        X = E_in.embed(Gv_in, y0 = y0v)
        Y = E_in.embed(Ge_in, y0 = y0e)
        Z = E_in.embed(Gtw_in, y0 = y0tw)

        # Step 3: Extract vertex and edge embeddings for each approach
        X_v, X_e, _ = Gv_in.vertex_and_edge_embeddings(X)
        Y_v, Y_e, _ = Ge_in.vertex_and_edge_embeddings(Y)
        Z_v, Z_e, _ = Gtw_in.vertex_and_edge_embeddings(Z)

        vertex_point_dist_metric_v += point_distance_metric(X_v, d)
        vertex_point_dist_metric_e += point_distance_metric(Y_v, d)
        vertex_point_dist_metric_tw += point_distance_metric(Z_v, d)

        edge_point_dist_metric_v += point_distance_metric(X_e, d)
        edge_point_dist_metric_e += point_distance_metric(Y_e, d)
        edge_point_dist_metric_tw += point_distance_metric(Z_e, d)

        neighbor_preservation_metric_v += knn_neighbor_preservation_accuracy(X_v, G, n_neighbors)
        neighbor_preservation_metric_e += knn_neighbor_preservation_accuracy(Y_v, G, n_neighbors)
        neighbor_preservation_metric_tw += knn_neighbor_preservation_accuracy(Z_v, G, n_neighbors)

        if target is not None:
            classification_metric_v += knn_classification_accuracy(X_v, target, n_neighbors)
            classification_metric_e += knn_classification_accuracy(Y_v, target, n_neighbors)
            classification_metric_tw += knn_classification_accuracy(Z_v, target, n_neighbors)

    vertex_point_dist_metric_v = vertex_point_dist_metric_v / n_samples
    vertex_point_dist_metric_e = vertex_point_dist_metric_e / n_samples
    vertex_point_dist_metric_tw = vertex_point_dist_metric_tw / n_samples

    edge_point_dist_metric_v = edge_point_dist_metric_v / n_samples
    edge_point_dist_metric_e = edge_point_dist_metric_e / n_samples
    edge_point_dist_metric_tw = edge_point_dist_metric_tw / n_samples

    neighbor_preservation_metric_v = neighbor_preservation_metric_v / n_samples
    neighbor_preservation_metric_e = neighbor_preservation_metric_e / n_samples
    neighbor_preservation_metric_tw = neighbor_preservation_metric_tw / n_samples

    if target is not None:
        classification_metric_v = classification_metric_v / n_samples
        classification_metric_e = classification_metric_e / n_samples
        classification_metric_tw = classification_metric_tw / n_samples

    metrics_dict = {
        'n_neighbors': n_neighbors,
        'n_samples': n_samples,

        'VertexPointDistance': {
            'Vertex': vertex_point_dist_metric_v,
            'Edge': vertex_point_dist_metric_e,
            'Twin': vertex_point_dist_metric_tw
        },

        'EdgePointDistance': {
            'Vertex': edge_point_dist_metric_v,
            'Edge': edge_point_dist_metric_e,
            'Twin': edge_point_dist_metric_tw
        },

        'KNeighborsPreservation': {
            'Vertex': neighbor_preservation_metric_v,
            'Edge': neighbor_preservation_metric_e,
            'Twin': neighbor_preservation_metric_tw
        }
    }

    if target is not None:
        metrics_dict['KNeighborsClassifier'] = {
            'Vertex': classification_metric_v,
            'Edge': classification_metric_e,
            'Twin': classification_metric_tw
        }

    return metrics_dict


def metrics_to_df(metrics: dict, graph_label: str = None):
    """
    Convert embedding metrics to structured pandas DataFrames.

    Parameters
    ----------
    metrics : dict
        Output from embedding_metrics_all function
    graph_label : str, optional
        Label for the graph (e.g., 'Football 2023', 'Tabula Sapiens')
        Used as index level for stacking multiple graphs

    Returns
    -------
    dict
        Dictionary with keys:
        - 'PointDistance': DataFrame with rows for VertexPointDistance and EdgePointDistance
        - 'KNeighborsPreservation': DataFrame with rows for k values
        - 'KNeighborsClassifier': DataFrame with rows for k values (if present)
    """
    embedding_types = ['Vertex', 'Edge', 'Twin']

    # 1. Point Distance DataFrame
    point_dist_data = {
        emb_type: [
            metrics['VertexPointDistance'][emb_type],
            metrics['EdgePointDistance'][emb_type]
        ]
        for emb_type in embedding_types
    }

    df_point_dist = pd.DataFrame(
        point_dist_data,
        index=['Vertex', 'Edge']
    )
    df_point_dist.index.name = 'Point type'

    # 2. KNeighbors Preservation DataFrame
    knn_pres_data = {
        emb_type: metrics['KNeighborsPreservation'][emb_type]
        for emb_type in embedding_types
    }

    df_knn_preservation = pd.DataFrame(
        knn_pres_data,
        index=metrics['n_neighbors']
    )
    df_knn_preservation.index.name = 'k'

    result = {
        'PointDistance': df_point_dist,
        'KNeighborsPreservation': df_knn_preservation
    }

    # 3. KNeighbors Classifier DataFrame (if present)
    if 'KNeighborsClassifier' in metrics:
        knn_class_data = {
            emb_type: metrics['KNeighborsClassifier'][emb_type]
            for emb_type in embedding_types
        }

        df_knn_classifier = pd.DataFrame(
            knn_class_data,
            index=metrics['n_neighbors']
        )

        df_knn_classifier.index.name = 'k'
        result['KNeighborsClassifier'] = df_knn_classifier

    # Add graph label as MultiIndex if provided
    if graph_label is not None:
        for key in result:
            df = result[key]
            df.index = pd.MultiIndex.from_product(
                [[graph_label], df.index],
                names=['Graph', df.index.name or 'Metric']
            )

    return result

