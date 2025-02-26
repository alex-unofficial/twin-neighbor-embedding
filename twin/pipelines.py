
import networkx as nx
import scipy.sparse as sp
import numpy as np
from twin import graphmatrix as gm
from twin import embedding as emb

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
