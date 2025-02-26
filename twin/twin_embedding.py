import scipy.sparse as sparse
from sgtsnepi import sgtsnepi
import networkx as nx
import numpy as np


def twin_embedding(inc_matrix, alpha=0.85, n=2, **kwargs):

    num_nodes = inc_matrix.shape[0]
    num_edges = inc_matrix.shape[1]

    twin_adj = twin_adjacency(inc_matrix, alpha, n)

    y = sgtsnepi(twin_adj, **kwargs)

    edge_embeddings = y[:, :num_edges]
    node_embeddings = y[:, num_edges:]

    return node_embeddings, edge_embeddings


def twin_adjacency(inc_matrix, alpha=0.85, n=2):
    '''
    inc_matrix: matrix-edge incidence matrix of graph G
    alpha: a real between 0 and 1
    n: the number of iterations
    '''

    assert n >= 1, "n argument should be n >= 1"
    #assert 0 < alpha < 1, "alpha parameter should be in (0,1) for convergence"

    num_nodes = inc_matrix.shape[0]
    num_edges = inc_matrix.shape[1]

    adj_matrix = sparse.block_array(
        [[None, inc_matrix.T],
         [inc_matrix, None]],
        format='csc', dtype=inc_matrix.dtype
    )

    # identity matrix of the same size and type as A_twin
    eye = sparse.identity(num_edges + num_nodes, format='csc', dtype=adj_matrix.dtype)

    # initial A_twin for k = 1
    twin_adj = alpha * adj_matrix

    # calculate k-th power from recursive formula
    for k in range(1, n):
        twin_adj = alpha * adj_matrix @ (twin_adj + eye)

    return twin_adj


def generate_plots(G, n, alpha, lambda_par, axes, edge_transparency=0.3):
    A_vertex = nx.adjacency_matrix(G)
    A_edge = nx.adjacency_matrix(nx.line_graph(G))
    B = nx.incidence_matrix(G)
    
    # Calculate Twin Embedding
    A_twin = twin_adjacency(B, alpha=alpha, n=n)
    twin_yn, twin_ye = twin_embedding(B, alpha=alpha, n=n, lambda_par=lambda_par, silent=True)
    
    # Calculate vertex and edge embeddings with SGtSNE
    y_vertex = sgtsnepi(A_vertex, lambda_par=lambda_par, silent=True)
    y_edge = sgtsnepi(A_edge, lambda_par=lambda_par, silent=True)

    # Calculate vertex positions using spring layout model
    spring_pos = nx.spring_layout(G)
    spring_yn = np.zeros_like(y_vertex)
    for node, pos in spring_pos.items():
        spring_yn[:, node] = pos

    # Prepare axes for plotting
    for ax in axes.flatten():
        ax.clear()
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        
    
    # Plot ye, yn from twin embedding
    ax_twin = axes[0,0]
    ax_twin.scatter(*twin_ye, s=3, c='r')
    ax_twin.scatter(*twin_yn, s=3, c='k')
    ax_twin.set_title(f"Twin Embedding (n={n}, α={alpha}, λ={lambda_par})")
    ax_twin.autoscale_view(True, True, True)

    # Plot y_vertex from SGtSNE
    ax_vertex = axes[0,1]
    ax_vertex.scatter(*y_vertex, s=3, c='k')
    ax_vertex.set_title(f"Vertex Embedding (λ={lambda_par})")
    ax_vertex.autoscale_view(True, True, True)

     # Plot y_edge from SGtSNE
    ax_edge = axes[0,2]
    ax_edge.scatter(*y_edge, s=3, c='r')
    ax_edge.set_title(f"Edge Embedding (λ={lambda_par})")
    ax_edge.autoscale_view(True, True, True)

    # Plot vertices from twin embedding with edges drawn as lines
    ax_lines = axes[1,0]
    nx.draw(G, pos={i: twin_yn[:,i] for i in range(twin_yn.shape[1])}, ax=ax_lines, node_size=0, node_color='k', edge_color='r', alpha=edge_transparency)
    ax_lines.scatter(*twin_yn, s=3, c='k')
    ax_lines.set_title(f"Twin Embedding with edges as lines")
    ax_lines.autoscale_view(True, True, True)

    # Plot spring layout model
    ax_spring = axes[1,1]
    nx.draw(G, pos=spring_pos, ax=ax_spring, node_size=0, node_color='k', edge_color='r', alpha=edge_transparency)
    ax_spring.scatter(*spring_yn, s=3, c='k')
    ax_spring.set_title(f"Spring Layout model")
    ax_spring.autoscale_view(True, True, True)
    
    # Show spy plot of Twin adjacency matrix.
    ax_spy = axes[1,2]
    ax_spy.imshow(A_twin.todense(), interpolation=None, cmap='BuPu')
    ax_spy.set_title(f"Twin Adjacency matrix (n={n}, α={alpha})")

    