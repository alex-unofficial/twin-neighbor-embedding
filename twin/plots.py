from pathlib import Path

from loguru import logger
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import distinctipy

import seaborn as sns

from sklearn.metrics import pairwise_distances

from shapely.geometry import LineString

def block_adj(A, b = 7):

    N = A.shape[0]             # total size of the matrix (N x N)

    # Number of blocks along each dimension
    n_blocks = int( np.ceil( N / b ) )

    # Initialize the "blocked" matrix, B, where each entry is a sum over a b-by-b block
    B = np.zeros((n_blocks, n_blocks), dtype=np.float64)

    # If A is binary, you could just count nonzeros; if it has numeric values, sum them.
    rows, cols = A.nonzero()
    vals = A.data

    for (r, c, val) in zip(rows, cols, vals):
        br = r // b  # block-row index
        bc = c // b  # block-col index

        # find the 

        B[br, bc] += val

    return B


def custom_cmap(n_labels, seed=0, pastel_factor=0.0, factor_brightness=1.0):
    color_base = distinctipy.get_colors(n_labels, pastel_factor=pastel_factor, colorblind_type='Deuteranomaly',rng=seed)

    rng = np.random.default_rng(seed=seed)
    color_base = rng.permutation(color_base)

    def adjust_brightness(color, factor):
        return np.clip(np.array(color) * factor, 0, 1)

    colors = np.array([
        adjust_brightness(color, factor_brightness) 
        for color in color_base
    ])

    black = np.array([0.0, 0.0, 0.0])
    colors = np.vstack([colors] + [black])

    return colors

def drawsegments(Xe, Se, ax, alpha = 0.9, linewidth = 2, color="b", segment_length=1.0, zorder=None):

    # make sure Xe is 2-D, otherwise throw error
    if Xe.shape[0] != 2:
        raise ValueError("Xe must be a 2-D embedding")

    # find the diagonal of the bounding box of Xe
    min_x, min_y = np.min(Xe, axis=1)
    max_x, max_y = np.max(Xe, axis=1)
    diag_length = np.sqrt((max_x - min_x)**2 + (max_y - min_y)**2) * 0.02

    segment_length = segment_length * diag_length

    # Compute line segment endpoints
    dx = segment_length / np.sqrt(1 + Se**2)  # delta x based on slope
    dy = Se * dx  # delta y based on slope

    # Plot line segments
    for i in range(Xe.shape[1]):
        # if color is a list of N items, use the i-th color
        if isinstance(color, list) and len(color) == Xe.shape[1]:
            color_i = color[i]
        else:
            color_i = color

        ax.plot(
            [Xe[0, i] - dx[i], Xe[0, i] + dx[i]],
            [Xe[1, i] - dy[i], Xe[1, i] + dy[i]],
            color=color_i,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder
        )


def drawedges(G, dim_emb, pos, ax, alpha, linewidth, color='b'):
    if dim_emb == 2:
        nx.draw_networkx_edges(
            G,
            pos={i: pos[:, i] for i in range(G.number_of_nodes())},
            ax=ax,
            edge_color=color,
            alpha=alpha,
            width=linewidth,
        )
    elif dim_emb == 3:
        pos = {i: pos[:, i] for i in range(G.number_of_nodes())}
        for edge in G.edges():
            x_edge = [pos[edge[0]][0], pos[edge[1]][0]]
            y_edge = [pos[edge[0]][1], pos[edge[1]][1]]
            z_edge = [pos[edge[0]][2], pos[edge[1]][2]]
            ax.plot(x_edge, y_edge, z_edge, c=color, alpha=alpha, linewidth=linewidth, zorder=1)
    else:
        raise ValueError("Embedding dimension must be 2 or 3")

def plot_all_embeddings(
    experiment,
    markersize=18,
    linewidth=2,
    alpha=0.1,
    show_edges_original=True,
    show_edges_line=False,
    show_edges_incidence=False,
    show_edges_segments=False,
    segment_alpha=0.9,
    segment_width=2,
    segment_length=1.0
):
    dim_emb = experiment["VertexEmbedding"]["V"].shape[0]

    if dim_emb == 2:
        fig, axes = plt.subplots(
            3, 3, figsize=(3*6, 3*6), squeeze=False,
        )  # 2 rows, 3 columns
    elif dim_emb == 3:
        fig, axes = plt.subplots(
            3, 3, figsize=(3*6, 3*6), squeeze=False, subplot_kw={'projection':'3d'}
        )  # 2 rows, 3 columns
    else:
        raise ValueError("Embedding dimension must be 2 or 3")

    G  = experiment["OriginalGraph"]
    Ge = experiment["LineGraph"]
    Gb = experiment["IncidenceGraph"]

    for i, method in enumerate(["VertexEmbedding", "EdgeEmbedding", "TwinEmbedding"]):
        axes[0, i].scatter(*experiment[method]["V"], c="r", s=markersize, zorder=3)
        axes[1, i].scatter(*experiment[method]["E"], c="b", s=markersize)
        axes[2, i].scatter(*experiment[method]["E"], c="b", s=markersize, zorder=1)
        axes[2, i].scatter(*experiment[method]["V"], c="r", s=markersize, zorder=2)

        if show_edges_segments:
            ax = axes[2, i]
            drawsegments(
                experiment[method]["E"],
                experiment[method]["S"],
                ax = ax,
                alpha = segment_alpha,
                linewidth = segment_width,
                segment_length = segment_length,
                color = "b",
                zorder = -1
            )

        if show_edges_original:
            drawedges(G, dim_emb, experiment[method]["V"], axes[0, i], alpha, linewidth)
        if show_edges_line:
            drawedges(Ge, dim_emb, experiment[method]["E"], axes[1, i], alpha, linewidth)
        if show_edges_incidence:
            Xb = np.hstack((experiment[method]["E"], experiment[method]["V"]))
            drawedges(Gb, dim_emb, Xb, axes[2, i], alpha, linewidth)

        for ax in axes[:,i]:
            ax.set_box_aspect([1,1,1] if dim_emb == 3 else 1)

    for spine in axes[0,0].spines.values():
        spine.set_linewidth(4)
        spine.set_color('black') 

    for spine in axes[1,1].spines.values():
        spine.set_linewidth(4)
        spine.set_color('black') 

    for spine in axes[2,2].spines.values():
        spine.set_linewidth(4)
        spine.set_color('black') 

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([]) if dim_emb == 3 else None

    texts = []

    for j, label in enumerate(["$A_V$", "$A_E$", "$A_{TW}$"]):
        axes[0,j].set_title(label, fontsize=12, fontweight='bold')

    for i, label in enumerate(["Vertices", "Edges", "Vertex+Edge"]):
        texts.append(
            fig.text(0.01, 1 - (i+0.5) * 1/3, label, ha='center', va='center', fontsize=12, fontweight='normal', rotation=90)
        )
    fig.tight_layout()

    fig.tight_layout()

    return fig, axes, texts


def application_plot(
    experiment,
    markersize_vertex=18,
    markersize_edge=18,
    linewidth=2,
    alpha=0.1,
    show_edges_original=True,
    show_edges_segments=False,
    segment_alpha=0.9,
    segment_width=2,
    segment_length=1.0,
    hidevertexpoints = False,
    horizontal = False,
    colors_vertices = None,
    colors_edges = None,
    perm = None,
):

    dim_emb = experiment["VertexEmbedding"]["V"].shape[0]
    G  = experiment["OriginalGraph"]

    frame_colors = [plt.get_cmap('tab10')(i) for i in [1, 4, 2] ]

    if horizontal:
        fig, axes = plt.subplots(1,6, figsize=(6*6, 1*6), squeeze=True)
    else:
        fig, axes = plt.subplots(2,3, figsize=(6*3, 2*6), squeeze=False)

    # build a matrix N by 3 with color red "r"
    if colors_vertices is None:
        colors_vertices = np.array( [ [1.0, 0.0, 0.0] ] * G.number_of_nodes())
    if colors_edges is None:    
        colors_edges = np.array( [ [0.0, 0.0, 1.0] ] * G.number_of_edges())

    # order the nodes so that the subset_background is drawn first
    if perm is None:
        perm = np.arange(G.number_of_nodes())

    for i, method in enumerate(["VertexEmbedding", "EdgeEmbedding", "TwinEmbedding"]):
        ax = axes[i*2] if horizontal else axes[0,i]
        ax.scatter(*experiment[method]["V"][:,perm], c=colors_vertices[perm,:], s=markersize_vertex, zorder=3)
        if show_edges_original:
            drawedges(G, dim_emb, experiment[method]["V"], ax, alpha, linewidth, color = colors_edges)
        
        ax.set_box_aspect(1)
        
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_linewidth(8)
            spine.set_color(frame_colors[i])


        ax = axes[i*2+1] if horizontal else axes[1,i]
        if not hidevertexpoints:
            ax.scatter(*experiment[method]["V"], c=colors_vertices, s=markersize_vertex, zorder=1)
        ax.scatter(*experiment[method]["E"], c=colors_edges, s=markersize_edge, zorder=2)
        if show_edges_segments:
            drawsegments(
                experiment[method]["E"],
                experiment[method]["S"],
                ax = ax,
                alpha = segment_alpha,
                linewidth = segment_width,
                segment_length = segment_length,
                color = "b",
                zorder = -1
            )
        
        ax.set_box_aspect(1)

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_linewidth(8)
            spine.set_color(frame_colors[i])



    fig.tight_layout()

    return fig


def compute_metric(experiment, method, space, adjacency = False):

    if space == "V":
        G = experiment["OriginalGraph"]
        adj = nx.to_scipy_sparse_array(G,format='csc')
    elif space == "E":
        G = experiment["LineGraph"]
        adj = nx.to_scipy_sparse_array(G,format='csc')
            
    V = experiment[method][space]
    # Compute the distance matrix
    distance_matrix = pairwise_distances(V.T);

    if adjacency:
        # Hadamard the distance matrix with the graph adjacency matrix
        distance_matrix_adj = adj.multiply(distance_matrix)

        # Get a vector of nonzero elements
        distances_adj = distance_matrix_adj.data
        distances_adj = distances_adj / distance_matrix.max()

        return distances_adj
    else:

        # make all diagonal elements NaN
        np.fill_diagonal(distance_matrix, np.nan)

        # Flatten the distance matrix
        distances = distance_matrix.flatten()
        # Remove NaNs
        distances = distances[~np.isnan(distances)]
        distances = distances / distances.max()

        return distances


def cumdeg(G, ax):
    degrees = [u for _,u in G.degree()]

    # histcounts
    # Define bin edges from 0 to n (e.g., n = 10)
    n = np.max(degrees)
    bins = np.arange(0, n+2)  # +2 ensures last bin includes n

    # Compute histogram counts
    counts, bin_edges = np.histogram(degrees, bins=bins)

    cdf = np.cumsum(counts) / np.sum(counts)

    # CCDF = 1 - CDF
    ccdf = 1 - cdf

    ax.loglog(bin_edges[:-1], ccdf,  label='CCDF (steps)', linewidth = 2)
    # ax.set_xlabel('Degree')
    # ax.set_ylabel('P(X ≥ degree)')
    # ax.set_title('Inverse Cumulative Distribution (Degree) on Log-Log Scale')
    # ax.legend()
    ax.grid(True, which="major", ls="--", alpha=0.5)
    ax.set_xlim( (np.min(degrees)-2, np.max(degrees)) )

    return None


def plot_single_metric(experiment, space, ax, adjacency=False, legend=False, labels=True, orientation='vertical'):

    colors = [plt.get_cmap('tab10')(i) for i in [1, 4, 2] ]
    ax.set_prop_cycle(color=colors)
    
    # # depending on method, pick color
    # color = colors[["VertexEmbedding", "EdgeEmbedding", "TwinEmbedding"].index(method)]

    for method in ["VertexEmbedding", "EdgeEmbedding", "TwinEmbedding"]:
        distances = compute_metric(experiment, method, space, adjacency=adjacency)

        # Create equi-weight vector
        weights = np.zeros_like(distances) + 1. / distances.size
        assert np.abs(weights.sum() - 1) < 1e-10

        # Plot the histogram
        ax.hist(
            distances,
            bins=100,
            alpha=1.0,
            label=method,
            weights=weights * 100,
            histtype="step",
            linewidth=2,
            orientation=orientation
        )

    # add common labels
    if labels:
        ax.set_ylabel("Relative frequency (%)")
        ax.set_xlabel("Distance in embedding space")

    # add major and minor grid lines
    ax.grid(which='major', axis='y', linestyle='-', linewidth=0.5, alpha=0.8)
    ax.grid(which='minor', axis='y', linestyle='--', linewidth=0.3, alpha=0.8)
    # show minor grid lines
    # ax.minorticks_on()

    ax.legend() if legend else None

    return None

def draw_metrics(experiment, figsize=(12 * 2/3, 8 * 2/3)):

    # Draw a histogram of all distances in the matrix
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=False, sharex=False)
    plt.subplots_adjust(wspace=0.1)  # Adjust the width space between subplots

    # Define a color cycle
    # colors = plt.cm.Set1(np.linspace(0, 1, 10))
    colors = [plt.get_cmap('tab10')(i) for i in [1, 2, 4] ]

    for ax in fig.axes:
        ax.set_prop_cycle(color=colors)


    # for each embedding method, draw histograms, overlayed
    for method in ["VertexEmbedding", "EdgeEmbedding", "TwinEmbedding"]:

        for i, space in enumerate(["V", "E"]):
            
            distances = compute_metric(experiment, method, space)

            # Create equi-weight vector
            weights = np.zeros_like(distances) + 1. / distances.size
            assert np.abs(weights.sum() - 1) < 1e-10


            # Plot the histogram
            axes[0,i].hist(distances, bins=100, alpha=1.0, label=method, weights=weights * 100, histtype='step', linewidth = 3)

            distances = compute_metric(experiment, method, space, adjacency=True)

            # Create equi-weight vector
            weights = np.zeros_like(distances) + 1. / distances.size
            assert np.abs(weights.sum() - 1) < 1e-10

            axes[1,i].hist(distances, bins=100, alpha=1.0, label=method, weights=weights * 100, histtype='step', linewidth = 3)

    # add common labels
    # axes[0].set_ylabel("Relative frequency (%)")


    # axes[1,0].set_xlabel("$X$", labelpad=0)
    # axes[1,1].set_xlabel("$Y$", labelpad=0)

    # add major and minor grid lines
    for ax in axes.flatten():
        ax.grid(which='major', axis='y', linestyle='-', linewidth=0.5, alpha=0.8)
        ax.grid(which='minor', axis='y', linestyle='--', linewidth=0.3, alpha=0.8)
        # show minor grid lines
        ax.minorticks_on()

    # add superlable y
    fig.supylabel("Relative frequency (%)", x=0.075)
    fig.supxlabel("Distance in embedding space", y=0.02)


    fig.suptitle("Embedding quality measures", fontsize=12, y=0.93)

    ax.legend()

    return fig


def count_intersections(G, y):
    edges = []
    lines = []
    for e in G.edges():
        edges += [e]
        lines += [LineString([y[:, e[0]], y[:, e[1]]])]

    lengths = np.array(list(map(lambda l: l.length, lines)))
    sorted_by_len = (np.argsort(lengths))[::-1]

    intersections = np.zeros(len(edges))
    cumulative_intersections = np.zeros(len(edges))

    for base, i in enumerate(sorted_by_len):
        ei = edges[i]
        li = lines[i]

        cumulative_intersections[i] = cumulative_intersections[sorted_by_len[base - 1]]

        for j in sorted_by_len[:base]:
            ej = edges[j]
            lj = lines[j]

            # If edges share a node, continue
            if len(np.intersect1d(ei, ej)) > 0:
                continue

            # If edges geometrically cross, add a crossing and continue
            if li.intersects(lj):
                intersection_point = li.intersection(lj)

                if intersection_point.is_empty:
                    continue
                if intersection_point.geom_type == 'Point':
                    intersections[i] += 1
                    intersections[j] += 1

                    cumulative_intersections[i] += 1

    return lengths[sorted_by_len], intersections[sorted_by_len], cumulative_intersections[sorted_by_len]
