from pathlib import Path

from loguru import logger

from twin.config import REAL_WORLD_DIR
from twin.pipelines import graph_embedding_all
from twin.plots import (
    plot_all_embeddings,
    draw_metrics,
    drawedges,
    drawsegments,
    cumdeg,
    plot_single_metric,
    count_intersections,
    custom_cmap,
    block_adj,
)
import twin.dataset as dataset
import twin.metadata as metadata

import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
import distinctipy
import networkx as nx
import igraph as ig
import leidenalg as leiden
import seaborn as sns
from itertools import count

from timeit import default_timer as timer

import scipy.io as sio

def tabula_sapiens_blockadj(ax, experiment, partition, vertex_labels, label2id, label_colors, b = 7):
    # Create a random sparse adjacency matrix in CSR format
    A = experiment["Gv_in"].get_adj()

    # permute A so that clusters from vertex_labels are contiguous
    perm = np.lexsort( (np.array( partition.membership ).reshape(-1,1), vertex_labels.reshape(-1,1)), axis = 0 ).reshape(-1)
    A_sub = A[perm, :][:, perm]

    unique_clusters, cluster_boundaries = np.unique(vertex_labels[perm], return_index=True)
    cluster_boundaries = cluster_boundaries // b
    cluster_boundaries = np.append(cluster_boundaries, A_sub.shape[0])

    B = block_adj( A_sub, b )
    n_blocks = B.shape[0]

    B_masked = np.where(B == 0, np.nan, B)

    # Optionally, you could average or take fraction of nonzero by dividing by b^2:
    # B /= (b**2)

    # get the colormap of gray in distinct colors (extract only 6 levels) and keep the last 3 levels
    base_cmap = plt.get_cmap('gray_r', 6)

    cmap = colors.LinearSegmentedColormap.from_list(
        "truncated_cmap", 
        base_cmap(np.linspace(1/4, 1, 256))  # Extract upper 1/3 of colormap
    )
    cmap.set_bad('white')

    ims = ax.imshow(B_masked, cmap=cmap, origin='upper', interpolation='nearest', vmin=0, vmax=3)
    # plt.colorbar(ims, label='Sum in each b-by-b block')
    # ax.set_title('Blocked Visualization of Sparse Matrix')
    ax.set_xticks([])
    ax.set_yticks([])

    blocks_lines = []

    # Draw diagonal blocks using cluster boundaries
    for i in range(len(cluster_boundaries) - 1):
        start = cluster_boundaries[i]
        end = cluster_boundaries[i+1]
        
        # Define square corners for the diagonal blocks
        square_x = [start, end, end, start, start]
        square_y = [start, start, end, end, start]
        
        # Plot the block
        blocks_lines.append( ax.plot(square_x, square_y, color=label_colors[i,:], linewidth=1.0)[0] )

        ax.set_xlim(0, n_blocks)
        ax.set_ylim(n_blocks, 0)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    legend_labels = [str(i) for i in label2id.keys()]
    # put a legend of the colors in label_colors along with the label in legend_labels
    legend = ax.legend(blocks_lines, legend_labels, loc='upper right', title='Cell types', fontsize=6, title_fontsize=6)

def tabula_sapiens_teaser(
    organ : str = "liver",
    dim_embedding: int = 2,
    lambda_par: float = 8,
    run_exact: bool = False
):

    logger.info(f"Creating Tabula {organ.title()} figure")

    tabula = dataset.tabula(organ)

    try:
        start = timer()

        # Extract embedding from Graph
        experiment = graph_embedding_all(
            tabula, # input graph
            twinmatrix_kw={"alpha": 0.85, "k": 2}, # twin matrix parameters
            #
            # --- select one of the following embedding methods ---
            embedding=emb.SGtSNELayout, embedding_kw={"lambda_par": lambda_par, "run_exact": run_exact},
            # embedding=emb.SpectralLayout, embedding_kw={"normed": True},
            # embedding=emb.SpringLayout, embedding_kw={},
            # 
            # --- the rest are common parameters ---
            seed=10,
            d=dim_embedding,
            common_init=True,
            normalize=True
        )

        end = timer()

        logger.success(f"Generated embeddings in {end - start} seconds")

    except Exception as e:
        logger.error("Exception occured while generating embeddings: {e}")

    tabula_ig = ig.Graph.from_networkx(tabula)

    partition = leiden.find_partition(
        tabula_ig,
        leiden.RBConfigurationVertexPartition,
        resolution_parameter=1.0,
        seed = 10
    )

    s = sio.loadmat(str(REAL_WORLD_DIR / f'tabula-{organ}.mat'))

    Xumap_V, Xumap_E, Xumap_S = experiment['Gv_in'].vertex_and_edge_embeddings( s['X_umap'].T )

    orig_labels = s['cell_ontology_class'][0]
    vertex_labels_leiden = partition.membership
    extracted_labels = np.array([elem[0] for elem in orig_labels])
    unique_labels = np.unique(extracted_labels)

    # 3. Create a label -> ID mapping
    label2id = {label: idx for idx, label in enumerate(unique_labels)}

    vertex_labels = np.array([label2id[x] for x in extracted_labels])

    n_labels = len( np.unique( vertex_labels ) )

    # Generate labels for edges based on the edge labels
    edge_alpha        = []
    edge_alpha_all    = []
    edge_marker_size  = []
    alpha_edge_points = []

    # Generate labels for edges based on the edge labels
    edge_labels = []
    for u, v in tabula.edges():
        if vertex_labels[u] == vertex_labels[v]:
            edge_labels.append(vertex_labels[u])
        else:
            edge_labels.append(-1)

    label_colors = custom_cmap(n_labels, seed=7)
    rgb0 = label_colors[ 0, : ].copy()
    rgb5 = label_colors[ 5, : ].copy()
    label_colors[ 0, : ] = rgb5
    label_colors[ 5, : ] = rgb0

    vertex_colors = np.array([np.array(label_colors[l]) for l in vertex_labels])
    edge_colors = np.array([np.array(label_colors[l]) for l in edge_labels])

    for u, v, d in tabula.edges(data=True):
        edge_alpha_all.append(d['weight'])
        if vertex_labels[u] == vertex_labels[v]:
            edge_alpha.append(0.4)
            edge_marker_size.append(0.5)
            alpha_edge_points.append(0.15)
        else:
            # edge_alpha.append(0.2 if d['weight'] > 0.8 else 0)
            # edge_alpha_all.append(d['weight'])
            edge_alpha.append(0.2)
            edge_marker_size.append(1)
            alpha_edge_points.append(0.7)

    alpha_edge_points = np.array(alpha_edge_points)

    logger.info("Creating Tabula plots..")

    fig, axes = plt.subplots(1, 5, figsize=(5*6, 6))
    # Plot Vertex and Edge positions from twin embedding
    frame_colors = [distinctipy.BLACK if i == -1 else plt.get_cmap('tab10')(i) for i in [-1, -1, -1, 2, 2]]
    for i in range(5):

        ax = axes[i]

        match i:
            case 0:
                ax.scatter(*Xumap_V,  s=5, zorder=-1, alpha = 1, c = np.array([[.5, .5, .5]]))
            case 1:
                ax.scatter(*Xumap_E, c=edge_colors, s=edge_marker_size, zorder=-1, alpha = alpha_edge_points)
            case 2:
                tabula_sapiens_blockadj(ax, experiment, partition, vertex_labels, label2id, label_colors, b = 7)
                continue
            case 3:
                ax.scatter(*experiment["TwinEmbedding"]["V"], s=8, c = np.array([[.5, .5, .5]]), zorder=-1, alpha = 1.0)
            case 4:
                ax.scatter(*experiment["TwinEmbedding"]["E"], c=edge_colors, s=edge_marker_size, zorder=-1, alpha = np.clip( 2*alpha_edge_points, max=1 ))


        ax.set_box_aspect(1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', length=0)
        ax.grid()
        for spine in ax.spines.values():
            spine.set_linewidth(8)
            spine.set_color(frame_colors[i])


    fig.tight_layout()

    return fig


def tree_of_cliques(
    h: int = 3,
    k: int = 2,
    n: int = 4,
    dim_embedding: int = 2,
    lambda_par: int = 5,
    run_exact: bool = False
):

    frame_colors = [plt.get_cmap('tab10')(i) for i in [1, 4, 2] ]

    logger.info(f"Creating Tree of Cliques figure")

    G = dataset.tree_of_cliques( h=h, k=k, n=n )

    try:
        start = timer()

        experiment = graph_embedding_all(
            G, # input graph
            twinmatrix_kw={"alpha": 0.85, "k": 2}, # twin matrix parameters
            #
            # --- select one of the following embedding methods ---
            embedding=emb.SGtSNELayout,
            embedding_kw={"lambda_par": lambda_par, "run_exact": run_exact},
            seed=0,
            d=dim_embedding,
            normalize=False,
            common_init=True,
        )

        end = timer()

        logger.success(f"Generated embeddings in {end - start} seconds")

    except Exception as e:
        logger.error("Exception occured while generating embeddings: {e}")

    lv, _, cv = count_intersections(G, experiment["VertexEmbedding"]["V"])
    le, _, ce = count_intersections(G, experiment["EdgeEmbedding"]["V"])
    lt, _, ct = count_intersections(G, experiment["TwinEmbedding"]["V"])


    alpha = 0.3
    linewidth = 4

    segment_alpha=0.9
    segment_width=2
    segment_length=1.0

    logger.info("Creating Tree of Cliques plots...")

    fig, axes = plt.subplots(1, 4, figsize=(4*6, 6))

    # Suplots 1 and 3
    for i, method in enumerate(["VertexEmbedding", "TwinEmbedding"]):
        ax = axes[2*i]

        ax.scatter(*experiment[method]["V"], c="r", s=64, zorder=3)
        drawedges(G, dim_embedding, experiment[method]["V"], ax, alpha, linewidth)

        ax.set_box_aspect(1)

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_linewidth(8)
            spine.set_color(frame_colors[2*i])

    # Suplot 2
    axes[1].scatter(*experiment["VertexEmbedding"]["V"], c="r", s=64, zorder=1)
    axes[1].scatter(*experiment["VertexEmbedding"]["E"], c="b", s=64, zorder=2)

    drawsegments(
        experiment["VertexEmbedding"]["E"],
        experiment["VertexEmbedding"]["S"],
        ax = axes[1],
        alpha = segment_alpha,
        linewidth = segment_width,
        segment_length = segment_length,
        color = "b",
        zorder = -1
    )

    axes[1].set_xticks([])
    axes[1].set_yticks([])

    for spine in axes[1].spines.values():
        spine.set_linewidth(8)
        spine.set_color(frame_colors[0])

    # Subplot 4
    axes[3].step(lv / np.max(lv), cv, c=frame_colors[0], label=r'$A_V$', linewidth=4)
    # axes[3].step(le / np.max(le), ce, c=frame_colors[1], label=r'$A_E$', linewidth=4)
    axes[3].step(lt / np.max(lt), ct, c=frame_colors[2], label=r'$A_{tne}$', linewidth=4)

    axes[3].set_xlabel(r'length threshold $\tau$')
    axes[3].set_ylabel(r'# edge crossings')

    for spine in axes[3].spines.values():
        spine.set_linewidth(4)

    axes[3].set_xlim((-0.01, 0.5))
    axes[3].set_ylim((0, None))

    axes[3].yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    axes[3].grid()
    axes[3].legend()

    fig.tight_layout()

    return fig


def small_world(
    n : int = 150,
    k : int = 12,
    p : float = 0.01,
    dim_embedding: int = 2,
    lambda_par: int = 1,
    run_exact: bool = False
):

    frame_colors = [plt.get_cmap('tab10')(i) for i in [1, 4, 2] ]

    logger.info("Creating Small World figure")

    G = dataset.small_world(n, k, p)

    try:
        start = timer()

        experiment = graph_embedding_all(
            G, # input graph
            twinmatrix_kw={"alpha": 0.85, "k": 2}, # twin matrix parameters
            #
            # --- select one of the following embedding methods ---
            embedding=emb.SGtSNELayout,
            embedding_kw={"lambda_par": lambda_par, "run_exact": run_exact},
            seed=0,
            d=dim_embedding,
            normalize=True,
            common_init=True,
        )

        end = timer()

        logger.success(f"Generated embeddings in {end - start} seconds")

    except Exception as e:
        logger.error("Exception occured while generating embeddings: {e}")

    alpha = 0.3
    linewidth = 1

    logger.info("Creating Small World plots...")

    fig, axes = plt.subplots(1, 3, figsize=(3*4, 4))
    for i, method in enumerate(["VertexEmbedding", "EdgeEmbedding", "TwinEmbedding"]):
        ax = axes[i]

        ax.scatter(*experiment[method]["V"], c="r", s=15, zorder=3)
        drawedges(G, dim_embedding, experiment[method]["V"], ax, alpha, linewidth)

        ax.set_box_aspect(1)

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_linewidth(4)
            spine.set_color(frame_colors[i])

    fig.tight_layout()

    return fig


def football_figure(
    dim_embedding: int = 2,
    lambda_par: int = 5,
    run_exact: bool = False
    # -----------------------------------------
):
    frame_colors = [plt.get_cmap('tab10')(i) for i in [1, 4, 2] ]

    logger.info(f"Creating Football figure")

    football = dataset.football_2023()

    try:
        start = timer()

        # Extract embedding from Graph
        experiment = graph_embedding_all(
            football, # input graph
            twinmatrix_kw={"alpha": 0.85, "k": 2}, # twin matrix parameters
            # -- embedding method --
            embedding=emb.SGtSNELayout, embedding_kw={"lambda_par": lambda_par, "run_exact": run_exact},
            # --- the rest are common parameters ---
            seed=10,
            d=dim_embedding,
            normalize=True,
            common_init=True,
        )

        end = timer()

        logger.success(f"Generated embeddings in {end - start} seconds")

    except Exception as e:
        logger.error("Exception occured while generating embeddings: {e}")

    # Get Ground-Truth labels from metadata
    conference = metadata.football_2023["conference"]
    conference_standing = metadata.football_2023["conference_standing"]

    counter = count(0)
    conf_labels = {
        conf: (-1 if conf == "NCAA Division I FBS independents" else next(counter))
        for conf in dict.fromkeys(conference)
    }

    stand_labels = []
    for stand in conference_standing:
        if stand[-5:-1] == "East":
            stand_labels.append(1)
        elif stand[-5:-1] == "West":
            stand_labels.append(-1)
        else:
            stand_labels.append(0)

    vertex_labels = [conf_labels[conf] for conf in conference]
    unique_labels = conf_labels.keys()

    # Generate labels for edges based on the edge labels

    intra_cluster_alpha = 0.6
    inter_cluster_alpha = 0.1

    edge_labels = []
    edge_standing = []
    edge_alpha = []
    for u, v in football.edges():
        if (
            vertex_labels[u] == vertex_labels[v] and
            stand_labels[u] == stand_labels[v] and
            vertex_labels[u] != conf_labels["NCAA Division I FBS independents"]
        ):
            edge_labels.append(vertex_labels[u])
            edge_standing.append(stand_labels[u])
            edge_alpha.append(intra_cluster_alpha)
        else:
            edge_labels.append(-1)
            edge_standing.append(0)
            edge_alpha.append(inter_cluster_alpha)

    # Convert vertex, edge labels to unique colors
    n_labels = len(unique_labels)

    # Convert vertex, edge labels to unique colors
    n_labels = len(unique_labels)

    color_base = sns.color_palette("husl", n_labels, desat=1)

    rng = np.random.default_rng(seed=1)
    color_base = rng.permutation(color_base)

    # Generate light and dark variations
    factor_light = 1.3  # Brighten factor
    factor_dark = 0.7   # Darken factor

    def adjust_brightness(color, factor):
        return np.clip(np.array(color) * factor, 0, 1)

    colors = np.array([
        adjust_brightness(color, {-1: factor_dark, 0: 1.0, 1: factor_light}[i]) 
        for color in color_base for i in [-1, 0, 1]
    ])

    black = np.array([0.0, 0.0, 0.0])
    colors = np.vstack([colors] + 3 * [black])

    vertex_colors = np.array([
        colors[3*l + 1 + s, :]
        for l, s in zip(vertex_labels, stand_labels)
    ])

    edge_colors = np.array([
        colors[3*l + 1 + s, :]
        for l, s in zip(edge_labels, edge_standing)
    ])

    logger.info("Creating Football plots...")
    # Plot Vertex and Edge positions from twin embedding
    fig, axes = plt.subplots(1, 3, figsize=(3*4, 4))
    for i, method in enumerate(["VertexEmbedding", "EdgeEmbedding", "TwinEmbedding"]):

        ax = axes[i]

        ax.scatter(*experiment[method]["V"], c=vertex_colors, s=15, zorder=3)

        drawedges(
            football,
            dim_embedding,
            experiment[method]["V"],
            ax,
            edge_alpha, 1, edge_colors
        )

        ax.set_box_aspect(1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        for spine in ax.spines.values():
            spine.set_linewidth(4)
            spine.set_color(frame_colors[i])

    fig.tight_layout()

    return None


'''
@app.command()
def barabasi_albert(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    output_path: Path = FIGURES_DIR / "metrics",
    output_name: str = "ba-n{n}-d{d}.pdf",
    output_ba_path: Path = FIGURES_DIR / "barabasi-albert",
    n: int = 1000,
    d: int = 7,
    dim_embedding: int = 2,
    lambda_par: int = 5,
    run_exact: bool = False
    # -----------------------------------------
):
    logger.info(f"Exporting Barabasi-Albert metrics at: {output_path}")
    initial_graph = nx.cycle_graph(7)
    G = dataset.barabasi_albert( n, d, initial_graph = initial_graph )

    experiment = graph_embedding_all(
        G, # input graph
        twinmatrix_kw={"alpha": 0.85, "k": 2}, # twin matrix parameters
        #
        # --- select one of the following embedding methods ---
        embedding=emb.SGtSNELayout, embedding_kw={"lambda_par": lambda_par, "run_exact": run_exact},
        # embedding=emb.SpectralLayout, embedding_kw={"normed": True},
        # embedding=emb.SpringLayout, embedding_kw={},
        # 
        # --- the rest are common parameters ---
        seed=6,
        d=dim_embedding,
        common_init=True,
    )

    # Create a figure with a GridSpec layout
    fig = plt.figure(figsize=(8, 2))
    gs = gridspec.GridSpec(1, 6, width_ratios=[1, 1, 0.2, 1, 0.2, 1], wspace=0.2)  # Adjust width_ratios

    # Create subplots using the GridSpec
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 3])
    ax4 = fig.add_subplot(gs[0, 5])  # Span the entire row


    plot_single_metric(experiment, "V", ax=ax1, adjacency=False, labels=False)
    plot_single_metric(experiment, "E", ax=ax2, adjacency=False, labels=False)
    plot_single_metric(experiment, "V", ax=ax3, adjacency=True, labels=False)
    cumdeg(G, ax4)

    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticklabels([])

    def format_func_pdf(value, tick_number):
        return f"{value:.0f}%"  # Format to 2 decimal places

    def format_func(value, tick_number):
        return f"{value:.0f}"  # Format to 2 decimal places

    def format_func_cdf(value, tick_number):
        if value >= 0.01:
            return f"{value*100:.0f}%"  # Format to 2 decimal places
        else:
            return f"{value*100:.1f}%"  # Format to 2 decimal places

    # yticklabels format of ax1 should by {:d}%
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_func_pdf))
    ax3.yaxis.set_major_formatter(ticker.FuncFormatter(format_func_pdf))
    ax4.yaxis.set_major_formatter(ticker.FuncFormatter(format_func_cdf))
    ax4.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))


    # reduce spaceing between axis and labels
    ax1.tick_params(axis='y', pad=0)
    ax3.tick_params(axis='y', pad=0)
    ax4.tick_params(axis='y', pad=0)

    ax1.set_facecolor( (1.0, 0.4745098039215686, 0.4235294117647059, 0.2) )
    ax2.set_facecolor( (0.011764705882352941, 0.2627450980392157, 0.8745098039215686, 0.1) )

    # add a text at the top-right of the axes ax1
    ax1.text(0.95, 0.95, "PMF", transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right')
    ax2.text(0.95, 0.95, "PMF", transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right')
    ax3.text(0.95, 0.95, "PMF", transform=ax3.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right')
    ax4.text(0.95, 0.95, "CCDF", transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right')

    ax4.set_xlim(6, 140)

    ax1.set_xlabel("distance", labelpad=-10)
    ax2.set_xlabel("distance", labelpad=-10)
    ax3.set_xlabel("edge-length", labelpad=-10)
    ax4.set_xlabel("degree", labelpad=-10)

    os.makedirs(output_path, exist_ok=True)
    filename = output_path / output_name.format(n=n, d=d)

    fig.savefig(filename, bbox_inches="tight")
    logger.success(f"Exported file {filename}")

    # define a vector of vertex_colors so that the top-10 degree nodes are colored differently
    vertex_colors = []
    vertex_alpha = []
    vertex_size = []
    sorted_indices = np.argsort([deg for _, deg in G.degree()])
    # extract only nodes from the tuple
    top_degree_idx = sorted_indices[-10:]

    for i in range(len(G.nodes)):
        if i in top_degree_idx:
            vertex_colors.append( colors.to_rgb(colors.XKCD_COLORS["xkcd:brick red"]))
            vertex_alpha.append(1)
            vertex_size.append(25)
        else:
            vertex_colors.append( colors.to_rgb(colors.XKCD_COLORS["xkcd:salmon"]) )
            vertex_alpha.append(0.4)
            vertex_size.append(5)

    # for the edge colors, color the edges incident to the top-10 degree nodes
    high_deg_alpha = 0.2
    low_deg_alpha = 0.02

    edge_colors = []
    edge_alpha = []
    for u, v in G.edges():
        if u in top_degree_idx or v in top_degree_idx:
            edge_colors.append(colors.to_rgb(colors.XKCD_COLORS["xkcd:blue"]))
            edge_alpha.append(high_deg_alpha)
        else:
            edge_colors.append(colors.to_rgb(colors.XKCD_COLORS["xkcd:cement"]))
            edge_alpha.append(low_deg_alpha)

    frame_colors = [plt.get_cmap('tab10')(i) for i in [1, 4, 2] ]

    for i, method in enumerate(["VertexEmbedding", "EdgeEmbedding", "TwinEmbedding"]):
        # Plot Vertex and Edge positions from twin embedding
        fig, ax = plt.subplots(figsize=(6, 6))    

        Xv = experiment[method]["V"][:,sorted_indices]
        clr = [vertex_colors[i] for i in sorted_indices]
        alpha = [vertex_alpha[i] for i in sorted_indices]
        s = [vertex_size[i] for i in sorted_indices]
        ax.scatter(*Xv, c=clr, s=s, zorder=3, alpha=alpha)

        drawedges(G, dim_embedding, experiment[method]["V"], ax, edge_alpha, 1, edge_colors)

        ax.set_box_aspect(1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        for spine in ax.spines.values():
            spine.set_linewidth(8)
            spine.set_color(frame_colors[i])

        filename = output_ba_path / "ba-n{n}-d{d}_0-{i}.png".format(n=n, d=d,i=i)
        fig.savefig(filename, bbox_inches="tight", dpi=300)

        fig, ax = plt.subplots(figsize=(6, 6))    

        ax.scatter(*Xv, c=clr, s=s, zorder=3, alpha=alpha)
        ax.scatter(*experiment[method]["E"], c=edge_colors, s=3, zorder=1, alpha=edge_alpha)

        ax.set_box_aspect(1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # remove xticks and yticks
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_linewidth(8)
            spine.set_color(frame_colors[i])

        filename = output_ba_path / "ba-n{n}-d{d}_1-{i}.png".format(n=n, d=d,i=i)
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    logger.success("Exported files")

    return None
'''
