from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from twin.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, REAL_WORLD_DIR, SYNTHETIC_NET_DIR

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import networkx as nx
from math import ceil

app = typer.Typer()

# ---------------------------------------------------------------------------- #
#                                 Sets of data                                 #
# ---------------------------------------------------------------------------- #

def ring_of_cliques(n, k):
    return nx.ring_of_cliques(n, k)


def tree_of_cliques(h, k, n, root_cluster=False):
    '''Generate Tree of Cliques graph

    Args:
        h (int) : height of the tree structure
        k (int) : number of children of each clique
        n (int) : number of nodes in each clique

    Returns:
        networkx.Graph: The Tree of Cliques graph.
    '''
    
    # Construct root clique
    if root_cluster:
        tok = nx.complete_graph(n)
    else:
        tok = nx.Graph()
        tok.add_node(0)

    # Base case: return Clique of size n
    if h == 0:
        return tok

    # Recursive case: build subtrees and join with root clique
    for i in range(k):

        # Construct child subtree
        child_tok = tree_of_cliques(h - 1, k, n, root_cluster=True)

        # Join with root clique
        subtree_start_idx = len(tok)
        tok = nx.disjoint_union(tok, child_tok)

        # Connect root to child subtree
        if root_cluster:
            tok.add_edge((i + 1) % n, subtree_start_idx)
        else:
            tok.add_edge(0, subtree_start_idx)

    return tok


def barabasi_albert(n, k, seed = 0, **kwargs):
    return nx.barabasi_albert_graph(n, k, seed = seed, **kwargs)


def small_world(n, k, p, seed = 0, **kwargs):
    return nx.watts_strogatz_graph(n, k, p, seed = seed, **kwargs)


def karate():
    return nx.from_scipy_sparse_array(load_spmatfile(str(REAL_WORLD_DIR / 'karate.mat')))

def homer():
    return nx.from_scipy_sparse_array(load_spmatfile(str(REAL_WORLD_DIR / 'homer.mat')))


def football():
    return nx.from_scipy_sparse_array(load_spmatfile(str(REAL_WORLD_DIR / 'football.mat')))


def football_2023():
    return nx.from_scipy_sparse_array(load_spmatfile(str(REAL_WORLD_DIR / 'football_2023.mat')))


def tabula(organ="vasculature"):
    return nx.from_scipy_sparse_array(load_spmatfile(str(REAL_WORLD_DIR / f'tabula-{organ}.mat')))


def polblogs():
    return nx.from_scipy_sparse_array(load_spmatfile(str(REAL_WORLD_DIR / 'polblogs-lwcc.mat')))


def sbmer3modes():
    return nx.from_scipy_sparse_array(load_spmatfile(str(SYNTHETIC_NET_DIR / 'sbm-er-3modes.mat')))


def mycielski(n = 7):
    """Generate the Mycielski graph of order n.

    Args:
        n (int, optional): Order of the Mycielski graph. Defaults to 7.

    Returns:
        networkx.Graph: The Mycielski graph.

    Notes:
        The Mycielski graph is a triangle-free graph with chromatic number n.
        The embedding does not show any particular structure.
    """
    return nx.mycielski_graph(n)


def lfr():
    G = nx.LFR_benchmark_graph(
        n=1000, tau1=3, tau2=1.5, mu=0.1, average_degree=10, min_community=200, seed=0
    )


    # keep only the largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    largest_subgraph = G.subgraph(largest_cc).copy()

    largest_subgraph.remove_edges_from(nx.selfloop_edges(largest_subgraph))
    relabeled_subgraph = nx.convert_node_labels_to_integers(largest_subgraph, first_label=0)

    return relabeled_subgraph


# ---------------------------------------------------------------------------- #
#                               Utility functions                              #
# ---------------------------------------------------------------------------- #


def load_spmatfile(filename: Path):

    mat_data = sio.loadmat(filename)

    # read sparse matrix
    A = mat_data["A"]

    # transform to CSC matrix
    A = sp.csc_matrix(A)

    return A


def stick_sculpture(n):
    """
    Generate a stick sculpture graph with n sticks across a path on the wall and a path on the floor.

    Args:
        n (int): Number of sticks.

    Returns:
        networkx.Graph: The resulting stick sculpture graph.
    """
    ident = np.eye(n)

    Apath = ident.copy()
    Apath[n-1, n-1] = 0
    Apath = Apath[:, [n-1] + list(range(n-1))]

    Astar = np.zeros((n, n))
    Astar[0, 1:ceil(2*n/3)] = 1

    Awall = Apath + Astar
    Awall = Awall + Awall.T

    Astar[0, 1:ceil(3*n/4)] = 1
    Afloor = Apath + Astar
    Afloor = Afloor + Afloor.T

    twist = ident[:, [0] + list(range(n-1, 0, -1))]

    A = np.block([[Awall, twist], [twist.T, Afloor]])

    G = nx.from_numpy_array(A)
    return G


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
