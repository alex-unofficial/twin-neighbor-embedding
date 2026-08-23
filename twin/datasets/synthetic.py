"""Synthetic generators, all combinatorial (spec §3.4).

Labels that are **known by construction** are returned by the generator rather than
reconstructed afterwards: the clique index of a vertex in ``ring_of_cliques(n, k)`` is ``i // k``
and should not be re-derived by eye. Where a generator has no unambiguous ground-truth
partition — the tree of cliques' overlapping structure, a Watts-Strogatz ring, a
Barabasi-Albert graph — it registers none, rather than inventing one.
"""

import numpy as np

from twin.dataset import barabasi_albert as _barabasi_albert
from twin.dataset import ring_of_cliques as _ring_of_cliques
from twin.dataset import small_world as _small_world
from twin.dataset import tree_of_cliques as _tree_of_cliques
from twin.datasets.base import Dataset, DatasetStyle, LabelSet, Unweighted


def _combinatorial(name, graph, params, description, label_sets=None) -> Dataset:
    return Dataset(
        name=name,
        graph=graph,
        affinity=Unweighted(),
        distance=Unweighted(),
        label_sets=label_sets or {},
        style=DatasetStyle(palette="husl", palette_seed=0),
        params=params,
        description=description,
    )


def ring_of_cliques(n: int = 10, k: int = 4) -> Dataset:
    graph = _ring_of_cliques(n, k)
    clique = np.arange(graph.number_of_nodes()) // k

    return _combinatorial(
        "ring_of_cliques",
        graph,
        {"n": n, "k": k},
        f"{n} cliques of {k} vertices joined in a ring. The cleanest case where the twin "
        "transform should separate dense groups from the sparse links between them.",
        label_sets={
            "clique": LabelSet(
                name="clique",
                values=clique,
                display_names={i: f"clique {i}" for i in range(n)},
                derivation="generator",
                scoring_role="primary",
                display_role="hue",
            )
        },
    )


def tree_of_cliques(h: int = 3, k: int = 3, n: int = 4) -> Dataset:
    return _combinatorial(
        "tree_of_cliques",
        _tree_of_cliques(h=h, k=k, n=n),
        {"h": h, "k": k, "n": n},
        f"A depth-{h} tree of {n}-cliques with {k} children each. Hierarchical, and the case "
        "the crossing metrics are most visible on.",
    )


def small_world(n: int = 150, k: int = 12, p: float = 0.01) -> Dataset:
    return _combinatorial(
        "small_world",
        _small_world(n, k, p),
        {"n": n, "k": k, "p": p},
        f"Watts-Strogatz ring, n={n}, k={k}, rewiring p={p}. Locally dense with a known "
        "ground-truth geometry — the ring — so a layout can be judged by eye.",
    )


def barabasi_albert(n: int = 1000, k: int = 7) -> Dataset:
    return _combinatorial(
        "barabasi_albert",
        _barabasi_albert(n, k),
        {"n": n, "k": k},
        f"Preferential attachment, n={n}, k={k}. Heavy-tailed degree, which is the regime where "
        "the twin transform's cost is worst and hub crowding is most visible.",
    )
