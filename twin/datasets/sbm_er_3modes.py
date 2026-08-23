"""sbm-er-3modes — a stochastic block model with one constant edge weight.

All 5753 edges carry the same weight, 8.691e-05. A constant weight carries no information, so
both sides are declared `Unweighted()` (spec §0.5) — the kind table has no row for "constant",
and treating it as a `strength` would hand the distance derivation a constant 1/w that says
exactly as little.

Block membership is not stored in the `.mat`, so no label set is registered. If the generator
parameters are ever recovered, the blocks belong here as `derivation="generator"`.
"""

import networkx as nx

from twin.config import SYNTHETIC_NET_DIR
from twin.dataset import load_spmatfile
from twin.datasets.base import Dataset, DatasetStyle, Unweighted


def sbm_er_3modes() -> Dataset:
    graph = nx.from_scipy_sparse_array(
        load_spmatfile(str(SYNTHETIC_NET_DIR / "sbm-er-3modes.mat"))
    )
    return Dataset(
        name="sbm_er_3modes",
        graph=graph,
        affinity=Unweighted(),
        distance=Unweighted(),
        style=DatasetStyle(palette="husl", palette_seed=0),
        params={},
        description=(
            "Stochastic block model with Erdos-Renyi background, three modes. Dense "
            "(mean degree 55) relative to the rest of the roster, which makes it the case "
            "where the line graph is most expensive."
        ),
    )
