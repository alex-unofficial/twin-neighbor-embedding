"""Zachary's karate club — two different graphs under two names (spec §0.5 decision 2).

The registry previously exposed only the *weighted* one, as `karate_club`, while the spec's
table described the unweighted one. They are separate datasets:

* `karate` reads `karate.mat` and is unweighted — the combinatorial small case;
* `karate_weighted` is `nx.karate_club_graph`, whose 1-7 interaction strengths make it the
  genuinely strength-weighted small test case that football turned out not to be.
"""

import networkx as nx
import numpy as np

from twin.config import REAL_WORLD_DIR
from twin.dataset import load_spmatfile
from twin.datasets.base import Dataset, DatasetStyle, LabelSet, Stored, Unweighted

CLUBS = {0: "Mr. Hi", 1: "Officer"}


def karate() -> Dataset:
    graph = nx.from_scipy_sparse_array(load_spmatfile(str(REAL_WORLD_DIR / "karate.mat")))
    return Dataset(
        name="karate",
        graph=graph,
        affinity=Unweighted(),
        distance=Unweighted(),
        style=DatasetStyle(palette="husl", palette_seed=0),
        params={},
        description=(
            "Zachary's karate club as stored in karate.mat: unweighted, and the smallest "
            "combinatorial case in the roster. The club split is not in the file — see "
            "karate_weighted for the labelled variant."
        ),
    )


def karate_weighted() -> Dataset:
    graph = nx.karate_club_graph()
    club = np.array([0 if graph.nodes[i]["club"] == "Mr. Hi" else 1 for i in graph.nodes()])

    return Dataset(
        name="karate_weighted",
        graph=graph,
        # Interaction strengths: counts of shared activities. They compose in series like
        # conductances, so the distance side is derived as 1/w (spec §0.5 decision 1).
        affinity=Stored(meaning="strength"),
        distance=None,
        label_sets={
            "club": LabelSet(
                name="club",
                values=club,
                display_names=CLUBS,
                derivation="ground truth",
                scoring_role="primary",
                display_role="hue",
            )
        },
        style=DatasetStyle(palette="husl", palette_seed=0),
        params={},
        description=(
            "nx.karate_club_graph, carrying Zachary's 1-7 interaction strengths. The small "
            "count-weighted case, and the only one that exercises the strength -> 1/w "
            "derivation."
        ),
    )
