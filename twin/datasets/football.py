"""football_2023 — NCAA Division I FBS, 2023 season (spec §3.4).

Effectively unweighted: weights are games played, and 732 of 735 edges weigh 1 (spec §0.5), so
both sides are declared `Unweighted()` and the `strength` path is irrelevant for it.

The interesting part is the label hierarchy. `conference` is the primary scoring label *and*
supplies hue; the East/West division subdivides it and supplies shade. That is why the existing
figures pick hue from the conference and vary only shade by division. The division itself is
**derived** — from the trailing parenthesis of the `conference_standing` string — which used to
be a slicing expression buried inside a plotting function.
"""

from itertools import count

import networkx as nx
import numpy as np

from twin.config import REAL_WORLD_DIR
from twin.dataset import load_spmatfile
from twin.datasets.base import Dataset, DatasetStyle, LabelSet, Unweighted
from twin.datasets.football_metadata import CONFERENCE, CONFERENCE_STANDING

#: The independents are not a conference; they are the absence of one.
INDEPENDENTS = "NCAA Division I FBS independents"

#: West < none < East, so that the ordered sub-label maps onto the three shade factors.
DIVISION_CODES = {"West": -1, "East": 1}
DIVISION_NAMES = {-1: "West", 0: "none", 1: "East"}


def conference_codes() -> tuple[np.ndarray, dict[str, int]]:
    """Integer-coded conference, in first-appearance order, independents at ``-1``."""
    counter = count(0)
    code_of = {
        conf: (-1 if conf == INDEPENDENTS else next(counter))
        for conf in dict.fromkeys(CONFERENCE)
    }
    return np.array([code_of[conf] for conf in CONFERENCE]), code_of


def division_codes() -> np.ndarray:
    """East / none / West, from the parenthesised suffix of ``conference_standing``.

    ``"Big Ten (West)"[-5:-1] == "West"``. Lifted out of `football_figure` verbatim, where it
    was an unnamed slice; pinned in ``tests/test_step2_characterisation.py``.
    """
    return np.array([DIVISION_CODES.get(stand[-5:-1], 0) for stand in CONFERENCE_STANDING])


def football_2023() -> Dataset:
    graph = nx.from_scipy_sparse_array(
        load_spmatfile(str(REAL_WORLD_DIR / "football_2023.mat"))
    )

    conference, code_of = conference_codes()
    division = division_codes()

    return Dataset(
        name="football_2023",
        graph=graph,
        affinity=Unweighted(),
        distance=Unweighted(),
        label_sets={
            "conference": LabelSet(
                name="conference",
                values=conference,
                display_names={code: name for name, code in code_of.items()},
                derivation="ground truth",
                scoring_role="primary",
                display_role="hue",
                noise_label=-1,
            ),
            "division": LabelSet(
                name="division",
                values=division,
                display_names=DIVISION_NAMES,
                derivation='derived:conference_standing[-5:-1] in {"East", "West"}',
                scoring_role="sub",
                display_role="shade",
                parent="conference",
                ordered=True,
            ),
        },
        composite_of={"conference_division": ("conference", "division")},
        style=DatasetStyle(
            palette="husl",
            palette_seed=1,
            shade_factors=(0.7, 1.0, 1.3),
            noise_color=(0.0, 0.0, 0.0),
            vertex_size=15.0,
            edge_alpha_intra=0.6,
            edge_alpha_inter=0.1,
        ),
        params={},
        description=(
            "NCAA Division I FBS 2023 season; an edge is a game played. The conference / "
            "division hierarchy is the worked example for composite-label scoring: TNE claims "
            "to reveal the sub-conference sub-clusters, and the composite label is what tests "
            "that directly."
        ),
    )
