"""Which valuation a bare `nx.Graph` carries, and where it came from.

One support, three valuations — combinatorial (1), affinity (*w*), distance (*d*) — is a
`LayoutGraph` property (CLAUDE.md), but the `nx.Graph` objects that layer hands out, and the
graph a `Dataset` stores, carry no trace of it: the edge attribute is called ``weight`` either
way. That is the shape of bug §1.6 (a distance consumed as an affinity) and of the step-2
graphviz regression (the declared form was right; what arrived was not).

This module is deliberately tiny and dependency-free so that both the `Dataset` layer and the
`LayoutGraph` layer can stamp without either importing the other — `twin.layoutgraph` pulls in
`sgtsnepi` through `twin.twin_embedding`, which has no business being loaded to build a dataset.
"""

from typing import Literal

import networkx as nx

#: Graph attributes stamped on every `nx.Graph` this package hands out.
SIDE_KEY = "twin_side"
SOURCE_KEY = "twin_source"

Side = Literal["affinity", "distance", "combinatorial"]


def stamp_form(G: nx.Graph, *, side: Side, source: str) -> nx.Graph:
    """Record which valuation ``G``'s ``weight`` attribute carries, and where it came from.

    Two step-3 guards read the stamp, and both guard against a *silent* misuse rather than a
    loud one: :func:`~twin.metrics.common.ideal_distances` refuses an affinity-valued graph, and
    refuses any stamped graph handed to it together with an explicit ``weight_transform``
    (`DECISIONS.md` (10)); :func:`~twin.metrics.structural.adjacency_distance_ratio` refuses a
    distance-valued one. A bare unstamped graph stays acceptable to both, so the analytic tests
    of §6 can build a $K_4$ by hand without a `Dataset`.

    Args:
        G: the graph to stamp, modified in place.
        side: what its ``weight`` attribute holds.
        source: free-form provenance, for the error messages and the manifest.

    Returns:
        ``G``.
    """
    G.graph[SIDE_KEY] = side
    G.graph[SOURCE_KEY] = source
    return G


def graph_side(G: nx.Graph) -> Side | None:
    """The stamped side of a graph, or None if nothing ever declared one for it."""
    return G.graph.get(SIDE_KEY)


# `graph_source(G)` lived here and was deleted: it had no callers, because every guard that
# names a graph's provenance in an error message reads `G.graph.get(SOURCE_KEY)` inline
# (`metrics/common.py`, `metrics/structural.py`). `graph_side` is the accessor that carries
# meaning — it is what the guards branch on — and `SOURCE_KEY` is exported for the message.
