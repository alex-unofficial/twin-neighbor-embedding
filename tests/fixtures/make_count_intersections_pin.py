"""Regenerate ``count_intersections_pin.npz`` — the characterisation pin for step 5 commit 8.

Run from the repository root with the project interpreter::

    venv/bin/python tests/fixtures/make_count_intersections_pin.py

`plots.count_intersections` feeds the paper's Figure 3 (crossings against a length threshold).
Step 5 reimplements it on the crossing pipeline's layer 2, and the reimplementation is meant to be
**behaviour-preserving** — so its three return arrays were pinned from the shapely-based
implementation *before* the rewrite. The figure smoke test only catches a raise, not a moved
curve; this is what catches a moved curve.

Two fixtures, because they fail differently: a random karate layout (all edge lengths distinct)
and the {12/5} star polygon (**every** edge the same length, so the descending-length sort is
entirely ties and any change in tie-breaking shows up in the cumulative curve).

Regenerating this file is only ever correct in a commit that *intends* to move Figure 3.
"""

from pathlib import Path

import networkx as nx
import numpy as np

from twin.plots import count_intersections

OUT = Path(__file__).with_name("count_intersections_pin.npz")


def fixtures() -> dict[str, tuple[nx.Graph, np.ndarray]]:
    """Each layout is ``(d, n)`` — `plots.py` is still the legacy orientation (CLAUDE.md)."""
    karate = nx.karate_club_graph()
    random_layout = np.random.default_rng(20260914).normal(size=(2, karate.number_of_nodes()))

    # The {12/5} star polygon: a 12-cycle whose vertices sit on a circle in the order 0, 5, 10,
    # ... so every edge is the same chord and they cross each other constantly.
    cycle = nx.cycle_graph(12)
    angles = 2 * np.pi * (5 * np.arange(12) % 12) / 12
    star = np.stack([np.cos(angles), np.sin(angles)])

    return {"karate": (karate, random_layout), "cycle_ties": (cycle, star)}


def build() -> dict[str, np.ndarray]:
    arrays = {}
    for name, (G, y) in fixtures().items():
        lengths, per_edge, cumulative = count_intersections(G, y)
        arrays[f"{name}_lengths"] = lengths
        arrays[f"{name}_per_edge"] = per_edge
        arrays[f"{name}_cumulative"] = cumulative
    return arrays


if __name__ == "__main__":
    arrays = build()
    np.savez_compressed(OUT, **arrays)
    for name, value in arrays.items():
        print(f"{name:22s} {value.shape}  sum={value.sum():g}")
    print(f"\nwrote {OUT}")
