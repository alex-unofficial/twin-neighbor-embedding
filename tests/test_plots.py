"""`plots.count_intersections` did not move when it was rebuilt on the crossing pipeline.

It feeds the paper's Figure 3 — crossings against a length threshold — and step 5 reimplemented
it on `twin.metrics.crossings` layer 2 so the tree holds one intersection routine instead of two.
`tests/test_paper_figures.py` would not have noticed a moved curve: it checks that the figure
renders, not what it shows. These are the pins that notice.

⚠️ The reimplementation passes `include_adjacent=False` **permanently and by intent** (DECISIONS
(31a)). The shapely version skipped edge pairs sharing a vertex; eq. (78) does not. That
divergence is deliberate for this figure and is recorded in spec §0.5 as standing, not scheduled.
In a *straight* drawing it turns out to change no count except for collinear overlap — see the
test below, which is where that claim is pinned rather than asserted in prose.
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from twin.metrics.crossings import crossing_pairs
from twin.plots import count_intersections

from tests.fixtures.make_count_intersections_pin import fixtures

PINS = Path(__file__).parent / "fixtures" / "count_intersections_pin.npz"


@pytest.mark.parametrize("name", ["karate", "cycle_ties"])
def test_count_intersections_reproduces_its_pre_rewrite_output(name):
    """Exact equality on all three arrays. `cycle_ties` is the {12/5} star polygon, where every
    edge has the same length — so the descending sort is entirely ties and a change in
    tie-breaking would move the cumulative curve even though the counts were right."""
    pinned = np.load(PINS)
    G, y = fixtures()[name]

    lengths, per_edge, cumulative = count_intersections(G, y)

    np.testing.assert_array_equal(lengths, pinned[f"{name}_lengths"])
    np.testing.assert_array_equal(per_edge, pinned[f"{name}_per_edge"])
    np.testing.assert_array_equal(cumulative, pinned[f"{name}_cumulative"])


def test_the_three_arrays_are_mutually_consistent():
    """`per_edge` counts each crossing twice, `cumulative` once, and both sort by length."""
    G, y = fixtures()["karate"]
    lengths, per_edge, cumulative = count_intersections(G, y)

    assert (np.diff(lengths) <= 0).all()
    assert (np.diff(cumulative) >= 0).all()
    assert per_edge.sum() == 2 * cumulative[-1]
    assert cumulative[-1] == len(crossing_pairs(G, np.asarray(y).T, None, include_adjacent=False))


def test_include_adjacent_is_a_no_op_here_except_for_collinear_overlap():
    """What `include_adjacent=False` actually buys in a **straight** drawing, which is all this
    function draws.

    Two distinct straight edges sharing a vertex meet only at that vertex, and layer 2 excludes
    that contact by identity whichever way the flag is set — so for ordinary configurations the
    flag changes no count and only saves work. It bites in exactly one case: two adjacent edges
    drawn along the same line and overlapping along a stretch. The thesis convention counts that
    as one crossing; the shapely version counted zero, because it only ever counted `Point`
    intersections; and `include_adjacent=False` keeps the zero.
    """
    ordinary = nx.Graph([(0, 1), (0, 2), (3, 4)])
    y = np.array([[0.0, 2.0, 1.0, 0.5, 1.5], [0.0, 0.0, 1.0, -1.0, 2.0]])
    V = np.asarray(y).T
    assert len(crossing_pairs(ordinary, V, None, include_adjacent=True)) == len(
        crossing_pairs(ordinary, V, None, include_adjacent=False)
    )

    overlapping = nx.Graph([(0, 1), (0, 2)])
    on_a_line = np.array([[0.0, 3.0, 2.0], [0.0, 0.0, 0.0]])
    assert len(crossing_pairs(overlapping, on_a_line.T, None, include_adjacent=True)) == 1
    assert count_intersections(overlapping, on_a_line)[2][-1] == 0


def test_the_one_residual_difference_from_the_shapely_version():
    """⚠️ Recorded rather than discovered later. Two **non-adjacent** edges drawn along the same
    line and overlapping: the shapely version counted 0, because it took only `Point`
    intersections and this one is a `LineString`; layer 2 applies spec §2.4's convention and
    counts 1. No layout in the roster produces the configuration — it needs four collinear
    vertices — and the new answer is the one the spec asks for, so it is left standing.
    """
    G = nx.Graph([(0, 1), (2, 3)])
    on_a_line = np.array([[0.0, 2.0, 1.0, 3.0], [0.0, 0.0, 0.0, 0.0]])

    assert count_intersections(G, on_a_line)[2][-1] == 1


def test_an_edgeless_graph_returns_three_empty_arrays():
    lengths, per_edge, cumulative = count_intersections(nx.empty_graph(3), np.zeros((2, 3)))
    assert lengths.shape == per_edge.shape == cumulative.shape == (0,)
