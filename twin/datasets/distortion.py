"""How much of the ambient metric survives sparsification (spec §3.4).

For a point-cloud-derived dataset, the graph is the **stipulated ground truth**: sparsification
happens in the factory, every method and every metric operates inside the same universe, and a
layout is not expected to recover distances the method was never shown. That stipulation is
legitimate — sparse input is the regime the whole method family is defined on — but it is only
*honest* if its cost is stated.

For `digits` and `tabula` the unsparsified source is available and is deliberately declined, so
the cost is measurable rather than merely acknowledged. This is the analogue of
:func:`~twin.datasets.consistency.check_weight_consistency`: that one asks whether two stored
sides agree with each other, this one asks whether the stored side agrees with the source it was
sparsified from.

What the shortest-path closure of a *k*-NN graph is, precisely:

* **exact on the *k*-NN edges.** An edge's weight *is* the ambient distance, and no path can beat
  it by the triangle inequality, so $d_{geo} = d_{full}$ there;
* **strictly larger elsewhere**, for the same reason — graph geodesics only ever overestimate.
  The distortion is therefore a one-sided inflation of *long-range* distances, which is where it
  lands on the metrics: local ones (kNN recall, NND, edge lengths) are undistorted, and the
  global ones (SNS, ADR, Sil\\*) score against a proxy;
* **monotone in *k***, converging to the ambient metric as the graph fills in.

So SNS's target on a *k*-NN graph with Euclidean edge weights is an Isomap distance, and that is
a nameable target rather than a spoiled one.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

#: Source vertices to run shortest paths from. 100 x n pairs is ample for a median and a rank
#: correlation, and keeps the whole report to a few seconds.
DEFAULT_SOURCES = 100


@dataclass
class DistortionReport:
    """Agreement between the graph geodesic and the ambient metric it was sparsified from.

    Attributes:
        dataset: the dataset's name.
        n_sources / n_pairs: sampled source vertices, and the pairs they produced.
        spearman: rank correlation between geodesic and ambient distance. This is the headline:
            it says whether the graph preserves the *ordering* of the ambient metric, which is
            what every rank-based metric depends on.
        relative_stress: $\\sqrt{\\sum (d_{geo} - d_{full})^2 / \\sum d_{full}^2}$ — the
            magnitude of the inflation, not just its ordering.
        scaled_relative_stress: the same after the optimal uniform rescaling
            $\\alpha^{*} = \\langle d_{geo}, d_{full}\\rangle / \\langle d_{geo}, d_{geo}\\rangle$.
            This is the number to read for SNS, which has its own closed-form $\\alpha^{*}$ and
            therefore absorbs a uniform inflation entirely. The gap between the two says how much
            of the raw stress is *just* a change of scale.
        median_ratio / max_ratio: $d_{geo} / d_{full}$. Both are >= 1 by the triangle
            inequality; a median near 1 means the geodesic tracks the ambient metric closely.
        edge_weight_max_error: largest discrepancy between a stored edge weight and the ambient
            distance between its endpoints. Non-zero would mean the edge weights are not
            distances in the declared space at all, which would invalidate everything above.
        n_unreachable: sampled pairs with no path. Non-zero on a connected graph is impossible
            and would indicate the distance form disagrees with the graph.
    """

    dataset: str
    n_sources: int
    n_pairs: int
    spearman: float
    relative_stress: float
    scaled_relative_stress: float
    median_ratio: float
    max_ratio: float
    edge_weight_max_error: float
    n_unreachable: int

    def summary(self) -> str:
        return (
            f"rho={self.spearman:+.4f}, stress={self.relative_stress:.4f} "
            f"({self.scaled_relative_stress:.4f} after optimal rescaling), "
            f"d_geo/d_full median={self.median_ratio:.4f} max={self.max_ratio:.4f} "
            f"over {self.n_pairs} sampled pairs"
        )


def sparsification_distortion(
    dataset,
    distance: nx.Graph,
    *,
    n_sources: int = DEFAULT_SOURCES,
    seed: int = 0,
) -> DistortionReport | None:
    """Compare graph geodesics against the ambient metric. None unless a source cloud is stored.

    Args:
        dataset: a :class:`~twin.datasets.Dataset` with ``source_points``.
        distance: the **vertex graph's** distance form — the ideal-distance source. Passed in
            rather than derived here, so that this module does not depend on the layout layer.
        n_sources: vertices to run single-source shortest paths from.
        seed: for the source sample.
    """
    points = dataset.source_points
    if points is None:
        return None

    points = np.asarray(points, dtype=np.float64)
    n = dataset.n

    csgraph = nx.to_scipy_sparse_array(distance, nodelist=range(n), weight="weight")
    sources = np.sort(
        np.random.default_rng(seed).choice(n, size=min(n_sources, n), replace=False)
    )

    geodesic = dijkstra(csgraph, directed=False, indices=sources)
    ambient = cdist(points[sources], points)

    # Drop each source's distance to itself; keep everything else.
    keep = np.ones_like(geodesic, dtype=bool)
    keep[np.arange(sources.size), sources] = False

    unreachable = int(np.isinf(geodesic[keep]).sum())
    finite = keep & np.isfinite(geodesic)
    d_geo, d_full = geodesic[finite], ambient[finite]

    ratio = d_geo / d_full
    total = float(np.sum(d_full**2))
    stress = float(np.sqrt(np.sum((d_geo - d_full) ** 2) / total))

    # SNS fits its own global scale, so the fair comparison rescales first.
    alpha = float(np.dot(d_geo, d_full) / np.dot(d_geo, d_geo))
    scaled_stress = float(np.sqrt(np.sum((alpha * d_geo - d_full) ** 2) / total))

    return DistortionReport(
        dataset=dataset.name,
        n_sources=int(sources.size),
        n_pairs=int(d_geo.size),
        spearman=float(spearmanr(d_geo, d_full).statistic),
        relative_stress=stress,
        scaled_relative_stress=scaled_stress,
        median_ratio=float(np.median(ratio)),
        max_ratio=float(ratio.max()),
        edge_weight_max_error=_edge_weight_error(distance, points),
        n_unreachable=unreachable,
    )


def _edge_weight_error(distance: nx.Graph, points: np.ndarray) -> float:
    """Largest |stored edge weight − ambient distance between its endpoints|.

    Zero is the expected answer and the precondition for everything else: it confirms the edge
    weights really are distances in the space `source_points` lives in.
    """
    if distance.number_of_edges() == 0:
        return 0.0
    edges = np.array([(u, v) for u, v in distance.edges()], dtype=int)
    stored = np.array([w for _, _, w in distance.edges(data="weight")], dtype=float)
    ambient = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    return float(np.abs(stored - ambient).max())
