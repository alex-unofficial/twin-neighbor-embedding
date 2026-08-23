"""How a dataset's graph got its edge set, declared and checkable (spec §3.4).

**Sparse input is the operating regime, not a compromise.** Every method under study consumes a
sparse graph, and the twin transform *requires* one: on a complete graph over tabula's 5007
vertices, $A_E = B^{\\top}B$ would carry $\\sum_v \\deg(v)(\\deg(v)-1) \\approx 10^{11}$ nonzeros
and $A_{tne}$ would span $n + m \\approx 12.5$ M nodes. A dense-input comparison is not an
experiment that could be run at any budget.

So sparsification happens in the **dataset factory**, before any representation or layout method
exists. Two consequences that make that the right place:

* the graph's edge set defines the *original edge indexing* that `vertex_and_edge_points` returns
  into — if a layout-layer knob could change the support, "original edge *e*" would become a
  function of the representation;
* $k$ sweeps through the existing batch axis for free, as
  ``("digits", {"n_neighbors": [5, 12, 30]})``, which is correct semantically: a different $k$ is
  a different dataset, not a different view of one.

This module holds the **declaration**, not a builder. Where a dataset computes its own kNN graph
it does so in its factory (digits, via `twin.dataset.construct_knn_graph`); where the export
already shipped one it does not compute anything at all (tabula). A general sparsifier for a
dense stored distance is deliberately not written until a dataset needs one — same call as
`largest_component()` and `disconnected="components"`.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class KNearest:
    """The graph is a symmetrised *k*-nearest-neighbour graph.

    Chosen over an $r$-neighbourhood because it bounds degree and cannot leave a vertex
    isolated, and because it is what UMAP, *t*-SNE and tabula's own export already do. Note that
    for a globally monotone affinity derivation such as ``exp(-d/d_scale)``, taking the *k*
    nearest by distance and the *k* largest by affinity select the **same** set — so there is
    only one place to sparsify, and it is the distance side.

    Args:
        k: neighbours per vertex, **before** symmetrisation. This is a budget parameter and is
            declared rather than derived from a target density: §4.2's λ criterion is defined at
            *k* = mean degree, so it is already a first-class experimental quantity, and
            choosing it via a density proxy would hide the axis one most wants visible.
        symmetrise: ``"or"`` keeps an edge present in either direction, ``"and"`` only mutual
            neighbours.
        space: where the distances were measured, when that is not the raw feature space —
            tabula's are Euclidean in a 50-d scVI latent representation, not in gene space.
        metric: the distance used.
    """

    k: int
    symmetrise: Literal["or", "and"] = "or"
    space: str = ""
    metric: str = "euclidean"

    def describe(self) -> str:
        where = f" in {self.space}" if self.space else ""
        return f"{self.k}-NN ({self.metric}{where}), symmetrised by {self.symmetrise}"

    def check(self, dataset) -> list[str]:
        """Violations of what this declaration implies, empty if it holds.

        Two invariants, both structural rather than statistical:

        * **every vertex has degree at least k.** A *k*-NN graph gives each vertex *k* out-edges
          and or-symmetrisation only adds; this is the hard floor that distinguishes a *k*-NN
          graph from an $r$-neighbourhood one, and it is what fails if a file is re-exported at
          a different *k*.
        * **nk/2 ≤ m ≤ nk.** The lower bound is every edge reciprocated, the upper bound none.
        """
        if self.symmetrise != "or":
            return []  # "and" bounds differ; no registered dataset uses it, so it is not guessed at

        degrees = np.array([d for _, d in dataset.graph.degree()])
        n, m = dataset.n, dataset.m
        problems = []

        if degrees.size and degrees.min() < self.k:
            problems.append(
                f"declared {self.k}-NN but {int((degrees < self.k).sum())} vertices have degree "
                f"below {self.k} (minimum {int(degrees.min())})"
            )
        if not (n * self.k / 2 <= m <= n * self.k):
            problems.append(
                f"declared {self.k}-NN over {n} vertices implies "
                f"{int(n * self.k / 2)} <= m <= {n * self.k}, got m = {m}"
            )
        return problems


#: What a dataset declares when its edge set is the data rather than a choice — every
#: combinatorial graph in the roster, where there is no point cloud and nothing was discarded.
SparsityModel = KNearest
