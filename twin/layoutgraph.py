"""The `LayoutGraph` layer and the three mappings (spec §3.4, §3.5).

A `LayoutGraph` is an affinity form, a distance form, and a way to derive vertex/edge points.
It is laid out and is **never labelled** — $A_E$'s nodes are edges and $A_{tne}$'s are edges
*and* vertices, so a length-$n$ label vector does not apply to either. Labels do not need to
ride along: :meth:`LayoutGraph.vertex_and_edge_points` returns points in the **original**
indexing, so a layout of any representation is scored against the dataset's own labels with no
length mismatch possible.

Three mappings, matching the algebra::

    vertex_graph(dataset)  :  Dataset     -> LayoutGraph(A_V)     <- derivation happens here
    edge_graph(vg)         :  LayoutGraph -> LayoutGraph(A_E)
    twin_graph(vg, ...)    :  LayoutGraph -> LayoutGraph(A_tne)

Both derived graphs are built from the vertex graph's **affinity** form, always:
$A_E = B^{\\top}B$ and $A_{tne} = \\sum_k \\alpha^k \\tilde A^k$ are affinity algebra, and feeding
them distances inverts the meaning of every term in the polynomial. Building them off the
vertex graph also means the incidence matrix is constructed once rather than three times, and
that the derivation chain can be read off a `parent` link instead of maintained as a string.

This module replaces the `GraphMatrixMethod` classes in :mod:`twin.graphmatrix`, which survive
only until the plotting refactor (spec §3.6, step 10) because `graph_embedding_all` and
`paper_figures` still use them. Do not add callers of those.
"""

from dataclasses import dataclass, field
from typing import Callable, Literal

import networkx as nx
import numpy as np
import scipy.sparse as sp

from twin.forms import SIDE_KEY, SOURCE_KEY, Side, graph_side, stamp_form  # noqa: F401
from twin.graphmatrix import compute_slopes
from twin.twin_embedding import twin_adjacency

Kind = Literal["vertex", "edge", "twin"]

#: How the affinity side is derived when only a distance side is stored.
AffinityRule = Literal["exp_decay", "exp_decay_sq"]


# --------------------------------------------------------------------------- #
#                                  the type                                    #
# --------------------------------------------------------------------------- #


@dataclass
class LayoutGraph:
    """One representation of one dataset, ready to be laid out. Never labelled.

    Attributes:
        name: ``"A_V"``, ``"A_E"`` or ``"A_tne"``.
        kind: which of the three mappings produced it.
        affinity: the matrix that is laid out, ``(n_points, n_points)``.
        incidence / incidence_weighted: the original graph's incidence matrices, built **once**
            on the vertex graph and shared by the derived ones. Columns are in
            ``dataset.graph.edges()`` order, which is what makes edge points indexable by edge.
        n_vertices / n_edges: of the **original** graph, whatever this representation's node
            set is.
        point_provenance: per point set, whether those points were laid out directly
            ("primary") or reconstructed from the other set ("derived") — spec §0.5 decision 5.
        parent: the `LayoutGraph` this one was derived from, or None for the vertex graph.
        dataset_name: the dataset at the root of the chain.
        is_faithful: whether the affinity reproduces the dataset's own edge weights.
            ``normalize=True`` perturbs $A_V$ and breaks this; it is declared rather than
            inferred.
        derivation: how this graph's *affinity* was obtained.
        distance_derivation: how its *distance* form is obtained.
        params: the method knobs (`normalize`, `alpha`, `k`) for the manifest.
    """

    name: str
    kind: Kind
    affinity: sp.sparray
    incidence: sp.csc_array
    incidence_weighted: sp.csc_array
    n_vertices: int
    n_edges: int
    point_provenance: dict[str, str]

    parent: "LayoutGraph | None" = None
    dataset_name: str | None = None
    is_faithful: bool = False
    derivation: str = ""
    distance_derivation: str = ""
    params: dict = field(default_factory=dict)

    _build_distance: Callable[[], nx.Graph] | None = field(
        default=None, repr=False, compare=False
    )
    _distance: nx.Graph | None = field(default=None, repr=False, compare=False)
    _affinity_graph: nx.Graph | None = field(default=None, repr=False, compare=False)
    _combinatorial: sp.sparray | None = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------- the forms

    @property
    def n_points(self) -> int:
        """Rows of the matrix actually laid out: n for A_V, m for A_E, n + m for A_tne."""
        return self.affinity.shape[0]

    @property
    def combinatorial(self) -> sp.sparray:
        """The 0/1 structure: presence or absence of a connection, ``(n_points, n_points)``.

        A `LayoutGraph` is **one support carrying up to three valuations** — combinatorial (1),
        affinity (*w*) and distance (*d*) — and this is the first of them. Some methods want no
        valuation at all: Yifan Hu's is a combinatorial force method, attractive $d^2/K$ along
        edges and repulsive $-CK^2/d$ between all pairs with $K$ a *global* constant, so it has
        no per-edge target to read.

        Derived by rerunning the same algebra on the **unweighted** incidence, not by
        thresholding the affinity. Those agree today, but `normalize=True` scales entries by
        $1/\\sqrt{\\deg}$ and a float comparison against zero is the wrong thing to hang a
        support on.
        """
        if self._combinatorial is None:
            B = self.incidence
            if self.kind == "vertex":
                A = B @ B.T
            elif self.kind == "edge":
                A = B.T @ B
            else:
                A = twin_adjacency(B, self.params.get("alpha", 0.85), n=self.params.get("k", 2))

            A = sp.csr_array(A - sp.diags_array(A.diagonal()))
            A.eliminate_zeros()
            A.data = np.ones_like(A.data)
            self._combinatorial = A
        return self._combinatorial

    @property
    def distance(self) -> nx.Graph:
        """The distance form, as a graph. Non-edges are **absent**, never present at 1.

        For the **vertex graph** this is the *ideal-distance source*: the dataset's stored
        distance side if it has one, otherwise the single derivation from its stored affinity.
        It is by construction the least-derived distance form available, and ideal distances
        for SNS and Sil* come from here — computed once per dataset, shared by every
        representation, owned by the runner (spec §2.0, `DECISIONS.md`).

        **Never take ideal distances from `edge_graph` or `twin_graph`.** Their distance form is
        a generic fallback out of a matrix that is not in the dataset's units, so scoring
        against it would measure $A_V$, $A_E$ and $A_{tne}$ against three *different* targets
        and make the central ablation — which is the point of the whole experiment — mean
        nothing.
        """
        if self._distance is None:
            self._distance = self._build_distance()
        return self._distance

    def affinity_graph(self) -> nx.Graph:
        """The affinity form as a graph, stamped ``twin_side="affinity"``.

        The counterpart of :attr:`distance`, for the metrics that weight a pair by how strongly
        it is coupled rather than by how far apart it should be: ADR and `weighted_recall`
        (spec §2.1). Its node set is this representation's, so it is the **vertex** graph's
        affinity a vertex-indexed metric wants — $A_E$'s nodes are edges.

        Why not just ``dataset.graph``: on `digits` that graph's ``weight`` is a stored
        Euclidean distance, so a metric reading it scores the inverse of what it means
        (`DECISIONS.md`, step-3 decisions). This form is an affinity by construction on every
        dataset, derived or stored.
        """
        if self._affinity_graph is None:
            G = nx.from_scipy_sparse_array(sp.csr_array(self.affinity))
            self._affinity_graph = stamp_form(
                G,
                side="affinity",
                source=f"{self.name} of {self.dataset_name}: {self.derivation}",
            )
        return self._affinity_graph

    def chain(self) -> list[str]:
        """The derivation chain, root first — structural, not a maintained string."""
        above = (
            self.parent.chain()
            if self.parent is not None
            else [f"dataset:{self.dataset_name}"]
        )
        return above + [f"{self.name}: {self.derivation}"]

    def distance_chain(self) -> list[str]:
        """The chain the manifest records for a distance-consuming layout method."""
        return self.chain() + [f"distance: {self.distance_derivation}"]

    # --------------------------------------------------------------- points

    def vertex_and_edge_points(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split a raw ``(d, n_points)`` layout into vertex and edge points.

        Both come back in the **original** indexing — vertex ``i`` of ``dataset.graph``, and the
        ``e``-th entry of ``dataset.graph.edges()`` — which is what lets the dataset's labels
        apply to a layout of any representation.
        """
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_points:
            raise ValueError(
                f"{self.name}: expected a (d, {self.n_points}) layout, got {X.shape}. The "
                "embedding layer is (d, n); LayoutResult.from_raw is the only conversion point."
            )

        if self.kind == "vertex":
            X_v, X_e = X, _edge_points_from_vertices(self.incidence, X)
        elif self.kind == "edge":
            X_v, X_e = _vertex_points_from_edges(self.incidence, self.affinity, X), X
        else:
            X_v, X_e = X[:, self.n_edges :], X[:, : self.n_edges]

        return X_v, X_e, compute_slopes(self.incidence, X_v)


def _edge_points_from_vertices(incidence: sp.sparray, X_v: np.ndarray) -> np.ndarray:
    """Each edge point is a degree-weighted convex combination of its endpoints."""
    degree = np.asarray(incidence.sum(axis=1)).reshape(-1, 1)
    inc_deg = incidence * degree
    inc_deg = inc_deg / np.asarray(inc_deg.sum(axis=0)).reshape(1, -1)
    return (inc_deg.T @ X_v.T).T


def _vertex_points_from_edges(
    incidence: sp.sparray, line_adjacency: sp.sparray, X_e: np.ndarray
) -> np.ndarray:
    """Each vertex is the edge-degree-weighted centroid of its incident edge points."""
    edge_degree = np.asarray(line_adjacency.sum(axis=0)).reshape(1, -1)
    inc_deg = edge_degree * incidence
    inc_deg = inc_deg / np.asarray(inc_deg.sum(axis=1)).reshape(-1, 1)
    return (inc_deg @ X_e.T).T


# --------------------------------------------------------------------------- #
#                            the derivation rules                              #
# --------------------------------------------------------------------------- #


def affinity_matrix_to_distance(A: sp.sparray) -> sp.sparray:
    """Generic affinity -> distance fallback: normalise, **symmetrise**, then $-\\log$.

    Used by `edge_graph` and `twin_graph`, whose matrices are not in the dataset's units —
    neither has entries that are "strengths" or "distances" in the original sense.

    Row normalisation alone leaves the matrix asymmetric, and `nx.from_scipy_sparse_array` then
    silently keeps one value of each pair. That is a live bug in
    `KamadaKawaiLayout(mode="similarity")` today (spec §0.5 decision 7), and the reason this is
    lifted out rather than copied.
    """
    A = sp.csr_array(A).astype(float)
    row_sums = np.asarray(A.sum(axis=1)).ravel()
    inv = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums != 0)

    normalised = sp.diags_array(inv) @ A
    symmetric = sp.csr_array((normalised + normalised.T) * 0.5)
    symmetric.eliminate_zeros()

    distance = symmetric.copy()
    distance.data = -np.log(distance.data)
    return distance


def _edge_weights(matrix: sp.sparray, graph: nx.Graph, what: str) -> np.ndarray:
    """Pull one weight per graph edge out of a matrix, in ``graph.edges()`` order."""
    if graph.number_of_edges() == 0:
        return np.zeros(0)

    endpoints = np.array(list(graph.edges()), dtype=int)
    weights = np.asarray(
        sp.csr_array(matrix)[endpoints[:, 0], endpoints[:, 1]]
    ).ravel().astype(float)

    missing = int((weights == 0).sum())
    if missing:
        raise ValueError(
            f"{what} is zero on {missing} of the graph's {graph.number_of_edges()} edges. A "
            "stored side must span the graph's edge set; a gap here would silently drop those "
            "edges from the representation."
        )
    return weights


def _derive_distance(weights: np.ndarray, meaning: str) -> tuple[np.ndarray, str]:
    """The affinity -> distance rules of spec §3.4, applied edgewise.

    Each rule names a model under which the distance form is *derived*, not merely monotone.
    $1/w$ is the resistance model, not a fallback: weights that compose in series like
    conductances give it a derivation, in the same way surprisal gives $-\\log$ one.
    """
    if meaning == "none":
        return np.ones_like(weights), "hop count (combinatorial)"
    if meaning in ("strength", "count"):
        return 1.0 / weights, "1/w (resistance model: weights compose in series)"
    if meaning == "kernel":
        w_max = weights.max()
        return -np.log(weights / w_max), f"-log(w/w_max) (latent metric), w_max={w_max:.6g}"
    if meaning == "probability":
        return -np.log(weights), "-log(w) (surprisal)"
    if meaning == "adaptive_kernel":
        raise ValueError(
            "an adaptive kernel is globally non-invertible: its per-point rho_i and sigma_i are "
            "not stored, so no single transform of w recovers d. A dataset declaring "
            "meaning='adaptive_kernel' must supply a stored distance side. Refusing, rather "
            "than applying -log(w/w_max), which does not apply here (spec §3.4)."
        )
    raise ValueError(f"no distance derivation for meaning {meaning!r}")


def _derive_affinity(
    weights: np.ndarray, meaning: str, rule: AffinityRule
) -> tuple[np.ndarray, str]:
    """The distance -> affinity rule of spec §3.4, applied edgewise.

    The exponential is the exact inverse of $-\\log(w/w_{\\max})$, so the affinities the layout
    is given and the ideal distances the metrics score against describe the *same* geometry; a
    Gaussian forward with a log reverse would not round-trip. ``d_scale`` is required, not
    optional: raw kNN distances span an order of magnitude, and $\\exp(-d)$ on unnormalised
    input underflows to zero for most edges.
    """
    if meaning == "none":
        return np.ones_like(weights), "indicator (combinatorial)"

    d_scale = float(np.median(weights))
    if d_scale <= 0:
        raise ValueError("median edge distance is zero; exp(-d/d_scale) has no scale to use")

    if rule == "exp_decay":
        return np.exp(-weights / d_scale), f"exp(-d/d_scale), d_scale={d_scale:.6g}"
    if rule == "exp_decay_sq":
        return (
            np.exp(-((weights / d_scale) ** 2)),
            f"exp(-(d/d_scale)^2), d_scale={d_scale:.6g}",
        )
    raise ValueError(f"unknown affinity rule {rule!r}")


def _incidence_matrices(
    graph: nx.Graph, weights: np.ndarray, normalize: bool
) -> tuple[sp.csc_array, sp.csc_array]:
    """The unweighted incidence $B$ and the weighted $B_w^{1/2}$, in ``graph.edges()`` order.

    Built directly from ``graph.edges()`` rather than from a matrix, so that column ``e`` *is*
    edge ``e`` by construction — every consumer that pairs an edge point with an edge (curve
    construction, slopes, crossings) depends on that.
    """
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    endpoints = np.array(list(graph.edges()), dtype=int).reshape(m, 2)

    rows = endpoints.ravel()
    cols = np.repeat(np.arange(m), 2)

    incidence = sp.csc_array((np.ones(2 * m), (rows, cols)), shape=(n, m))
    values = np.repeat(np.sqrt(weights), 2)

    if normalize:
        # sqrt(w_e) / sqrt(weighted degree), which is what makes A_V symmetric-normalised.
        weighted_degree = np.zeros(n)
        np.add.at(weighted_degree, rows, np.repeat(weights, 2))
        values = values / np.sqrt(weighted_degree)[rows]

    return incidence, sp.csc_array((values, (rows, cols)), shape=(n, m))


# --------------------------------------------------------------------------- #
#                               the three mappings                             #
# --------------------------------------------------------------------------- #


def vertex_graph(
    dataset, *, normalize: bool = False, affinity_rule: AffinityRule = "exp_decay"
) -> LayoutGraph:
    """$A_V$ — and the single place a missing side is derived (spec §3.4).

    $A_V = B_w^{1/2}(B_w^{1/2})^{\\top}$ reproduces $w_{ij}$ on edges, so it is a **faithful**
    view of the dataset and a classical vertex layout can use the dataset's own distances —
    tabula's *measured* `D` drives a Kamada-Kawai vertex layout, which is the point of storing
    it. ``normalize=True`` perturbs $A_V$ and breaks faithfulness.

    Args:
        dataset: a :class:`~twin.datasets.Dataset`.
        normalize: symmetric degree normalisation of the incidence.
        affinity_rule: which distance -> affinity rule to use when the affinity side has to be
            derived. ``"exp_decay_sq"`` exists so the choice can be compared empirically once
            (spec §3.4) rather than left as an assumption.
    """
    graph = dataset.graph

    if dataset.affinity is not None:
        affinity_w = _edge_weights(
            dataset.stored_matrix("affinity"), graph, f"{dataset.name}: the stored affinity"
        )
        affinity_derivation = f"stored ({dataset.affinity.describe()})"
    else:
        stored_d = _edge_weights(
            dataset.stored_matrix("distance"), graph, f"{dataset.name}: the stored distance"
        )
        affinity_w, rule = _derive_affinity(stored_d, dataset.distance.meaning, affinity_rule)
        affinity_derivation = f"derived from distance: {rule}"

    if dataset.distance is not None:
        distance_w = _edge_weights(
            dataset.stored_matrix("distance"), graph, f"{dataset.name}: the stored distance"
        )
        distance_derivation = f"stored ({dataset.distance.describe()})"
    else:
        distance_w, rule = _derive_distance(affinity_w, dataset.affinity.meaning)
        distance_derivation = f"derived from affinity: {rule}"

    incidence, incidence_weighted = _incidence_matrices(graph, affinity_w, normalize)

    A_V = incidence_weighted @ incidence_weighted.T
    A_V = A_V - sp.diags_array(A_V.diagonal())

    edges = list(graph.edges())

    return LayoutGraph(
        name="A_V",
        kind="vertex",
        affinity=A_V,
        incidence=incidence,
        incidence_weighted=incidence_weighted,
        n_vertices=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        # Vertices are laid out; edge points are a convex combination of their endpoints.
        point_provenance={"V": "primary", "E": "derived"},
        dataset_name=dataset.name,
        is_faithful=not normalize,
        derivation=affinity_derivation + (" [degree-normalised]" if normalize else ""),
        distance_derivation=distance_derivation,
        params={"normalize": normalize, "affinity_rule": affinity_rule},
        _build_distance=lambda: _weighted_graph(
            graph.number_of_nodes(),
            edges,
            distance_w,
            source=f"A_V of {dataset.name}: {distance_derivation}",
        ),
    )


def edge_graph(vertex: LayoutGraph) -> LayoutGraph:
    """$A_E = B^{\\top}B$ — the line graph, off the vertex graph's affinity form."""
    _require_vertex_graph(vertex, "edge_graph")

    B = vertex.incidence_weighted
    A_E = B.T @ B
    A_E = A_E - sp.diags_array(A_E.diagonal())

    return _derived(vertex, "A_E", "edge", A_E, {"V": "derived", "E": "primary"}, "B^T B", {})


def twin_graph(vertex: LayoutGraph, alpha: float = 0.85, k: int = 2) -> LayoutGraph:
    """$A_{tne} = \\sum_k \\alpha^k \\tilde A^k$ over the incidence — the twin transform.

    Row order is **[edges (m); vertices (n)]**, because `twin_adjacency` builds
    ``[[0, B^T], [B, 0]]``. Both point sets are laid out directly, which is the point of it.
    """
    _require_vertex_graph(vertex, "twin_graph")

    A_tne = twin_adjacency(vertex.incidence_weighted, alpha, n=k)
    A_tne = A_tne - sp.diags_array(A_tne.diagonal())

    return _derived(
        vertex,
        "A_tne",
        "twin",
        A_tne,
        {"V": "primary", "E": "primary"},
        f"sum_k alpha^k A~^k, alpha={alpha}, k={k}",
        {"alpha": alpha, "k": k},
    )


def _derived(
    vertex: LayoutGraph,
    name: str,
    kind: Kind,
    affinity: sp.sparray,
    point_provenance: dict[str, str],
    derivation: str,
    params: dict,
) -> LayoutGraph:
    return LayoutGraph(
        name=name,
        kind=kind,
        affinity=affinity,
        incidence=vertex.incidence,
        incidence_weighted=vertex.incidence_weighted,
        n_vertices=vertex.n_vertices,
        n_edges=vertex.n_edges,
        point_provenance=point_provenance,
        parent=vertex,
        dataset_name=vertex.dataset_name,
        # Neither is in the dataset's units, so neither reproduces its weights.
        is_faithful=False,
        derivation=derivation,
        distance_derivation="generic fallback: normalise, symmetrise, -log",
        params={**vertex.params, **params},
        _build_distance=lambda: stamp_form(
            nx.from_scipy_sparse_array(affinity_matrix_to_distance(affinity)),
            side="distance",
            # Named as the fallback it is: this is the form ideal distances must never come
            # from, and the stamp is what lets a consumer say so in its error message.
            source=f"{name} of {vertex.dataset_name}: generic fallback (normalise, symmetrise, -log)",
        ),
    )


# --------------------------------------------------------------------------- #
#                        Dataset -> LayoutGraph, by name                        #
# --------------------------------------------------------------------------- #
#
# The batch's "representation" axis. Each builder is `Dataset -> LayoutGraph` with the method
# knobs as keyword arguments, so the three read the same at the call site even though A_E and
# A_tne are two-step derivations. Caching an expensive A_tne across a lambda sweep belongs in
# the runner, not here.


def vertex_representation(dataset, **kw) -> LayoutGraph:
    return vertex_graph(dataset, **kw)


def edge_representation(dataset, **kw) -> LayoutGraph:
    return edge_graph(vertex_graph(dataset, **kw))


def twin_representation(dataset, *, alpha: float = 0.85, k: int = 2, **kw) -> LayoutGraph:
    return twin_graph(vertex_graph(dataset, **kw), alpha=alpha, k=k)


REPRESENTATION_REGISTRY = {
    "vertex": vertex_representation,
    "edge": edge_representation,
    "twin": twin_representation,
}


def _require_vertex_graph(graph: LayoutGraph, where: str) -> None:
    if graph.kind != "vertex":
        raise ValueError(
            f"{where} takes the vertex graph, got {graph.name}. A_E and A_tne are built from "
            "A_V's affinity form — they are affinity algebra, and chaining them off each other "
            "or off a distance form inverts the meaning of every term (spec §3.5)."
        )


def _weighted_graph(
    n: int, edges: list[tuple[int, int]], weights: np.ndarray, source: str
) -> nx.Graph:
    """A distance graph: every edge present with its weight, every non-edge **absent**.

    Absence is the point. `Unweighted` distance must give hop counts, so non-edges cannot be
    materialised at 1 — and a graph expresses absence naturally where a matrix has to encode it.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_weighted_edges_from((u, v, float(w)) for (u, v), w in zip(edges, weights))
    return stamp_form(G, side="distance", source=source)
