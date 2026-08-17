import networkx as nx
import scipy.sparse as sp
import numpy as np
import pandas as pd

from twin import graphmatrix as gm
from twin import embedding as emb
from twin import dataset as data
from twin.batching import *
from twin.metrics import *
from twin.results import LayoutResult

from typeguard import typechecked
import textwrap
import time
import tracemalloc


def _require_provenance(Gm) -> dict[str, str]:
    """Read the point provenance a representation declares, or say clearly that it must."""
    provenance = getattr(Gm, "point_provenance", None)
    if not provenance:
        raise TypeError(
            f"{type(Gm).__name__} does not declare point_provenance. Every "
            "GraphMatrixMethod must state, per point set, whether its points were laid out "
            "directly or reconstructed from the other set (spec §0.5 decision 5)."
        )
    return provenance


def graph_embedding(
    G: nx.Graph,
    representation=gm.VertexMatrix,
    embedding=emb.SpringLayout,
    y0=None,
    seed: int = 0,
    d: int = 2,
    embedding_kw: dict = {},
    representation_kw: dict = {},
    normalize: bool = False,
    representation_name: str | None = None,
    embedding_name: str | None = None,
    measure_memory: bool = False,
) -> LayoutResult:
    """Run the embedding pipeline for one representation and one layout method.

    Args:
        G: the graph to lay out.
        representation: a :class:`~twin.graphmatrix.GraphMatrixMethod` subclass.
        embedding: a :class:`~twin.embedding.GraphEmbeddingMethod` subclass.
        y0: initial positions, ``(d, n_points)``, or None.
        seed, d: passed to the layout method.
        embedding_kw, representation_kw, normalize: constructor arguments.
        representation_name, embedding_name: registry names for the result record. Default to
            the class names.
        measure_memory: track peak Python-side allocation during the embed. Off by default:
            :mod:`tracemalloc` roughly doubles allocation cost, which would corrupt the
            ``runtime_s`` measurement sitting next to it, and it cannot see allocations made
            inside a C extension such as ``sgtsnepi`` anyway. Enable it deliberately, for a
            memory run, rather than leaving it on during a timing run.

    Returns:
        A :class:`~twin.results.LayoutResult` in the canonical ``(n, d)`` orientation.

    Note:
        Timing is split per spec §3.2: ``representation_s`` covers building the matrix (for
        A_tne that is the (n+m) construction, a first-order reported cost) and ``runtime_s``
        covers the embed alone.
    """
    E = embedding(seed=seed, d=d, **embedding_kw)

    t0 = time.perf_counter()
    Gm = representation(G, normalize=normalize, **representation_kw)
    representation_s = time.perf_counter() - t0

    tracing = measure_memory and not tracemalloc.is_tracing()
    if tracing:
        tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0] if measure_memory else 0

    t0 = time.perf_counter()
    Y = E.embed(Gm, y0)
    runtime_s = time.perf_counter() - t0

    peak_mem_mb = None
    if measure_memory:
        peak_mem_mb = max(tracemalloc.get_traced_memory()[1] - baseline, 0) / 1e6
        if tracing:
            tracemalloc.stop()

    y_v, y_e, y_s = Gm.vertex_and_edge_embeddings(Y)

    return LayoutResult.from_raw(
        V=y_v,
        E=y_e,
        slopes=y_s,
        G=G,
        representation=representation_name or representation.__name__,
        embedding=embedding_name or embedding.__name__,
        point_provenance=_require_provenance(Gm),
        params={
            "d": d,
            "normalize": normalize,
            "representation_kw": dict(representation_kw),
            "embedding_kw": dict(embedding_kw),
        },
        seed=seed,
        runtime_s=runtime_s,
        representation_s=representation_s,
        n_points=Gm.get_adj().shape[0],
        peak_mem_mb=peak_mem_mb,
    )


batch_registries = {
    "graph": {
        "ring_of_cliques": data.ring_of_cliques,
        "tree_of_cliques": data.tree_of_cliques,
        "karate_club": nx.karate_club_graph,
        "football_2023": data.football_2023,
        "small_world": data.small_world,
        "barabasi_albert": data.barabasi_albert,
        "mnist": data.mnist,
        "tabula": data.tabula,
    },

    "representation": {
        "vertex": gm.VertexMatrix,
        "edge": gm.EdgeMatrix,
        "twin": gm.TwinEmbeddingMatrix,
    },

    "embedding": {
        "spring": emb.SpringLayout,
        "yifanhu": emb.YifanHuLayout,
        "forceatlas2": emb.ForceAtlas2Layout,
        "kamada-kawai": emb.KamadaKawaiLayout,
        "neato": emb.NeatoLayout,
        "spectral": emb.SpectralLayout,
        "sgtsne": emb.SGtSNELayout,
    },
}


def batch_graph_embedding(
    batch_input,
    *,
    registries=batch_registries,
    y0=None,
    seed: int = 0,
    d: int = 2,
    normalize: bool = False,
    measure_memory: bool = False,
    duplicate_policy: str = "error",
    progress: bool = False,
    progress_every: int = 1,
    progress_verbose: bool = False,
    progress_fn=print,
) -> dict[BatchKey, LayoutResult]:
    """Run ``graph_embedding`` over the expanded batch spec.

    Returns:
        One :class:`~twin.results.LayoutResult` per case, keyed by ``BatchKey``.
    """

    def runner(case: BatchCase):
        c = BatchCaseView(case)
        c.require("graph", "representation", "embedding")

        G = resolve_object_choice(c.choice("graph"))

        return graph_embedding(
            G,
            representation=c.obj("representation"),
            embedding=c.obj("embedding"),
            y0=y0,
            seed=seed,
            d=d,
            representation_kw=c.kwargs("representation"),
            embedding_kw=c.kwargs("embedding"),
            normalize=normalize,
            representation_name=c.name("representation"),
            embedding_name=c.name("embedding"),
            measure_memory=measure_memory,
        )

    return run_batch(
        batch_input,
        registries,
        runner,
        duplicate_policy=duplicate_policy,
        axis_order=["graph", "representation", "embedding"],
        progress=progress,
        progress_every=progress_every,
        progress_verbose=progress_verbose,
        progress_fn=progress_fn,
    )


def graph_embedding_all(
    G: nx.Graph,
    embedding=emb.SpringLayout,
    seed: int = 0,
    d: int = 2,
    embedding_kw: dict = {},
    twinmatrix_kw: dict = {},
    common_init: bool = True,
    normalize: bool = False,
):
    """Runs the full pipeline for all graph matrix methods."""

    E_in = embedding(seed=seed, d=d, **embedding_kw)
    Gv_in = gm.VertexMatrix(G, normalize=False)
    Ge_in = gm.EdgeMatrix(G, normalize=False)
    Gtw_in = gm.TwinEmbeddingMatrix(G, normalize=normalize, **twinmatrix_kw)

    # number of nodes + edges
    nv = Gtw_in.incidence.shape[0]
    ne = Gtw_in.incidence.shape[1]

    # Step 1: Create initial positions for the embeddings
    y0 = E_in.init_embedding(nv + ne, d)

    y0e = y0[:, :ne] if common_init else None
    y0v = y0[:, ne:] if common_init else None
    y0tw = y0 if common_init else None

    # Step 2: Embed using all three approaches
    X = E_in.embed(Gv_in, y0=y0v)
    Y = E_in.embed(Ge_in, y0=y0e)
    Z = E_in.embed(Gtw_in, y0=y0tw)

    # Step 3: Extract vertex and edge embeddings for each approach
    X_v, X_e, X_s = Gv_in.vertex_and_edge_embeddings(X)
    Y_v, Y_e, Y_s = Ge_in.vertex_and_edge_embeddings(Y)
    Z_v, Z_e, Z_s = Gtw_in.vertex_and_edge_embeddings(Z)

    # Optional: form line graph & incidence for postprocess
    L = Ge_in.incidence.T @ Ge_in.incidence
    L = L - sp.diags(L.diagonal())

    inc_emb = sp.block_array(
        [[None, Gv_in.incidence.T], [Gv_in.incidence, None]],
        format="csc",
        dtype=Gv_in.incidence.dtype,
    )

    dict_result = {
        "VertexEmbedding": {"V": X_v, "E": X_e, "S": X_s},
        "EdgeEmbedding": {"V": Y_v, "E": Y_e, "S": Y_s},
        "TwinEmbedding": {"V": Z_v, "E": Z_e, "S": Z_s},
        "OriginalGraph": G,
        "Gv_in": Gv_in,
        "Ge_in": Ge_in,
        "Gtw_in": Gtw_in,
        "LineGraph": nx.from_scipy_sparse_array(L),
        "IncidenceGraph": nx.from_scipy_sparse_array(inc_emb),
    }
    return dict_result


def _result_from(Gm, Y, G, representation, E_in, seed) -> LayoutResult:
    """Split a raw ``(d, n_points)`` layout and cross the orientation boundary once."""
    y_v, y_e, y_s = Gm.vertex_and_edge_embeddings(Y)
    return LayoutResult.from_raw(
        V=y_v,
        E=y_e,
        slopes=y_s,
        G=G,
        representation=representation,
        embedding=getattr(E_in, "method_name", type(E_in).__name__),
        point_provenance=Gm.point_provenance,
        seed=seed,
        n_points=Gm.get_adj().shape[0],
    )


def embedding_metrics_all(
    G: nx.Graph,
    embedding=emb.SpringLayout,
    d: int = 2,
    n_samples: int = 1,
    n_neighbors: int = 10,
    target=None,
    mode="recall",
    embedding_kw: dict = {},
    twinmatrix_kw: dict = {},
    common_init: bool = True,
    normalize: bool = False,
):
    """Calculates all metric types given a graph and embedding

    Note:
        The ``VertexPointDistance`` / ``EdgePointDistance`` entries were dropped with
        ``point_distance_metric`` (spec §1.2). ADR (spec §2.1) fills the same slot once it
        lands; nothing stands in for it meanwhile.

        This whole function is the old hardcoded vertex/edge/twin triple that bypasses the
        batch engine, and is superseded by ``batch_layout_metrics`` (spec §3.7).
    """

    neighbor_preservation_metric_v = 0.0
    neighbor_preservation_metric_e = 0.0
    neighbor_preservation_metric_tw = 0.0

    if target is not None:
        classification_metric_v = 0.0
        classification_metric_e = 0.0
        classification_metric_tw = 0.0

    Gv_in = gm.VertexMatrix(G, normalize=False)
    Ge_in = gm.EdgeMatrix(G, normalize=False)
    Gtw_in = gm.TwinEmbeddingMatrix(G, normalize=normalize, **twinmatrix_kw)

    nv = Gtw_in.incidence.shape[0]
    ne = Gtw_in.incidence.shape[1]

    for seed in range(n_samples):
        E_in = embedding(seed=seed, d=d, **embedding_kw)
        y0 = E_in.init_embedding(nv + ne, d)

        y0e = y0[:, :ne] if common_init else None
        y0v = y0[:, ne:] if common_init else None
        y0tw = y0 if common_init else None

        X = E_in.embed(Gv_in, y0=y0v)
        Y = E_in.embed(Ge_in, y0=y0e)
        Z = E_in.embed(Gtw_in, y0=y0tw)

        # Cross the (d, n) -> (n, d) boundary exactly once, here (spec §1.3, §3.1).
        res_v = _result_from(Gv_in, X, G, "vertex", E_in, seed)
        res_e = _result_from(Ge_in, Y, G, "edge", E_in, seed)
        res_tw = _result_from(Gtw_in, Z, G, "twin", E_in, seed)

        neighbor_preservation_metric_v += knn_neighbor_preservation_accuracy(
            res_v.V, G, n_neighbors, mode=mode
        )
        neighbor_preservation_metric_e += knn_neighbor_preservation_accuracy(
            res_e.V, G, n_neighbors, mode=mode
        )
        neighbor_preservation_metric_tw += knn_neighbor_preservation_accuracy(
            res_tw.V, G, n_neighbors, mode=mode
        )

        if target is not None:
            classification_metric_v += knn_accuracy(res_v.V, target, n_neighbors)
            classification_metric_e += knn_accuracy(res_e.V, target, n_neighbors)
            classification_metric_tw += knn_accuracy(res_tw.V, target, n_neighbors)

    neighbor_preservation_metric_v = neighbor_preservation_metric_v / n_samples
    neighbor_preservation_metric_e = neighbor_preservation_metric_e / n_samples
    neighbor_preservation_metric_tw = neighbor_preservation_metric_tw / n_samples

    if target is not None:
        classification_metric_v = classification_metric_v / n_samples
        classification_metric_e = classification_metric_e / n_samples
        classification_metric_tw = classification_metric_tw / n_samples

    metrics_dict = {
        "n_neighbors": n_neighbors,
        "n_samples": n_samples,
        "KNeighborsPreservation": {
            "Vertex": neighbor_preservation_metric_v,
            "Edge": neighbor_preservation_metric_e,
            "Twin": neighbor_preservation_metric_tw,
        },
    }

    if target is not None:
        metrics_dict["KNeighborsClassifier"] = {
            "Vertex": classification_metric_v,
            "Edge": classification_metric_e,
            "Twin": classification_metric_tw,
        }

    return metrics_dict


def metrics_to_typ(metrics):
    """Emit Typst tables for whichever metric families are present.

    Note:
        The point-distance table is emitted only if the key is present; it went away with
        ``point_distance_metric`` (spec §1.2) and returns as ADR in spec §2.1. This whole
        function is rewritten against the long DataFrame in spec §5.2.
    """
    has_point_distance = any(
        "VertexPointDistance" in m and "EdgePointDistance" in m for m in metrics.values()
    )

    point_dist_tab = (
        textwrap.dedent("""
        #table(
        \tcolumns: 5,
        \ttable.header(
        \t\ttable.cell(colspan: 2)[],
        \t\ttable.cell(colspan:3)[Embedding Type],
        \t\t[Dataset], table.vline(stroke: thinstroke, start: 2),
        \t\t[Dist. Type], table.vline(stroke: thinstroke, start: 2),
        \t\t[Vertex], [Edge], [Twin]
        \t),
        {content}
        )
        """)
        .format(
            content="\n\ttable.hline(stroke: 1pt),\n".join(
                [
                    (
                        f"\tgraph(rowspan: 2)[{graph_label}],\n{
                            "\n".join([
                                f"\t[{point_type}], ..hlmin({vert:.6f}, {edge:.6f}, {twin:.6f}),"
                                for point_type, vert, edge, twin in zip(
                                    ['Vertex', 'Edge'],
                                    [vdist['Vertex'], edist['Vertex']],
                                    [vdist['Edge'], edist['Edge']],
                                    [vdist['Twin'], edist['Twin']]
                                )
                            ])
                        }"
                        if (vdist := metric.get("VertexPointDistance"))
                        and (edist := metric.get("EdgePointDistance"))
                        else ""
                    )
                    for graph_label, metric in metrics.items()
                ]
            )
        )
        .expandtabs(2)
    ) if has_point_distance else None

    knn_adj_tab = (
        textwrap.dedent("""
        #table(
        \tcolumns: 5,
        \ttable.header(
        \t\ttable.cell(colspan: 2)[],
        \t\ttable.cell(colspan:3)[Embedding Type],
        \t\t[Dataset], table.vline(stroke: thinstroke, start: 2),
        \t\t[$k$], table.vline(stroke: thinstroke, start: 2),
        \t\t[Vertex], [Edge], [Twin]
        \t),
        {content}
        )
        """)
        .format(
            content="\n\ttable.hline(stroke: 1pt),\n".join(
                [
                    f"\tgraph(rowspan: {len(metric['n_neighbors'])})[{graph_label}],\n{
                        "\n".join([
                            f"\t[{k}], ..hlmax({vert:.6f}, {edge:.6f}, {twin:.6f}),"
                            for k, vert, edge, twin in zip(metric['n_neighbors'], knn_adj['Vertex'], knn_adj['Edge'], knn_adj['Twin'])
                        ])
                    }"
                    if (knn_adj := metric["KNeighborsPreservation"]) else ""
                    for graph_label, metric in metrics.items()
                ]
            )
        )
        .expandtabs(2)
    )

    for m in metrics.values():
        if "KNeighborsClassifier" in m:
            knn_class_tab = (
                textwrap.dedent("""
                #table(
                \tcolumns: 5,
                \ttable.header(
                \t\ttable.cell(colspan: 2)[],
                \t\ttable.cell(colspan:3)[Embedding Type],
                \t\t[Dataset], table.vline(stroke: thinstroke, start: 2),
                \t\t[$k$], table.vline(stroke: thinstroke, start: 2),
                \t\t[Vertex], [Edge], [Twin]
                \t),
                {content}
                )
                """)
                .format(
                    content="\n\ttable.hline(stroke: 1pt),\n".join(
                        [
                            (
                                f"\tgraph(rowspan: {len(metric['n_neighbors'])})[{graph_label}],\n{
                                    "\n".join([
                                        f"\t[{k}], ..hlmax({vert:.6f}, {edge:.6f}, {twin:.6f}),"
                                        for k, vert, edge, twin in zip(metric['n_neighbors'], knn_class['Vertex'], knn_class['Edge'], knn_class['Twin'])
                                    ])
                                }"
                                if ("KNeighborsClassifier" in metric)
                                and (knn_class := metric["KNeighborsClassifier"])
                                else ""
                            )
                            for graph_label, metric in metrics.items()
                        ]
                    )
                )
                .expandtabs(2)
            )
            break
    else:
        knn_class_tab = None

    return tuple(tab for tab in (point_dist_tab, knn_adj_tab, knn_class_tab) if tab is not None)


def metrics_to_df(metrics: dict, graph_label: str = None):
    """
    Convert embedding metrics to structured pandas DataFrames.

    Parameters
    ----------
    metrics : dict
        Output from embedding_metrics_all function
    graph_label : str, optional
        Label for the graph (e.g., 'Football 2023', 'Tabula Sapiens')
        Used as index level for stacking multiple graphs

    Returns
    -------
    dict
        Dictionary with keys:
        - 'PointDistance': DataFrame with rows for VertexPointDistance and EdgePointDistance
        - 'KNeighborsPreservation': DataFrame with rows for k values
        - 'KNeighborsClassifier': DataFrame with rows for k values (if present)
    """
    embedding_types = ["Vertex", "Edge", "Twin"]

    result = {}

    # 1. Point Distance DataFrame -- present only if the caller supplied it. Removed with
    #    `point_distance_metric` (spec §1.2); returns as ADR in spec §2.1.
    if "VertexPointDistance" in metrics and "EdgePointDistance" in metrics:
        point_dist_data = {
            emb_type: [
                metrics["VertexPointDistance"][emb_type],
                metrics["EdgePointDistance"][emb_type],
            ]
            for emb_type in embedding_types
        }

        df_point_dist = pd.DataFrame(point_dist_data, index=["Vertex", "Edge"])
        df_point_dist.index.name = "Point type"
        result["PointDistance"] = df_point_dist

    # 2. KNeighbors Preservation DataFrame
    knn_pres_data = {
        emb_type: metrics["KNeighborsPreservation"][emb_type] for emb_type in embedding_types
    }

    df_knn_preservation = pd.DataFrame(knn_pres_data, index=metrics["n_neighbors"])
    df_knn_preservation.index.name = "k"
    result["KNeighborsPreservation"] = df_knn_preservation

    # 3. KNeighbors Classifier DataFrame (if present)
    if "KNeighborsClassifier" in metrics:
        knn_class_data = {
            emb_type: metrics["KNeighborsClassifier"][emb_type] for emb_type in embedding_types
        }

        df_knn_classifier = pd.DataFrame(knn_class_data, index=metrics["n_neighbors"])

        df_knn_classifier.index.name = "k"
        result["KNeighborsClassifier"] = df_knn_classifier

    # Add graph label as MultiIndex if provided
    if graph_label is not None:
        for key in result:
            df = result[key]
            df.index = pd.MultiIndex.from_product(
                [[graph_label], df.index], names=["Graph", df.index.name or "Metric"]
            )

    return result
