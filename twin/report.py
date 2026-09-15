r"""The dataset connectivity / weight / label report (spec §3.4).

    python -m twin.report                 # markdown to stdout
    python -m twin.report -o docs/DATASET_REPORT.md
    python -m twin.report tabula digits   # just these
    python -m twin.report --no-label-scores   # skip the graph-side scores (~4x faster)

One row per registered dataset. It exists because several decisions cannot be made without it:
§4.4's disconnected-graph policy needs to know which datasets are actually disconnected, §4.2's
lambda criterion is defined at *k* = mean degree, and whether a dataset's stored sides can be
trusted against each other is a measurement, not an assumption.

**The graph-side label scores are here and their layout-side counterparts are not**, which is
§5.3 guard 2 rather than an omission: Sil\* and Acc\* answer "are these labels reflected in the
topology at all?" on a scale incomparable with a layout's, and column adjacency is an invitation
to difference two incomparable quantities. Nothing in this file needs a layout.

Nothing here branches on a dataset name; every column comes off the `Dataset` object. Adding a
dataset adds a row and nothing else.

⚠️ **The label-score section dominates the runtime** — about 11 s without it and 2 minutes with.
Most of that is the permutation nulls: `tabula`'s four label sets at 39 draws each are ~145 s,
against 10 s for its all-pairs closure. It is on by default so that a regenerated
`docs/DATASET_REPORT.md` is never silently missing a section; `--no-label-scores` is the escape.
Datasets with no label sets cost nothing either way, because the closure is only built when
there is something to score against it.

The numbers this file generates are pinned in `tests/test_report_numbers.py`, the same
generated-plus-pinned arrangement `DATASET_PROVENANCE.md` has — `DECISIONS.md` (42) and (44)
quote them, and a design record whose numbers nothing checks goes stale silently.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp

from twin.datasets import (
    DATASET_REGISTRY,
    DistortionReport,
    WeightConsistencyReport,
    check_weight_consistency,
    load_dataset,
    sparsification_distortion,
)
from twin.layoutgraph import vertex_graph


#: Leiden resolutions swept for the compactness ceiling. The ceiling is the **best** Sil\* over
#: the sweep, so a resolution that splits the graph too finely cannot depress it — the reference
#: is meant to be optimistic, since its job is to answer "how high does Sil\* go on this graph".
CEILING_RESOLUTIONS = (0.5, 1.0, 2.0)

#: Leiden's seed, matching `paper_figures.tabula_sapiens_teaser` so the two agree.
CEILING_SEED = 10


@dataclass
class PermutationNull:
    r"""A metric's reference distribution under random relabelling, and the standardised gap.

    **A z-score, not a min and a max.** The two order statistics are the least robust summaries
    available — they are the only two that cannot converge, so their spread grows with the
    permutation count and a table using them cannot be compared across rows with different
    counts. $z = (\text{observed} - \mu_0) / \sigma_0$ is stable at modest counts and puts every
    row on one scale.

    ⚠️ **The z-score makes "above chance" comparable across label sets; it does *not* make
    magnitudes comparable.** $\sigma_0$ itself depends on the class count and balance — a
    2-class balanced null has a tiny spread, so `tabula`'s `donor` reaches $z = 471$ against
    `cell_ontology_class`'s $z = 17$ while its Sil\* is only 0.31 against 0.28. That is a
    statement about how far each sits above its *own* null, and reading it as "donor is 27×
    better encoded" is wrong. For magnitude, compare within a stratum at matched class counts
    (`DECISIONS.md` (42)).

    Attributes:
        mean: $\mu_0$, the null mean. For Sil\* this is **negative** by an amount that grows
            with the class count, because $b^*_i$ is a min over $\lvert\mathcal{L}\rvert - 1$
            class means and so sits below the overall mean while $a^*_i$ is at it.
        sd: $\sigma_0$. **Exactly zero is a real case, not a guard**: `tabula`'s `method` splits
            213 against 4794, so every relabelling leaves almost every 1-hop neighbourhood
            majority-`10X` and Acc\* is the majority rate on all permutations.
        z: ``nan`` when ``sd`` is zero — there is no permutation test, and the honest statement
            is the margin over the majority-class baseline instead.
        n_permutations: how many were drawn. Scaled with the cost of one evaluation; see
            :func:`_permutation_count`.
    """

    mean: float
    sd: float
    z: float
    n_permutations: int

    @classmethod
    def from_samples(cls, observed: float, samples: np.ndarray) -> "PermutationNull":
        r"""⚠️ Degeneracy is decided by **counting distinct values**, not by ``sd > 0``.

        On `tabula`'s `method` the null takes exactly one value — 0.957459556620731, bit-identical
        to the majority rate on all permutations — and ``samples.std(ddof=1)`` still returns
        $2.3\times10^{-16}$, because it subtracts a rounded mean from a constant array. Dividing
        by that gave $z = 8.7\times10^{13}$: a point-mass null reported as the most significant
        result in the table.

        The distinct-value count is exact and needs no tolerance, which is the fix this project
        prefers (`DECISIONS.md` (32)) — a threshold on ``sd`` would have had to be placed, and
        placed against a quantity whose scale differs per metric.
        """
        point_mass = samples.size < 2 or np.unique(samples).size == 1
        sd = 0.0 if point_mass else float(samples.std(ddof=1))
        return cls(
            mean=float(samples.mean()),
            sd=sd,
            z=float("nan") if point_mass else float((observed - samples.mean()) / sd),
            n_permutations=int(samples.size),
        )

    @property
    def degenerate(self) -> bool:
        """No permutation test exists — the null is a point mass."""
        return not np.isfinite(self.z)


def cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    r"""$V = \sqrt{\chi^2 / (n \min(r-1, c-1))}$ — association between two partitions.

    0 for independence, 1 when either determines the other. Used against the dataset's scoring
    primary, as the confound column: a covariate scoring highly on a graph is only evidence
    *about that covariate* if it is not already a proxy for the biology. `tabula`'s `donor`
    reaches $V = 0.917$ against `cell_ontology_class`.

    Preferred to mutual information here because it is bounded in $[0, 1]$ with the endpoints
    meaning something fixed, where NMI's value depends on the class counts — the same reason
    the raw Sil\* column needed a reference in the first place. On this roster the two disagree
    in the direction that matters: NMI(donor, cell type) is 0.413, which reads as moderate,
    where $V = 0.917$ correctly reads as near-determination.
    """
    from scipy.stats import chi2_contingency

    a, b = np.asarray(a), np.asarray(b)
    classes_a, classes_b = np.unique(a), np.unique(b)
    if classes_a.size < 2 or classes_b.size < 2:
        return float("nan")

    table = np.zeros((classes_a.size, classes_b.size), dtype=np.int64)
    for i, ai in enumerate(classes_a):
        in_a = a == ai
        for j, bj in enumerate(classes_b):
            table[i, j] = int((in_a & (b == bj)).sum())

    chi2 = float(chi2_contingency(table).statistic)
    return float(np.sqrt(chi2 / (a.size * min(table.shape[0] - 1, table.shape[1] - 1))))


@dataclass
class LabelScoreRow:
    r"""The graph-side scores for one label set of one dataset.

    Per *label set*, not per dataset, because a dataset can carry several and they are the unit
    that gets scored. `tabula` has five counting its composite.

    Attributes:
        silhouette_graph: Sil\*, eqs. (53)-(55), under ``disconnected="lcc"``.
        silhouette_graph_null: the permutation reference for Sil\* — see :class:`PermutationNull`.
        graph_accuracy: Acc\*, eqs. (61)-(64). **Uniform voting and the fractional tie rule**,
            which is the headline convention and is not recoverable from the number; see
            ``label.py`` and `DECISIONS.md` (41).
        graph_accuracy_affinity: Acc\* voting with $w_{ij}$ (eq. 62). Equal to the above on a
            dataset declared unweighted — the declaration wins over any stray edge attribute.
        majority_class_rate: the chance floor. Acc\* exceeds 0.94 on every registered dataset,
            so this column is what distinguishes a strong result from a vacuous one: 0.987
            against a 0.958 floor on `tabula`'s `method` is nearly nothing.
        cramers_v_vs_primary: association with the dataset's **scoring primary** label set,
            $V = \sqrt{\chi^2 / (n \min(r-1, c-1))}$. The confound column: a covariate that
            nearly determines the primary cannot be read as an independent measurement of it.
            `tabula`'s `donor` scores **0.917** against `cell_ontology_class`, which is what
            makes its high Sil\* unattributable without conditioning (`DECISIONS.md` (42)).
            ``nan`` for the primary itself and where no primary is declared.
        n_excluded: vertices with no vote mass, dropped from Acc\*. Zero on the whole current
            roster, since every registered dataset is connected.
    """

    dataset: str
    label_set: str
    role: str
    derivation: str

    n_classes: int
    min_class_size: int
    median_class_size: float
    n_singletons: int

    silhouette_graph: float
    silhouette_graph_null: "PermutationNull"
    silhouette_graph_excluded: int

    graph_accuracy: float
    graph_accuracy_affinity: float
    graph_accuracy_null: "PermutationNull"
    majority_class_rate: float
    cramers_v_vs_primary: float
    n_ties: int
    n_excluded: int


@dataclass
class DatasetRow:
    """Everything the report knows about one dataset."""

    name: str
    n: int
    m: int
    density: float
    connected: bool
    n_components: int
    largest_component_fraction: float
    mean_degree: float

    affinity_source: str
    distance_source: str
    affinity_derivation: str
    distance_derivation: str

    weight_min: float
    weight_median: float
    weight_max: float
    weight_distinct: int

    sparsity: str
    sparsity_violations: list[str]
    degree_min: int
    degree_median: float
    degree_max: int
    n_isolated: int
    stored_nnz: dict[str, int]

    zero_distance_pair_fraction: float
    labels: list[tuple[str, int, str, str, str]] = field(default_factory=list)
    composites: list[str] = field(default_factory=list)
    consistency: WeightConsistencyReport | None = None
    distortion: DistortionReport | None = None
    description: str = ""

    label_scores: list[LabelScoreRow] = field(default_factory=list)
    compactness_ceiling: float | None = None
    compactness_ceiling_classes: int | None = None


def zero_distance_pair_fraction(distance: nx.Graph, n: int) -> float:
    """Fraction of **vertex pairs** whose ideal distance is exactly zero (spec §0.5).

    Not the fraction of edges at $w_{\\max}$: zero-weight edges compose, so whole groups can
    become mutually distance-zero. Since every derivation produces non-negative weights,
    $d(u, v) = 0$ exactly when $u$ and $v$ are joined by a path of zero-weight edges — so this
    is the connected components of the zero-weight subgraph, not an all-pairs shortest path.

    It matters because zero ideal distances damage different metrics differently: SNS tolerates
    them, silhouette is distorted, and shortest-path derivations collapse. A small fraction
    makes the question academic; a large one is evidence about the data.
    """
    weights = np.array([w for _, _, w in distance.edges(data="weight")], dtype=float)
    if (weights < 0).any():
        raise ValueError(
            "negative edge weights in the distance form; the zero-pair count assumes "
            "non-negativity, and shortest paths would not be well defined either"
        )
    if n < 2:
        return 0.0

    zero = nx.Graph()
    zero.add_nodes_from(range(n))
    zero.add_edges_from((u, v) for u, v, w in distance.edges(data="weight") if w == 0)

    coincident = sum(len(c) * (len(c) - 1) // 2 for c in nx.connected_components(zero))
    return coincident / (n * (n - 1) / 2)


def _permutation_count(n: int) -> int:
    r"""How many permutations the reference distribution is estimated from, given $n$.

    Two competing pressures, both measured. A z-score needs enough draws for $\sigma_0$ to be
    usable — below about 30 the standard-deviation estimate is itself noisy — and one evaluation
    costs 0.001 s at karate's size but 0.38 s (Sil\*) plus 0.49 s (Acc\*) at tabula's, where
    five label sets multiply it.

    ``200000 // n``, clamped to [30, 1000]: **1000** on `karate_weighted` and `football_2023`,
    where it is free and where twenty draws visibly straddled zero; 111 on `digits`; **30** on
    `tabula`, five label sets of which cost about 130 s between them. Thirty is tolerable there
    only because the z-scores are enormous — a 13 % error in $\sigma_0$ does not change the
    reading of $z = 17$ — and that is a property of this roster rather than a general licence.
    """
    return int(np.clip(200000 // max(n, 1), 30, 1000))


def compactness_ceiling(labels_free_graph: nx.Graph, ideal: np.ndarray, n: int):
    r"""The best Sil\* any partition of this graph reaches, as far as Leiden can find one.

    **The column that makes the Sil\* numbers readable.** Sil\* has no fixed ceiling at 1 and no
    floor at 0, and both bounds are properties of the *graph*: $a^*_i$ is the mean distance to
    all of one's own class and $b^*_i$ the mean to all of the nearest other one, so once classes
    have internal extent comparable to the gap between them — which any connected $k$-NN graph
    guarantees — the two means converge and $s^*_i$ is small however clean the labels are. The
    measured $b^*/a^*$ ratio is 1.50 on football's conferences, 1.64 on `digits` and 1.50 on
    `tabula`, which caps Sil\* near $1 - 1/1.5 \approx 0.33$ before any label quality enters.

    So the question "is 0.33 low?" cannot be answered against 1. It is answered against a
    partition chosen to *maximise* topological compactness on the same graph, which is what
    modularity optimisation does. Measured: `ring_of_cliques` **0.5325**, identical to its ground
    truth to four digits; `football_2023` 0.3433 against a ground truth of 0.3304; `digits`
    0.3950 against 0.3670; `tabula` 0.3890 against 0.2839. The real label sets sit at 73-100 %
    of it.

    ⚠️ **It is a reference level, not a baseline and not a target.** A modularity partition is
    free to choose its own classes, so it should win; a ground truth that matched it exactly
    would mean the labels add nothing to the topology. The number to read is the gap, and a
    *small* gap says the labels are about as recoverable as any partition of this graph is.

    Args:
        labels_free_graph: the affinity-valued graph, for the community detection.
        ideal: the ``(n, n)`` ideal distances Sil\* is computed against.
        n: the vertex count.

    Returns:
        ``(best Sil*, class count of the partition that achieved it)``.
    """
    # Imported here rather than at module scope so that `--no-label-scores` does not need
    # igraph at all; both are hard dependencies, but this section is the only consumer.
    import igraph as ig
    import leidenalg as la

    from twin.metrics import silhouette_graph

    edges = list(labels_free_graph.edges(data="weight"))
    graph = ig.Graph(n=n, edges=[(u, v) for u, v, _ in edges])
    weights = [1.0 if w is None else float(w) for _, _, w in edges]

    best, best_classes = -np.inf, 0
    for resolution in CEILING_RESOLUTIONS:
        partition = la.find_partition(
            graph,
            la.RBConfigurationVertexPartition,
            weights=weights,
            resolution_parameter=resolution,
            seed=CEILING_SEED,
        )
        membership = np.asarray(partition.membership)
        value = silhouette_graph(membership, ideal=ideal)
        if value > best:
            best, best_classes = value, int(np.unique(membership).size)
    return float(best), best_classes


def score_label_sets(dataset, ideal: np.ndarray, affinity: nx.Graph) -> list[LabelScoreRow]:
    r"""Sil\* and Acc\* for every label set and composite of one dataset, with their nulls.

    Both metrics take the graph from the evaluation context — ``affinity`` here — and never
    ``dataset.graph``: on `digits` that attribute is a stored Euclidean *distance*, so eq. (62)
    over it would count the least-coupled neighbours most (CLAUDE.md, bug §1.6 one layer up).

    **The headline Acc\* is uniform voting with the fractional tie rule**, matching eq. (61) and
    the layout-side default; the affinity-weighted value is reported beside it rather than in
    place of it. That is a decision and not an inheritance — `DECISIONS.md` (41) records that
    affinity weighting is better on all seven weighted label sets of this roster and equal on
    the four unweighted ones, and why the headline is uniform anyway.

    ⚠️ **A degenerate composite is scored once, not twice.** `tabula`'s `compartment_cell_type`
    *is* `cell_ontology_class` renumbered — the product partition equals its finer part — so a
    second row would be a duplicate column wearing a different name. The redundancy is stated in
    the rendered section instead.
    """
    from twin.metrics import (
        graph_neighborhood_accuracy,
        graph_neighborhood_accuracy_report,
        majority_class_rate,
        silhouette_graph,
        silhouette_graph_report,
    )

    degenerate = dataset.degenerate_composites()
    primaries = dataset.label_sets_by_scoring_role("primary")
    primary = primaries[0].values if len(primaries) == 1 else None

    n_permutations = _permutation_count(dataset.n)
    rows = []
    for name in list(dataset.label_sets) + sorted(dataset.composite_of):
        if name in degenerate:
            continue
        labels = dataset.labels(name)
        declared = dataset.label_sets.get(name)

        silhouette = silhouette_graph_report(labels, ideal=ideal)
        accuracy = graph_neighborhood_accuracy_report(labels, affinity)

        # One generator, so both nulls run over the *same* permutations. They are read side by
        # side and an unexplained disagreement must not be able to come from the draw.
        rng = np.random.default_rng(0)
        null_silhouette = np.empty(n_permutations)
        null_accuracy = np.empty(n_permutations)
        for i in range(n_permutations):
            permuted = rng.permutation(labels)
            null_silhouette[i] = silhouette_graph(permuted, ideal=ideal)
            null_accuracy[i] = graph_neighborhood_accuracy(permuted, affinity)

        rows.append(
            LabelScoreRow(
                dataset=dataset.name,
                label_set=name,
                role=declared.scoring_role if declared else "composite",
                derivation=declared.derivation if declared else "derived",
                n_classes=silhouette.n_classes,
                min_class_size=silhouette.min_class_size,
                median_class_size=silhouette.median_class_size,
                n_singletons=silhouette.n_singletons,
                silhouette_graph=silhouette.value,
                silhouette_graph_null=PermutationNull.from_samples(
                    silhouette.value, null_silhouette
                ),
                silhouette_graph_excluded=silhouette.n_excluded,
                graph_accuracy=accuracy.accuracy,
                graph_accuracy_affinity=graph_neighborhood_accuracy(
                    labels, affinity, weighting="affinity"
                ),
                graph_accuracy_null=PermutationNull.from_samples(
                    accuracy.accuracy, null_accuracy
                ),
                majority_class_rate=majority_class_rate(labels),
                cramers_v_vs_primary=(
                    float("nan")
                    if primary is None or (declared is not None and declared.scoring_role == "primary")
                    else cramers_v(labels, primary)
                ),
                n_ties=accuracy.n_ties,
                n_excluded=accuracy.n_excluded,
            )
        )
    return rows


def describe_dataset(name: str, *, label_scores: bool = True, **params) -> DatasetRow:
    """Build one row. Loads the dataset and its vertex graph; nothing is cached.

    Args:
        name: the registry name.
        label_scores: compute the graph-side Sil\\* and Acc\\* section. Costs an all-pairs
            closure — 9.6 s on `tabula`, 1.0 s on `digits` — so it is skippable, but it is on by
            default; see the module docstring. A dataset with no label sets pays nothing either
            way, because the closure is only built when there is something to score.
        **params: forwarded to the dataset factory.
    """
    dataset = load_dataset(name, **params)
    vg = vertex_graph(dataset)

    scores: list[LabelScoreRow] = []
    ceiling = ceiling_classes = None
    if label_scores and dataset.label_sets:
        from twin.metrics.context import EvaluationContext

        ctx = EvaluationContext.for_dataset(dataset)
        scores = score_label_sets(dataset, ctx.ideal, ctx.affinity)
        ceiling, ceiling_classes = compactness_ceiling(ctx.affinity, ctx.ideal, dataset.n)
        # tabula's closure is 200 MB and its condensed form another 100 MB, and the report walks
        # ten datasets in one process.
        ctx.release()

    weights = np.array([w for _, _, w in vg.distance.edges(data="weight")], dtype=float)
    sources = dataset.describe_sources()
    degrees = np.array([d for _, d in dataset.graph.degree()])

    # nnz of the raw stored matrices, before any symmetrisation the mapping applies. For tabula,
    # A's 98736 against D's 70098 against the graph's 49368 undirected edges is the consistency
    # a re-export would break, and it is invisible from the graph alone.
    stored_nnz = {
        key: int(sp.csr_array(matrix).nnz) for key, matrix in sorted(dataset.matrices.items())
    }

    return DatasetRow(
        name=name,
        n=dataset.n,
        m=dataset.m,
        density=nx.density(dataset.graph),
        connected=dataset.is_connected(),
        n_components=dataset.n_components(),
        largest_component_fraction=dataset.largest_component_fraction(),
        mean_degree=dataset.mean_degree(),
        affinity_source=sources["affinity"],
        distance_source=sources["distance"],
        affinity_derivation=vg.derivation,
        distance_derivation=vg.distance_derivation,
        weight_min=float(weights.min()) if weights.size else float("nan"),
        weight_median=float(np.median(weights)) if weights.size else float("nan"),
        weight_max=float(weights.max()) if weights.size else float("nan"),
        weight_distinct=int(np.unique(weights).size),
        sparsity=dataset.sparsity.describe() if dataset.sparsity else "the graph is the data",
        sparsity_violations=dataset.sparsity.check(dataset) if dataset.sparsity else [],
        degree_min=int(degrees.min()) if degrees.size else 0,
        degree_median=float(np.median(degrees)) if degrees.size else 0.0,
        degree_max=int(degrees.max()) if degrees.size else 0,
        n_isolated=int((degrees == 0).sum()),
        stored_nnz=stored_nnz,
        zero_distance_pair_fraction=zero_distance_pair_fraction(vg.distance, dataset.n),
        labels=[
            (
                label.name,
                label.n_classes,
                label.scoring_role,
                label.display_role,
                label.derivation,
            )
            for label in dataset.label_sets.values()
        ],
        composites=sorted(dataset.composite_of),
        consistency=check_weight_consistency(dataset),
        distortion=sparsification_distortion(dataset, vg.distance),
        description=dataset.description,
        label_scores=scores,
        compactness_ceiling=ceiling,
        compactness_ceiling_classes=ceiling_classes,
    )


def build_report(
    names: list[str] | None = None, *, label_scores: bool = True
) -> list[DatasetRow]:
    return [
        describe_dataset(name, label_scores=label_scores)
        for name in (names or list(DATASET_REGISTRY))
    ]


# --------------------------------------------------------------------------- #
#                                  rendering                                   #
# --------------------------------------------------------------------------- #


def _fmt(x: float, places: int = 4) -> str:
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "—"
    return f"{x:.{places}g}"


def _derivation_note(row: DatasetRow) -> str:
    """Which side had to be derived, and by what rule. Empty when both are stored."""
    for side, text in (("affinity", row.affinity_derivation), ("distance", row.distance_derivation)):
        if text.startswith("derived"):
            return f"{side}: " + text.split(": ", 1)[1]
    return "both sides stored"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _label_score_sections(rows: list[DatasetRow]) -> list[str]:
    r"""The graph-side label scores: Sil\*, Acc\*, and what each is read against.

    Empty when the section was skipped, so ``--no-label-scores`` produces a report with the
    heading absent rather than present and blank.
    """
    scored = [r for r in rows if r.label_scores]
    if not scored:
        return []

    def null_cell(null: PermutationNull) -> str:
        """The standardised gap, or an honest blank where no permutation test exists.

        Two significant figures at most, and none beyond the units place above 100. At
        $n = 30$ permutations $\sigma_0$ itself carries about 13 % error, so a $z$ printed to
        one decimal claims a precision it does not have — and it misleads concretely: at
        ``.1f`` two unrelated `digits` rows read as an identical 131.7 where they are 131.706
        and 131.723.
        """
        if null.degenerate:
            return "*no test*"
        if abs(null.z) >= 100:
            return f"{null.z:,.0f}"
        return f"{null.z:.1f}"

    out = [
        "## Graph-side label scores — Sil\\* and Acc\\*",
        "",
        "**Label-validity checks, not baselines** (spec §2.3, §5.3 guard 2). These score the",
        "*graph*: they answer whether the labels are reflected in the topology at all, which is",
        "what says when a low layout score is uninformative. Their layout-side counterparts are",
        "deliberately not in this table — the scales are incomparable, and column adjacency is",
        "an invitation to difference them.",
        "",
        "Every score below is under `disconnected='lcc'`, and **Acc\\* is uniform voting with",
        "the fractional tie rule** (eq. 61) unless the column says otherwise — a convention that",
        "cannot be recovered from the numbers, and one that `DECISIONS.md` (41) argues for rather",
        "than inherits. `z` is the standardised gap from a null that permutes the label vector,",
        "preserving every class size; `V` is Cramér's V against the dataset's scoring primary.",
        "",
        "### Sil\\* (eqs. 53-55)",
        "",
        _table(
            [
                "dataset",
                "label set",
                "role",
                "Sil\\*",
                "null mean",
                "null sd",
                "z",
                "ceiling",
                "% of ceiling",
                "V vs primary",
            ],
            [
                [
                    f"`{s.dataset}`" if i == 0 else "",
                    f"`{s.label_set}`",
                    s.role,
                    f"{s.silhouette_graph:.4f}",
                    f"{s.silhouette_graph_null.mean:+.4f}",
                    f"{s.silhouette_graph_null.sd:.5f}",
                    null_cell(s.silhouette_graph_null),
                    f"{r.compactness_ceiling:.4f}" if i == 0 else "",
                    f"{100 * s.silhouette_graph / r.compactness_ceiling:.0f} %"
                    if r.compactness_ceiling and r.compactness_ceiling > 0
                    else "—",
                    "—"
                    if not np.isfinite(s.cramers_v_vs_primary)
                    else f"{s.cramers_v_vs_primary:.3f}",
                ]
                for r in scored
                for i, s in enumerate(r.label_scores)
            ],
        ),
        "",
        "⚠️ **Do not read Sil\\* on a 0-to-1 scale.** Its bounds are properties of the graph, not",
        "of the labels: $a^*_i$ averages over all of one's own class and $b^*_i$ over all of the",
        "nearest other one, so once classes have internal extent comparable to the gap between",
        "them — which any connected $k$-NN graph guarantees — the two means converge. The",
        "measured $b^*/a^*$ is 1.50 on football's conferences, 1.64 on `digits` and 1.50 on",
        "`tabula`, capping Sil\\* near $1 - 1/1.5 \\approx 0.33$ before label quality enters at",
        "all. The implementation reaches 0.80 on a graph built to allow it (two cliques joined",
        "by a path), so the ceiling is the data's and not the metric's.",
        "",
        "The **ceiling** column is therefore the reference: the best Sil\\* a Leiden partition",
        "of the same graph reaches, over resolutions "
        + ", ".join(str(x) for x in CEILING_RESOLUTIONS)
        + ".",
        "A modularity partition picks its own classes and so *should* win — a ground truth",
        "matching it exactly would mean the labels add nothing to the topology. What the column",
        "says is how much of the available compactness the labels capture.",
        "",
        "⚠️ **`z` says how far a score sits above its own null; it does not compare magnitudes",
        "across label sets.** $\\sigma_0$ depends on the class count and balance, so a 2-class",
        "balanced split has a tiny null spread and reaches a large $z$ at a modest Sil\\*. Read",
        "`z` for *whether* a label set is encoded and the Sil\\* and ceiling columns for *how",
        "strongly*; for a magnitude comparison between two label sets, condition on a stratum",
        "at matched class counts (`DECISIONS.md` (42)). Read it to two significant figures at",
        "most — $\\sigma_0$ carries ~13 % error at 30 permutations — and past about 50 only as",
        "\"far above chance\".",
        "",
        "⚠️ **`V vs primary` is the confound column.** A covariate is only evidence about itself",
        "if it is not already a proxy for the biology, and a high $z$ cannot distinguish the two.",
        "A `sub` label or a composite reaching $V = 1$ is expected rather than informative — one",
        "coarsens or refines the primary by construction. The row to read is an **`auxiliary`**",
        "one: `tabula`'s `donor` at **0.917** against `cell_ontology_class` is a technical",
        "covariate that very nearly determines the biology, so neither its Sil\* nor its $z$ can",
        "be attributed to batch effect without conditioning — `DECISIONS.md` (42) does that.",
        "",
        "### Acc\\* (eqs. 61-64), 1-hop neighbourhood",
        "",
        _table(
            [
                "dataset",
                "label set",
                "Acc\\* (uniform)",
                "affinity-weighted",
                "floor",
                "margin over floor",
                "z",
                "ties",
                "excluded",
            ],
            [
                [
                    f"`{s.dataset}`" if i == 0 else "",
                    f"`{s.label_set}`",
                    f"{s.graph_accuracy:.4f}",
                    f"{s.graph_accuracy_affinity:.4f}",
                    f"{s.majority_class_rate:.4f}",
                    f"{s.graph_accuracy - s.majority_class_rate:+.4f}",
                    null_cell(s.graph_accuracy_null),
                    f"{s.n_ties}",
                    f"{s.n_excluded}",
                ]
                for r in scored
                for i, s in enumerate(r.label_scores)
            ],
        ),
        "",
        "⚠️ **Acc\\* exceeds 0.94 on every registered dataset, so the raw value says almost",
        "nothing** — a 1-hop neighbourhood of a $k$-NN graph is label-homogeneous nearly by",
        "construction. Read the margin over the floor: +0.86 on football's conference is",
        "decisive, **+0.03** on tabula's `method` is not, and the two raw scores differ by less",
        "than 0.002.",
        "",
        "**`method` is also where the permutation test does not exist**, and the column says so",
        "rather than printing a number. Its split is 213 against 4794, so every relabelling",
        "leaves almost every neighbourhood majority-`10X`: the null is a point mass at the",
        "majority rate, $\\sigma_0 = 0$, and $z$ is undefined. The honest statement there is the",
        "margin over the majority-class baseline, which is +0.029.",
        "",
        "`ties` counts vertices with more than one class tied for the argmax, which a",
        "variable-size neighbourhood produces far more readily than a fixed $k$ — the reported",
        "score resolves them by the fractional rule and is an expectation over tie-breakings to",
        "that extent. `affinity-weighted` is eq. (62); it equals the uniform column exactly on a",
        "dataset declared unweighted, because the declaration wins over any stray edge",
        "attribute, and it is **better on every weighted label set here** — which is why the",
        "uniform headline is a stated choice and not a default nobody examined.",
        "",
        "### What the two disagree about",
        "",
        "Where Sil\\* and Acc\\* diverge they are both right, and the pair says more than either:",
        "Sil\\* is global compactness and Acc\\* is local homogeneity (thesis §3.1.2). The clearest",
        "case in the roster is football's `division` — Acc\\* near the top of the table and Sil\\*",
        "at chance, because teams play mostly within their division while \"West\" spans eleven",
        "conferences that sit far apart in the graph. A multi-modal class is locally clean and",
        "globally diffuse, and only reporting both shows it.",
        "",
    ]

    degenerate = [
        (r.name, composite, part)
        for r in scored
        for composite, part in load_dataset(r.name).degenerate_composites().items()
    ]
    if degenerate:
        out += [
            "### Composites omitted as degenerate",
            "",
            "A composite is the product partition of its parts, which always refines each part —"
            " so it",
            "**equals** a part exactly when their class counts agree, and scoring it then emits a",
            "duplicate column under a different name. Omitted here and flagged for step 7's",
            "enumeration:",
            "",
        ] + [
            f"- `{name}` / `{composite}` **is** `{part}` renumbered, so it is scored once, as "
            f"`{part}`."
            for name, composite, part in degenerate
        ] + [
            "",
            "⚠️ This is **not** readable off `LabelSet.parent`, which was the obvious criterion.",
            "Both datasets that declare a parent compose it with its child, and only one is",
            "degenerate: tabula's `cell_ontology_class` does refine `compartment`, while",
            "football's `division` (West/East/none, applied across conferences) does not — so its",
            "composite is a genuine 15-class product. `parent` has no consumer in the tree and",
            "carries no refinement guarantee; `DECISIONS.md` (43) records that.",
            "",
        ]

    out += [
        "Permutation counts scale with $n$ ("
        + ", ".join(
            f"`{r.name}` {r.label_scores[0].silhouette_graph_null.n_permutations}"
            for r in scored
        )
        + "),",
        "because one evaluation costs 0.001 s at karate's size and 0.87 s at tabula's, where",
        "five label sets multiply it — while a z-score needs about thirty draws before",
        "$\\sigma_0$ is usable. Thirty is tolerable on `tabula` only because its $z$ values are",
        "large enough that a 13 % error in $\\sigma_0$ cannot change the reading.",
        "",
    ]
    return out


def to_markdown(rows: list[DatasetRow]) -> str:
    out = [
        "# Dataset report",
        "",
        "Generated by `python -m twin.report`. One row per entry in `DATASET_REGISTRY`; every",
        "column comes off the `Dataset` object, so adding a dataset adds a row and nothing else.",
        "",
        "## Shape and connectivity",
        "",
        _table(
            ["dataset", "n", "m", "density", "connected", "components", "largest cc", "mean degree"],
            [
                [
                    f"`{r.name}`",
                    f"{r.n}",
                    f"{r.m}",
                    _fmt(r.density, 3),
                    "yes" if r.connected else "**no**",
                    f"{r.n_components}",
                    _fmt(r.largest_component_fraction, 3),
                    _fmt(r.mean_degree, 4),
                ]
                for r in rows
            ],
        ),
        "",
        "`mean degree` is the *k* at which the λ selection criterion of spec §4.2 is defined.",
        "",
        "## Sparsity",
        "",
        "Sparse input is the operating regime, not a compromise: the twin transform *requires*",
        "it. On a complete graph over tabula's 5007 vertices A_E would carry ~10^11 nonzeros and",
        "A_tne would span 12.5 M nodes, so a dense-input comparison is not an experiment that",
        "could be run at any budget. Where the edge set was therefore a *choice*, the choice is",
        "declared here and checked against the graph rather than trusted.",
        "",
        _table(
            ["dataset", "sparsity model", "degree min / median / max", "isolated", "stored nnz", "declaration holds"],
            [
                [
                    f"`{r.name}`",
                    r.sparsity,
                    f"{r.degree_min} / {_fmt(r.degree_median, 4)} / {r.degree_max}",
                    f"{r.n_isolated}",
                    ", ".join(f"{k}: {v}" for k, v in r.stored_nnz.items()) or "—",
                    "yes" if not r.sparsity_violations else "**NO** — " + "; ".join(r.sparsity_violations),
                ]
                for r in rows
            ],
        ),
        "",
        "A *k*-NN graph gives every vertex degree at least *k* after or-symmetrisation, and",
        "nk/2 <= m <= nk. Those are the invariants a file re-exported at a different *k* breaks,",
        "which is why they are checked rather than printed.",
        "",
        "## Stored sides, and what is derived",
        "",
        _table(
            ["dataset", "affinity", "distance", "derivation"],
            [
                [
                    f"`{r.name}`",
                    r.affinity_source,
                    r.distance_source,
                    _derivation_note(r),
                ]
                for r in rows
            ],
        ),
        "",
        "`adaptive_kernel` has **no** derivation rule and raises: UMAP's per-point parameters",
        "are not stored, so no global transform of the affinity recovers a distance. Such a",
        "dataset must supply a stored distance side, and tabula fails loudly rather than falling",
        "back if `D` is ever dropped.",
        "",
        "## Ideal distances",
        "",
        "Distribution of the vertex graph's distance form over edges — the stored distance side",
        "where there is one, otherwise the single derivation from the stored affinity. This is",
        "the *ideal-distance source*: one target per dataset, shared by every representation.",
        "",
        _table(
            ["dataset", "min", "median", "max", "distinct", "pairs at d = 0"],
            [
                [
                    f"`{r.name}`",
                    _fmt(r.weight_min),
                    _fmt(r.weight_median),
                    _fmt(r.weight_max),
                    f"{r.weight_distinct}",
                    _fmt(100 * r.zero_distance_pair_fraction, 3) + " %",
                ]
                for r in rows
            ],
        ),
        "",
        "A zero ideal distance is not automatically a defect — under a probability or kernel",
        "model it is the model saying two vertices coincide — but it distorts silhouette and",
        "collapses shortest-path derivations, so the fraction of **vertex pairs** it affects is",
        "reported rather than the fraction of edges at the maximum weight.",
        "",
        "## Labels",
        "",
    ]

    label_rows = []
    for r in rows:
        if not r.labels:
            label_rows.append([f"`{r.name}`", "*none*", "", "", "", ""])
            continue
        for i, (name, n_classes, scoring, display, derivation) in enumerate(r.labels):
            label_rows.append(
                [
                    f"`{r.name}`" if i == 0 else "",
                    f"`{name}`",
                    f"{n_classes}",
                    scoring,
                    display,
                    derivation,
                ]
            )
    out += [
        _table(
            ["dataset", "label set", "classes", "scoring role", "display role", "derivation"],
            label_rows,
        ),
        "",
        "Class counts exclude the noise class. Scoring and display roles are independent:",
        "football scores the coarse label and takes hue from it, tabula inverts that. An",
        "`auxiliary` set is a technical covariate scored **deliberately** — a layout that",
        "separates cells by batch rather than by cell type is displaying an artifact, so a low",
        "score there is the good outcome.",
        "",
        "⚠️ **That reading holds for `method` and fails for `donor`.** It assumes the covariate",
        "is independent of the biology, and tabula's two donors sampled almost disjoint cell",
        "populations: NMI(`donor`, `cell_ontology_class`) is **0.413**, against 0.059 for",
        "`method`. So `donor`'s score is largely cell-type structure read through a confounded",
        "label — it is the *highest* of tabula's four on both Sil and Sil\\*, and a low score on",
        "it is not an achievable target. `method` is the covariate the argument applies to.",
        "",
    ]

    composites = [(r.name, c) for r in rows for c in r.composites]
    if composites:
        out += [
            "Declared composites: "
            + ", ".join(f"`{name}` / `{c}`" for name, c in composites)
            + ".",
            "",
        ]

    out += _label_score_sections(rows)

    both_stored = [r for r in rows if r.consistency is not None]
    checked = [r for r in both_stored if r.consistency.comparable]
    skipped = [r for r in both_stored if not r.consistency.comparable]

    out += ["## Weight-source consistency", ""]
    if not checked:
        out += ["No dataset stores two sides that carry rank information.", ""]
    else:
        out += [
            _table(
                [
                    "dataset",
                    "shared pairs",
                    "same edge set",
                    "global ρ",
                    "per-row ρ",
                    "unmixed per-row ρ",
                ],
                [
                    [
                        f"`{r.name}`",
                        f"{r.consistency.n_shared}",
                        "yes" if r.consistency.spans_same_edge_set else "**no**",
                        _fmt(r.consistency.global_spearman),
                        _fmt(r.consistency.median_row_spearman),
                        _fmt(r.consistency.median_row_spearman_unmixed),
                    ]
                    for r in checked
                ],
            ),
            "",
            "`unmixed per-row ρ` restricts to the pairs the raw stored matrix recorded in one",
            "direction only, where a symmetrisation cannot have mixed one row's parameters into",
            "another's. It is what separates *the two sources disagree* from *symmetrisation",
            "blurred an otherwise exact relationship*, and on tabula it settles the question:",
            "exactly −1 wherever it can be seen unmixed.",
            "",
        ]
    if skipped:
        out += [
            "Not comparable, because a side is an indicator or a single constant weight and so"
            " carries no ranks — nothing could disagree: "
            + ", ".join(f"`{r.name}`" for r in skipped)
            + ".",
            "",
        ]

    distorted = [r for r in rows if r.distortion is not None]
    out += ["## What sparsification costs", ""]
    if not distorted:
        out += ["No dataset in this selection retains the point cloud it was built from.", ""]
    else:
        out += [
            "For a point-cloud-derived dataset the graph is the **stipulated ground truth**, and",
            "a layout is not expected to recover distances the method was never shown. That is a",
            "deliberate stipulation rather than a limitation of the data — the unsparsified",
            "source is available for both of these and is declined — so what it costs is",
            "measured here rather than merely acknowledged.",
            "",
            _table(
                ["dataset", "rank ρ", "stress", "stress after rescaling", "d_geo/d_full median", "max", "edge weight error"],
                [
                    [
                        f"`{r.name}`",
                        _fmt(r.distortion.spearman),
                        _fmt(r.distortion.relative_stress),
                        _fmt(r.distortion.scaled_relative_stress),
                        _fmt(r.distortion.median_ratio),
                        _fmt(r.distortion.max_ratio),
                        _fmt(r.distortion.edge_weight_max_error, 2),
                    ]
                    for r in distorted
                ],
            ),
            "",
            "Graph geodesics are exact on the *k*-NN edges and overestimate strictly elsewhere,",
            "so the ratio is >= 1 everywhere and the distortion is a one-sided inflation of",
            "*long-range* distances. Local metrics — kNN recall, NND, edge lengths — are",
            "therefore undistorted; the global ones score against a proxy.",
            "",
            "**Read the rescaled column for SNS.** It fits its own global scale, so a uniform",
            "inflation costs it nothing; most of the raw stress here is exactly that. What",
            "remains is the ordering, and the rank correlation is what says how much of it",
            "survives. All of these improve monotonically with *k*, which is the same knob that",
            "makes the twin transform tractable — the trade is explicit and both ends are in",
            "this report.",
            "",
        ]

    out += ["## Why each dataset is in the roster", ""]
    for r in rows:
        out += [f"**`{r.name}`** — {r.description}", ""]

    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m twin.report", description=__doc__)
    parser.add_argument("names", nargs="*", help="datasets to report on (default: all)")
    parser.add_argument("-o", "--out", type=Path, help="write markdown here instead of stdout")
    parser.add_argument(
        "--no-label-scores",
        action="store_true",
        help="skip the graph-side Sil* / Acc* section, whose permutation nulls are most of the "
        "runtime (11 s instead of ~2 min)",
    )
    args = parser.parse_args(argv)

    unknown = [n for n in args.names if n not in DATASET_REGISTRY]
    if unknown:
        parser.error(f"unknown dataset(s) {unknown}; registered: {sorted(DATASET_REGISTRY)}")

    markdown = to_markdown(
        build_report(args.names or None, label_scores=not args.no_label_scores)
    )

    if args.out:
        args.out.write_text(markdown)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
