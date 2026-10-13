r"""Reference levels for the graph-side label scores — what Sil\* and Acc\* are read against.

Moved here from `report.py` at spec §7 step 7, **unchanged**. The move is forced, not
tidying: `report.py` imports from `twin.metrics`, so a metric registry that imported the
reporting module for these would be metrics → report → metrics — a cycle surviving only because
report's imports were function-local, and backwards layering regardless. The metric layer must
not depend on the reporting layer. `tests/test_report_numbers.py` pins every value these
produce and passes unchanged across the move, which is the proof that it preserved behaviour.

Why they exist at all, in one line each (the measurements are in `DECISIONS.md` (40)-(42)):

* a **permutation null** for Sil\* and Acc\*, because Sil\*'s chance level is *negative* by a
  class-count-dependent amount and Acc\* exceeds 0.94 everywhere;
* the **majority-class rate**, the floor every accuracy is read against (it lives in
  `label.py`; it is listed here because it is one of the four);
* the **compactness ceiling**, the best Sil\* any Leiden partition of the graph reaches —
  ~0.53 even where the labels *are* the topology, so 0.33 is not "low";
* **Cramér's V against the scoring primary**, the confound column.
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np

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
