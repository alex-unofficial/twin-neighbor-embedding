"""The per-dataset evaluation context (spec §2.0, `DECISIONS.md` (5)).

    A metric is a function of the embedded points, the `Dataset`, and the **vertex**
    `LayoutGraph`.

"points plus a `Dataset`" is not sufficient: kNN recall and crossings need graph structure, SNS
and Sil\\* need ideal distances, ADR and weighted recall need affinities. Nothing is lost by
saying so, because the vertex graph is a pure function of the dataset — it just has to be
*reachable*, and it has to be the un-normalised faithful one rather than whatever the case was
laid out with.

**A context, not a field on `LayoutResult`.** Attaching this to each result would duplicate it
per representation, per seed and per λ, and would re-couple the two layers step 2 separated —
a `LayoutGraph` is never labelled and a `Dataset` is never laid out, and this object is where
the two meet without either being contaminated.

This module is the noun. The verb — `METRIC_REGISTRY`, `MetricSpec`, `evaluate_layout`,
`batch_layout_metrics`, persistence — is spec §7 step 7, and is what this exists to make
trivial.
"""

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from twin.layoutgraph import LayoutGraph, vertex_graph
from twin.metrics.common import condensed, default_k, ideal_distances


@dataclass(frozen=True)
class EvaluationContext:
    """Everything about one dataset that every metric over it shares.

    Built once per dataset and threaded through, so that the expensive parts — the
    $O(n^2)$ ideal-distance closure above all — are paid for once rather than per case.

    Attributes:
        dataset: the source of truth and the labels.
        vertex: ``vertex_graph(dataset)``, un-normalised and faithful. Not the representation
            that was laid out: $A_E$ and $A_{tne}$ are affinity algebra over a different node
            set, and their distance form is a generic fallback in nobody's units.
        disconnected: the policy the ideal distances are computed under.
        max_n: the guard passed to :func:`~twin.metrics.common.ideal_distances`.
    """

    dataset: object
    vertex: LayoutGraph
    disconnected: str = "lcc"
    max_n: int = 20000

    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------- construction

    @classmethod
    def for_dataset(cls, dataset, **kw) -> "EvaluationContext":
        """The only constructor worth calling: build the vertex graph from the dataset."""
        vertex = vertex_graph(dataset)
        if not vertex.is_faithful:
            raise ValueError(
                f"{dataset.name}: the evaluation context needs the faithful vertex graph, and "
                "this one is degree-normalised. Normalisation perturbs A_V, so scoring against "
                "it would compare a layout to a target the dataset does not have."
            )
        return cls(dataset=dataset, vertex=vertex, **kw)

    def __post_init__(self):
        if self.vertex.kind != "vertex":
            raise ValueError(
                f"the evaluation context takes the vertex graph, got {self.vertex.name}. A_E's "
                "nodes are edges and A_tne's are edges and vertices, so neither can be scored "
                "against a length-n label vector or a vertex-indexed ideal distance."
            )

    # ------------------------------------------------------------------ shape

    @property
    def n(self) -> int:
        return self.dataset.n

    @property
    def k(self) -> int:
        """$k$ for the neighbourhood metrics: the **input** graph's mean degree (§0.5 dec. 6).

        The same $k$ for every representation of this dataset, because $R^k$ is evaluated
        against $G$'s neighbourhoods whichever matrix produced the layout.
        """
        return default_k(self.dataset.graph)

    # ------------------------------------------------------------- the forms

    @property
    def affinity(self) -> nx.Graph:
        """The affinity-valued graph ADR and ``weighted_recall`` read.

        Stamped, so those two refuse it if it is ever replaced by something that is not an
        affinity. Never ``dataset.graph``: on digits that graph's ``weight`` attribute is a
        stored Euclidean distance.
        """
        return self.vertex.affinity_graph()

    @property
    def distance(self) -> nx.Graph:
        """The least-derived distance form: the ideal distances' source, as a graph."""
        return self.vertex.distance

    @property
    def ideal(self) -> np.ndarray:
        r"""The all-pairs ideal distances, computed **once** and shared (spec §2.0).

        Lazy, and that matters twice: a metric set of ADR and kNN recall must not pay for a
        closure it never reads, and at step 7 the resume path builds contexts for cases whose
        shards already exist.

        This is the single target the whole ablation rests on. $A_V$, $A_E$ and $A_{tne}$ are
        scored against **this** array, not against one closure each — otherwise the comparison
        between the three representations, which is the point of the experiment, would be
        between three different questions.
        """
        if "ideal" not in self._cache:
            self._cache["ideal"] = ideal_distances(
                self.distance, disconnected=self.disconnected, max_n=self.max_n
            )
        return self._cache["ideal"]

    @property
    def ideal_condensed(self) -> np.ndarray:
        r"""The ideal distances in :data:`~twin.metrics.common.CONDENSED_ORDERING`, condensed once.

        Beside :attr:`ideal` for the same reason :attr:`ideal` is here: derived, per-dataset,
        expensive, and shared by every representation. Without it SNS, ``raw_stress``,
        ``normalized_stress`` and the finite-pair mask each condense the same array — four times
        per case, for a quantity that cannot vary within a dataset.

        The two forms are held together and cost 100 MB each at tabula's $n = 5007$; the square
        one is what ``silhouette_graph`` will need at step 6 as a precomputed metric, and
        :meth:`release` drops both.
        """
        if "ideal_condensed" not in self._cache:
            self._cache["ideal_condensed"] = condensed(self.ideal)
        return self._cache["ideal_condensed"]

    @property
    def finite_pairs(self) -> np.ndarray:
        """Which condensed pairs the disconnection policy actually scored.

        ``False`` marks a pair excluded by ``disconnected`` or unreachable in the graph. Every
        registered dataset is connected, so this is all-True on the current roster — it exists
        so that no stress metric grows a second code path for the disconnected case.
        """
        if "finite_pairs" not in self._cache:
            self._cache["finite_pairs"] = np.isfinite(self.ideal_condensed)
        return self._cache["finite_pairs"]

    @property
    def ideal_is_computed(self) -> bool:
        """Whether :attr:`ideal` has been materialised yet. For tests and for a run planner."""
        return "ideal" in self._cache

    def release(self) -> None:
        """Drop the cached closure and everything derived from it.

        For a runner moving to the next dataset — tabula's square closure is 200 MB and its
        condensed form another 100 MB. Clears the cache wholesale rather than popping ``ideal``
        alone: a derived array that outlives the array it was derived from is a stale target,
        and this is the one place the ablation's shared target can go stale.
        """
        self._cache.clear()
