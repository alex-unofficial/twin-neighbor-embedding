r"""`MetricSpec` and `METRIC_REGISTRY` — the metric suite as data (spec §2.5).

**Declaration only.** Nothing here runs a metric; :mod:`twin.metrics.evaluate` is the driver.
What this module owns is *what each metric needs, what it returns, and how its rows are named*,
so that the driver can size a run, skip what cannot be computed, and key every row the way
``tests/fixtures/metric_golden.json`` already does.

**The row naming is read off the golden file, not invented.** A row is
``metric[@variant][@k=K]``: ``knn_recall@k=5``, ``crossing_number@curved``,
``graph_neighborhood_accuracy@uniform``. A distribution's *reduction* is folded into the variant
with an underscore — ``detour_ratio@curved_mean``, ``edge_lengths@mean`` — which is what makes
11 of the file's 44 keys come out of declared :attr:`MetricSpec.reductions` rather than out of a
naming scheme chosen here. One quirk is reproduced rather than tidied:
``node_resolution_vertexwise`` keys its default normaliser with **no** variant suffix, where
``graph_neighborhood_accuracy`` spells out ``@uniform``. Hence ``None`` is a legal variant name.

Five fields go beyond the spec's sketch, each forced by something measured — see
:class:`MetricSpec`. Two more are this module's own: ``params``, the static keyword arguments
that are part of a metric's identity (γ, the variant's mode) and so belong in the metric-set
digest; and ``reported``, which says a metric is a *foil* — computed and pinned so an argument
can be demonstrated, never tabulated (``node_resolution``, ``raw_stress``,
``normalized_stress``).

⚠️ **Which graph a binder hands over is not uniform, and must not be made uniform.**

* Everything that reads a valuation takes it from the context — ``ctx.affinity``, ``ctx.ideal``
  — never ``dataset.graph``, whose ``weight`` is a stored *distance* on `digits` (CLAUDE.md).
* Everything **per-edge** — the crossing family, ``detour_ratio``, ``edge_lengths`` — takes
  ``result.G``, which *is* ``dataset.graph``. ``result.E`` is indexed by
  ``dataset.graph.edges()``, and ``ctx.affinity`` is rebuilt from a CSR matrix, so its
  ``edges()`` come out in CSR order: on `digits` the two orders disagree in 12377 of 14732
  positions. Handing ``ctx.affinity`` to ``crossing_pairs`` would pair each edge's curve with
  another edge's edge point, and nothing would raise. None of these metrics reads a weight, so
  ``dataset.graph`` is safe for them in a way it is not for ADR.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from twin.metrics.crossings import (
    crossing_number,
    crossing_pairs_report,
    crossings_per_edge_pair,
    detour_ratio,
    length_weighted_crossings,
)
from twin.metrics.label import (
    graph_neighborhood_accuracy,
    graph_neighborhood_accuracy_report,
    knn_accuracy,
    knn_accuracy_report,
    majority_class_rate,
    silhouette,
    silhouette_graph,
    silhouette_graph_report,
    silhouette_report,
)
from twin.metrics.structural import (
    adjacency_distance_ratio,
    knn_jaccard,
    knn_jaccard_report,
    knn_precision,
    knn_precision_report,
    knn_recall,
    knn_recall_report,
    normalized_stress,
    raw_stress,
    scale_normalized_stress,
    weighted_recall,
    weighted_recall_report,
)
from twin.metrics.visual import (
    edge_length_cv,
    edge_lengths,
    nn_distance_distribution,
    node_resolution,
    node_resolution_vertexwise,
    node_resolution_vertexwise_report,
    normalized_nn_distance,
    pairwise_distance_distribution,
)

__all__ = [
    "CROSSING_GAMMA",
    "EXCLUDED",
    "METRIC_REGISTRY",
    "MetricSpec",
    "PROVIDERS",
    "Provider",
    "REDUCTIONS",
    "declared_rows",
    "metric_key",
    "metric_set_digest",
    "row_variant",
]

Kind = Literal["structural", "visual", "label", "crossing"]
Cost = Literal["cheap", "quadratic", "crossing"]

#: The curvature of the ``curved`` variant (spec §4.3). One value, shared by the provider that
#: solves the crossings and by ``detour_ratio``, so the two can never disagree about which
#: drawing they measured.
CROSSING_GAMMA = 0.5

_NO_PARAMS: Mapping[str, Any] = MappingProxyType({})


def _frozen(mapping: Mapping) -> Mapping:
    return MappingProxyType(dict(mapping))


# --------------------------------------------------------------------------- #
#                                 The spec                                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class MetricSpec:
    r"""Everything the driver needs to know about one metric without running it.

    Attributes:
        fn: the metric.
        bind: ``(ctx, result, variant, params) -> (args, kwargs)``, the adapter from the
            evaluation's nouns to this metric's signature. A signature match on ``needs`` is
            impossible: the argument orders are deliberately inconsistent and pinned that way —
            ``adjacency_distance_ratio(X, G)``, ``silhouette(X, labels)``,
            ``graph_neighborhood_accuracy(labels, G)`` (labels first, (39c)),
            ``crossing_number(G, V, E)``. ``params`` holds the variant's kwargs merged with every
            resolved input: ``k`` (a tuple) for a swept metric, ``labels`` for a label metric,
            and one entry per provider named in ``needs``. ``result`` is ``None`` for a
            ``graph_side`` metric, which has no layout.
        kind: the §2 family.
        needs: **providers, not argument names** — what has to exist before the metric can run.
            ``"labels"`` makes the driver enumerate label sets; ``"ideal"`` says the metric
            triggers the context's $O(n^2)$ closure; ``"edge_points"`` says the curved variant
            needs ``result.E``; ``"crossing_pairs"`` names a per-case :data:`PROVIDERS` entry,
            which is how three metrics share one 40 s solve instead of running three.
        report_fn: the ``*_report`` sibling, called **instead of** ``fn`` wherever one exists.
            It is free — ``knn_accuracy`` literally returns one field of
            ``knn_accuracy_report`` — and it carries ``n_excluded``, ``n_ties`` and
            ``majority_class_rate``, which (40) says are mandatory beside the score.
        value_field: the field of ``report_fn``'s result that is the value.
        sweeps: ``("k",)`` for the neighbourhood family; the result is one row per ``k``.
        returns: ``"scalar"`` or ``"distribution"``. Declared, never inferred from the shape —
            a swept metric and a distribution both return an ``ndarray`` (`DECISIONS.md` (20)).
        reductions: for a distribution, the summaries that become rows — read off the golden
            file: ``detour_ratio`` ``("mean", "max")``, ``edge_lengths`` ``("mean",)``, both
            distance distributions ``("min", "median", "max")``. This is what makes
            ``distribution_policy="off"`` safe: the summaries always land in the table, and
            the policy governs only whether the raw array is also written.
        swept_diagnostics: report fields that are **arrays over k** and are unpacked per row;
            every other non-value report field is a scalar diagnostic. Declared rather than
            sniffed by shape, which is what (20) forbids in spirit.
        variants: variant name → the kwargs it adds. ``None`` is the unsuffixed default.
        params: static kwargs shared by every variant, and part of the metric's identity.
        applies_to: the point sets the metric's ``X`` may be. ``("V",)`` for every entry, and
            load-bearing: nothing in ``twin/metrics/`` takes the edge cloud as its ``X``, and a
            pairwise metric on `tabula`'s ``E`` (m = 49368) is a 9.7 GB condensed vector — an
            OOM, not a slow case.
        cost: §2.5's cost class. Its first real use is deciding whether to sample RSS. ⚠️
            §2.5's *assignment* is wrong in both directions and these are the measured ones:
            crossings are 40 s and +428 MB per `tabula` layout, Sil\* 0.58 s, the label family
            under 4 s; the expensive metrics scale with $m$, not $n$.
        within_graph_only: §5.3 guard 1 — the exporter must refuse a cross-graph mean.
        graph_side: a function of ``(dataset, label set)`` and of nothing in the layout. Run
            once per dataset into a table of its own — §5.3 guard 2 — never once per case.
        reported: ``False`` for a foil: computed and pinned so that an argument can be shown,
            never tabulated.
        version: bumped by a commit that intentionally changes this metric's output, in the
            same commit that regenerates ``metric_golden.json``. Part of the metric-set digest,
            so the bump invalidates exactly the shards that hold the stale values. ⚠️ **Not a
            hash of the source**: a docstring edit would invalidate every shard.
    """

    fn: Callable
    bind: Callable
    kind: Kind
    needs: tuple[str, ...] = ()
    report_fn: Callable | None = None
    value_field: str | None = None
    sweeps: tuple[str, ...] = ()
    returns: Literal["scalar", "distribution"] = "scalar"
    reductions: tuple[str, ...] = ()
    swept_diagnostics: tuple[str, ...] = ()
    variants: Mapping[str | None, Mapping[str, Any]] = field(
        default_factory=lambda: _frozen({None: _NO_PARAMS})
    )
    params: Mapping[str, Any] = _NO_PARAMS
    applies_to: tuple[str, ...] = ("V",)
    cost: Cost = "cheap"
    within_graph_only: bool = False
    graph_side: bool = False
    reported: bool = True
    version: int = 1

    def __post_init__(self):
        if (self.report_fn is None) != (self.value_field is None):
            raise ValueError("report_fn and value_field are declared together or not at all")
        if self.returns == "distribution" and not self.reductions:
            raise ValueError(
                "a distribution needs declared reductions, or distribution_policy='off' would "
                "leave nothing of it in the table"
            )
        if self.returns == "scalar" and self.reductions:
            raise ValueError("reductions apply to distributions only")
        if self.swept_diagnostics and not self.sweeps:
            raise ValueError("swept_diagnostics without a sweep")
        unknown = set(self.reductions) - set(REDUCTIONS)
        if unknown:
            raise ValueError(f"unknown reductions {sorted(unknown)}; known: {sorted(REDUCTIONS)}")
        if not self.variants:
            raise ValueError("at least one variant; use {None: {}} for an unvaried metric")
        if self.graph_side and self.applies_to != ("V",):
            raise ValueError("a graph-side metric has no point set")
        object.__setattr__(self, "variants", _frozen({k: _frozen(v) for k, v in self.variants.items()}))
        object.__setattr__(self, "params", _frozen(self.params))

    @property
    def call(self) -> Callable:
        """What the driver actually invokes: the report sibling when there is one."""
        return self.report_fn or self.fn

    def variant_params(self, variant: str | None) -> dict[str, Any]:
        return {**self.params, **self.variants[variant]}


# --------------------------------------------------------------------------- #
#                                   Naming                                     #
# --------------------------------------------------------------------------- #

#: The reductions a distribution may declare. NaN-aware: a ``detour_ratio`` is ``nan`` for an
#: edge whose endpoints coincide, and that edge is *excluded* (and counted in ``n_excluded``),
#: not allowed to turn the whole summary into ``nan``.
REDUCTIONS: Mapping[str, Callable] = MappingProxyType(
    {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "min": np.nanmin,
        "max": np.nanmax,
    }
)


def row_variant(variant: str | None, reduction: str | None = None) -> str | None:
    """The ``variant`` column: a reduction is folded in with ``_`` — ``curved_mean``, ``mean``."""
    if reduction is None:
        return variant
    return reduction if variant is None else f"{variant}_{reduction}"


def declared_rows(spec: MetricSpec, variant: str | None, ks: tuple[int, ...]):
    """The ``(variant column, k)`` pairs one call of ``spec`` under ``variant`` emits.

    The single source of row naming: the driver emits exactly these for a computed metric, and
    exactly these with ``status="skipped"`` or ``"error"`` for one that could not be — so a
    metric that fails is visible in the table as the rows it *would* have produced.
    """
    columns = (
        [row_variant(variant, r) for r in spec.reductions]
        if spec.returns == "distribution"
        else [variant]
    )
    k_values = tuple(ks) if "k" in spec.sweeps else (None,)
    return [(column, k) for column in columns for k in k_values]


def metric_key(metric: str, variant: str | None = None, k: int | None = None) -> str:
    """``metric[@variant][@k=K]`` — the golden file's key for one row."""
    key = metric
    if variant is not None:
        key += f"@{variant}"
    if k is not None:
        key += f"@k={k}"
    return key


# --------------------------------------------------------------------------- #
#                                 Providers                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Provider:
    """An expensive intermediate, computed once per ``(case, variant)`` and shared.

    Attributes:
        fn: ``(ctx, result, variant_params) -> value``.
        cost: as on :class:`MetricSpec`; decides whether the driver samples RSS around it.
        report_fields: the value's diagnostic fields, written once per ``(case, variant)``
            under the provider's own name rather than once per consuming metric.
    """

    fn: Callable
    cost: Cost = "cheap"
    report_fields: tuple[str, ...] = ()


def _edge_points(result, params) -> Any:
    """``None`` for the straight variant, ``result.E`` for the curved one."""
    return result.E if params["curved"] else None


def _crossing_pairs(ctx, result, params):
    return crossing_pairs_report(
        result.G,
        result.V,
        _edge_points(result, params),
        gamma=params["gamma"],
        include_adjacent=params["include_adjacent"],
    )


#: Per-case intermediates, by the name a :attr:`MetricSpec.needs` entry uses.
PROVIDERS: Mapping[str, Provider] = MappingProxyType(
    {
        "crossing_pairs": Provider(
            fn=_crossing_pairs,
            cost="crossing",
            report_fields=(
                "n_candidates",
                "n_filtered",
                "n_near_tangential",
                "n_degenerate",
                "n_shared_endpoint",
            ),
        ),
    }
)

#: Needs that are not providers: they are resolved by the driver itself.
INPUT_NEEDS = frozenset({"labels", "ideal", "edge_points"})


# --------------------------------------------------------------------------- #
#                                   Binders                                    #
# --------------------------------------------------------------------------- #


def _k(params) -> dict[str, Any]:
    return {"k": tuple(params["k"])} if "k" in params else {}


def _bind_v(ctx, result, variant, params):
    return (result.V,), {}


def _bind_v_k(ctx, result, variant, params):
    return (result.V,), _k(params)


def _bind_v_affinity(ctx, result, variant, params):
    return (result.V, ctx.affinity), {}


def _bind_v_affinity_k(ctx, result, variant, params):
    return (result.V, ctx.affinity), _k(params)


def _bind_v_ideal(ctx, result, variant, params):
    return (result.V,), {"ideal": ctx}


def _bind_v_edges(ctx, result, variant, params):
    """Per-edge metrics take ``result.G`` — see the module docstring for why not ``ctx``."""
    return (result.V, result.G), {}


def _bind_vnr(ctx, result, variant, params):
    return (result.V, params["normalize"]), {}


def _bind_knn_accuracy(ctx, result, variant, params):
    return (result.V, params["labels"]), {**_k(params), "weighting": params["weighting"]}


def _bind_silhouette(ctx, result, variant, params):
    return (result.V, params["labels"]), {}


def _bind_crossing_metric(ctx, result, variant, params):
    return (result.G, result.V, _edge_points(result, params)), {
        "gamma": params["gamma"],
        "include_adjacent": params["include_adjacent"],
        "crossings": params["crossing_pairs"].crossings,
    }


def _bind_detour(ctx, result, variant, params):
    return (result.G, result.V, _edge_points(result, params)), {"gamma": params["gamma"]}


def _bind_silhouette_graph(ctx, result, variant, params):
    return (params["labels"],), {"ideal": ctx}


def _bind_graph_accuracy(ctx, result, variant, params):
    return (params["labels"], ctx.affinity), {
        "weighting": params["weighting"],
        "tie_rule": params["tie_rule"],
    }


def _bind_labels(ctx, result, variant, params):
    return (params["labels"],), {}


# --------------------------------------------------------------------------- #
#                                 The registry                                 #
# --------------------------------------------------------------------------- #

_CROSSING_VARIANTS = {"straight": {"curved": False}, "curved": {"curved": True}}
_CROSSING_PARAMS = {"gamma": CROSSING_GAMMA, "include_adjacent": True}

_NEIGHBOURHOOD = dict(
    kind="structural",
    sweeps=("k",),
    value_field="value",
    swept_diagnostics=("n_excluded", "n_scored"),
)

METRIC_REGISTRY: Mapping[str, MetricSpec] = MappingProxyType(
    {
        # ---------------------------------------------------------- structural (§2.1)
        "adjacency_distance_ratio": MetricSpec(
            fn=adjacency_distance_ratio,
            bind=_bind_v_affinity,
            kind="structural",
            cost="quadratic",
            within_graph_only=True,
        ),
        "knn_recall": MetricSpec(
            fn=knn_recall, report_fn=knn_recall_report, bind=_bind_v_affinity_k, **_NEIGHBOURHOOD
        ),
        "knn_precision": MetricSpec(
            fn=knn_precision,
            report_fn=knn_precision_report,
            bind=_bind_v_affinity_k,
            **_NEIGHBOURHOOD,
        ),
        "knn_jaccard": MetricSpec(
            fn=knn_jaccard, report_fn=knn_jaccard_report, bind=_bind_v_affinity_k, **_NEIGHBOURHOOD
        ),
        "weighted_recall": MetricSpec(
            fn=weighted_recall,
            report_fn=weighted_recall_report,
            bind=_bind_v_affinity_k,
            **_NEIGHBOURHOOD,
        ),
        "scale_normalized_stress": MetricSpec(
            fn=scale_normalized_stress,
            bind=_bind_v_ideal,
            kind="structural",
            needs=("ideal",),
            cost="quadratic",
        ),
        # The two stresses SNS is contrasted with: computed so §3's scale-dependence argument
        # can be shown, never tabulated (`raw_stress`'s docstring).
        "raw_stress": MetricSpec(
            fn=raw_stress,
            bind=_bind_v_ideal,
            kind="structural",
            needs=("ideal",),
            cost="quadratic",
            reported=False,
        ),
        "normalized_stress": MetricSpec(
            fn=normalized_stress,
            bind=_bind_v_ideal,
            kind="structural",
            needs=("ideal",),
            cost="quadratic",
            reported=False,
        ),
        # -------------------------------------------------------------- visual (§2.2)
        "normalized_nn_distance": MetricSpec(
            fn=normalized_nn_distance,
            bind=_bind_v_k,
            kind="visual",
            sweeps=("k",),
            cost="quadratic",
        ),
        # NR is the foil VNR is defined against, not a reported metric (CLAUDE.md).
        "node_resolution": MetricSpec(
            fn=node_resolution, bind=_bind_v, kind="visual", cost="quadratic", reported=False
        ),
        # The unsuffixed default is the per-vertex median, the [THESIS EDIT]; `max` is eq. (69)
        # as written; `mean` is exactly NND_1 and is kept only so that identity stays
        # demonstrable. The golden file keys the default with no suffix.
        "node_resolution_vertexwise": MetricSpec(
            fn=node_resolution_vertexwise,
            report_fn=node_resolution_vertexwise_report,
            value_field="value",
            bind=_bind_vnr,
            kind="visual",
            variants={
                None: {"normalize": "median"},
                "max": {"normalize": "max"},
                "mean": {"normalize": "mean"},
            },
            cost="quadratic",
        ),
        "pairwise_distance_distribution": MetricSpec(
            fn=pairwise_distance_distribution,
            bind=_bind_v,
            kind="visual",
            returns="distribution",
            reductions=("min", "median", "max"),
            cost="quadratic",
        ),
        "nn_distance_distribution": MetricSpec(
            fn=nn_distance_distribution,
            bind=_bind_v,
            kind="visual",
            returns="distribution",
            reductions=("min", "median", "max"),
            cost="quadratic",
        ),
        "edge_lengths": MetricSpec(
            fn=edge_lengths,
            bind=_bind_v_edges,
            kind="visual",
            returns="distribution",
            reductions=("mean",),
        ),
        "edge_length_cv": MetricSpec(fn=edge_length_cv, bind=_bind_v_edges, kind="visual"),
        # --------------------------------------------------------------- label (§2.3)
        "knn_accuracy": MetricSpec(
            fn=knn_accuracy,
            report_fn=knn_accuracy_report,
            value_field="accuracy",
            bind=_bind_knn_accuracy,
            kind="label",
            needs=("labels",),
            sweeps=("k",),
            swept_diagnostics=("n_ties", "n_zero_distance"),
            # Uniform is the headline (eq. 57); `distance` is eq. (58), the layout half of the
            # weighted pairing (41) reports beside it and warns against reading as neutral.
            variants={None: {"weighting": "uniform"}, "distance": {"weighting": "distance"}},
        ),
        "silhouette": MetricSpec(
            fn=silhouette,
            report_fn=silhouette_report,
            value_field="value",
            bind=_bind_silhouette,
            kind="label",
            needs=("labels",),
            cost="quadratic",
        ),
        # ------------------------------------------------------------ crossings (§2.4)
        # One solve, three summaries: `needs=("crossing_pairs",)` is what stops a flat registry
        # tripling the dominant cost of the suite (40 s and +428 MB per tabula layout).
        "crossing_number": MetricSpec(
            fn=crossing_number,
            bind=_bind_crossing_metric,
            kind="crossing",
            needs=("crossing_pairs", "edge_points"),
            variants=_CROSSING_VARIANTS,
            params=_CROSSING_PARAMS,
            cost="crossing",
            within_graph_only=True,
        ),
        "crossings_per_edge_pair": MetricSpec(
            fn=crossings_per_edge_pair,
            bind=_bind_crossing_metric,
            kind="crossing",
            needs=("crossing_pairs", "edge_points"),
            variants=_CROSSING_VARIANTS,
            params=_CROSSING_PARAMS,
            cost="crossing",
        ),
        "length_weighted_crossings": MetricSpec(
            fn=length_weighted_crossings,
            bind=_bind_crossing_metric,
            kind="crossing",
            needs=("crossing_pairs", "edge_points"),
            variants=_CROSSING_VARIANTS,
            params=_CROSSING_PARAMS,
            cost="crossing",
            within_graph_only=True,
        ),
        # What separates "the transform reduces crossings" from "the drawing style does" (§4.3).
        "detour_ratio": MetricSpec(
            fn=detour_ratio,
            bind=_bind_detour,
            kind="crossing",
            needs=("edge_points",),
            variants=_CROSSING_VARIANTS,
            params={"gamma": CROSSING_GAMMA},
            returns="distribution",
            reductions=("mean", "max"),
        ),
        # -------------------------------------------- graph side: label-validity checks
        # Functions of (dataset, label set) alone. Run once per dataset into their own table,
        # which is §5.3 guard 2 by construction rather than by a rule someone remembers.
        "silhouette_graph": MetricSpec(
            fn=silhouette_graph,
            report_fn=silhouette_graph_report,
            value_field="value",
            bind=_bind_silhouette_graph,
            kind="label",
            needs=("labels", "ideal"),
            graph_side=True,
        ),
        # Uniform is the headline (41); `affinity` is eq. (62) beside it; `first` exists
        # because the fractional tie rule is untested without a rule to disagree with.
        "graph_neighborhood_accuracy": MetricSpec(
            fn=graph_neighborhood_accuracy,
            report_fn=graph_neighborhood_accuracy_report,
            value_field="accuracy",
            bind=_bind_graph_accuracy,
            kind="label",
            needs=("labels",),
            variants={
                "uniform": {"weighting": "uniform", "tie_rule": "fractional"},
                "affinity": {"weighting": "affinity", "tie_rule": "fractional"},
                "first": {"weighting": "uniform", "tie_rule": "first"},
            },
            graph_side=True,
        ),
        # The floor every accuracy is read against: a pinned accuracy without it says nothing.
        "majority_class_rate": MetricSpec(
            fn=majority_class_rate,
            bind=_bind_labels,
            kind="label",
            needs=("labels",),
            graph_side=True,
        ),
    }
)

#: Every ``twin.metrics.__all__`` name that is deliberately **not** a registry entry, and why.
#: ``tests/test_metric_registry.py`` asserts the export list is covered exactly, so a new
#: metric is either registered or excused here — never simply never run.
EXCLUDED: Mapping[str, str] = MappingProxyType(
    {
        "EvaluationContext": "infrastructure: the per-dataset noun every binder reads",
        "condensed": "infrastructure: the pair ordering",
        "finite_pair_mask": "infrastructure: the stress metrics' pair mask",
        "ideal_distances": "infrastructure: owned by EvaluationContext.ideal",
        "pairwise_distances": "infrastructure: the shared pdist cache",
        "mean_pairwise_distance": "a component of ADR and NND, not a score",
        "optimal_scale": "SNS's closed-form alpha*, a component rather than a score",
        "require_n_by_d": "infrastructure: the orientation guard",
        "default_k": "infrastructure: the driver's k",
        "clear_pairwise_cache": "infrastructure: cache hygiene, called by the runner",
        "clear_loo_cache": "infrastructure: cache hygiene, called by the runner",
        "edge_index_array": "infrastructure: the per-edge order of record",
        "edge_curves": "layer 1 of the crossing pipeline: control points, not a score",
        "crossing_pairs": "layer 2: the 'crossing_pairs' provider, shared by three metrics",
        "crossing_pairs_report": "layer 2: the 'crossing_pairs' provider, shared by three metrics",
        "CrossingReport": "the provider's return type",
        "GraphAccuracyReport": "a report type",
        "KnnAccuracyReport": "a report type",
        "NeighborPreservationReport": "a report type",
        "NodeResolutionReport": "a report type",
        "SilhouetteReport": "a report type",
    }
)


def metric_set_digest(specs: Mapping[str, MetricSpec] = METRIC_REGISTRY, **settings: Any) -> str:
    r"""What a shard records, and what :func:`~twin.runs.keys.shard_is_current` compares.

    Over ``{name, variant, params, version}`` for every entry, plus the declarations that
    change which rows a metric emits (``returns``, ``sweeps``, ``reductions``, ``graph_side``),
    plus ``settings`` — the evaluation choices that are not part of a case's identity but do
    change its rows, such as an explicit ``k`` tuple.

    ⚠️ **``version`` is why this is not just the names.** Without it, editing a metric's
    internals and re-running resumes onto the stale values — the failure the golden-file rule
    exists to prevent, except that the stale numbers would reach the results table. The rule
    that ties the two together, *a commit that changes a metric's output regenerates
    ``metric_golden.json`` and bumps that metric's ``version``*, is enforced by
    ``tests/test_metric_versions.py``.

    Functions are not digested — neither their identity (it differs per process) nor their
    source (a docstring edit would invalidate every shard).
    """
    payload = {
        "metrics": {
            name: {
                "variants": {
                    "" if v is None else v: dict(spec.variant_params(v)) for v in spec.variants
                },
                "version": spec.version,
                "returns": spec.returns,
                "sweeps": list(spec.sweeps),
                "reductions": list(spec.reductions),
                "graph_side": spec.graph_side,
            }
            for name, spec in sorted(specs.items())
        },
        "settings": settings,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=repr)
    return hashlib.blake2b(canonical.encode()).hexdigest()[:16]
