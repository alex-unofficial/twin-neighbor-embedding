r"""`evaluate_layout` — the per-case driver of `METRIC_REGISTRY` (spec §3.7).

⚠️ **The signature departs from §3.7**, which takes ``(result, dataset, metric_spec)``.
``LayoutResult.dataset`` *is* the dataset, so passing it again opens a path where a layout is
scored against a different dataset object — which is exactly what resuming from a persisted
layout introduces. It takes the :class:`~twin.metrics.context.EvaluationContext` instead: that
carries the dataset, the faithful vertex graph, the shared ``k`` and the shared ideal, and it is
where the disconnection policy lives. The two are checked against each other.

Rows are dicts, not a frame: the caller stamps the case's identity on them (`twin.runs.keys`)
and writes them. Two lists come back — the metric rows and the diagnostics — because (18) keeps
counts with different denominators out of the value column.

**Nothing raises past this function for a metric's sake.** A 40 s crossing metric failing on one
case must not abort an eight-hour batch, and must not be silently absent either: it becomes rows
with ``status="error"`` and the exception in ``skip_reason``. A metric that *cannot* run — no
labels, no edge points, a 3-D layout for a planar metric — becomes rows with
``status="skipped"`` and the reason. Either way the table holds every row the metric would have
produced, which is what §5.3 guard 3 ("no silent caps") needs to footnote it.
"""

from __future__ import annotations

import dataclasses
import time
from collections import Counter
from collections.abc import Mapping
from typing import Any

import numpy as np

from twin.instrumentation import RSSSampler
from twin.metrics.context import EvaluationContext
from twin.metrics.registry import (
    METRIC_REGISTRY,
    PROVIDERS,
    REDUCTIONS,
    MetricSpec,
    declared_rows,
)

__all__ = [
    "ProviderCache",
    "evaluate_graph_side",
    "evaluate_layout",
    "resolve_ks",
    "scored_label_sets",
]

#: An error message is kept to this many characters; the traceback is not persisted.
REASON_MAX = 400

#: The disconnection policies a run may use. `lcc` only, by the author's decision at plan
#: review: every registered dataset is connected, `silhouette_graph` refuses `mask`
#: unconditionally, and two-contexts-per-dataset plumbing for §4.4's SNS-only `mask` allowance
#: would be untested code for a case that does not occur. It arrives with the first
#: disconnected dataset.
SUPPORTED_DISCONNECTED = ("lcc",)


# --------------------------------------------------------------------------- #
#                             What gets enumerated                             #
# --------------------------------------------------------------------------- #


def scored_label_sets(dataset) -> tuple[list[str], dict[str, str]]:
    """The label sets a label metric is evaluated over, and the composites dropped from them.

    Every declared set — primary, then sub, then auxiliary — and every composite, **minus**
    :meth:`~twin.datasets.Dataset.degenerate_composites`. Without the subtraction `tabula` emits
    ``compartment_cell_type`` as a duplicate column of ``cell_ontology_class``: the product
    partition equals its finer part, so the two score identically for every metric.

    Returns:
        ``(names, dropped)`` — ``dropped`` maps each removed composite to the part it collapses
        to, for the manifest (§5.3 guard 3: no silent caps).
    """
    names = [
        label_set.name
        for role in ("primary", "sub", "auxiliary")
        for label_set in dataset.label_sets_by_scoring_role(role)
    ]
    dropped = dataset.degenerate_composites()
    names += [name for name in sorted(dataset.composite_of) if name not in dropped]
    return names, dropped


def resolve_ks(ctx: EvaluationContext, k=None) -> tuple[int, ...]:
    """The neighbourhood sizes: ``default_k`` unless the caller asks for a sweep.

    ``default_k`` is the input graph's mean degree (§0.5 decision 6) and §4.2's λ criterion is
    defined at it, so it is the headline — 5 on `karate_weighted`, 16 on `digits`, 55 on
    `sbm_er_3modes`. An explicit tuple is for a robustness appendix, and the manifest has to
    say which was used, because the rows cannot.
    """
    if k is None:
        return (ctx.k,)
    ks = tuple(int(v) for v in np.atleast_1d(k))
    if not ks or min(ks) < 1:
        raise ValueError(f"k must be positive integers, got {k!r}")
    return ks


def check_policy(ctx: EvaluationContext) -> None:
    if ctx.disconnected not in SUPPORTED_DISCONNECTED:
        raise ValueError(
            f"disconnected={ctx.disconnected!r} is not supported by the runner; use 'lcc'. "
            "Every registered dataset is connected, and silhouette_graph refuses 'mask' "
            "unconditionally, so a second policy would be untested code for a case that does "
            "not occur (plan step 7, decision 3)."
        )


def _check_same_dataset(result, ctx: EvaluationContext) -> None:
    """``ctx.dataset is result.dataset``, or the same registry name and factory params.

    Identity is the normal case. Name-and-params equality is what a layout rehydrated from disk
    can offer, since it is rebuilt through ``DATASET_REGISTRY`` rather than unpickled.
    """
    a, b = result.dataset, ctx.dataset
    if a is b:
        return
    if (a.name, a.params) != (b.name, b.params):
        raise ValueError(
            f"the layout is of {a.name}{a.params} and the context is for {b.name}{b.params}. "
            "A layout must be scored against the dataset it is a layout of."
        )


# --------------------------------------------------------------------------- #
#                               The provider cache                             #
# --------------------------------------------------------------------------- #


class ProviderCache:
    r"""Per-case intermediates, computed once per ``(provider, variant)`` and shared.

    The per-case analogue of :class:`~twin.metrics.context.EvaluationContext`, and deliberately
    a **second object**: the context is per-dataset, frozen and lazy by design, and adding a
    per-case field to it would re-couple the layers step 2 separated. It lives exactly as long
    as one :func:`evaluate_layout` call.

    A provider that raises is remembered as having raised, so three consumers of a failed
    40 s solve do not each retry it.
    """

    def __init__(self, ctx: EvaluationContext, result, *, measure_memory: bool = True):
        self.ctx = ctx
        self.result = result
        self.measure_memory = measure_memory
        self.calls: Counter = Counter()
        self.diagnostics: list[dict] = []
        self._values: dict[tuple[str, str | None], Any] = {}

    def get(self, name: str, variant: str | None, params: Mapping[str, Any]):
        key = (name, variant)
        if key not in self._values:
            provider = PROVIDERS[name]
            self.calls[key] += 1
            try:
                value, elapsed, mem = _timed(
                    provider.fn,
                    (self.ctx, self.result, params),
                    {},
                    sample=self.measure_memory and provider.cost != "cheap",
                )
            except Exception as exc:  # noqa: BLE001 — recorded, re-raised to each consumer
                self._values[key] = exc
            else:
                self._values[key] = value
                base = {"point_set": "V", "label_set": None, "metric": name, "variant": variant}
                self.diagnostics.append({**base, "k": None, "diagnostic": "elapsed_s", "value": elapsed})
                if mem is not None:
                    self.diagnostics.append(
                        {**base, "k": None, "diagnostic": "rss_delta_mb", "value": mem}
                    )
                for field_name in provider.report_fields:
                    self.diagnostics.append(
                        {
                            **base,
                            "k": None,
                            "diagnostic": field_name,
                            "value": float(getattr(value, field_name)),
                        }
                    )
        value = self._values[key]
        if isinstance(value, Exception):
            raise ProviderError(name, value)
        return value


class ProviderError(RuntimeError):
    def __init__(self, name: str, cause: Exception):
        super().__init__(f"provider {name!r} failed: {type(cause).__name__}: {cause}")
        self.name = name
        self.cause = cause


def _timed(fn, args, kwargs, *, sample: bool):
    """Call ``fn``, returning ``(value, elapsed_s, rss_delta_mb or None)``.

    RSS is sampled only for a non-cheap cost class — the first real use of §2.5's field. The
    sampler is never nested: evaluating a metric is not done inside an embed, and nested
    samplers on one process would measure the same process twice. ⚠️ The delta is from a
    baseline that drifts upward under glibc across a batch, so it is a lower bound.
    """
    with RSSSampler(enabled=sample) as mem:
        t0 = time.perf_counter()
        value = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
    return value, elapsed, (mem.delta_mb if mem.enabled else None)


# --------------------------------------------------------------------------- #
#                                  The driver                                  #
# --------------------------------------------------------------------------- #


def _reason(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= REASON_MAX else text[: REASON_MAX - 1] + "…"


def _unmet(spec: MetricSpec, params: Mapping[str, Any], result) -> str | None:
    """Why this metric cannot run on this case, or None. Decided before any work is done."""
    if result is None:
        return None
    if spec.kind == "crossing" and result.d != 2:
        return f"crossing geometry is planar; this layout is {result.d}-dimensional"
    if "edge_points" in spec.needs and params.get("curved") and result.E is None:
        return f"the {result.embedding} layout of {result.representation} has no edge points"
    return None


def _status_rows(spec, variant, ks, base, status, reason) -> list[dict]:
    return [
        {
            **base,
            "variant": column,
            "k": k,
            "value": float("nan"),
            "status": status,
            "skip_reason": reason,
            "n_excluded": None,
        }
        for column, k in declared_rows(spec, variant, ks)
    ]


def _numeric_fields(report, exclude: set[str]) -> dict[str, Any]:
    """The report's diagnostic fields: everything numeric that is not the value itself."""
    out = {}
    for f in dataclasses.fields(report):
        if f.name in exclude:
            continue
        value = getattr(report, f.name)
        if isinstance(value, (bool, str)) or value is None:
            continue
        if isinstance(value, (int, float, np.integer, np.floating, np.ndarray)):
            out[f.name] = value
    return out


def evaluate_spec(
    name: str,
    spec: MetricSpec,
    variant: str | None,
    *,
    ctx: EvaluationContext,
    result,
    ks: tuple[int, ...],
    label_set: str | None = None,
    providers: ProviderCache | None = None,
    measure_memory: bool = True,
    distributions: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """One metric, one variant, one label set — every row it emits, and its diagnostics.

    The unit both drivers share: :func:`evaluate_layout` per case, and the graph-side table per
    dataset (``result=None``).
    """
    base = {"point_set": None if result is None else "V", "label_set": label_set, "metric": name}
    params = spec.variant_params(variant)

    reason = _unmet(spec, params, result)
    if reason is not None:
        return _status_rows(spec, variant, ks, base, "skipped", reason), []

    if "k" in spec.sweeps:
        params["k"] = ks
    if "labels" in spec.needs:
        params["labels"] = ctx.dataset.labels(label_set)

    try:
        for need in spec.needs:
            if need in PROVIDERS:
                params[need] = providers.get(need, variant, params)
        args, kwargs = spec.bind(ctx, result, variant, params)
        output, elapsed, mem = _timed(
            spec.call, args, kwargs, sample=measure_memory and spec.cost != "cheap"
        )
        rows, diagnostics = _unpack(spec, variant, ks, base, output, distributions)
    except Exception as exc:  # noqa: BLE001 — see the module docstring
        return _status_rows(spec, variant, ks, base, "error", _reason(exc)), []

    call = {**base, "variant": variant, "k": None}
    diagnostics.append({**call, "diagnostic": "elapsed_s", "value": elapsed})
    if mem is not None:
        diagnostics.append({**call, "diagnostic": "rss_delta_mb", "value": mem})
    return rows, diagnostics


def _unpack(spec, variant, ks, base, output, distributions):
    """Turn one call's return value into rows, per the spec's declaration — never its shape."""
    n_excluded = None
    diag: dict[str, Any] = {}
    if spec.report_fn is not None:
        value = getattr(output, spec.value_field)
        n_excluded = getattr(output, "n_excluded", None)
        diag = _numeric_fields(output, {spec.value_field, "k", "n_excluded"})
    else:
        value = output

    rows: list[dict] = []
    diagnostics: list[dict] = []
    ok = {"status": "ok", "skip_reason": None}

    if spec.returns == "distribution":
        array = np.asarray(value, dtype=float).ravel()
        excluded = int(np.isnan(array).sum())
        if distributions is not None:
            distributions[(base["point_set"], base["label_set"], base["metric"], variant)] = array
            rows.append(
                {**base, "variant": variant, "k": None, "value": float("nan"),
                 "status": "distribution", "skip_reason": None, "n_excluded": excluded}
            )
        for column, k in declared_rows(spec, variant, ks):
            reduction = column if variant is None else column[len(variant) + 1 :]
            summary = (
                float(REDUCTIONS[reduction](array)) if array.size > excluded else float("nan")
            )
            rows.append({**base, "variant": column, "k": k, "value": summary,
                         "n_excluded": excluded, **ok})
        return rows, diagnostics

    if "k" in spec.sweeps:
        values = np.asarray(value, dtype=float).ravel()
        if values.size != len(ks):
            raise ValueError(f"{base['metric']} returned {values.size} values for {len(ks)} k")
        for i, k in enumerate(ks):
            excluded = None if n_excluded is None else int(np.asarray(n_excluded).ravel()[i])
            rows.append({**base, "variant": variant, "k": k, "value": float(values[i]),
                         "n_excluded": excluded, **ok})
            for field_name, field_value in diag.items():
                if field_name in spec.swept_diagnostics:
                    field_value = np.asarray(field_value).ravel()[i]
                diagnostics.append({**base, "variant": variant, "k": k,
                                    "diagnostic": field_name, "value": float(field_value)})
        return rows, diagnostics

    rows.append({**base, "variant": variant, "k": None, "value": float(value),
                 "n_excluded": None if n_excluded is None else int(n_excluded), **ok})
    for field_name, field_value in diag.items():
        diagnostics.append({**base, "variant": variant, "k": None,
                            "diagnostic": field_name, "value": float(field_value)})
    return rows, diagnostics


def evaluate_layout(
    result,
    ctx: EvaluationContext,
    *,
    specs: Mapping[str, MetricSpec] = METRIC_REGISTRY,
    k=None,
    measure_memory: bool = True,
    distributions: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """Every per-case metric in ``specs`` on one layout (spec §3.7).

    Graph-side entries are skipped here — they are functions of the dataset alone and run once
    per dataset, in :func:`evaluate_graph_side`.

    Args:
        result: the :class:`~twin.results.LayoutResult` to score.
        ctx: the evaluation context **of the dataset ``result`` is a layout of**.
        specs: the metrics. Defaults to the whole registry.
        k: ``None`` for ``default_k`` (the headline), or explicit sizes for a sweep.
        measure_memory: sample RSS around the non-cheap metrics.
        distributions: if a dict, each distribution's raw array is stored in it keyed by
            ``(point_set, label_set, metric, variant)`` and a ``status="distribution"`` row is
            emitted beside its summaries. ``None`` — ``distribution_policy="off"`` — keeps the
            summaries only.

    Returns:
        ``(metric_rows, diagnostic_rows)``, without the case's identity columns.
    """
    check_policy(ctx)
    _check_same_dataset(result, ctx)
    ks = resolve_ks(ctx, k)
    label_sets, _ = scored_label_sets(ctx.dataset)
    providers = ProviderCache(ctx, result, measure_memory=measure_memory)

    rows: list[dict] = []
    diagnostics: list[dict] = []
    for name, spec in specs.items():
        if spec.graph_side:
            continue
        for variant in spec.variants:
            if "labels" in spec.needs and not label_sets:
                base = {"point_set": "V", "label_set": None, "metric": name}
                rows += _status_rows(
                    spec, variant, ks, base, "skipped", "the dataset declares no label sets"
                )
                continue
            for label_set in label_sets if "labels" in spec.needs else [None]:
                r, d = evaluate_spec(
                    name, spec, variant, ctx=ctx, result=result, ks=ks, label_set=label_set,
                    providers=providers, measure_memory=measure_memory,
                    distributions=distributions,
                )
                rows += r
                diagnostics += d
    return rows, diagnostics + providers.diagnostics


# --------------------------------------------------------------------------- #
#                   The graph-side table — once per dataset (§5.3 guard 2)     #
# --------------------------------------------------------------------------- #


def _null_rows(label_set: str, metric: str, null) -> list[dict]:
    r"""A :class:`~twin.metrics.reference.PermutationNull` as rows.

    ⚠️ A point-mass null has no permutation test — `tabula`'s `method` Acc\* null takes one
    value on every draw — so its ``z`` is a **skipped** row with the reason, not an ``ok`` NaN.
    """
    base = {"point_set": None, "label_set": label_set, "metric": metric, "k": None,
            "n_excluded": None}
    ok = {"status": "ok", "skip_reason": None}
    rows = [
        {**base, "variant": "mean", "value": null.mean, **ok},
        {**base, "variant": "sd", "value": null.sd, **ok},
        {**base, "variant": "n_permutations", "value": float(null.n_permutations), **ok},
    ]
    if null.degenerate:
        rows.append({**base, "variant": "z", "value": float("nan"), "status": "skipped",
                     "skip_reason": "the permutation null is a point mass; there is no test"})
    else:
        rows.append({**base, "variant": "z", "value": null.z, **ok})
    return rows


def reference_rows(ctx: EvaluationContext, label_sets: list[str]) -> list[dict]:
    r"""The reference levels Sil\* and Acc\* are read against, as rows of the graph-side table.

    Computed by :mod:`twin.metrics.reference` — never reimplemented here — and therefore over
    exactly the label sets `twin.report` scores, which the caller's enumeration must match.

    * ``silhouette_graph_null`` / ``graph_neighborhood_accuracy_null`` — ``mean``, ``sd``,
      ``z``, ``n_permutations``. The Acc\* null is of the **uniform** vote with the fractional
      tie rule, the headline of (41). ⚠️ $z$ answers "above chance", not "how strongly": it
      depends on class count and balance, so ranking label sets on it ranks class balance
      ((42)).
    * ``cramers_v_vs_primary`` — the confound column. Skipped, with the reason, for the
      primary itself and where no single primary is declared.
    * ``compactness_ceiling`` and ``compactness_ceiling_classes`` — per dataset, so
      ``label_set`` is empty.
    """
    from twin.metrics.reference import compactness_ceiling, score_label_sets

    scored = score_label_sets(ctx.dataset, ctx.ideal, ctx.affinity)
    if [row.label_set for row in scored] != list(label_sets):
        raise RuntimeError(
            f"the reference levels were computed over {[r.label_set for r in scored]} and the "
            f"metrics over {list(label_sets)}; the two enumerations have drifted apart"
        )

    rows: list[dict] = []
    for row in scored:
        rows += _null_rows(row.label_set, "silhouette_graph_null", row.silhouette_graph_null)
        rows += _null_rows(row.label_set, "graph_neighborhood_accuracy_null", row.graph_accuracy_null)
        base = {"point_set": None, "label_set": row.label_set, "metric": "cramers_v_vs_primary",
                "variant": None, "k": None, "n_excluded": None}
        if np.isfinite(row.cramers_v_vs_primary):
            rows.append({**base, "value": row.cramers_v_vs_primary, "status": "ok",
                         "skip_reason": None})
        else:
            reason = (
                "this is the scoring primary"
                if row.role == "primary"
                else "no single scoring primary is declared"
            )
            rows.append({**base, "value": float("nan"), "status": "skipped",
                         "skip_reason": reason})

    ceiling, classes = compactness_ceiling(ctx.affinity, ctx.ideal, ctx.dataset.n)
    base = {"point_set": None, "label_set": None, "variant": None, "k": None,
            "n_excluded": None, "status": "ok", "skip_reason": None}
    rows.append({**base, "metric": "compactness_ceiling", "value": ceiling})
    rows.append({**base, "metric": "compactness_ceiling_classes", "value": float(classes)})
    return rows


def evaluate_graph_side(
    ctx: EvaluationContext,
    *,
    specs: Mapping[str, MetricSpec] = METRIC_REGISTRY,
    reference_levels: bool = True,
) -> tuple[list[dict], list[dict]]:
    r"""Every graph-side metric over every scored label set of one dataset — **once**.

    Sil\*, Acc\* and their reference levels are functions of ``(dataset, label set, policy)``
    and of nothing in the layout. Emitted per case they would repeat one value per
    representation × embedding × seed — and at `tabula`'s 39 permutation draws across four label
    sets, cost ~145 s *per case* instead of once. So they get a table of their own, and §5.3
    guard 2 — "emit them in a separate table from their layout counterparts" — falls out of the
    structure rather than being a rule someone has to remember.

    Args:
        ctx: the dataset's evaluation context.
        specs: the registry; only its ``graph_side`` entries are run.
        reference_levels: also compute the permutation nulls, Cramér's V and the compactness
            ceiling. On by default because a bare Sil\* or Acc\* is unreadable (40); the cost
            is the nulls — ~145 s on `tabula`, seconds elsewhere.

    Returns:
        ``(rows, diagnostics)``. ``point_set`` is empty on every row: there is no layout.
    """
    check_policy(ctx)
    label_sets, _ = scored_label_sets(ctx.dataset)

    rows: list[dict] = []
    diagnostics: list[dict] = []
    for name, spec in specs.items():
        if not spec.graph_side:
            continue
        for variant in spec.variants:
            if "labels" in spec.needs and not label_sets:
                base = {"point_set": None, "label_set": None, "metric": name}
                rows += _status_rows(
                    spec, variant, (), base, "skipped", "the dataset declares no label sets"
                )
                continue
            for label_set in label_sets if "labels" in spec.needs else [None]:
                r, d = evaluate_spec(
                    name, spec, variant, ctx=ctx, result=None, ks=(), label_set=label_set,
                    measure_memory=False,
                )
                rows += r
                diagnostics += d

    if reference_levels and label_sets:
        rows += reference_rows(ctx, label_sets)
    return rows, diagnostics
