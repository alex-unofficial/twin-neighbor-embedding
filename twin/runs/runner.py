r"""`batch_layout_metrics` — the batch grammar driven through layouts and the metric registry.

**It drops one level below** :func:`~twin.batching.run_batch` — to
:func:`~twin.batching.expand_batch` and :func:`~twin.batching.make_batch_key` — **and that is
using `batching.py`, not rewriting it** (CLAUDE.md). ``run_batch_cases`` computes the key inside
its loop and immediately calls the runner, so there is no point at which to group cases by
dataset or (from Half B on) to skip a persisted one; and ``batch_graph_embedding`` accumulates
every ``LayoutResult`` in a dict, which is the opposite of what an overnight batch wants. Two
things ``run_batch_cases`` did are lost by that, and are re-implemented here deliberately:

* **the duplicate check** — on the case's *id* rather than its ``BatchKey``. Fixed kwargs are
  not in the key, so `tabula`/liver and `tabula`/lung share a ``BatchKey`` and are still two
  cases; ``run_batch_cases`` would have refused the pair as duplicates;
* **progress reporting**.

⚠️ **Cases run dataset-major.** ``expand_batch`` concatenates groups, and ``batches/batch.py``
has two groups per block over one ``common.graph`` list — so in spec order a one-dataset cache
would build, release and rebuild `tabula`'s 14.9 s closure. Results are order-independent, so a
stable sort by the graph's identity is free. The ``Dataset`` is memoised with its context:
``resolve_object_choice`` calls the registry factory per case, so without it every case would
reload `tabula`'s ``.mat``.

⚠️ **The cache unit is (case, point set), and caches are cleared with the context.** Do not
raise ``PAIRWISE_CACHE_MAXSIZE`` to make a slow batch faster — its docstring predicts the 12 GB
footgun; the fix for a thrashing cache is the loop order, which is this module's job. Budget
about 1 GB for one `tabula` case: the square ideal (200 MB), its condensed form (100 MB), two
pairwise entries (200 MB) and the crossing solve's +428 MB transient.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import pandas as pd
import pyarrow as pa

from twin.batching import (
    expand_batch,
    key_to_flat_record,
    make_batch_key,
    resolve_object_choice,
)
from twin.metrics.common import clear_loo_cache, clear_pairwise_cache
from twin.metrics.context import EvaluationContext
from twin.metrics.evaluate import (
    check_policy,
    evaluate_graph_side,
    evaluate_layout,
    failed_case_rows,
    resolve_ks,
    scored_label_sets,
)
from twin.metrics.registry import METRIC_REGISTRY, MetricSpec, metric_set_digest
from twin.runs.keys import (
    DIAGNOSTIC_ROW_SCHEMA,
    GRAPH_SIDE_ROW_SCHEMA,
    METRIC_ROW_SCHEMA,
    _canonical_scalar,
    case_identity,
    dataset_identity,
    decode_key_json,
    fixed_extra,
)

__all__ = ["AXES", "MetricTables", "batch_layout_metrics", "plan_cases", "to_frame"]

#: The key's axis order. Fixed, because the order is part of the digest: the same case typed with
#: its axes in a different order must not get a second id.
AXES = ("graph", "representation", "embedding")


@dataclasses.dataclass
class MetricTables:
    """The three tables one run produces.

    Attributes:
        metrics: the tidy long frame of §3.7 — one row per (case, point set, label set, metric,
            variant, k) — with the per-axis parameter columns materialised.
        diagnostics: (18)'s separate table: counts with their own denominators, and cost.
        graph_side: §5.3 guard 2's separate table — Sil\\*, Acc\\* and their reference levels,
            one row per (dataset, label set, metric, variant), never per case.
        dropped_composites: §5.3 guard 3 — ``{dataset: {composite: the part it collapses to}}``.
        k_per_dataset: the ``k`` actually used per dataset; the rows alone cannot say whether it
            was the default.
        metric_set_digest: what every row was computed under.
    """

    metrics: pd.DataFrame
    diagnostics: pd.DataFrame
    graph_side: pd.DataFrame
    dropped_composites: dict[str, dict[str, str]]
    k_per_dataset: dict[str, list[int]]
    metric_set_digest: str


@dataclasses.dataclass
class PlannedCase:
    case: dict
    key: tuple
    extra: dict
    identity: dict[str, str]

    @property
    def graph_identity(self) -> str:
        """A string identifying the dataset this case is built from: name plus its kwargs."""
        choice = self.case["graph"]
        kwargs = {k: _canonical_scalar(v) for k, v in sorted(choice["kwargs"].items())}
        return json.dumps([choice["name"], kwargs], sort_keys=True)


def plan_cases(
    batch_input,
    registries: Mapping[str, Mapping[str, Any]],
    *,
    duplicate_policy: str = "error",
    **runner_scalars: Any,
) -> list[PlannedCase]:
    """Expand, key, de-duplicate and order the cases — everything before any work is done.

    Args:
        duplicate_policy: ``"error"`` (the default, and the safe one) raises on two cases with
            one id; ``"skip"`` keeps the first. ``"overwrite"`` has no meaning without a store
            and is not offered.
        **runner_scalars: the arguments every case shares, which go into each case's ``extra``.
    """
    if duplicate_policy not in ("error", "skip"):
        raise ValueError(f"duplicate_policy must be 'error' or 'skip', got {duplicate_policy!r}")

    planned: list[PlannedCase] = []
    seen: set[str] = set()
    for case in expand_batch(batch_input, registries):
        missing = set(AXES) - set(case)
        if missing:
            raise KeyError(f"batch case is missing required axes: {sorted(missing)}")
        key = make_batch_key(case, axis_order=list(AXES))
        extra = fixed_extra(case, **runner_scalars)
        identity = case_identity(key, extra)
        if identity["key_id"] in seen:
            if duplicate_policy == "error":
                raise ValueError(f"two cases share the id {identity['key_id']}")
            continue
        seen.add(identity["key_id"])
        planned.append(PlannedCase(case=case, key=key, extra=extra, identity=identity))

    # Stable: within one dataset, cases keep the order the spec gave them.
    return sorted(planned, key=lambda p: p.graph_identity)


def _stamp(rows: Iterable[dict], base: dict) -> list[dict]:
    return [{**base, **row} for row in rows]


def batch_layout_metrics(
    batch_input,
    *,
    registries: Mapping[str, Mapping[str, Any]] | None = None,
    specs: Mapping[str, MetricSpec] = METRIC_REGISTRY,
    k=None,
    seed: int = 0,
    d: int = 2,
    normalize: bool = False,
    measure_memory: bool = True,
    reference_levels: bool = True,
    disconnected: str = "lcc",
    duplicate_policy: str = "error",
    progress: bool = False,
    progress_fn: Callable[[str], None] = print,
) -> MetricTables:
    """Lay out every case of ``batch_input`` and score it with ``specs`` (spec §3.7).

    Args:
        batch_input: a batch spec in the grammar of :mod:`twin.batching`.
        registries: the axis registries; ``twin.pipelines.batch_registries`` by default.
        specs: the metrics. Defaults to the whole registry.
        k: ``None`` for each dataset's ``default_k`` — the headline, and the only value §4.2's
            λ criterion is defined at — or explicit sizes for a robustness sweep.
        seed, d, normalize: as for :func:`~twin.pipelines.graph_embedding`, and part of every
            case's id.
        measure_memory: sample RSS during each embed and around each non-cheap metric.
        reference_levels: compute the graph-side permutation nulls, Cramér's V and the
            compactness ceiling.
        disconnected: only ``"lcc"`` is accepted (plan decision 3).
        duplicate_policy: see :func:`plan_cases`.
        progress, progress_fn: report one line per case.

    Returns:
        :class:`MetricTables`.
    """
    from twin.pipelines import batch_registries, graph_embedding

    registries = batch_registries if registries is None else registries
    planned = plan_cases(
        batch_input, registries, duplicate_policy=duplicate_policy,
        seed=seed, d=d, normalize=normalize, common_init=False,
    )
    digest = metric_set_digest(specs, k=None if k is None else tuple(resolve_ks(None, k)))

    metric_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    graph_side_rows: list[dict] = []
    dropped: dict[str, dict[str, str]] = {}
    k_used: dict[str, list[int]] = {}

    total = len(planned)
    if progress:
        progress_fn(f"{total} cases")

    ctx: EvaluationContext | None = None
    current: str | None = None
    for i, p in enumerate(planned, start=1):
        if p.graph_identity != current:
            if ctx is not None:
                _release(ctx)
            dataset = resolve_object_choice(p.case["graph"])
            ctx = EvaluationContext.for_dataset(dataset, disconnected=disconnected)
            check_policy(ctx)
            current = p.graph_identity

            label = _dataset_label(dataset)
            k_used[label] = list(resolve_ks(ctx, k))
            _, dropped_here = scored_label_sets(dataset)
            if dropped_here:
                dropped[label] = dict(dropped_here)
            rows, _ = evaluate_graph_side(ctx, specs=specs, reference_levels=reference_levels)
            identity = dataset_identity(dataset.name, dataset.params)
            graph_side_rows += [
                {**identity, **_without(row, "point_set", "k"), "metric_set_digest": digest}
                for row in rows
            ]

        base = {
            **p.identity,
            **{axis: p.case[axis]["name"] for axis in AXES},
            "metric_set_digest": digest,
        }
        t0 = time.perf_counter()
        try:
            result = graph_embedding(
                ctx.dataset,
                representation=p.case["representation"]["object"],
                embedding=p.case["embedding"]["object"],
                seed=seed,
                d=d,
                normalize=normalize,
                representation_kw=p.case["representation"]["kwargs"],
                embedding_kw=p.case["embedding"]["kwargs"],
                representation_name=p.case["representation"]["name"],
                embedding_name=p.case["embedding"]["name"],
                measure_memory=measure_memory,
            )
        except Exception as exc:  # noqa: BLE001 — one failed embed must not end the batch
            reason = f"embed failed: {type(exc).__name__}: {exc}"
            metric_rows += _stamp(failed_case_rows(ctx, specs=specs, k=k, reason=reason), base)
            if progress:
                progress_fn(f"case {i}/{total} FAILED {p.identity['key_id']}: {reason}")
            continue

        rows, diagnostics = evaluate_layout(
            result, ctx, specs=specs, k=k, measure_memory=measure_memory
        )
        metric_rows += _stamp(rows, base)
        diagnostic_rows += _stamp(diagnostics + _embed_diagnostics(result), base)
        if progress:
            progress_fn(
                f"case {i}/{total} ({100 * i / total:.0f}%) {p.identity['key_id']} "
                f"{time.perf_counter() - t0:.1f}s"
            )

    if ctx is not None:
        _release(ctx)

    return MetricTables(
        metrics=to_frame(metric_rows, METRIC_ROW_SCHEMA),
        diagnostics=to_frame(diagnostic_rows, DIAGNOSTIC_ROW_SCHEMA),
        graph_side=to_frame(graph_side_rows, GRAPH_SIDE_ROW_SCHEMA, materialise=False),
        dropped_composites=dropped,
        k_per_dataset=k_used,
        metric_set_digest=digest,
    )


def _dataset_label(dataset) -> str:
    return dataset.name if not dataset.params else f"{dataset.name}{dataset.params}"


def _without(row: dict, *names: str) -> dict:
    return {k: v for k, v in row.items() if k not in names}


def _release(ctx: EvaluationContext) -> None:
    """Drop a dataset's closure and every content-keyed cache entry made from its layouts."""
    ctx.release()
    clear_pairwise_cache()
    clear_loo_cache()


def _embed_diagnostics(result) -> list[dict]:
    """The embed's own cost, as diagnostics rows of metric ``"embed"``."""
    base = {"point_set": None, "label_set": None, "metric": "embed", "variant": None, "k": None}
    values = {
        "runtime_s": result.runtime_s,
        "representation_s": result.representation_s,
        "n_points": result.n_points,
        "peak_mem_mb": result.peak_mem_mb,
    }
    return [
        {**base, "diagnostic": name, "value": float(value)}
        for name, value in values.items()
        if value is not None
    ]


# --------------------------------------------------------------------------- #
#                                 Frames                                       #
# --------------------------------------------------------------------------- #


def _nullable(pa_type):
    """pandas' nullable dtypes for the integer and string columns, so `k` stays an integer."""
    if pa.types.is_integer(pa_type):
        return pd.Int64Dtype()
    return None


def to_frame(rows: list[dict], schema: pa.Schema, *, materialise: bool = True) -> pd.DataFrame:
    """Rows to a typed frame, with each case's parameters as columns.

    The per-axis parameter columns are materialised **here**, from ``key_json``, rather than
    stored in a shard: ``key_to_flat_record`` gives ``embedding.linlog`` only to the
    `forceatlas2` cases, so they are ragged across cases and only the concatenation knows the
    full union (§3.7: use ``key_to_flat_record``; do not hand-roll the flattening). The
    ``extra`` — fixed kwargs and the runner's scalars — is materialised beside them, which is
    how ``seed`` reaches a column although it is not a batch axis.
    """
    table = pa.Table.from_pylist(rows, schema=schema)
    frame = table.to_pandas(types_mapper=_nullable)
    if not materialise or frame.empty:
        return frame

    records = {}
    for key_json in frame["key_json"].unique():
        key, extra = decode_key_json(key_json)
        record = key_to_flat_record(key)
        for axis in AXES:
            record.pop(axis, None)
        record.update(extra)
        records[key_json] = record
    params = pd.DataFrame.from_dict(records, orient="index")
    params.index.name = "key_json"
    return frame.merge(params, left_on="key_json", right_index=True, how="left")
