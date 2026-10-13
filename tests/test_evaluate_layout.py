"""`evaluate_layout` (spec §3.7) — the registry driven over one layout.

The strongest test here is the first: the registry, driven through its binders, reproduces every
per-case value in ``metric_golden.json`` to 1e-12. That is what proves the binders hand each
metric the right arguments — the signature conformance test in `test_metric_registry.py` checks
arity, and cannot see two arguments of the same type in the wrong order.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import twin.metrics.registry as registry
from twin.datasets import load_dataset
from twin.metrics.context import EvaluationContext
from twin.metrics.evaluate import (
    ProviderCache,
    evaluate_layout,
    resolve_ks,
    scored_label_sets,
)
from twin.metrics.registry import METRIC_REGISTRY, MetricSpec, metric_key
from twin.results import LayoutResult

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "metric_golden.json").read_text())

_spec = importlib.util.spec_from_file_location(
    "make_metric_golden", Path(__file__).parent / "fixtures" / "make_metric_golden.py"
)
generator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generator)


def _result(dataset, V, E=None, **kw):
    return LayoutResult(
        V=V,
        E=E,
        slopes=None,
        dataset=dataset,
        representation=kw.pop("representation", "vertex"),
        embedding=kw.pop("embedding", "fixture"),
        point_provenance={"V": "primary", "E": "derived"},
        **kw,
    )


@pytest.fixture(scope="module")
def karate():
    dataset = load_dataset("karate_weighted")
    ctx = EvaluationContext.for_dataset(dataset)
    V = generator.karate_structured_layout(dataset.graph)
    return ctx, _result(dataset, V, generator.karate_edge_points(dataset.graph, V))


@pytest.fixture(scope="module")
def golden_rows(karate):
    ctx, result = karate
    return evaluate_layout(result, ctx, k=tuple(GOLDEN["k"]))


# --------------------------------------------------------------------------- #
#                         The golden values, through the driver                #
# --------------------------------------------------------------------------- #


def test_the_registry_reproduces_every_per_case_golden_value(golden_rows):
    rows, _ = golden_rows
    by_key = {metric_key(r["metric"], r["variant"], r["k"]): r for r in rows}
    per_case = {
        key
        for key in GOLDEN["values"]
        if not METRIC_REGISTRY[key.split("@")[0]].graph_side
    }
    assert per_case <= set(by_key), sorted(per_case - set(by_key))
    for key in sorted(per_case):
        row = by_key[key]
        assert row["status"] == "ok", (key, row["skip_reason"])
        assert row["value"] == pytest.approx(GOLDEN["values"][key], rel=1e-12), key
    assert len(per_case) == 39  # the 44 less the five graph-side keys: Sil*, Acc* x3, the floor


def test_no_row_errored_on_the_golden_case(golden_rows):
    rows, _ = golden_rows
    bad = [(r["metric"], r["variant"], r["skip_reason"]) for r in rows if r["status"] != "ok"]
    assert bad == []


def test_graph_side_metrics_are_not_emitted_per_case(golden_rows):
    """§5.3 guard 2: they are functions of the dataset, and a per-case row would repeat them."""
    rows, _ = golden_rows
    graph_side = {n for n, s in METRIC_REGISTRY.items() if s.graph_side}
    assert not {r["metric"] for r in rows} & graph_side


def test_rows_carry_the_report_diagnostics(golden_rows):
    """(18): counts with their own denominators go to the diagnostics, not the value column."""
    rows, diagnostics = golden_rows
    names = {(d["metric"], d["diagnostic"]) for d in diagnostics}
    assert ("knn_accuracy", "majority_class_rate") in names  # (40): mandatory beside Acc
    assert ("knn_accuracy", "n_ties") in names
    assert ("knn_recall", "n_scored") in names
    assert ("crossing_pairs", "n_near_tangential") in names
    assert ("silhouette", "n_classes") in names
    assert ("node_resolution_vertexwise", "n_zero_denominator") in names
    # `n_excluded` is the metric's *own* exclusions and has a column of its own.
    assert not any(d["diagnostic"] == "n_excluded" for d in diagnostics)
    recall = [r for r in rows if r["metric"] == "knn_recall"]
    assert [r["n_excluded"] for r in recall] == [0, 0]


def test_every_call_is_timed(golden_rows):
    rows, diagnostics = golden_rows
    timed = {(d["metric"], d["variant"], d["label_set"]) for d in diagnostics
             if d["diagnostic"] == "elapsed_s"}
    called = {(r["metric"], r["variant"], r["label_set"]) for r in rows
              if not METRIC_REGISTRY[r["metric"]].reductions}
    assert called <= timed


def test_memory_is_sampled_only_above_the_cheap_class(golden_rows):
    _, diagnostics = golden_rows
    sampled = {d["metric"] for d in diagnostics if d["diagnostic"] == "rss_delta_mb"}
    cheap = {n for n, s in METRIC_REGISTRY.items() if s.cost == "cheap"}
    assert sampled and not sampled & cheap
    assert "crossing_pairs" in sampled


# --------------------------------------------------------------------------- #
#                           One solve feeds three metrics                      #
# --------------------------------------------------------------------------- #


def test_the_crossing_solve_runs_once_per_variant_for_three_metrics(karate, monkeypatch):
    ctx, result = karate
    calls = []
    real = registry.crossing_pairs_report

    def counting(*args, **kwargs):
        calls.append(args[2] is None)
        return real(*args, **kwargs)

    monkeypatch.setattr(registry, "crossing_pairs_report", counting)
    evaluate_layout(result, ctx)
    assert sorted(calls) == [False, True], "one solve per variant: curved, straight"


def test_a_failing_provider_fails_its_consumers_once(karate, monkeypatch):
    ctx, result = karate
    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("solver blew up")

    monkeypatch.setattr(registry, "crossing_pairs_report", broken)
    rows, _ = evaluate_layout(result, ctx)
    crossing = [r for r in rows if "crossing_pairs" in METRIC_REGISTRY[r["metric"]].needs]
    assert len(crossing) == 6 and all(r["status"] == "error" for r in crossing)
    assert all("solver blew up" in r["skip_reason"] for r in crossing)
    assert len(calls) == 2, "a failed solve is remembered, not retried per consumer"
    # Everything else still ran.
    assert any(r["metric"] == "adjacency_distance_ratio" and r["status"] == "ok" for r in rows)


def test_the_provider_cache_counts_its_calls(karate):
    ctx, result = karate
    cache = ProviderCache(ctx, result, measure_memory=False)
    params = METRIC_REGISTRY["crossing_number"].variant_params("straight")
    first = cache.get("crossing_pairs", "straight", params)
    assert cache.get("crossing_pairs", "straight", params) is first
    assert cache.calls[("crossing_pairs", "straight")] == 1


# --------------------------------------------------------------------------- #
#                         Skipped and errored, never absent                    #
# --------------------------------------------------------------------------- #


def test_a_dataset_with_no_labels_skips_the_label_metrics_with_a_reason():
    dataset = load_dataset("karate")
    assert not dataset.label_sets
    ctx = EvaluationContext.for_dataset(dataset)
    V = np.random.default_rng(0).normal(size=(dataset.n, 2))
    rows, _ = evaluate_layout(_result(dataset, V), ctx, measure_memory=False)

    label = [r for r in rows if r["metric"] in ("silhouette", "knn_accuracy")]
    assert label, "a skipped metric still emits the rows it would have produced"
    assert {r["status"] for r in label} == {"skipped"}
    assert all("no label sets" in r["skip_reason"] for r in label)
    assert all(np.isnan(r["value"]) for r in label)


def test_a_raising_metric_is_an_error_row_and_does_not_abort():
    dataset = load_dataset("karate_weighted")
    ctx = EvaluationContext.for_dataset(dataset)

    def explode(X):
        raise FloatingPointError("deliberate")

    specs = {
        "explode": MetricSpec(fn=explode, bind=lambda c, r, v, p: ((r.V,), {}), kind="visual"),
        "node_resolution": METRIC_REGISTRY["node_resolution"],
    }
    V = np.random.default_rng(0).normal(size=(dataset.n, 2))
    rows, _ = evaluate_layout(_result(dataset, V), ctx, specs=specs, measure_memory=False)
    assert [(r["metric"], r["status"]) for r in rows] == [
        ("explode", "error"),
        ("node_resolution", "ok"),
    ]
    assert rows[0]["skip_reason"] == "FloatingPointError: deliberate"


def test_the_curved_variant_is_skipped_without_edge_points(karate):
    ctx, result = karate
    no_edges = _result(result.dataset, result.V, None)
    rows, _ = evaluate_layout(no_edges, ctx, measure_memory=False)
    curved = [r for r in rows if (r["variant"] or "").startswith("curved")]
    assert curved and {r["status"] for r in curved} == {"skipped"}
    straight = [r for r in rows if (r["variant"] or "").startswith("straight")]
    assert straight and {r["status"] for r in straight} == {"ok"}


def test_a_three_dimensional_layout_skips_the_planar_metrics():
    dataset = load_dataset("karate_weighted")
    ctx = EvaluationContext.for_dataset(dataset)
    V = np.random.default_rng(0).normal(size=(dataset.n, 3))
    rows, _ = evaluate_layout(_result(dataset, V), ctx, measure_memory=False)
    crossing = [r for r in rows if METRIC_REGISTRY[r["metric"]].kind == "crossing"]
    assert {r["status"] for r in crossing} == {"skipped"}
    assert all("planar" in r["skip_reason"] for r in crossing)
    assert all(r["status"] == "ok" for r in rows if r["metric"] == "adjacency_distance_ratio")


def test_an_excluded_entry_of_a_distribution_is_counted_not_propagated(karate):
    """Coincident endpoints give a `nan` detour ratio; the summaries exclude it and say so."""
    ctx, result = karate
    V = result.V.copy()
    u, v = next(iter(result.G.edges()))
    V[v] = V[u]
    rows, _ = evaluate_layout(
        _result(result.dataset, V), ctx,
        specs={"detour_ratio": METRIC_REGISTRY["detour_ratio"]}, measure_memory=False,
    )
    straight = {r["variant"]: r for r in rows if r["variant"].startswith("straight")}
    assert straight["straight_mean"]["value"] == pytest.approx(1.0)
    assert straight["straight_mean"]["n_excluded"] >= 1


# --------------------------------------------------------------------------- #
#                         Policy, identity and enumeration                     #
# --------------------------------------------------------------------------- #


def test_a_layout_is_refused_against_another_datasets_context(karate):
    _, result = karate
    other = EvaluationContext.for_dataset(load_dataset("karate"))
    with pytest.raises(ValueError, match="must be scored against the dataset"):
        evaluate_layout(result, other)


def test_a_rehydrated_dataset_is_accepted_by_name_and_params(karate):
    ctx, result = karate
    rebuilt = _result(load_dataset("karate_weighted"), result.V, result.E)
    assert rebuilt.dataset is not ctx.dataset
    rows, _ = evaluate_layout(
        rebuilt, ctx, specs={"node_resolution": METRIC_REGISTRY["node_resolution"]}
    )
    assert rows[0]["status"] == "ok"


def test_only_the_lcc_policy_is_accepted(karate):
    _, result = karate
    masked = EvaluationContext.for_dataset(result.dataset, disconnected="mask")
    with pytest.raises(ValueError, match="'lcc'"):
        evaluate_layout(result, masked)


def test_k_defaults_to_the_mean_degree(karate):
    ctx, _ = karate
    assert resolve_ks(ctx) == (5,)
    assert resolve_ks(ctx, 7) == (7,)
    assert resolve_ks(ctx, [5, 10]) == (5, 10)
    with pytest.raises(ValueError):
        resolve_ks(ctx, 0)


def test_the_degenerate_composite_is_dropped_and_reported():
    """tabula's `compartment_cell_type` *is* `cell_ontology_class` renumbered."""
    names, dropped = scored_label_sets(load_dataset("tabula"))
    assert "compartment_cell_type" not in names
    assert dropped == {"compartment_cell_type": "cell_ontology_class"}
    assert names[0] == "cell_ontology_class"  # the primary first


def test_a_genuine_composite_is_kept():
    """football's `division` does not refine `conference`; its composite is a real 15 classes."""
    names, dropped = scored_label_sets(load_dataset("football_2023"))
    assert names == ["conference", "division", "conference_division"]
    assert dropped == {}


def test_label_metrics_run_once_per_label_set():
    dataset = load_dataset("football_2023")
    ctx = EvaluationContext.for_dataset(dataset)
    V = np.random.default_rng(0).normal(size=(dataset.n, 2))
    rows, _ = evaluate_layout(
        _result(dataset, V), ctx, specs={"silhouette": METRIC_REGISTRY["silhouette"]}
    )
    assert [r["label_set"] for r in rows] == ["conference", "division", "conference_division"]


# --------------------------------------------------------------------------- #
#                         Which graph the per-edge metrics get                 #
# --------------------------------------------------------------------------- #


def test_the_per_edge_metrics_are_handed_the_datasets_own_graph(karate):
    """`result.E` is in `dataset.graph.edges()` order; the context's graph is not."""
    ctx, result = karate
    for name, spec in METRIC_REGISTRY.items():
        if spec.kind != "crossing" and name not in ("edge_lengths", "edge_length_cv"):
            continue
        params = spec.variant_params(next(iter(spec.variants)))
        params["crossing_pairs"] = registry.crossing_pairs_report(result.G, result.V)
        args, _ = spec.bind(ctx, result, None, params)
        assert any(a is result.G for a in args), name


def test_the_contexts_edge_order_is_not_the_datasets_on_digits():
    """Why the rule above exists. On karate the two orders happen to agree, so this is checked
    on digits, where they do not — a test on the small graphs would prove nothing."""
    dataset = load_dataset("digits")
    ctx = EvaluationContext.for_dataset(dataset)
    ours = [tuple(sorted(e)) for e in dataset.graph.edges()]
    theirs = [tuple(sorted(e)) for e in ctx.affinity.edges()]
    assert sorted(ours) == sorted(theirs)
    assert sum(a != b for a, b in zip(ours, theirs)) > 10000


def test_distributions_are_kept_only_when_asked(karate):
    ctx, result = karate
    specs = {"edge_lengths": METRIC_REGISTRY["edge_lengths"]}
    off, _ = evaluate_layout(result, ctx, specs=specs)
    assert [r["status"] for r in off] == ["ok"]

    kept: dict = {}
    on, _ = evaluate_layout(result, ctx, specs=specs, distributions=kept)
    assert [r["status"] for r in on] == ["distribution", "ok"]
    (array,) = kept.values()
    assert array.shape == (result.m,)
    assert on[1]["value"] == pytest.approx(array.mean())
