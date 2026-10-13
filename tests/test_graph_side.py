r"""The graph-side table (§5.3 guard 2) — Sil\*, Acc\* and their reference levels, once per dataset.

These are functions of ``(dataset, label set)`` and nothing in the layout, so they live in a
table of their own. The values come from two places — the registry's graph-side entries and
`twin.metrics.reference` — and the tests below are what keep those two from drifting apart.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from twin.datasets import load_dataset
from twin.metrics.context import EvaluationContext
from twin.metrics.evaluate import _null_rows, evaluate_graph_side
from twin.metrics.reference import PermutationNull, score_label_sets
from twin.metrics.registry import METRIC_REGISTRY, metric_key
from twin.runs.keys import GRAPH_SIDE_ROW_SCHEMA, dataset_identity, params_from_json

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "metric_golden.json").read_text())


@pytest.fixture(scope="module")
def football():
    ctx = EvaluationContext.for_dataset(load_dataset("football_2023"))
    rows, diagnostics = evaluate_graph_side(ctx)
    return ctx, rows, diagnostics


def _by(rows, metric, label_set=None, variant=None):
    (row,) = [
        r for r in rows
        if r["metric"] == metric and r["label_set"] == label_set and r["variant"] == variant
    ]
    return row


def test_the_graph_side_golden_values_reproduce():
    ctx = EvaluationContext.for_dataset(load_dataset("karate_weighted"))
    rows, _ = evaluate_graph_side(ctx, reference_levels=False)
    by_key = {metric_key(r["metric"], r["variant"]): r["value"] for r in rows}
    graph_side = {
        key for key in GOLDEN["values"] if METRIC_REGISTRY[key.split("@")[0]].graph_side
    }
    assert len(graph_side) == 5
    for key in graph_side:
        assert by_key[key] == pytest.approx(GOLDEN["values"][key], rel=1e-12), key


def test_the_registry_and_the_report_agree(football):
    """Two code paths compute Sil* and Acc*; if they diverge, a table and the dataset report
    would print different numbers for the same quantity."""
    ctx, rows, _ = football
    for scored in score_label_sets(ctx.dataset, ctx.ideal, ctx.affinity):
        ls = scored.label_set
        assert _by(rows, "silhouette_graph", ls)["value"] == scored.silhouette_graph
        assert _by(rows, "graph_neighborhood_accuracy", ls, "uniform")["value"] == (
            scored.graph_accuracy
        )
        assert _by(rows, "graph_neighborhood_accuracy", ls, "affinity")["value"] == (
            scored.graph_accuracy_affinity
        )
        assert _by(rows, "majority_class_rate", ls)["value"] == scored.majority_class_rate


def test_every_label_set_is_scored_once(football):
    _, rows, _ = football
    sil = [r["label_set"] for r in rows if r["metric"] == "silhouette_graph"]
    assert sil == ["conference", "division", "conference_division"]


def test_the_reference_levels_travel_with_the_scores(football):
    """(40): a bare Sil* or Acc* is unreadable. Pinned against `test_report_numbers.py`."""
    _, rows, _ = football
    assert _by(rows, "compactness_ceiling")["value"] == pytest.approx(0.3433, abs=5e-5)
    for ls in ("conference", "division", "conference_division"):
        for metric in ("silhouette_graph_null", "graph_neighborhood_accuracy_null"):
            variants = {r["variant"] for r in rows if r["metric"] == metric and r["label_set"] == ls}
            assert variants == {"mean", "sd", "z", "n_permutations"}
        assert _by(rows, "silhouette_graph_null", ls, "n_permutations")["value"] == 1000
    # Sil*'s chance level is negative by a class-count-dependent amount.
    assert _by(rows, "silhouette_graph_null", "conference", "mean")["value"] < 0


def test_cramers_v_is_skipped_for_the_primary_with_a_reason(football):
    _, rows, _ = football
    primary = _by(rows, "cramers_v_vs_primary", "conference")
    assert primary["status"] == "skipped" and "primary" in primary["skip_reason"]
    assert _by(rows, "cramers_v_vs_primary", "division")["status"] == "ok"


def test_a_point_mass_null_has_no_z_and_says_so():
    """tabula's `method` Acc* null takes one value on every draw; `np.std` returns 2.3e-16."""
    null = PermutationNull.from_samples(0.9868, np.full(39, 0.957459556620731))
    rows = {r["variant"]: r for r in _null_rows("method", "graph_neighborhood_accuracy_null", null)}
    assert rows["z"]["status"] == "skipped"
    assert "point mass" in rows["z"]["skip_reason"]
    assert rows["sd"]["value"] == 0.0 and rows["sd"]["status"] == "ok"


def test_a_dataset_with_no_labels_emits_skipped_rows_and_no_references():
    ctx = EvaluationContext.for_dataset(load_dataset("karate"))
    rows, _ = evaluate_graph_side(ctx)
    assert {r["status"] for r in rows} == {"skipped"}
    assert {r["metric"] for r in rows} == {
        "silhouette_graph", "graph_neighborhood_accuracy", "majority_class_rate"
    }
    assert not ctx.ideal_is_computed, "nothing to score, so no closure"


def test_tabulas_degenerate_composite_never_appears():
    """§5.3 guard 2's sibling: one row per label set, and the duplicate is not one of them."""
    ctx = EvaluationContext.for_dataset(load_dataset("tabula"))
    rows, _ = evaluate_graph_side(
        ctx, specs={"majority_class_rate": METRIC_REGISTRY["majority_class_rate"]},
        reference_levels=False,
    )
    assert [r["label_set"] for r in rows] == [
        "cell_ontology_class", "compartment", "donor", "method"
    ]


def test_graph_side_rows_have_no_point_set(football):
    _, rows, _ = football
    assert {r["point_set"] for r in rows} == {None}


def test_the_rows_fit_the_graph_side_schema(football, tmp_path):
    import pyarrow as pa

    ctx, rows, _ = football
    identity = dataset_identity(ctx.dataset.name, ctx.dataset.params)
    table = pa.Table.from_pylist(
        [
            {**identity, **{k: v for k, v in r.items() if k not in ("point_set", "k")},
             "metric_set_digest": "x"}
            for r in rows
        ],
        schema=GRAPH_SIDE_ROW_SCHEMA,
    )
    assert table.num_rows == len(rows)
    assert params_from_json(identity["graph_params"]) == ctx.dataset.params
