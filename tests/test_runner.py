"""`batch_layout_metrics` — the batch grammar through layouts and the registry, into tables.

The row counts below are computed **by hand** from the registry's declarations, not by calling
`declared_rows`: a test that predicted the count with the function that produced it would pass
against any bug the two shared.

Every test writes under ``tmp_path``; none touches the repository's ``runs/``.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from twin.datasets import DATASET_REGISTRY
from twin.embedding import EMBEDDING_REGISTRY, GraphEmbeddingMethod
from twin.layoutgraph import REPRESENTATION_REGISTRY
from twin.metrics.registry import METRIC_REGISTRY
from twin.runs import runner as runner_module
from twin.runs.runner import batch_layout_metrics, plan_cases

_spec = importlib.util.spec_from_file_location(
    "smoke", Path(__file__).parents[1] / "batches" / "smoke.py"
)
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)

#: Rows one case emits per label set it scores, read off the registry declarations by hand:
#: ADR 1, the four kNN criteria 4, three stresses 3, NND 1, NR 1, VNR 3 variants, two
#: distributions x 3 reductions, edge_lengths 1 reduction, edge_length_cv 1, three crossing
#: metrics x 2 variants, detour 2 variants x 2 reductions — and per label set, knn_accuracy's
#: 2 variants and silhouette's 1. At the default k, one value of k.
UNLABELLED_ROWS = 1 + 4 + 3 + 1 + 1 + 3 + 6 + 1 + 1 + 6 + 4
PER_LABEL_SET_ROWS = 2 + 1

#: Graph-side rows per label set: Sil* 1, Acc* 3 variants, the floor 1; then the reference
#: levels: two nulls x 4 statistics, Cramér's V 1. Plus the ceiling's 2 rows per dataset.
GRAPH_SIDE_PER_LABEL_SET = 1 + 3 + 1 + 8 + 1
GRAPH_SIDE_PER_DATASET = 2

LABEL_SETS = {"karate_weighted": 1, "football_2023": 3}

BATCH = [
    {
        "graph": ["karate_weighted", "football_2023"],
        "representation": ["vertex", "twin"],
        "embedding": ["spectral"],
    }
]


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke")
    smoke.run(out, batch=BATCH, progress_fn=lambda _: None)
    return {
        name: pd.read_parquet(out / f"{name}.parquet")
        for name in ("metrics", "diagnostics", "graph_side")
    }


def test_the_registry_count_is_the_hand_count():
    """If this fails the registry changed and the hand count above must be redone, not the
    other way round."""
    assert sum(1 for s in METRIC_REGISTRY.values() if not s.graph_side) == 21


def test_the_metric_row_count_is_predicted(tables):
    cases_per_dataset = 2  # two representations x one embedding
    expected = sum(
        cases_per_dataset * (UNLABELLED_ROWS + PER_LABEL_SET_ROWS * n)
        for n in LABEL_SETS.values()
    )
    assert len(tables["metrics"]) == expected == 148


def test_the_graph_side_row_count_is_predicted(tables):
    expected = sum(
        GRAPH_SIDE_PER_LABEL_SET * n + GRAPH_SIDE_PER_DATASET for n in LABEL_SETS.values()
    )
    assert len(tables["graph_side"]) == expected == 60


def test_every_row_is_ok_on_the_smoke_batch(tables):
    metrics = tables["metrics"]
    bad = metrics[metrics.status != "ok"]
    assert bad.empty, bad[["metric", "variant", "skip_reason"]].to_string()


def test_sil_star_appears_once_per_dataset_and_label_set_never_per_case(tables):
    graph_side = tables["graph_side"]
    sil = graph_side[graph_side.metric == "silhouette_graph"]
    assert sil.groupby(["graph", "label_set"]).size().eq(1).all()
    assert len(sil) == sum(LABEL_SETS.values())
    assert not set(tables["metrics"].metric) & {
        "silhouette_graph", "graph_neighborhood_accuracy", "majority_class_rate"
    }


def test_the_parquet_round_trip_keeps_the_types(tables):
    metrics = tables["metrics"]
    assert str(metrics["k"].dtype) == "Int64", "k must stay an integer column with nulls"
    assert str(metrics["n_excluded"].dtype) == "Int64"
    assert metrics["value"].dtype == np.float64
    # The runner's scalars are materialised from `extra`, with their types.
    assert set(metrics["seed"]) == {0}
    assert metrics["normalize"].dtype == bool


def test_each_case_has_one_id_and_every_row_carries_it(tables):
    metrics = tables["metrics"]
    per_case = metrics.groupby(["graph", "representation", "embedding"])["key_id"].nunique()
    assert per_case.eq(1).all()
    assert metrics["key_id"].nunique() == 4
    assert metrics["metric_set_digest"].nunique() == 1


def test_the_embed_cost_is_a_diagnostic(tables):
    diagnostics = tables["diagnostics"]
    embed = diagnostics[diagnostics.metric == "embed"]
    assert set(embed.diagnostic) >= {"runtime_s", "representation_s", "n_points"}
    n_points = embed[embed.diagnostic == "n_points"].pivot(
        index="graph", columns="representation", values="value"
    )
    assert (n_points["twin"] > n_points["vertex"]).all(), "A_tne lays out n + m points"


def test_the_parameter_columns_are_materialised_from_the_key():
    """A swept parameter reaches a column of its own; a fixed one does too, via `extra`."""
    batch = [
        {
            "graph": ["karate_weighted"],
            "representation": ["vertex"],
            "embedding": [("spring", {"iterations": [10, 20]})],
        }
    ]
    tables = batch_layout_metrics(
        batch, specs={"node_resolution": METRIC_REGISTRY["node_resolution"]},
        reference_levels=False, measure_memory=False,
    )
    assert sorted(tables.metrics["embedding.iterations"]) == [10, 20]


# --------------------------------------------------------------------------- #
#                         Ordering, identity, duplicates                       #
# --------------------------------------------------------------------------- #


def test_cases_run_dataset_major_and_each_dataset_is_built_once(monkeypatch):
    """`batches/batch.py` has two groups per block over one graph list; in spec order a
    one-dataset cache would rebuild tabula's closure. Count the factory calls."""
    calls = []

    def counting(name):
        factory = DATASET_REGISTRY[name]

        def build(**kw):
            calls.append(name)
            return factory(**kw)

        return build

    registries = {
        "graph": {n: counting(n) for n in ("karate_weighted", "karate")},
        "representation": REPRESENTATION_REGISTRY,
        "embedding": {"spectral": EMBEDDING_REGISTRY["spectral"].cls},
    }
    batch = [{"common": {"graph": ["karate_weighted", "karate"], "embedding": ["spectral"]},
              "groups": [{"representation": ["vertex"]}, {"representation": ["edge"]}]}]
    order = [p.case["graph"]["name"] for p in plan_cases(batch, registries)]
    assert order == ["karate", "karate", "karate_weighted", "karate_weighted"]

    released = []
    real_release = runner_module._release
    monkeypatch.setattr(
        runner_module,
        "_release",
        lambda ctx: (released.append(ctx.dataset.name), real_release(ctx)),
    )
    batch_layout_metrics(
        batch, registries=registries, specs={"node_resolution": METRIC_REGISTRY["node_resolution"]},
        reference_levels=False, measure_memory=False,
    )
    assert sorted(calls) == ["karate", "karate_weighted"], "one build per dataset"
    assert released == ["karate", "karate_weighted"], "one release per dataset, caches with it"


#: Trivial factories: plan_cases is pure bookkeeping, so the real registries would only slow it.
FAKE = {
    "graph": {"tabula": lambda **kw: kw},
    "representation": {"vertex": lambda **kw: kw},
    "embedding": {"sgtsne": lambda **kw: kw},
}


def test_cases_differing_only_in_a_fixed_kwarg_are_not_duplicates():
    """Their `BatchKey`s are equal — `run_batch_cases` would have refused the pair."""
    batch = [
        {"graph": [("tabula", {"organ": "liver"}), ("tabula", {"organ": "lung"})],
         "representation": ["vertex"], "embedding": ["sgtsne"]}
    ]
    planned = plan_cases(batch, FAKE)
    assert planned[0].key == planned[1].key
    assert planned[0].identity["key_id"] != planned[1].identity["key_id"]


def test_a_true_duplicate_is_refused_or_skipped():
    batch = [{"graph": ["tabula"], "representation": ["vertex"], "embedding": ["sgtsne"]}] * 2
    with pytest.raises(ValueError, match="share the id"):
        plan_cases(batch, FAKE)
    assert len(plan_cases(batch, FAKE, duplicate_policy="skip")) == 1


def test_the_seed_is_part_of_the_id():
    batch = [{"graph": ["tabula"], "representation": ["vertex"], "embedding": ["sgtsne"]}]
    ids = {plan_cases(batch, FAKE, seed=s)[0].identity["key_id"] for s in (0, 1)}
    assert len(ids) == 2


# --------------------------------------------------------------------------- #
#                            A failure is rows, not an abort                   #
# --------------------------------------------------------------------------- #


class Boom(GraphEmbeddingMethod):
    def __init__(self, seed=0, d=2, **kw):
        self.d = d

    def embed(self, layout_graph, y0=None):
        raise RuntimeError("the backend aborted")


def test_a_failed_embed_becomes_error_rows_and_the_batch_continues():
    registries = {
        "graph": DATASET_REGISTRY,
        "representation": REPRESENTATION_REGISTRY,
        "embedding": {"boom": Boom, "spectral": EMBEDDING_REGISTRY["spectral"].cls},
    }
    batch = [{"graph": ["karate_weighted"], "representation": ["vertex"],
              "embedding": ["boom", "spectral"]}]
    tables = batch_layout_metrics(
        batch, registries=registries, reference_levels=False, measure_memory=False
    )
    metrics = tables.metrics
    boom = metrics[metrics.embedding == "boom"]
    assert len(boom) == UNLABELLED_ROWS + PER_LABEL_SET_ROWS, "every row it would have produced"
    assert set(boom.status) == {"error"}
    assert boom.skip_reason.str.contains("the backend aborted").all()
    assert set(metrics[metrics.embedding == "spectral"].status) == {"ok"}


def test_only_the_lcc_policy_is_accepted():
    with pytest.raises(ValueError, match="'lcc'"):
        batch_layout_metrics(
            [{"graph": ["karate"], "representation": ["vertex"], "embedding": ["spectral"]}],
            disconnected="mask",
        )
