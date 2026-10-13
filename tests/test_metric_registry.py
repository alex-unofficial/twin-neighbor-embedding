"""`METRIC_REGISTRY` as a declaration (spec §2.5) — its coverage, its naming, its binders.

Three things here fail silently if they are wrong, which is why each is asserted directly:

* a metric that is never registered is simply never run, and nobody notices until §5;
* a renamed variant produces a well-formed table with the wrong column names;
* a binder that hands a metric the wrong argument order raises only on the cases that reach it.

The evaluation itself — that the bound calls reproduce the golden values — is
`tests/test_evaluate_layout.py`.
"""

import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import twin.metrics as metrics
from twin.datasets import load_dataset
from twin.metrics.context import EvaluationContext
from twin.metrics.crossings import CrossingReport
from twin.metrics.registry import (
    EXCLUDED,
    INPUT_NEEDS,
    METRIC_REGISTRY,
    PROVIDERS,
    MetricSpec,
    declared_rows,
    metric_key,
)
from twin.results import LayoutResult

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "metric_golden.json").read_text())

_spec = importlib.util.spec_from_file_location(
    "make_metric_golden", Path(__file__).parent / "fixtures" / "make_metric_golden.py"
)
generator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generator)

#: Keys the registry produces at the golden file's k that the file does not pin. Listed, so that
#: a rename shows up on *both* sides of the comparison rather than as a silent extra.
NOT_IN_GOLDEN = {
    # Eq. (58), the layout half of the weighted pairing (41) — registered at step 7, after the
    # golden file was last extended.
    "knn_accuracy@distance@k=5",
    "knn_accuracy@distance@k=10",
}


def registry_keys(ks=tuple(GOLDEN["k"])) -> set[str]:
    return {
        metric_key(name, column, k)
        for name, spec in METRIC_REGISTRY.items()
        for variant in spec.variants
        for column, k in declared_rows(spec, variant, ks)
    }


# --------------------------------------------------------------------------- #
#                                  Coverage                                    #
# --------------------------------------------------------------------------- #


def test_every_export_is_registered_or_excluded_with_a_reason():
    """The mirror of `test_metrics_surface.py`'s one-sided pin."""
    registered = {spec.fn.__name__ for spec in METRIC_REGISTRY.values()} | {
        spec.report_fn.__name__ for spec in METRIC_REGISTRY.values() if spec.report_fn
    }
    exported = set(metrics.__all__)
    unaccounted = exported - registered - set(EXCLUDED)
    assert not unaccounted, f"exported, never run, and not excused: {sorted(unaccounted)}"
    assert not (registered & set(EXCLUDED)), "both registered and excluded"
    assert set(EXCLUDED) <= exported, f"stale exclusions: {sorted(set(EXCLUDED) - exported)}"
    assert all(reason.strip() for reason in EXCLUDED.values())


def test_every_registered_function_is_exported():
    """A registered metric outside `__all__` is reachable only through the registry."""
    for name, spec in METRIC_REGISTRY.items():
        assert spec.fn.__name__ in metrics.__all__, name
        assert spec.fn is getattr(metrics, spec.fn.__name__), name


def test_registry_names_are_the_function_names():
    """The `metric` column *is* the function name — the golden file's convention."""
    for name, spec in METRIC_REGISTRY.items():
        assert spec.fn.__name__ == name


def test_every_golden_key_is_produced_by_a_registry_entry():
    """Catches a renamed variant or a dropped metric immediately. 11 of the 44 come out of
    declared reductions rather than names — `detour_ratio@curved_mean`, `edge_lengths@mean`,
    the six distribution order statistics."""
    golden = set(GOLDEN["values"])
    produced = registry_keys()
    assert golden - produced == set(), f"golden keys no entry produces: {sorted(golden - produced)}"
    assert produced - golden == NOT_IN_GOLDEN


def test_eleven_golden_keys_come_from_declared_reductions():
    reduced = {
        metric_key(name, column, k)
        for name, spec in METRIC_REGISTRY.items()
        if spec.reductions
        for variant in spec.variants
        for column, k in declared_rows(spec, variant, (5, 10))
    }
    assert len(reduced & set(GOLDEN["values"])) == 11


def test_every_need_is_a_provider_or_an_input():
    for name, spec in METRIC_REGISTRY.items():
        unknown = set(spec.needs) - set(PROVIDERS) - INPUT_NEEDS
        assert not unknown, f"{name}: unknown needs {sorted(unknown)}"


def test_the_three_crossing_metrics_share_one_provider():
    """A flat registry of independent entries would solve three times — 40 s each on tabula."""
    sharing = [n for n, s in METRIC_REGISTRY.items() if "crossing_pairs" in s.needs]
    assert sorted(sharing) == [
        "crossing_number",
        "crossings_per_edge_pair",
        "length_weighted_crossings",
    ]
    variants = {tuple(METRIC_REGISTRY[n].variants) for n in sharing}
    params = {tuple(sorted(METRIC_REGISTRY[n].params.items())) for n in sharing}
    assert len(variants) == len(params) == 1, "a shared provider needs identical variant params"


def test_section_5_3_guard_1_marks_adr_cr_and_lwcr():
    marked = {n for n, s in METRIC_REGISTRY.items() if s.within_graph_only}
    assert marked == {"adjacency_distance_ratio", "crossing_number", "length_weighted_crossings"}


def test_graph_side_metrics_are_the_label_validity_checks():
    """§5.3 guard 2: these go to their own table, computed once per dataset."""
    graph_side = {n for n, s in METRIC_REGISTRY.items() if s.graph_side}
    assert graph_side == {"silhouette_graph", "graph_neighborhood_accuracy", "majority_class_rate"}


def test_the_foils_are_not_reported():
    """NR is the foil VNR is defined against; the two stresses are what SNS is contrasted with."""
    foils = {n for n, s in METRIC_REGISTRY.items() if not s.reported}
    assert foils == {"node_resolution", "raw_stress", "normalized_stress"}


def test_every_metric_applies_to_the_vertex_layout_only():
    """A pairwise metric on tabula's E is a 9.7 GB condensed vector: an OOM, not a slow case."""
    assert {s.applies_to for s in METRIC_REGISTRY.values()} == {("V",)}


def test_the_headline_graph_vote_is_uniform():
    """(41): the reported Acc* is uniform voting with the fractional tie rule."""
    variants = METRIC_REGISTRY["graph_neighborhood_accuracy"].variants
    assert dict(variants["uniform"]) == {"weighting": "uniform", "tie_rule": "fractional"}


# --------------------------------------------------------------------------- #
#                         Declarations are self-consistent                     #
# --------------------------------------------------------------------------- #


def test_a_distribution_without_reductions_is_refused():
    with pytest.raises(ValueError, match="reductions"):
        MetricSpec(fn=len, bind=len, kind="visual", returns="distribution")


def test_report_fn_and_value_field_come_together():
    with pytest.raises(ValueError, match="together"):
        MetricSpec(fn=len, bind=len, kind="visual", report_fn=len)


def test_a_spec_is_immutable():
    spec = METRIC_REGISTRY["knn_recall"]
    with pytest.raises(Exception):
        spec.version = 2
    with pytest.raises(TypeError):
        spec.variants[None]["k"] = 3


# --------------------------------------------------------------------------- #
#                      Every binder satisfies its signature                    #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def golden_case():
    dataset = load_dataset("karate_weighted")
    ctx = EvaluationContext.for_dataset(dataset)
    V = generator.karate_structured_layout(dataset.graph)
    result = LayoutResult(
        V=V,
        E=generator.karate_edge_points(dataset.graph, V),
        slopes=None,
        dataset=dataset,
        representation="vertex",
        embedding="fixture",
        point_provenance={"V": "primary", "E": "derived"},
    )
    return ctx, result


@pytest.mark.parametrize("name", sorted(METRIC_REGISTRY))
def test_every_binder_satisfies_its_functions_signature(name, golden_case):
    ctx, result = golden_case
    spec = METRIC_REGISTRY[name]
    for variant in spec.variants:
        params = spec.variant_params(variant)
        if "k" in spec.sweeps:
            params["k"] = (5, 10)
        if "labels" in spec.needs:
            params["labels"] = ctx.dataset.labels("club")
        if "crossing_pairs" in spec.needs:
            params["crossing_pairs"] = CrossingReport(crossings=np.empty((0, 6)))
        args, kwargs = spec.bind(ctx, None if spec.graph_side else result, variant, params)
        inspect.signature(spec.call).bind(*args, **kwargs)
        inspect.signature(spec.fn).bind(*args, **kwargs)
