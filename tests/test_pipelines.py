"""Spec §3.1 / §3.2 — graph_embedding returns a LayoutResult, and instruments the run."""

import networkx as nx
import numpy as np
import pytest

from twin import embedding as emb
from twin.layoutgraph import REPRESENTATION_REGISTRY
from twin.pipelines import batch_graph_embedding, graph_embedding
from twin.results import LayoutResult

REPRESENTATIONS = [(name, REPRESENTATION_REGISTRY[name]) for name in ("vertex", "edge", "twin")]


@pytest.mark.parametrize(("name", "cls"), REPRESENTATIONS)
def test_returns_layout_result_in_canonical_orientation(karate_dataset, name, cls):
    res = graph_embedding(karate_dataset, representation=cls, embedding=emb.SpringLayout)

    assert isinstance(res, LayoutResult)
    assert res.V.shape == (karate_dataset.n, 2)
    assert res.E.shape == (karate_dataset.m, 2)
    assert res.slopes.shape == (karate_dataset.m,)


@pytest.mark.parametrize(("name", "cls"), REPRESENTATIONS)
def test_n_points_reflects_what_was_actually_laid_out(karate_dataset, name, cls):
    """Spec §3.2: n for A_V, m for A_E, n+m for A_tne, visible without recomputation."""
    n, m = karate_dataset.n, karate_dataset.m
    expected = {"vertex": n, "edge": m, "twin": n + m}[name]

    res = graph_embedding(karate_dataset, representation=cls, embedding=emb.SpringLayout)
    assert res.n_points == expected


@pytest.mark.parametrize(("name", "cls"), REPRESENTATIONS)
def test_provenance_is_carried_into_the_result(karate_dataset, name, cls):
    res = graph_embedding(karate_dataset, representation=cls, embedding=emb.SpringLayout)
    assert res.point_provenance == cls(karate_dataset).point_provenance


def test_timings_are_split_between_representation_and_embed(karate_dataset):
    res = graph_embedding(
        karate_dataset, representation=REPRESENTATION_REGISTRY["twin"],
        embedding=emb.SpringLayout,
    )
    assert res.representation_s > 0
    assert res.runtime_s > 0
    # Distinct measurements, not the same clock reading twice.
    assert res.representation_s != res.runtime_s


def test_memory_is_measured_by_default(karate_dataset):
    """RSS sampling is on by default: it is cheap and it sees into the C backend."""
    res = graph_embedding(karate_dataset, embedding=emb.SGtSNELayout)
    assert res.peak_mem_mb is not None and res.peak_mem_mb >= 0

    assert graph_embedding(karate_dataset, embedding=emb.SGtSNELayout, measure_memory=False).peak_mem_mb is None


def test_rss_sampler_sees_allocation_outside_python_bookkeeping():
    """The property tracemalloc could not provide: a large allocation shows up as RSS."""
    from twin.instrumentation import RSSSampler

    with RSSSampler(interval=0.001) as mem:
        block = np.ones(12_000_000, dtype=np.float64)  # ~96 MB, touched
        block[::1000] = 2.0
    del block

    assert mem.n_samples >= 2
    assert mem.peak_mb >= mem.baseline_mb
    assert mem.delta_mb > 50, f"expected to observe the ~96 MB allocation, saw {mem.delta_mb:.1f}"


def test_rss_sampler_brackets_even_a_very_short_block():
    from twin.instrumentation import RSSSampler

    with RSSSampler(interval=10.0) as mem:
        pass
    assert mem.n_samples == 2, "entry and exit samples must always be taken"
    assert mem.delta_mb >= 0


def test_rss_sampler_can_be_disabled():
    from twin.instrumentation import RSSSampler

    with RSSSampler(enabled=False) as mem:
        pass
    assert mem.n_samples == 0 and mem.delta_mb == 0.0


def test_records_identity_and_parameters(karate_dataset):
    res = graph_embedding(
        karate_dataset,
        representation=REPRESENTATION_REGISTRY["twin"],
        embedding=emb.SGtSNELayout,
        representation_kw={"alpha": 0.5, "k": 3},
        embedding_kw={"lambda_par": 2.0},
        representation_name="twin",
        embedding_name="sgtsne",
        seed=7,
    )
    assert (res.representation, res.embedding, res.seed) == ("twin", "sgtsne", 7)
    assert res.params["representation_kw"] == {"alpha": 0.5, "k": 3}
    assert res.params["embedding_kw"] == {"lambda_par": 2.0}


def test_undeclared_provenance_is_refused(karate_dataset):
    def undeclared(dataset, **kw):
        graph = REPRESENTATION_REGISTRY["vertex"](dataset, **kw)
        graph.point_provenance = {}
        return graph

    with pytest.raises(TypeError, match="does not declare point_provenance"):
        graph_embedding(karate_dataset, representation=undeclared, embedding=emb.SpringLayout)


def test_batch_returns_layout_results_with_registry_names():
    spec = [
        {
            "graph": "karate_weighted",
            "representation": ["vertex", "edge", "twin"],
            "embedding": "spring",
        }
    ]
    results = batch_graph_embedding(spec)

    assert len(results) == 3
    assert all(isinstance(r, LayoutResult) for r in results.values())
    assert {r.representation for r in results.values()} == {"vertex", "edge", "twin"}
    assert {r.embedding for r in results.values()} == {"spring"}
    assert {r.n_points for r in results.values()} == {34, 78, 34 + 78}


def test_labels_apply_to_a_layout_of_any_representation():
    """The load-bearing check for the whole layering (spec §3.4, `DECISIONS.md` (11)).

    A_E's nodes are edges and A_tne's are edges *and* vertices, so a length-n label vector never
    reaches them — but `vertex_and_edge_points` returns points in the **original** indexing, so
    the dataset's own labels apply afterwards with no length mismatch and no special case at the
    call site. If this needed one, the layering would be wrong.
    """
    from twin.datasets import load_dataset
    from twin.metrics import knn_accuracy

    dataset = load_dataset("karate_weighted")
    labels = dataset.labels("club")

    for name, builder in REPRESENTATION_REGISTRY.items():
        res = graph_embedding(dataset, representation=builder, embedding=emb.SpringLayout)

        assert res.V.shape[0] == labels.shape[0] == dataset.n
        assert res.n_points == {"vertex": 34, "edge": 78, "twin": 112}[name]
        assert 0.0 <= knn_accuracy(res.V, labels, 5) <= 1.0


def test_the_result_records_the_derivation_chain():
    """§3.5: the manifest records the real chain, read off `parent`, not a flag kept in sync."""
    from twin.datasets import load_dataset

    res = graph_embedding(
        load_dataset("karate_weighted"),
        representation=REPRESENTATION_REGISTRY["twin"],
        embedding=emb.SpringLayout,
    )
    assert res.derivation_chain[0] == "dataset:karate_weighted"
    assert res.derivation_chain[-1].startswith("A_tne:")
