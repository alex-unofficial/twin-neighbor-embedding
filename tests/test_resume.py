r"""Resume, end to end (`DECISIONS.md` (22), spec §4.6) — the failure this step can least afford.

A resume that silently serves stale data produces a well-formed table with the wrong numbers in
it, and nothing downstream can tell. So each of (22)'s promises is asserted by **counting the
work done**, never by timing it:

* a killed and restarted run produces the same rows as one that was never interrupted;
* adding a metric re-runs metrics and **not embeds** — the entire reason layouts persist;
* a bumped ``version`` recomputes exactly that metric, and not a 40 s crossing solve beside it;
* a layout produced under one SHA is reused under another, and the manifest says so.

Every test writes under ``tmp_path``, never the repository's ``runs/``.
"""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

import twin.metrics.registry as registry
import twin.pipelines as pipelines
from twin.metrics.registry import METRIC_REGISTRY
from twin.runs import runner as runner_module
from twin.runs import store as store_module
from twin.runs.runner import batch_layout_metrics
from twin.runs.store import DistributionPolicy

BATCH = [
    {
        "graph": ["karate_weighted"],
        "representation": ["vertex", "edge", "twin"],
        "embedding": ["spectral"],
    }
]

#: The row identity a comparison runs over; values compared exactly — spectral is deterministic.
ROW = ["key_id", "point_set", "label_set", "metric", "variant", "k"]


def run(root, **kw):
    kw.setdefault("reference_levels", False)
    kw.setdefault("measure_memory", False)
    kw.setdefault("allow_dirty", True)
    return batch_layout_metrics(BATCH, root=root, **kw)


def values(tables) -> pd.DataFrame:
    frame = tables.metrics[ROW + ["value", "status"]].copy()
    frame["k"] = frame["k"].astype("float")
    return frame.sort_values(ROW, na_position="first").reset_index(drop=True)


@pytest.fixture
def embeds(monkeypatch):
    """Count `graph_embedding` calls — the runner resolves it from `twin.pipelines` per call."""
    calls = []
    real = pipelines.graph_embedding

    def counting(*args, **kwargs):
        calls.append(kwargs.get("representation_name"))
        return real(*args, **kwargs)

    monkeypatch.setattr(pipelines, "graph_embedding", counting)
    return calls


@pytest.fixture
def solves(monkeypatch):
    """Count crossing solves — the one cost a careless invalidation would multiply."""
    calls = []
    real = registry.crossing_pairs_report
    monkeypatch.setattr(
        registry, "crossing_pairs_report", lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    )
    return calls


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """An uninterrupted run, in memory, that every persisted variant must reproduce."""
    return values(
        batch_layout_metrics(BATCH, reference_levels=False, measure_memory=False)
    )


# --------------------------------------------------------------------------- #
#                         What a persisted run writes                          #
# --------------------------------------------------------------------------- #


def test_a_persisted_run_reproduces_the_in_memory_one(tmp_path, reference):
    tables = run(tmp_path, run_id="r1")
    pd.testing.assert_frame_equal(values(tables), reference)

    run_dir = tmp_path / "r1"
    assert tables.run_dir == run_dir
    for name in ("manifest.json", "metrics.parquet", "diagnostics.parquet",
                 "graph_side.parquet", "log.txt"):
        assert (run_dir / name).is_file(), name
    assert len(list((tmp_path / "layouts").glob("*.npz"))) == 3
    assert len(list((run_dir / "shards").glob("*.metrics.parquet"))) == 3
    # The concatenated table on disk is the one returned.
    on_disk = pd.read_parquet(run_dir / "metrics.parquet")
    assert len(on_disk) == len(tables.metrics)


def test_layouts_live_beside_the_runs_not_inside_one(tmp_path):
    """Plan decision 1: the store is content-addressed, so a second run finds the first's."""
    run(tmp_path, run_id="r1")
    assert not (tmp_path / "r1" / "layouts").exists()
    assert (tmp_path / "layouts").is_dir()


def test_an_existing_run_is_not_silently_overwritten(tmp_path):
    run(tmp_path, run_id="r1")
    with pytest.raises(FileExistsError, match="resume="):
        run(tmp_path, run_id="r1")


def test_a_dirty_tree_is_refused_unless_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "git_state", lambda: {"sha": "a" * 40, "dirty": True})
    with pytest.raises(RuntimeError, match="clean tree"):
        run(tmp_path, run_id="r1", allow_dirty=False)
    tables = run(tmp_path, run_id="r2", allow_dirty=True)
    manifest = json.loads((tables.run_dir / "manifest.json").read_text())
    assert manifest["contributions"][-1]["dirty"] is True


def test_the_default_run_id_is_a_timestamp_and_the_short_sha(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "git_state", lambda: {"sha": "0123456789" * 4, "dirty": False})
    tables = run(tmp_path)
    assert tables.run_dir.name.endswith("Z-01234567")


# --------------------------------------------------------------------------- #
#                     Killed and restarted gives the same rows                 #
# --------------------------------------------------------------------------- #


def test_a_killed_run_resumes_to_the_same_rows(tmp_path, reference, monkeypatch, embeds):
    real = runner_module.evaluate_layout
    seen = []

    def dies_on_the_second_case(*args, **kwargs):
        seen.append(1)
        if len(seen) == 2:
            raise KeyboardInterrupt  # not an Exception: nothing in the runner may swallow it
        return real(*args, **kwargs)

    monkeypatch.setattr(runner_module, "evaluate_layout", dies_on_the_second_case)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, run_id="r1")
    assert not (tmp_path / "r1" / "manifest.json").exists()
    assert len(list((tmp_path / "r1" / "shards").glob("*.metrics.parquet"))) == 1
    assert len(embeds) == 2  # the second case's layout was persisted before it died

    monkeypatch.setattr(runner_module, "evaluate_layout", real)
    embeds.clear()
    tables = run(tmp_path, resume="r1")
    pd.testing.assert_frame_equal(values(tables), reference)
    assert len(embeds) == 1, "only the case that never reached its embed is embedded again"


def test_a_completed_run_resumes_without_any_work(tmp_path, embeds, solves):
    run(tmp_path, run_id="r1")
    embeds.clear(), solves.clear()
    tables = run(tmp_path, resume="r1")
    assert embeds == [] and solves == []
    manifest = json.loads((tables.run_dir / "manifest.json").read_text())
    assert {c["metrics"] for c in manifest["cases"]} == {"current"}
    assert {c["layout"]["source"] for c in manifest["cases"]} == {"reused"}
    assert len(manifest["contributions"]) == 2, "each invocation is a contribution"


# --------------------------------------------------------------------------- #
#                 Adding a metric re-runs metrics and not embeds — (22)        #
# --------------------------------------------------------------------------- #


def test_adding_a_metric_reruns_only_that_metric_and_no_embed(tmp_path, reference, embeds, solves):
    without = {n: s for n, s in METRIC_REGISTRY.items() if n != "edge_length_cv"}
    run(tmp_path, run_id="r1", specs=without)
    assert len(embeds) == 3 and len(solves) == 6  # 3 cases x (straight, curved)

    embeds.clear(), solves.clear()
    tables = run(tmp_path, resume="r1")
    assert embeds == [], "adding a metric must not re-embed — (22)'s entire promise"
    assert solves == [], "nor re-run a crossing solve whose rows cannot have moved"
    pd.testing.assert_frame_equal(values(tables), reference)
    manifest = json.loads((tables.run_dir / "manifest.json").read_text())
    assert {c["metrics"] for c in manifest["cases"]} == {"partial: recomputed ['edge_length_cv']"}


def test_a_new_run_reuses_another_runs_layouts(tmp_path, embeds):
    run(tmp_path, run_id="r1")
    embeds.clear()
    tables = run(tmp_path, run_id="r2")
    assert embeds == []
    assert len(list((tmp_path / "r2" / "shards").glob("*.metrics.parquet"))) == 3


def test_a_bumped_version_recomputes_exactly_that_metric(tmp_path, embeds, solves):
    run(tmp_path, run_id="r1")
    embeds.clear(), solves.clear()

    bumped = dict(METRIC_REGISTRY)
    bumped["silhouette"] = dataclasses.replace(bumped["silhouette"], version=2)
    tables = run(tmp_path, resume="r1", specs=bumped)

    assert embeds == [] and solves == []
    manifest = json.loads((tables.run_dir / "manifest.json").read_text())
    assert {c["metrics"] for c in manifest["cases"]} == {"partial: recomputed ['silhouette']"}
    assert manifest["metric_digests"]["silhouette"] != json.loads(
        (tables.run_dir / "manifest.json").read_text()
    )["metric_digests"]["knn_recall"]
    # Every row now carries the new set digest, kept rows included.
    assert tables.metrics["metric_set_digest"].nunique() == 1


def test_a_changed_crossing_version_reruns_the_solve_once_per_variant(tmp_path, solves):
    run(tmp_path, run_id="r1")
    solves.clear()
    bumped = dict(METRIC_REGISTRY)
    bumped["crossing_number"] = dataclasses.replace(bumped["crossing_number"], version=2)
    tables = run(tmp_path, resume="r1", specs=bumped)
    assert len(solves) == 6, "one solve per (case, variant), not per consumer"
    diagnostics = tables.diagnostics
    provider = diagnostics[(diagnostics.metric == "crossing_pairs")
                           & (diagnostics.diagnostic == "n_candidates")]
    assert len(provider) == 6, "the provider's diagnostics are replaced, not duplicated"


def test_switching_to_an_explicit_k_recomputes_only_the_swept_metrics(tmp_path, solves):
    run(tmp_path, run_id="r1")
    solves.clear()
    tables = run(tmp_path, resume="r1", k=(5, 10))
    assert solves == []
    manifest = json.loads((tables.run_dir / "manifest.json").read_text())
    recomputed = {c["metrics"] for c in manifest["cases"]}
    swept = sorted(n for n, s in METRIC_REGISTRY.items() if "k" in s.sweeps)
    assert recomputed == {f"partial: recomputed {swept}"}
    assert set(tables.metrics.dropna(subset=["k"])["k"]) == {5, 10}


def test_a_bumped_embedding_version_re_embeds(tmp_path, embeds, monkeypatch):
    """Layouts are reused across commits, so a change to a method's output must change the id."""
    from twin.embedding import EMBEDDING_REGISTRY

    run(tmp_path, run_id="r1")
    embeds.clear()
    spectral = EMBEDDING_REGISTRY["spectral"]
    monkeypatch.setitem(EMBEDDING_REGISTRY, "spectral", dataclasses.replace(spectral, version=2))
    run(tmp_path, run_id="r2")
    assert len(embeds) == 3
    assert len(list((tmp_path / "layouts").glob("*.npz"))) == 6


# --------------------------------------------------------------------------- #
#                  Provenance: which commit produced which artifact            #
# --------------------------------------------------------------------------- #


def test_a_layout_made_under_one_sha_is_reused_under_another(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "git_state", lambda: {"sha": "a" * 40, "dirty": False})
    run(tmp_path, run_id="r1")
    monkeypatch.setattr(store_module, "git_state", lambda: {"sha": "b" * 40, "dirty": False})
    tables = run(tmp_path, run_id="r2")

    manifest = json.loads((tables.run_dir / "manifest.json").read_text())
    assert [c["sha"] for c in manifest["contributions"]] == ["b" * 40]
    assert {c["layout"]["git_sha"] for c in manifest["cases"]} == {"a" * 40}
    assert {c["layout"]["source"] for c in manifest["cases"]} == {"reused"}


def test_a_resume_appends_its_contribution(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "git_state", lambda: {"sha": "a" * 40, "dirty": False})
    run(tmp_path, run_id="r1")
    monkeypatch.setattr(store_module, "git_state", lambda: {"sha": "b" * 40, "dirty": False})
    tables = run(tmp_path, resume="r1")
    manifest = json.loads((tables.run_dir / "manifest.json").read_text())
    assert [c["sha"] for c in manifest["contributions"]] == ["a" * 40, "b" * 40]


# --------------------------------------------------------------------------- #
#                                 The manifest                                 #
# --------------------------------------------------------------------------- #


def test_the_manifest_says_which_k_was_used(tmp_path):
    """Decision 5: the rows cannot say whether k was the default, so the manifest must."""
    default = run(tmp_path, run_id="a")
    explicit = run(tmp_path, run_id="b", k=(5,))
    m_default = json.loads((default.run_dir / "manifest.json").read_text())
    m_explicit = json.loads((explicit.run_dir / "manifest.json").read_text())
    assert m_default["k_source"] == "default_k" and m_explicit["k_source"] == "explicit"
    # Same k value — karate_weighted's mean degree is 5 — and still distinguishable.
    assert m_default["k_per_dataset"] == m_explicit["k_per_dataset"] == {"karate_weighted": [5]}


def test_the_manifest_carries_the_plans_fields(tmp_path):
    tables = run(tmp_path, run_id="r1")
    manifest = json.loads((tables.run_dir / "manifest.json").read_text())
    assert manifest["disconnected"] == "lcc"
    assert manifest["distribution_policy"] == "off"
    assert manifest["mem_measurement"] == "batch"
    assert manifest["metric_set_digest"] == tables.metric_set_digest
    assert manifest["metric_versions"]["silhouette"] == 1
    assert manifest["dropped_composites"] == []
    assert manifest["gamma"] == 0.5 and manifest["include_adjacent"] is True
    assert manifest["common_init"] == {"requested": False, "effective": False}
    assert set(manifest["datasets"]) == {"karate_weighted"}
    assert "affinity_derivation" in manifest["datasets"]["karate_weighted"]
    case = manifest["cases"][0]
    assert case["derivation_chain"][0] == "dataset:karate_weighted"
    assert set(case["point_provenance"]) == {"V", "E"}
    assert manifest["spec"] == [{"__dict__": {
        "graph": ["karate_weighted"], "representation": ["vertex", "edge", "twin"],
        "embedding": ["spectral"]}}]


def test_the_manifest_records_a_dropped_composite(tmp_path):
    """§5.3 guard 3 on the one dataset that has one. Minimal metric set: tabula is large."""
    tables = batch_layout_metrics(
        [{"graph": ["tabula"], "representation": ["vertex"], "embedding": ["spectral"]}],
        specs={"node_resolution": METRIC_REGISTRY["node_resolution"]},
        reference_levels=False, measure_memory=False, root=tmp_path, run_id="t",
        allow_dirty=True,
    )
    manifest = json.loads((tables.run_dir / "manifest.json").read_text())
    assert manifest["dropped_composites"] == [
        {"dataset": "tabula{'organ': 'liver'}", "composite": "compartment_cell_type",
         "collapses_to": "cell_ontology_class"}
    ]


# --------------------------------------------------------------------------- #
#                      Failures are retried, distributions kept            #
# --------------------------------------------------------------------------- #


def test_a_failed_embed_is_not_persisted_and_is_retried(tmp_path, monkeypatch):
    real = pipelines.graph_embedding
    def failing(*args, **kwargs):
        if kwargs.get("representation_name") == "edge":
            raise RuntimeError("backend aborted")
        return real(*args, **kwargs)

    monkeypatch.setattr(pipelines, "graph_embedding", failing)
    tables = run(tmp_path, run_id="r1")
    edge = tables.metrics[tables.metrics.representation == "edge"]
    assert set(edge.status) == {"error"}
    assert len(list((tmp_path / "r1" / "shards").glob("*.metrics.parquet"))) == 2

    monkeypatch.setattr(pipelines, "graph_embedding", real)
    tables = run(tmp_path, resume="r1")
    assert set(tables.metrics.status) == {"ok"}


@pytest.mark.parametrize("policy", ["full", "histogram(8)"])
def test_a_distribution_policy_writes_the_arrays(tmp_path, policy):
    tables = run(tmp_path, run_id="r1", distribution_policy=policy)
    files = list((tmp_path / "r1" / "distributions").glob("*.npz"))
    assert len(files) == 3
    with np.load(files[0], allow_pickle=False) as data:
        names = data.files
    assert any(n.startswith("edge_lengths|") for n in names)
    if policy == "full":
        assert "distribution" in set(tables.metrics.status)
    else:
        assert any(n.endswith("|counts") for n in names)


def test_distributions_are_not_written_by_default(tmp_path):
    run(tmp_path, run_id="r1")
    assert not (tmp_path / "r1" / "distributions").exists()


def test_the_policy_parser():
    assert DistributionPolicy.parse("off").keeps_arrays is False
    assert DistributionPolicy.parse("histogram").bins == 100
    assert DistributionPolicy.parse("histogram(16)").bins == 16
    with pytest.raises(ValueError):
        DistributionPolicy.parse("some")
