"""The content-addressed layout store (spec §4.6, plan decision 1).

A layout file must round-trip to a `LayoutResult` that validates, without unpickling anything,
and must rehydrate its dataset from the registry rather than carry it. Every test writes under
``tmp_path``.
"""

import json
import os

import numpy as np
import pytest

from twin import embedding as emb
from twin.datasets import load_dataset
from twin.layoutgraph import twin_representation
from twin.pipelines import graph_embedding
from twin.results import LayoutResult
from twin.runs.layouts import (
    INSTRUMENTATION,
    layout_path,
    load_layout,
    read_layout_meta,
    save_layout,
)


@pytest.fixture(scope="module")
def result():
    return graph_embedding(
        load_dataset("karate_weighted"),
        representation=twin_representation,
        embedding=emb.SpectralLayout,
        representation_name="twin",
        embedding_name="spectral",
    )


def _save(path, result, **kw):
    return save_layout(
        path, result, key_id="v1-x-0", key_json='{"k": 1}', git_sha="abc123",
        created="2026-10-14T00:00:00+00:00", **kw,
    )


def test_a_layout_round_trips_to_a_result_that_validates(tmp_path, result):
    path = _save(layout_path(tmp_path, "v1-x-0"), result)
    loaded = load_layout(path)

    assert isinstance(loaded, LayoutResult)  # __post_init__ ran: shapes validated
    np.testing.assert_array_equal(loaded.V, result.V)
    np.testing.assert_array_equal(loaded.E, result.E)
    np.testing.assert_array_equal(loaded.slopes, result.slopes)
    for name in ("representation", "embedding", "derivation_chain", "seed", "point_provenance",
                 "params", *INSTRUMENTATION):
        assert getattr(loaded, name) == getattr(result, name), name


def test_the_path_is_content_addressed_beside_the_runs(tmp_path):
    assert layout_path(tmp_path, "v1-abc-0123") == tmp_path / "layouts" / "v1-abc-0123.npz"


def test_nothing_in_the_file_needs_pickle(tmp_path, result):
    path = _save(tmp_path / "l.npz", result)
    with np.load(path, allow_pickle=False) as data:
        for name in data.files:
            assert data[name].dtype != object, name
        assert set(data.files) == {"V", "E", "slopes", "meta"}


def test_the_dataset_is_a_reference_not_a_copy(tmp_path, result):
    """§4.6: a Dataset in every layout file would carry every label vector and, for tabula, two
    ~50k-edge matrices — once per representation, seed and λ."""
    path = _save(tmp_path / "l.npz", result)
    meta = read_layout_meta(path)
    assert meta["dataset"] == {"name": "karate_weighted", "params": {"__dict__": {}}}
    # The file is the arrays plus a small JSON blob, not a serialised graph.
    assert os.path.getsize(path) < 20_000


def test_the_rehydrated_dataset_is_the_registry_one(tmp_path, result):
    loaded = load_layout(_save(tmp_path / "l.npz", result))
    fresh = load_layout(_save(tmp_path / "m.npz", result))
    assert loaded.dataset is not result.dataset  # rebuilt, not carried
    assert loaded.dataset.name == "karate_weighted"
    assert list(loaded.G.edges()) == list(result.G.edges())  # the order E is indexed by
    assert loaded.dataset is not fresh.dataset


def test_a_supplied_dataset_is_reused_if_it_matches(tmp_path, result):
    path = _save(tmp_path / "l.npz", result)
    assert load_layout(path, dataset=result.dataset).dataset is result.dataset
    with pytest.raises(ValueError, match="is a layout of karate_weighted"):
        load_layout(path, dataset=load_dataset("karate"))


def test_factory_params_round_trip_with_their_types(tmp_path):
    """digits records `n_neighbors=12` though nobody spelled it out; it must come back an int."""
    dataset = load_dataset("digits")
    fake = LayoutResult(
        V=np.zeros((dataset.n, 2)), E=None, slopes=None, dataset=dataset,
        representation="vertex", embedding="fixture", point_provenance={"V": "primary"},
    )
    loaded = load_layout(_save(tmp_path / "d.npz", fake), dataset=dataset)
    assert loaded.dataset.params == {"n_neighbors": 12}
    meta = read_layout_meta(tmp_path / "d.npz")
    assert meta["dataset"]["params"] == {"__dict__": {"n_neighbors": 12}}
    assert loaded.E is None and loaded.slopes is None


def test_the_producing_sha_and_time_are_recorded(tmp_path, result):
    meta = read_layout_meta(_save(tmp_path / "l.npz", result))
    assert meta["git_sha"] == "abc123"
    assert meta["created"].startswith("2026-10-14")
    assert meta["key_id"] == "v1-x-0"
    assert json.loads(meta["key_json"]) == {"k": 1}


def test_a_failed_write_leaves_no_layout(tmp_path, result, monkeypatch):
    """A kill mid-write must leave no file that looks complete."""
    def boom(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(np, "savez_compressed", boom)
    with pytest.raises(KeyboardInterrupt):
        _save(tmp_path / "layouts" / "l.npz", result)
    assert list((tmp_path / "layouts").iterdir()) == []


def test_an_unknown_format_is_refused(tmp_path, result):
    path = _save(tmp_path / "l.npz", result)
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    meta = json.loads(str(arrays["meta"][()]))
    meta["format"] = 99
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="format"):
        load_layout(path)


# --------------------------------------------------------------------------- #
#                 Decision 4: absolute RSS recorded beside the delta           #
# --------------------------------------------------------------------------- #


def test_the_embed_records_its_absolute_baseline_and_peak(result):
    assert result.rss_baseline_mb is not None and result.rss_peak_mb is not None
    assert result.rss_baseline_mb > 0
    assert result.peak_mem_mb == pytest.approx(
        max(result.rss_peak_mb - result.rss_baseline_mb, 0.0)
    )


def test_no_memory_fields_when_memory_is_not_measured():
    res = graph_embedding(
        load_dataset("karate_weighted"), embedding=emb.SpectralLayout, measure_memory=False
    )
    assert res.peak_mem_mb is None and res.rss_baseline_mb is None and res.rss_peak_mb is None
