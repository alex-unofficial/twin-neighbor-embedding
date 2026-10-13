"""The run manifest's building blocks (spec §4.6). End-to-end manifests are in `test_resume.py`."""

import json

import numpy as np
import pytest

from twin.datasets import load_dataset
from twin.runs.manifest import (
    MANIFEST_VERSION,
    _jsonable,
    dataset_entry,
    read_manifest,
    static_fields,
    write_manifest,
)
from twin.runs.provenance import git_state, now, package_versions


def test_the_dataset_entry_records_sources_and_the_derivation_rule():
    entry = dataset_entry(load_dataset("digits"))
    assert entry["name"] == "digits"
    assert entry["params"] == {"__dict__": {"n_neighbors": 12}}
    assert entry["stored_sources"]["affinity"] == "derived"
    # §4.6's d_scale lives in the derivation rule, which is where it is computed.
    assert "d_scale=" in entry["affinity_derivation"]
    assert entry["sparsity"].startswith("12-NN")
    json.dumps(entry)  # serialisable as it stands


def test_the_static_fields_are_section_4_6s():
    fields = static_fields()
    assert fields["gamma"] == 0.5
    assert fields["include_adjacent"] is True
    assert fields["vnr_normalisation"] == {
        "default": "median", "variants": {"max": "max", "mean": "mean"}
    }
    assert fields["metric_versions"]["silhouette"] == 1
    assert fields["key_version"] == "v1"


def test_a_manifest_round_trips_and_refuses_another_version(tmp_path):
    path = write_manifest(tmp_path / "manifest.json", {**static_fields(), "x": np.float64(1.5)})
    manifest = read_manifest(path)
    assert manifest["x"] == 1.5 and manifest["manifest_version"] == MANIFEST_VERSION
    path.write_text(json.dumps({**manifest, "manifest_version": 0}))
    with pytest.raises(ValueError, match="manifest version"):
        read_manifest(path)


def test_non_finite_values_do_not_produce_invalid_json(tmp_path):
    path = write_manifest(tmp_path / "m.json", {**static_fields(), "rho": float("nan")})
    json.loads(path.read_text())  # strict JSON has no NaN literal


def test_jsonable_handles_dataclasses_and_arrays():
    from twin.datasets import check_weight_consistency

    report = check_weight_consistency(load_dataset("tabula"))
    out = _jsonable(report)
    assert out["dataset"] == "tabula" and isinstance(out["global_spearman"], float)
    assert _jsonable(np.arange(3)) == [0, 1, 2]


def test_provenance_reads_this_repository():
    state = git_state()
    assert state["sha"] is None or len(state["sha"]) == 40
    assert "numpy" in package_versions() and "python" in package_versions()
    assert now().endswith("+00:00")
